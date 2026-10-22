"""Task 09 (T09.1): structural validation of finalist geometries.

``validate_structure`` runs the in-house checks on every finalist relaxed geometry and, when available, the
optional external tools. Each check records its tool/version, parameters, a parseable status and raw evidence.
In-house: post-relaxation slot/node mapping, residual force and stress (from the CalculationResult), minimum
interatomic distance, metal coordination numbers against the parent framework and cell distortion against the
built geometry. External (optional, never installed by this package): MOFChecker (``mofchecker`` Python package)
and Zeo++ (``network`` executable). A missing tool is SKIPPED_TOOL_UNAVAILABLE, never "passed"; tool success is
not proof of synthesizability. Phonon/elastic stability, PXRD and NMR comparison are not implemented.
"""
from __future__ import annotations

import importlib
import importlib.util
import math
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from .cif import write_cif
from .contract import StageStatus, make_record
from .framework import METALS, bond_graph, map_to_parent_slots, min_interatomic_distance

SKIPPED_TOOL = "SKIPPED_TOOL_UNAVAILABLE"
NOT_IMPLEMENTED = ("phonon_elastic_stability", "pxrd_comparison", "nmr_comparison", "crystalnets_topology")


def _structure(geometry: dict) -> dict:
    return {"lattice": geometry["lattice_A"],
            "atoms": [{"element": e, "frac": tuple(f)} for e, f in zip(geometry["elements"], geometry["frac_coords"])]}


def _lengths_angles(lattice) -> tuple:
    lengths = [math.sqrt(sum(x * x for x in v)) for v in lattice]

    def angle(u, v):
        return math.degrees(math.acos(sum(a * b for a, b in zip(u, v)) / (math.sqrt(sum(a * a for a in u)) * math.sqrt(sum(b * b for b in v)))))
    return lengths, [angle(lattice[1], lattice[2]), angle(lattice[0], lattice[2]), angle(lattice[0], lattice[1])]


def metal_coordination(geometry: dict, tolerance: float) -> dict:
    structure = _structure(geometry)
    adjacency = bond_graph(structure, tolerance)
    numbers = Counter()
    for atom, neighbours in zip(structure["atoms"], adjacency):
        if atom["element"] in METALS:
            numbers[(atom["element"], len(neighbours))] += 1
    return {f"{el}:{cn}": n for (el, cn), n in sorted(numbers.items())}


def mofchecker_check(cif_text: str) -> dict:
    if importlib.util.find_spec("mofchecker") is None:
        return {"status": SKIPPED_TOOL, "tool": "mofchecker"}
    try:
        module = importlib.import_module("mofchecker")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "structure.cif"
            path.write_text(cif_text)
            checker = module.MOFChecker.from_cif(str(path))
            descriptors = checker.get_mof_descriptors()
        return {"status": StageStatus.COMPLETED.value, "tool": "mofchecker",
                "version": getattr(module, "__version__", None),
                "raw": {k: (v if isinstance(v, (bool, int, float, str, type(None))) else str(v)) for k, v in descriptors.items()}}
    except Exception as exc:  # noqa: BLE001 - a tool failure is recorded, not hidden
        return {"status": StageStatus.FAILED.value, "tool": "mofchecker", "error": f"{type(exc).__name__}: {exc}"}


def zeopp_check(cif_text: str, probe_radius_A: float = 1.2) -> dict:
    exe = shutil.which("network")
    if exe is None:
        return {"status": SKIPPED_TOOL, "tool": "zeo++"}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            cif, res = Path(tmp) / "structure.cif", Path(tmp) / "structure.res"
            cif.write_text(cif_text)
            proc = subprocess.run([exe, "-ha", "-res", str(res), str(cif)], capture_output=True, text=True, timeout=600)
            raw = res.read_text() if res.exists() else ""
        fields = raw.split()
        values = {"largest_included_sphere_A": float(fields[1]), "largest_free_sphere_A": float(fields[2]),
                  "largest_included_sphere_along_free_path_A": float(fields[3])} if len(fields) >= 4 else None
        return {"status": StageStatus.COMPLETED.value if values else StageStatus.FAILED.value, "tool": "zeo++",
                "executable": exe, "parameters": {"mode": "-ha -res", "probe_radius_A": probe_radius_A},
                "raw": raw, "values": values, "stderr": proc.stderr[-500:]}
    except Exception as exc:  # noqa: BLE001
        return {"status": StageStatus.FAILED.value, "tool": "zeo++", "error": f"{type(exc).__name__}: {exc}"}


def validate_structure(spec, calculation: dict, built_geometry: dict) -> dict:
    """validate_structure(geometry_id, validation_protocol_id) -> StructureValidationReport for one finalist."""
    builder = spec.physical["protocols"]["builder"]
    tolerances = spec.physical["tolerances"]
    relaxed = calculation["relaxed_geometry"]
    checks = {}
    mapping = map_to_parent_slots(_structure(relaxed), spec.physical["builder_info"][calculation["topology_id"]],
                                  builder["bond_tolerance_A"], builder["slot_match_tolerance_A"])
    checks["post_relaxation_slot_mapping"] = {"status": StageStatus.COMPLETED.value if mapping["retained"] else StageStatus.FAILED.value,
                                              "tool": "mofpipe.framework.map_to_parent_slots", "reason": mapping["reason"],
                                              "parameters": {"bond_tolerance_A": builder["bond_tolerance_A"],
                                                             "slot_match_tolerance_A": builder["slot_match_tolerance_A"]}}
    conv = calculation.get("convergence") or {}
    relaxed_by_protocol = not str(calculation.get("geometry_purpose", "")).startswith("unrelaxed")
    checks["residual_force_stress"] = {
        "status": (StageStatus.COMPLETED.value if relaxed_by_protocol else "NOT_APPLICABLE_UNRELAXED_GEOMETRY"),
        "tool": "mofpipe.dft", "max_force_eV_per_A": conv.get("max_force_eV_per_A"),
        "max_stress_kbar": conv.get("max_stress_kbar"), "force_tolerance_eV_per_A": tolerances["force_tolerance_eV_per_angstrom"]}
    dmin = min_interatomic_distance(_structure(relaxed), 3.0)
    checks["min_interatomic_distance"] = {"status": (StageStatus.COMPLETED.value if dmin >= builder["min_interatomic_distance_A"]
                                                     else StageStatus.FAILED.value),
                                          "tool": "mofpipe.framework", "value_A": round(dmin, 4),
                                          "threshold_A": builder["min_interatomic_distance_A"]}
    parent = spec.physical["lattices"][calculation["topology_id"]]["structure"]
    parent_geometry = {"lattice_A": parent["lattice"], "elements": [a["element"] for a in parent["atoms"]],
                       "frac_coords": [a["frac"] for a in parent["atoms"]]}
    cn_relaxed = metal_coordination(relaxed, builder["bond_tolerance_A"])
    cn_parent = metal_coordination(parent_geometry, builder["bond_tolerance_A"])
    checks["metal_coordination"] = {"status": StageStatus.COMPLETED.value if cn_relaxed == cn_parent else StageStatus.FAILED.value,
                                    "tool": "mofpipe.framework.bond_graph", "relaxed": cn_relaxed, "parent": cn_parent}
    (l0, a0), (l1, a1) = _lengths_angles(built_geometry["lattice_A"]), _lengths_angles(relaxed["lattice_A"])
    checks["cell_distortion"] = {"status": StageStatus.COMPLETED.value, "tool": "mofpipe.validation",
                                 "length_change_percent": [round(100 * (b - a) / a, 3) for a, b in zip(l0, l1)],
                                 "angle_change_deg": [round(b - a, 3) for a, b in zip(a0, a1)]}
    cif = write_cif(calculation["configuration_id"].replace(":", "_"), relaxed["lattice_A"], relaxed["elements"],
                    relaxed["frac_coords"])
    checks["mofchecker"] = mofchecker_check(cif)
    checks["zeopp_pores"] = zeopp_check(cif)
    for name in NOT_IMPLEMENTED:
        checks[name] = {"status": StageStatus.NOT_RUN.value, "reason": "not implemented in this build"}
    failed = sorted(k for k, v in checks.items() if v["status"] == StageStatus.FAILED.value)
    return make_record("StructureValidationReport", spec.provenance, calculation_result_id=calculation["artifact_id"],
                       configuration_id=calculation["configuration_id"], topology_id=calculation["topology_id"],
                       checks=checks, failed_checks=failed,
                       status=StageStatus.COMPLETED.value if not failed else StageStatus.FAILED.value,
                       note="tool success is not proof of synthesizability; observations are not simulated here")
