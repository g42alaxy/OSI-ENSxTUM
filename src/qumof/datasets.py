"""Integrator glue: energy-record compatibility, label ingestion, frozen splits.

These are the producers of DatasetSnapshot and ValidationSplit that the task
briefs consume (fit_hamiltonian's dataset_snapshot_id, validate_hamiltonian's
split_id) but never define.
"""
from __future__ import annotations

import math
from collections import Counter
from fractions import Fraction

from .contract import PipelineError, content_hash, exact, make_record
from .structures import configuration_id, n_ones_of, orbit_id, validate_occupation

SUPPORTED_UNITS = ("eV",)
PER_FORMULA_UNIT = "per_abstract_12_slot_formula_unit"
PER_CELL = "per_cell"
PER_PHYSICAL_FORMULA_UNIT = "per_formula_unit"  # physical projects: verified formula-unit count from the inventory
UNIFORM_FIELDS = ("fidelity", "energy_unit", "normalization", "comparison_group")


def check_energy_compatibility(rows: list, provenance: str = "SYNTHETIC") -> None:
    """Refuse to pool energies with mixed fidelity/unit/normalisation/group or failed status."""
    if not rows:
        raise PipelineError("EMPTY_ENERGY_SET")
    for name in UNIFORM_FIELDS:
        values = {row.get(name) for row in rows}
        if len(values) > 1:
            raise PipelineError("INCOMPATIBLE_" + name.upper(), str(sorted(map(str, values))))
    for row in rows:
        if row.get("status") != "SUCCEEDED":
            raise PipelineError("NON_SUCCESSFUL_CALCULATION", f"{row.get('configuration_id')}: {row.get('status')}")
        if row.get("provenance") != provenance:
            raise PipelineError("PROVENANCE_MISMATCH", f"{row.get('configuration_id')}: {row.get('provenance')}")
    if rows[0].get("energy_unit") not in SUPPORTED_UNITS:
        raise PipelineError("UNSUPPORTED_ENERGY_UNIT", str(rows[0].get("energy_unit")))


def normalize_cell_energy(total_energy, n_formula_units) -> Fraction:
    if type(n_formula_units) is not int or n_formula_units <= 0:
        raise PipelineError("INVALID_FORMULA_UNIT_COUNT", repr(n_formula_units))
    return exact(total_energy) / n_formula_units


def ingest_labels(spec, rows: list, role: str, source: str) -> dict:
    """Integrator glue tool: validated labels -> immutable DatasetSnapshot."""
    check_energy_compatibility(rows, spec.provenance)
    normalization = rows[0]["normalization"]
    if normalization not in (PER_FORMULA_UNIT, PER_CELL, PER_PHYSICAL_FORMULA_UNIT):
        raise PipelineError("UNSUPPORTED_NORMALIZATION", normalization)
    if (normalization == PER_PHYSICAL_FORMULA_UNIT) != (spec.project_type != "synthetic_fixture"):
        raise PipelineError("INCOMPATIBLE_NORMALIZATION", f"{normalization} in a {spec.project_type} project")
    entries, seen = [], set()
    for row in rows:
        topo = spec.topologies.get(row.get("topology_id"))
        if topo is None:
            raise PipelineError("UNKNOWN_TOPOLOGY", str(row.get("topology_id")))
        bits = validate_occupation(row.get("bits"), topo.n_sites, n_ones_of(spec, topo))
        cid = configuration_id(topo, bits)
        if row.get("configuration_id") != cid:
            raise PipelineError("CONFIGURATION_ID_MISMATCH", str(row.get("configuration_id")))
        if row.get("orbit_id") != orbit_id(topo, bits):
            raise PipelineError("ORBIT_ID_MISMATCH", cid)
        if cid in seen:
            raise PipelineError("DUPLICATE_LABEL", cid)
        seen.add(cid)
        n_fu = row.get("n_formula_units")
        if n_fu != topo.n_formula_units:
            raise PipelineError("FORMULA_UNIT_MISMATCH", f"{cid}: {n_fu} != {topo.n_formula_units}")
        energy = exact(row["energy_eV"])
        normalized = normalize_cell_energy(energy, n_fu) if normalization == PER_CELL else energy
        entry = {"configuration_id": cid, "orbit_id": row["orbit_id"], "topology_id": topo.topology_id,
                 "bits": list(bits), "energy_exact": str(normalized), "energy_eV": float(normalized)}
        if row.get("calculation_result_id"):
            entry["calculation_result_id"] = row["calculation_result_id"]
        entries.append(entry)
    entries.sort(key=lambda e: e["configuration_id"])
    target = PER_PHYSICAL_FORMULA_UNIT if normalization == PER_PHYSICAL_FORMULA_UNIT else PER_FORMULA_UNIT
    return make_record(
        "DatasetSnapshot", spec.provenance, role=role, source=source, fidelity=rows[0]["fidelity"],
        energy_unit="eV", normalization=target, comparison_group=rows[0]["comparison_group"],
        n_labels=len(entries), per_topology=dict(sorted(Counter(e["topology_id"] for e in entries).items())),
        entries=entries,
    )


def merge_snapshots(spec, base: dict, extra: dict, role: str) -> dict:
    """Append newly acquired labels; target definition and fidelity must be identical."""
    if base["provenance"] != extra["provenance"]:
        raise PipelineError("PROVENANCE_MISMATCH", f"{base['provenance']} vs {extra['provenance']}")
    for name in ("fidelity", "energy_unit", "normalization", "comparison_group"):
        if base[name] != extra[name]:
            raise PipelineError("INCOMPATIBLE_" + name.upper(), f"{base[name]} vs {extra[name]}")
    entries = {e["configuration_id"]: e for e in base["entries"]}
    for e in extra["entries"]:
        if e["configuration_id"] in entries:
            raise PipelineError("DUPLICATE_LABEL", e["configuration_id"])
        entries[e["configuration_id"]] = e
    merged = sorted(entries.values(), key=lambda e: e["configuration_id"])
    return make_record("DatasetSnapshot", spec.provenance, role=role, source=f"merge({base['artifact_id']},{extra['artifact_id']})",
                       fidelity=base["fidelity"], energy_unit="eV", normalization=base["normalization"],
                       comparison_group=base["comparison_group"], n_labels=len(merged),
                       per_topology=dict(sorted(Counter(e["topology_id"] for e in merged).items())), entries=merged)


def freeze_split(spec, spaces: dict) -> dict:
    """Integrator glue tool: frozen symmetry-orbit split. Holdout IDs only; never holdout labels."""
    per_topology = {}
    for tid, space in spaces.items():
        manifest = spec.split_manifest.get(tid)
        if manifest is None:
            raise PipelineError("SCIENTIFIC_SPEC_MISSING", f"split for {tid}")
        groups = {k: set(manifest[k]) for k in ("training", "development", "holdout")}
        for k, ids in groups.items():
            if len(ids) != len(manifest[k]):
                raise PipelineError("DUPLICATE_SPLIT_ENTRY", f"{tid}:{k}")
        if (groups["training"] & groups["development"] or groups["training"] & groups["holdout"]
                or groups["development"] & groups["holdout"]):
            raise PipelineError("ORBIT_LEAKAGE", tid)
        all_orbits = set(space["orbit_sizes"])
        if set().union(*groups.values()) != all_orbits:
            raise PipelineError("SPLIT_DOES_NOT_PARTITION_ORBITS", tid)
        per_topology[tid] = {k: sorted(v) for k, v in groups.items()}
    return make_record("ValidationSplit", spec.provenance, split_unit="symmetry_orbit", per_topology=per_topology,
                       counts={t: {k: len(v) for k, v in g.items()} for t, g in per_topology.items()},
                       holdout_labels_available_to_pipeline=False)


def check_snapshot_against_split(snapshot: dict, split: dict, allowed_group: str) -> None:
    for e in snapshot["entries"]:
        groups = split["per_topology"][e["topology_id"]]
        if e["orbit_id"] in groups["holdout"]:
            raise PipelineError("HOLDOUT_LABEL_IN_LEARNER_DATA", e["configuration_id"])
        if e["orbit_id"] not in groups[allowed_group]:
            raise PipelineError("ORBIT_LEAKAGE", f"{e['configuration_id']} not in {allowed_group}")


def check_no_holdout(snapshot: dict, split: dict) -> None:
    """Acquired labels (refinement) may come from training or unlabelled orbits, never from the frozen holdout."""
    for e in snapshot["entries"]:
        if e["orbit_id"] in split["per_topology"][e["topology_id"]]["holdout"]:
            raise PipelineError("HOLDOUT_LABEL_IN_LEARNER_DATA", e["configuration_id"])


def freeze_physical_split(spec, spaces: dict) -> dict:
    """Seeded orbit split chosen BEFORE any label exists: labelled orbits -> training (CV) / frozen final holdout.

    Orbits not selected for labelling stay ``unlabelled``. The holdout IDs are fixed here, blind to energies.
    """
    policy = spec.physical["protocols"]["model_selection"]
    seed, fraction = policy["seed"], Fraction(str(policy["holdout_fraction"]))
    if not 0 < fraction < 1:
        raise PipelineError("SCIENTIFIC_SPEC_MISSING", "protocols.model_selection.holdout_fraction must be in (0, 1)")
    per_topology = {}
    for tid, space in spaces.items():
        orbits = sorted(space["orbit_sizes"], key=lambda o: content_hash([seed, "label", o]))
        n_label = policy["label_orbits_per_topology"]
        n_label = len(orbits) if n_label == "all" else n_label
        if type(n_label) is not int or n_label <= 0:
            raise PipelineError("SCIENTIFIC_SPEC_MISSING", "protocols.model_selection.label_orbits_per_topology")
        labelled = sorted(orbits[:n_label], key=lambda o: content_hash([seed, "holdout", o]))
        n_holdout = max(1, math.ceil(fraction * len(labelled)))
        if len(labelled) - n_holdout < 2:
            raise PipelineError("INSUFFICIENT_LABELS_FOR_SPLIT", f"{tid}: {len(labelled)} labelled orbits")
        per_topology[tid] = {"training": sorted(labelled[n_holdout:]), "development": [],
                             "holdout": sorted(labelled[:n_holdout]), "unlabelled": sorted(orbits[n_label:])}
    return make_record("ValidationSplit", spec.provenance, split_unit="symmetry_orbit", per_topology=per_topology,
                       counts={t: {k: len(v) for k, v in g.items()} for t, g in per_topology.items()},
                       policy={"seed": seed, "holdout_fraction": str(fraction), "selection": "seeded content-hash order"},
                       holdout_labels_available_to_pipeline=True,
                       holdout_access="sealed: labels are stored separately and read only after model selection")
