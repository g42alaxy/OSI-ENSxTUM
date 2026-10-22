"""Task 06: comparable periodic DFT evaluation through an asynchronous job interface.

``PeriodicDFTBackend`` mirrors the synthetic backend (submit -> job_status -> collect). The engine is
chosen by ``protocols.dft.engine`` and sits behind one adapter interface (write_input / parse_output):

  pyscf             PySCF GPW: Gaussian (GTH) orbitals with a plane-wave density (multigrid at Gamma, or
                    FFTDF with any k-mesh), DFT-D3/D4 from pyscf-dispersion, ASE relaxations; CPU or GPU
                    (gpu4pyscf). Runs ``pyscf_worker.py`` as a subprocess per calculation.
  quantum_espresso  ``pw.x`` (plane waves + pseudopotential files).

Every scientific setting, the engine's software/basis/pseudopotential identity included, is part of the
protocol hash; only the declared numerical retry ladder may change between attempts. Unconverged,
composition-changed or topology-changed results never become usable energies. Residual forces are checked
only on relaxed geometries, residual stress only after a variable-cell relaxation; a protocol may declare
``relaxation.calculation = "scf"`` (rigid builder geometries), which is recorded on every result.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

from .contract import JobState, PipelineError, StageStatus, content_hash, make_record
from .framework import _cart_to_frac, map_to_parent_slots

RY_TO_EV = Fraction("13.605693122994")  # CODATA 2018
HARTREE_TO_EV = 2 * RY_TO_EV
BOHR_TO_A = 0.529177210903
RY_PER_BOHR_TO_EV_PER_A = float(RY_TO_EV) / BOHR_TO_A
KBAR_PER_UNIT = {"kbar": 1.0, "GPa": 10.0}
RELAXATIONS = ("scf", "relax", "vc-relax")  # "scf": rigid geometry, a declared modelling choice
ALL_PROPERTIES = ("energy", "forces", "stress")
ATOMIC_MASSES = {"H": 1.008, "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Mg": 24.305,
                 "Al": 26.982, "Si": 28.085, "P": 30.974, "S": 32.06, "Cl": 35.45, "Ti": 47.867, "Cr": 51.996,
                 "Mn": 54.938, "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38, "Br": 79.904,
                 "Zr": 91.224, "Cd": 112.41}
COMMON_REQUIRED = ("protocol_id", "engine", "command", "xc_functional", "dispersion", "kpoint_spacing_inv_A",
                   "occupations", "zero_smearing_convention", "hubbard_u_eV", "relaxation", "final_single_point",
                   "numerical_retries", "difference_convergence", "n_cores", "timeout_s")
DIAGNOSTICS = ("software_version", "scf_converged", "ionic_converged", "max_force_eV_per_A", "max_stress_kbar",
               "pressure_kbar", "job_done")


def _kmesh(lattice, spacing: float) -> list:
    """Monkhorst-Pack divisions n_i = ceil(|b_i| / spacing), b_i reciprocal vectors including 2*pi."""
    a, b, c = lattice
    vol = abs(sum(a[i] * (b[(i + 1) % 3] * c[(i + 2) % 3] - b[(i + 2) % 3] * c[(i + 1) % 3]) for i in range(3)))
    mesh = []
    for u, v in ((b, c), (c, a), (a, b)):
        cross = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
        mesh.append(max(1, math.ceil(2 * math.pi * math.sqrt(sum(x * x for x in cross)) / vol / spacing)))
    return mesh


def stress_tolerance_kbar(tolerances: dict) -> float:
    stress = tolerances["stress_tolerance_and_units"]
    return float(stress["value"]) * KBAR_PER_UNIT[stress["units"]]


def _carts(geometry: dict) -> list:
    lattice = geometry["lattice_A"]
    return [[sum(f[k] * lattice[k][c] for k in range(3)) for c in range(3)] for f in geometry["frac_coords"]]


# -- Quantum ESPRESSO adapter -------------------------------------------------
class QuantumEspressoAdapter:
    #TODO: Task 06 — adapters for licensed codes (VASP/CP2K) behind the same write_input/parse_output interface;
    #TODO: k-point convergence of energy differences (only the basis cutoff is converged now).
    engine, name = "quantum_espresso", "quantum_espresso_pw"
    required = ("pseudo_dir", "pseudopotentials", "ecutwfc_Ry", "ecutrho_Ry", "conv_thr_Ry")
    retry_keys = {"mixing_beta", "mixing_mode", "electron_maxstep"}  # anything else changes the science
    cutoff_key, energy_unit, energy_to_eV = "ecutwfc_Ry", "Ry", RY_TO_EV
    conventions = ("total_energy_as_printed",)
    input_name, output_name = "pw.in", "pw.out"

    def protocol_errors(self, protocol, root: Path, elements, lattices) -> list:
        errors = []
        for el in sorted(elements):
            entry = protocol["pseudopotentials"].get(el)
            if not entry or not entry.get("file") or not entry.get("sha256"):
                errors.append(("SCIENTIFIC_SPEC_MISSING", f"pseudopotentials.{el}", ""))
                continue
            path = root / protocol["pseudo_dir"] / entry["file"]
            if not path.exists():
                errors.append(("MISSING_INPUT_FILE", f"pseudopotentials.{el}", str(path)))
            elif hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                errors.append(("PSEUDOPOTENTIAL_HASH_MISMATCH", f"pseudopotentials.{el}", entry["file"]))
            if el not in ATOMIC_MASSES:
                errors.append(("UNSUPPORTED_ELEMENT", f"pseudopotentials.{el}", ""))
        return errors

    def identity(self, backend) -> dict:
        return {}  # pseudopotential file hashes are part of the protocol itself; pw.x hash is recorded per job

    def cutoff_overrides(self, protocol, value) -> dict:
        ratio = Fraction(str(protocol["ecutrho_Ry"])) / Fraction(str(protocol["ecutwfc_Ry"]))
        return {"ecutwfc_Ry": value, "ecutrho_Ry": float(ratio * Fraction(str(value)))}

    def run_args(self, backend) -> list:
        return backend.command + ["-in", self.input_name]

    def env(self, backend):
        return None

    def write_input(self, geometry: dict, protocol: dict, tolerances: dict, calculation: str, magnetic_state: dict,
                    numerics: dict, overrides=None, pseudo_dir: str = ".", properties=ALL_PROPERTIES, backend=None) -> str:
        overrides = overrides or {}
        lattice, elements = geometry["lattice_A"], geometry["elements"]
        species = sorted(set(elements))
        occ = protocol["occupations"]
        system = ["ibrav = 0", f"nat = {len(elements)}", f"ntyp = {len(species)}",
                  f"ecutwfc = {overrides.get('ecutwfc_Ry', protocol['ecutwfc_Ry'])}",
                  f"ecutrho = {overrides.get('ecutrho_Ry', protocol['ecutrho_Ry'])}",
                  f"input_dft = '{protocol['xc_functional']}'"]
        if protocol["dispersion"] != "none":
            system.append(f"vdw_corr = '{protocol['dispersion']}'")
        if occ["kind"] == "smearing":
            system += ["occupations = 'smearing'", f"smearing = '{occ['smearing']}'", f"degauss = {occ['degauss_Ry']}"]
        else:
            system.append(f"occupations = '{occ['kind']}'")
        system.append(f"nspin = {magnetic_state['nspin']}")
        for i, el in enumerate(species, start=1):
            if el in magnetic_state.get("starting_magnetization", {}):
                system.append(f"starting_magnetization({i}) = {magnetic_state['starting_magnetization'][el]}")
        # pw.x tests every Cartesian component against forc_conv_thr; tol/sqrt(3) per component guarantees
        # the residual force norm checked in collect() is within the declared tolerance.
        force_ry = tolerances["force_tolerance_eV_per_angstrom"] / math.sqrt(3) / RY_PER_BOHR_TO_EV_PER_A
        control = [f"calculation = '{calculation}'", "prefix = 'mofpipe'", f"pseudo_dir = '{pseudo_dir}'",
                   "outdir = './tmp'", "tprnfor = .true.", "tstress = .true.", f"forc_conv_thr = {force_ry:.6e}"]
        electrons = [f"conv_thr = {protocol['conv_thr_Ry']}"] + [f"{k} = {v!r}" if isinstance(v, str) else f"{k} = {v}"
                                                                 for k, v in sorted(numerics.items())]
        blocks = [("CONTROL", control), ("SYSTEM", system), ("ELECTRONS", electrons)]
        if calculation in ("relax", "vc-relax"):
            blocks.append(("IONS", ["ion_dynamics = 'bfgs'"]))
        if calculation == "vc-relax":
            blocks.append(("CELL", ["cell_dynamics = 'bfgs'", f"press_conv_thr = {stress_tolerance_kbar(tolerances)}",
                                    f"cell_dofree = '{protocol['relaxation'].get('cell_dofree', 'all')}'"]))
        out = []
        for name, lines in blocks:
            out += [f"&{name}"] + [f"  {l}" for l in lines] + ["/"]
        out.append("ATOMIC_SPECIES")
        out += [f"  {el} {ATOMIC_MASSES[el]} {protocol['pseudopotentials'][el]['file']}" for el in species]
        out.append("CELL_PARAMETERS angstrom")
        out += ["  " + " ".join(f"{x:.8f}" for x in v) for v in lattice]
        out.append("ATOMIC_POSITIONS angstrom")
        out += [f"  {el} " + " ".join(f"{x:.8f}" for x in r) for el, r in zip(elements, _carts(geometry))]
        out.append("K_POINTS automatic")
        out.append("  " + " ".join(map(str, _kmesh(lattice, protocol["kpoint_spacing_inv_A"]))) + " 0 0 0")
        if protocol["hubbard_u_eV"]:
            out.append("HUBBARD (ortho-atomic)")
            out += [f"  U {orbital} {u}" for orbital, u in sorted(protocol["hubbard_u_eV"].items())]
        return "\n".join(out) + "\n"

    def store_output(self, run_dir: Path, proc) -> str:
        (run_dir / self.output_name).write_text(proc.stdout)
        return proc.stdout

    def parse_output(self, text: str, calculation: str, input_geometry: dict) -> dict:
        energies = re.findall(r"^!\s+total energy\s+=\s+(-?\d+\.\d+)\s+Ry", text, re.M)
        version = re.search(r"Program PWSCF (v\.\S+)", text)
        forces = [tuple(float(x) for x in m) for m in
                  re.findall(r"atom\s+\d+\s+type\s+\d+\s+force =\s+(\S+)\s+(\S+)\s+(\S+)", text)]
        pressure = re.findall(r"P=\s*(-?\d+\.\d+)", text)
        row = r"(?:[ \t]*\S+){6}[ \t]*"
        stress_blocks = re.findall(rf"total\s+stress[^\n]*\n({row}\n{row}\n{row})", text)
        smearing = re.findall(r"smearing contrib\. \(-TS\)\s+=\s+(-?\d+\.\d+)\s+Ry", text)
        out = {
            "job_done": "JOB DONE." in text, "software_version": version.group(1) if version else None,
            "scf_converged": "convergence has been achieved" in text and "convergence NOT achieved" not in text,
            "total_energy": energies[-1] if energies else None, "energy_unit": self.energy_unit,
            "smearing_contribution_Ry": smearing[-1] if smearing else None,
            "max_force_eV_per_A": None, "pressure_kbar": float(pressure[-1]) if pressure else None,
            "stress_kbar": None, "max_stress_kbar": None,
            "ionic_converged": True, "final_lattice_A": input_geometry["lattice_A"],
            "final_elements": input_geometry["elements"], "final_frac_coords": input_geometry["frac_coords"],
        }
        if stress_blocks:
            rows = [[float(x) for x in line.split()[3:6]] for line in stress_blocks[-1].strip().splitlines()]
            out["stress_kbar"] = [[-x for x in row] for row in rows]  # pw.x prints -dE/d(strain)/V; stored as (1/V) dE/d(strain)
            out["max_stress_kbar"] = max(abs(x) for row in rows for x in row)
        n_atoms = len(input_geometry["elements"])
        if len(forces) >= n_atoms:
            last = forces[-n_atoms:]
            out["max_force_eV_per_A"] = max(math.sqrt(sum(c * c for c in f)) for f in last) * RY_PER_BOHR_TO_EV_PER_A
        if calculation in ("relax", "vc-relax"):
            out["ionic_converged"] = "bfgs converged" in text and "End final coordinates" in text
            final = text.split("Begin final coordinates")[-1] if "Begin final coordinates" in text else ""
            cell = re.search(r"CELL_PARAMETERS \((\w+)\)\s*\n((?:\s*\S+\s+\S+\s+\S+\s*\n){3})", final)
            lattice = input_geometry["lattice_A"]
            if cell:
                if cell.group(1) != "angstrom":
                    raise PipelineError("DFT_OUTPUT_PARSE_ERROR", f"CELL_PARAMETERS units {cell.group(1)}")
                lattice = [[float(x) for x in row.split()] for row in cell.group(2).strip().splitlines()]
            elif calculation == "vc-relax" and final:
                raise PipelineError("DFT_OUTPUT_PARSE_ERROR", "vc-relax final coordinates without CELL_PARAMETERS")
            pos = re.search(r"ATOMIC_POSITIONS \((\w+)\)\s*\n(.*?)End final coordinates", final, re.S)
            if pos:
                rows = [l.split() for l in pos.group(2).strip().splitlines() if l.strip()]
                elements = [r[0] for r in rows]
                xyz = [tuple(float(x) for x in r[1:4]) for r in rows]
                if pos.group(1) == "angstrom":
                    fracs = [list(_cart_to_frac(lattice, r)) for r in xyz]
                elif pos.group(1) == "crystal":
                    fracs = [list(r) for r in xyz]
                else:
                    raise PipelineError("DFT_OUTPUT_PARSE_ERROR", f"ATOMIC_POSITIONS units {pos.group(1)}")
                out.update(final_lattice_A=lattice, final_elements=elements, final_frac_coords=fracs)
        return out


# -- PySCF adapter (GPW) ------------------------------------------------------
class PySCFAdapter:
    """PySCF periodic Gaussian-and-plane-waves DFT through ``pyscf_worker.py`` (one subprocess per job).

    ``integral_scheme``: ``multigrid`` (Gamma point, CP2K-style multigrid GPW; fastest for MOF-sized cells) or
    ``fftdf`` (FFT GPW with any Monkhorst-Pack mesh). ``device``: ``cpu`` or ``gpu`` (gpu4pyscf).
    ``basis``/``pseudo`` take one name or a per-element mapping, so heavier metal bases can be combined with
    lighter organic ones. ``ke_cutoff_Ha`` is the plane-wave cutoff of the density grid; it is the parameter of
    the energy-difference convergence study (``difference_convergence.ke_cutoff_Ha_series``).
    """
    #TODO: Task 06 — DFT+U (PySCF has k-point DFT+U but no periodic stress/multigrid gradients for it), hybrid
    #TODO: functionals (no periodic stress), user-supplied basis/pseudopotential files, Gaussian-basis convergence study.
    engine, name = "pyscf", "pyscf_gpw"
    required = ("basis", "pseudo", "ke_cutoff_Ha", "conv_tol_Ha", "integral_scheme", "device", "precision",
                "max_memory_MB")
    retry_keys = {"damp", "level_shift", "diis_space", "diis_start_cycle", "max_cycle"}
    cutoff_key, energy_unit, energy_to_eV = "ke_cutoff_Ha", "Ha", HARTREE_TO_EV
    conventions = ("internal_energy", "free_energy", "zero_temperature_extrapolated")
    input_name, output_name = "job.json", "result.json"
    worker = Path(__file__).with_name("pyscf_worker.py")
    DISPERSIONS = ("none", "d4") + tuple(f"{v}{atm}" for v in ("d3bj", "d3zero", "d3bjm", "d3zerom", "d3op")
                                         for atm in ("", "-atm"))

    def protocol_errors(self, protocol, root: Path, elements, lattices) -> list:
        errors = []
        if protocol["integral_scheme"] not in ("multigrid", "fftdf"):
            errors.append(("UNSUPPORTED_INTEGRAL_SCHEME", "integral_scheme", str(protocol["integral_scheme"])))
        if protocol["device"] not in ("cpu", "gpu"):
            errors.append(("UNSUPPORTED_DEVICE", "device", str(protocol["device"])))
        if protocol["dispersion"] not in self.DISPERSIONS:
            errors.append(("UNSUPPORTED_DISPERSION", "dispersion", str(protocol["dispersion"])))
        if protocol["hubbard_u_eV"]:
            errors.append(("UNSUPPORTED_HUBBARD_U", "hubbard_u_eV", "no periodic DFT+U gradients/stress in this engine"))
        occ = protocol["occupations"]
        if occ.get("kind") == "smearing" and (occ.get("method") not in ("fermi", "gaussian") or not occ.get("sigma_Ha")):
            errors.append(("SCIENTIFIC_SPEC_MISSING", "occupations", "smearing needs method fermi|gaussian and sigma_Ha"))
        elif occ.get("kind") not in ("fixed", "smearing"):
            errors.append(("UNSUPPORTED_OCCUPATIONS", "occupations", str(occ)))
        relax = protocol["relaxation"]
        if relax.get("calculation") in ("relax", "vc-relax"):
            for key in ("optimizer", "max_steps", "max_step_A"):
                if relax.get(key) is None:
                    errors.append(("SCIENTIFIC_SPEC_MISSING", f"relaxation.{key}", ""))
            if relax.get("optimizer") not in (None, "BFGS", "LBFGS", "FIRE"):
                errors.append(("UNSUPPORTED_OPTIMIZER", "relaxation.optimizer", str(relax["optimizer"])))
            if relax.get("cell_dofree", "all") != "all":
                errors.append(("UNSUPPORTED_RELAXATION", "relaxation.cell_dofree", "only 'all' in this engine"))
        for basis_key in ("basis", "pseudo"):
            value = protocol[basis_key]
            if isinstance(value, dict) and set(elements) - set(value):
                errors.append(("SCIENTIFIC_SPEC_MISSING", basis_key, f"no entry for {sorted(set(elements) - set(value))}"))
        if protocol["integral_scheme"] == "multigrid":
            for lattice in lattices:
                mesh = _kmesh(lattice, protocol["kpoint_spacing_inv_A"])
                if mesh != [1, 1, 1]:
                    errors.append(("UNSUPPORTED_KPOINTS_FOR_MULTIGRID", "kpoint_spacing_inv_A",
                                   f"k-mesh {mesh} for this cell; multigrid GPW is Gamma-only (use integral_scheme fftdf)"))
        return errors

    def settings(self, backend, geometry, calculation, magnetic_state, numerics, overrides, properties) -> dict:
        protocol, tolerances = backend.protocol, backend.tolerances
        relax = protocol["relaxation"]
        return {
            "xc": protocol["xc_functional"], "dispersion": protocol["dispersion"],
            "dispersion_xc": protocol.get("dispersion_xc") or protocol["xc_functional"].lower(),
            "basis": protocol["basis"], "pseudo": protocol["pseudo"],
            "ke_cutoff_Ha": overrides.get("ke_cutoff_Ha", protocol["ke_cutoff_Ha"]), "precision": protocol["precision"],
            "integral_scheme": protocol["integral_scheme"],
            "kmesh": _kmesh(geometry["lattice_A"], protocol["kpoint_spacing_inv_A"]),
            "occupations": protocol["occupations"], "energy_convention": protocol["zero_smearing_convention"],
            "conv_tol_Ha": protocol["conv_tol_Ha"], "magnetic_state": magnetic_state, "numerics": numerics,
            "device": protocol["device"], "n_threads": protocol["n_cores"], "max_memory_MB": protocol["max_memory_MB"],
            "cell_charge": 0, "compute_forces": "forces" in properties, "compute_stress": "stress" in properties,
            "relaxation_calculation": calculation, "optimizer": relax.get("optimizer"),
            "relaxation": {"optimizer": relax.get("optimizer"), "max_steps": relax.get("max_steps"),
                           "max_step_A": relax.get("max_step_A"),
                           "force_tolerance_eV_per_A": tolerances["force_tolerance_eV_per_angstrom"],
                           "stress_tolerance_kbar": stress_tolerance_kbar(tolerances)},
        }

    def identity(self, backend) -> dict:
        """Probe the worker interpreter: versions and basis/pseudopotential data hashes, or a typed refusal."""
        probe_dir = backend.store.root / "engine_probe"
        probe_dir.mkdir(exist_ok=True)
        dummy = {"lattice_A": [[10, 0, 0], [0, 10, 0], [0, 0, 10]]}
        settings = self.settings(backend, dummy, backend.protocol["relaxation"]["calculation"],
                                 backend.magnetic_states[0], {}, {}, ALL_PROPERTIES)
        (probe_dir / "probe.json").write_text(json.dumps({"settings": settings, "elements": sorted(backend.elements)}))
        try:
            proc = subprocess.run(backend.command + [str(self.worker), "--probe", "probe.json"], cwd=probe_dir,
                                  capture_output=True, text=True, timeout=600, env=self.env(backend))
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise PipelineError("BACKEND_UNAVAILABLE", f"PySCF probe: {exc}") from exc
        if proc.returncode != 0:
            raise PipelineError("BACKEND_UNAVAILABLE", f"PySCF probe failed: {proc.stderr[-500:]}")
        probe = json.loads(proc.stdout)
        if probe["errors"]:
            first = probe["errors"][0]
            code = "BACKEND_UNAVAILABLE" if first["code"] == "BACKEND_UNAVAILABLE" else "UNSUPPORTED_DFT_PROTOCOL"
            raise PipelineError(code, "; ".join(f"{e['code']}: {e['detail']}" for e in probe["errors"]))
        versions = probe["versions"]
        return {"engine": self.name, "data_hashes": probe["data_hashes"],
                "versions": {k: versions[k] for k in ("pyscf", "pyscf_dispersion", "gpu4pyscf", "worker") if k in versions},
                "environment": versions}

    def cutoff_overrides(self, protocol, value) -> dict:
        return {"ke_cutoff_Ha": value}

    def run_args(self, backend) -> list:
        return backend.command + [str(self.worker), self.input_name]

    def env(self, backend) -> dict:
        threads = str(backend.protocol["n_cores"])
        return {**os.environ, "OMP_NUM_THREADS": threads, "PYTHONHASHSEED": "0", "PYTHONNOUSERSITE": "1"}

    def write_input(self, geometry: dict, protocol: dict, tolerances: dict, calculation: str, magnetic_state: dict,
                    numerics: dict, overrides=None, pseudo_dir: str = ".", properties=ALL_PROPERTIES, backend=None) -> str:
        job = {"schema": "mofpipe_pyscf_job_v1", "calculation": calculation,
               "geometry": {k: geometry[k] for k in ("lattice_A", "elements", "frac_coords")},
               "settings": self.settings(backend, geometry, calculation, magnetic_state, numerics, overrides or {},
                                         properties)}
        return json.dumps(job, indent=1, sort_keys=True)

    def store_output(self, run_dir: Path, proc) -> str:
        (run_dir / "worker.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
        path = run_dir / self.output_name
        return path.read_text() if path.exists() else ""

    def parse_output(self, text: str, calculation: str, input_geometry: dict) -> dict:
        result = json.loads(text) if text.strip() else {"job_done": False}
        final = result.get("final_geometry") or {}
        return {
            "job_done": bool(result.get("job_done")), "error": result.get("error"),
            "software_version": (result.get("versions") or {}).get("pyscf"),
            "scf_converged": bool(result.get("scf_converged")),
            "ionic_converged": result.get("ionic_converged") if calculation in ("relax", "vc-relax") else True,
            "total_energy": result.get("total_energy_Ha"), "energy_unit": self.energy_unit,
            "energies_Ha": result.get("energies_Ha"), "max_force_eV_per_A": result.get("max_force_eV_per_A"),
            "stress_kbar": result.get("stress_kbar"), "max_stress_kbar": result.get("max_stress_kbar"),
            "pressure_kbar": result.get("pressure_kbar"),
            "final_lattice_A": final.get("lattice_A", input_geometry["lattice_A"]),
            "final_elements": final.get("elements", input_geometry["elements"]),
            "final_frac_coords": final.get("frac_coords", input_geometry["frac_coords"]),
            "engine_details": {k: result.get(k) for k in ("kmesh", "mesh", "nao", "nelectron", "n_scf", "spin_2S",
                                                          "energy_convention", "versions", "data_hashes", "wall_time_s")},
            "n_ionic_steps": len(result.get("trajectory") or []),
        }


ADAPTERS = {QuantumEspressoAdapter.engine: QuantumEspressoAdapter, PySCFAdapter.engine: PySCFAdapter}


# -- protocol -----------------------------------------------------------------
def dft_protocol_errors(protocol: dict, root: Path, elements: set, lattices=()) -> list:
    """SCIENTIFIC_SPEC_MISSING / UNSUPPORTED errors for a DFT protocol; [] when it is complete."""
    errors = []

    def err(code, path, detail=""):
        errors.append({"code": code, "check": f"protocols.dft.{path}", "detail": detail})

    adapter_cls = ADAPTERS.get(protocol.get("engine"))
    if protocol.get("engine") is not None and adapter_cls is None:
        err("UNSUPPORTED_DFT_ENGINE", "engine", f"{protocol['engine']} (supported: {sorted(ADAPTERS)})")
        return errors
    for key in COMMON_REQUIRED + (adapter_cls.required if adapter_cls else ()):
        if protocol.get(key) is None:
            err("SCIENTIFIC_SPEC_MISSING", key)
    if errors:
        return errors
    adapter = adapter_cls()
    if protocol["zero_smearing_convention"] not in adapter.conventions:
        err("UNSUPPORTED_ENERGY_CONVENTION", "zero_smearing_convention",
            f"{protocol['zero_smearing_convention']} (engine supports {list(adapter.conventions)})")
    for i, step in enumerate(protocol["numerical_retries"]):
        if set(step) - adapter.retry_keys:
            err("PROTOCOL_RETRY_CHANGES_SCIENCE", f"numerical_retries[{i}]", str(sorted(set(step) - adapter.retry_keys)))
    if not protocol["numerical_retries"]:
        err("SCIENTIFIC_SPEC_MISSING", "numerical_retries", "at least the default numerical settings")
    if protocol["relaxation"].get("calculation") not in RELAXATIONS:
        err("UNSUPPORTED_RELAXATION", "relaxation.calculation", str(protocol["relaxation"].get("calculation")))
    series_key = f"{adapter.cutoff_key}_series"
    series = protocol["difference_convergence"].get(series_key) or []
    if protocol[adapter.cutoff_key] not in series or max(series, default=0) <= protocol[adapter.cutoff_key]:
        err("SCIENTIFIC_SPEC_MISSING", f"difference_convergence.{series_key}",
            "must contain the protocol cutoff and at least one higher cutoff")
    for code, path, detail in adapter.protocol_errors(protocol, root, elements, lattices):
        err(code, path, detail)
    return errors


def tolerance_errors(tolerances: dict) -> list:
    stress = tolerances.get("stress_tolerance_and_units")
    if isinstance(stress, dict) and stress.get("units") not in KBAR_PER_UNIT:
        return [{"code": "UNSUPPORTED_UNIT", "check": "tolerances.stress_tolerance_and_units.units",
                 "detail": f"{stress.get('units')} (supported: {sorted(KBAR_PER_UNIT)})"}]
    return []


def protocol_fingerprint(protocol: dict, magnetic_policy: dict, tolerances: dict, engine_identity=None) -> str:
    """Hash of every setting that defines the scientific calculation (retry ladder and engine identity included)."""
    body = {"dft": protocol, "magnetic_state_policy": magnetic_policy,
            "force_tolerance": tolerances["force_tolerance_eV_per_angstrom"],
            "stress_tolerance": tolerances["stress_tolerance_and_units"]}
    if engine_identity:
        body["engine_identity"] = {k: engine_identity[k] for k in ("engine", "data_hashes", "versions")}
    return content_hash(body)


# -- backend ------------------------------------------------------------------
class PeriodicDFTBackend:
    #TODO: Task 10 — submit through AiiDA CalcJobs on HPC; this backend runs the engine locally and synchronously.
    name = "periodic_dft"

    def __init__(self, spec, store):
        physical = spec.physical
        self.spec, self.store = spec, store
        self.protocol = physical["protocols"]["dft"]
        self.tolerances = physical["tolerances"]
        self.magnetic_states = physical["chemistry"]["magnetic_state_policy"]["initial_states"]
        self.budgets = physical["budgets"]
        adapter_cls = ADAPTERS.get(self.protocol["engine"])
        if adapter_cls is None:
            raise PipelineError("UNSUPPORTED_DFT_PROTOCOL", f"engine {self.protocol['engine']}")
        self.adapter = adapter_cls()
        self.root = spec.root
        self.elements = set()
        for info in physical["lattices"].values():
            self.elements |= set(info["target_inventory"]) | {a["element"] for a in info["structure"]["atoms"]}
        command = list(self.protocol["command"])
        # abspath, not resolve(): a venv interpreter is a symlink whose target lacks the venv's packages.
        resolved = shutil.which(command[0]) or (os.path.abspath(self.root / command[0])
                                                if (self.root / command[0]).exists() else None)
        if resolved is None:
            raise PipelineError("BACKEND_UNAVAILABLE", f"DFT executable not found: {command[0]}")
        command[0] = resolved
        self.command = command
        files = command + ([str(self.adapter.worker)] if hasattr(self.adapter, "worker") else [])
        self.executable_sha256 = {c: hashlib.sha256(Path(c).read_bytes()).hexdigest() for c in files if Path(c).is_file()}
        self.engine_identity = self.adapter.identity(self)
        self.protocol_hash = protocol_fingerprint(self.protocol, physical["chemistry"]["magnetic_state_policy"],
                                                  self.tolerances, self.engine_identity)
        self.fidelity = f"DFT:{self.protocol['protocol_id']}:{self.protocol_hash[:12]}"
        self.comparison_group = f"{physical['chemistry']['formula_unit']}|{self.fidelity}"
        self.relaxation = self.protocol["relaxation"]["calculation"]

    def properties(self) -> tuple:
        """What the protocol's acceptance checks need from a labelling calculation."""
        return ALL_PROPERTIES if self.relaxation == "vc-relax" else ("energy", "forces")

    # submit_periodic_dft -> job_id; job_status(job_id); collect_periodic_dft(job_id)
    def submit(self, geometry: dict, calculation: str, magnetic_state: dict, purpose: str, overrides=None,
               properties=None) -> str:
        overrides = overrides or {}
        properties = tuple(sorted(properties or self.properties()))
        payload = {"backend": self.name, "protocol_hash": self.protocol_hash, "calculation": calculation,
                   "geometry_hash": geometry["geometry_hash"], "magnetic_state": magnetic_state, "overrides": overrides}
        if properties != tuple(sorted(ALL_PROPERTIES)):
            payload["properties"] = list(properties)
        job_id = "dftjob:" + content_hash(payload)[:20]  # idempotency key = scientific inputs
        job_dir = self.store.jobs / job_id.replace(":", "__")
        meta_path = job_dir / "job.json"
        if meta_path.exists() and json.loads(meta_path.read_text())["state"] == JobState.SUCCEEDED.value:
            self.store.bump("dft_cache_hits")
            self.store.log("job_reused", job_id=job_id, purpose=purpose)
            return job_id
        counters = self.store.state["counters"]
        if counters.get("dft_submissions", 0) >= self.budgets["max_dft_jobs"]:
            raise PipelineError("BUDGET_EXHAUSTED", f"max_dft_jobs={self.budgets['max_dft_jobs']}")
        if counters.get("dft_core_seconds", 0) / 3600 >= self.budgets["max_cpu_core_hours"]:
            raise PipelineError("BUDGET_EXHAUSTED", f"max_cpu_core_hours={self.budgets['max_cpu_core_hours']}")
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "geometry.json").write_text(json.dumps(geometry, sort_keys=True))
        meta = {"job_id": job_id, "purpose": purpose, "state": JobState.RUNNING.value, **payload,
                "properties": list(properties), "engine": self.adapter.name,
                "topology_id": geometry["topology_id"], "geometry_id": geometry["artifact_id"],
                "executable_sha256": self.executable_sha256, "attempts": []}
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
        self.store.bump("dft_submissions")
        self.store.log("job_submitted", job_id=job_id, purpose=purpose, calculation=calculation)
        pseudo_dir = str((self.root / self.protocol["pseudo_dir"]).resolve()) if "pseudo_dir" in self.protocol else "."
        for attempt, numerics in enumerate(self.protocol["numerical_retries"]):
            run_dir = job_dir / f"attempt_{attempt}"
            run_dir.mkdir(exist_ok=True)
            text = self.adapter.write_input(geometry, self.protocol, self.tolerances, calculation, magnetic_state,
                                            numerics, overrides, pseudo_dir, properties, backend=self)
            (run_dir / self.adapter.input_name).write_text(text)
            start = time.perf_counter()
            try:
                proc = subprocess.run(self.adapter.run_args(self), cwd=run_dir, capture_output=True, text=True,
                                      timeout=self.protocol["timeout_s"], env=self.adapter.env(self))
                outcome = {"returncode": proc.returncode, "stderr": proc.stderr[-2000:]}
                parsed = self.adapter.parse_output(self.adapter.store_output(run_dir, proc), calculation, geometry)
            except subprocess.TimeoutExpired:
                outcome, parsed = {"returncode": None, "stderr": "TIMEOUT"}, None
            wall = time.perf_counter() - start
            self.store.bump("dft_core_seconds", int(math.ceil(wall * self.protocol["n_cores"])))
            scf_ok = bool(parsed and parsed["scf_converged"])
            meta["attempts"].append({"attempt": attempt, "numerics": numerics, **outcome, "wall_time_s": round(wall, 3),
                                     "scf_converged": scf_ok, "error": (parsed or {}).get("error"),
                                     "input_sha256": hashlib.sha256(text.encode()).hexdigest()})
            if parsed is not None and outcome["returncode"] == 0 and scf_ok:
                break  # only SCF non-convergence is retried, and only with declared numerical settings
        meta["state"] = JobState.SUCCEEDED.value if meta["attempts"][-1]["returncode"] == 0 else JobState.FAILED.value
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
        self.store.log("job_finished", job_id=job_id, state=meta["state"], n_attempts=len(meta["attempts"]))
        return job_id

    def job_status(self, job_id: str) -> dict:
        path = self.store.jobs / job_id.replace(":", "__") / "job.json"
        if not path.exists():
            raise PipelineError("UNKNOWN_JOB", job_id)
        return json.loads(path.read_text())

    def read_output(self, job_id: str, geometry: dict) -> dict:
        meta = self.job_status(job_id)
        path = self.store.jobs / job_id.replace(":", "__") / f"attempt_{meta['attempts'][-1]['attempt']}" / self.adapter.output_name
        return self.adapter.parse_output(path.read_text() if path.exists() else "", meta["calculation"], geometry)

    def collect(self, job_id: str, decoration: dict) -> dict:
        """collect_periodic_dft(job_id) -> CalculationResult. Any failed check gives a non-SUCCEEDED status."""
        meta = self.job_status(job_id)
        job_dir = self.store.jobs / job_id.replace(":", "__")
        geometry = json.loads((job_dir / "geometry.json").read_text())
        last = meta["attempts"][-1]
        relaxed_by_protocol = self.relaxation in ("relax", "vc-relax")
        common = {"job_id": job_id, "backend": self.name, "engine": self.adapter.name, "calculation": meta["calculation"],
                  "configuration_id": decoration["configuration_id"], "orbit_id": decoration["orbit_id"],
                  "topology_id": decoration["topology_id"], "bits": decoration["bits"],
                  "input_hash": content_hash({k: meta[k] for k in ("protocol_hash", "calculation", "geometry_hash",
                                                                   "magnetic_state", "overrides")}),
                  "geometry_id": meta["geometry_id"], "geometry_hash": meta["geometry_hash"],
                  "magnetic_state": meta["magnetic_state"], "protocol_hash": self.protocol_hash,
                  "base_fidelity": self.fidelity, "comparison_group": self.comparison_group,
                  "recovery_history": meta["attempts"], "executable_sha256": meta["executable_sha256"],
                  "engine_identity": self.engine_identity,
                  "pseudopotential_sha256": ({el: p["sha256"] for el, p in sorted(self.protocol["pseudopotentials"].items())}
                                             if "pseudopotentials" in self.protocol else
                                             (self.engine_identity.get("data_hashes") or {}).get("pseudo")),
                  "geometry_purpose": (f"relaxed ({self.relaxation})" if relaxed_by_protocol else
                                       "unrelaxed builder geometry (protocol declares no relaxation)"),
                  "raw_output_ref": f"{job_dir.name}/attempt_{last['attempt']}/{self.adapter.output_name}"}

        def failed(status, code, detail="", parsed=None):
            return make_record("CalculationResult", self.spec.provenance, **common, status=status,
                               failure={"code": code, "detail": detail}, base_energy_exact=None, base_energy_eV=None,
                               diagnostics=parsed)

        if meta["state"] != JobState.SUCCEEDED.value:
            engine_error = last.get("error") or {}
            return failed(StageStatus.FAILED.value, "JOB_FAILED",
                          f"{engine_error.get('code', '')} {engine_error.get('detail', '')} {last.get('stderr', '')}".strip()[:300])
        parsed = self.read_output(job_id, geometry)
        diagnostics = {k: parsed.get(k) for k in DIAGNOSTICS}
        if not parsed["job_done"]:
            return failed(StageStatus.FAILED.value, "INCOMPLETE_OUTPUT", parsed=diagnostics)
        if not parsed["scf_converged"]:
            return failed(StageStatus.FAILED.value, "SCF_NOT_CONVERGED", parsed=diagnostics)
        if parsed["total_energy"] is None:
            return failed(StageStatus.FAILED.value, "INCOMPLETE_OUTPUT", parsed=diagnostics)
        if not parsed["ionic_converged"]:
            return failed(StageStatus.FAILED.value, "IONIC_NOT_CONVERGED", parsed=diagnostics)
        # Residual forces/stress are acceptance criteria only for what the protocol actually relaxed.
        force_tol = self.tolerances["force_tolerance_eV_per_angstrom"]
        if relaxed_by_protocol and (parsed["max_force_eV_per_A"] is None or parsed["max_force_eV_per_A"] > force_tol):
            return failed(StageStatus.FAILED.value, "RESIDUAL_FORCE_ABOVE_TOLERANCE",
                          f"{parsed['max_force_eV_per_A']} > {force_tol} eV/A", diagnostics)
        stress_tol = stress_tolerance_kbar(self.tolerances)
        if self.relaxation == "vc-relax" and (parsed["max_stress_kbar"] is None or parsed["max_stress_kbar"] > stress_tol):
            return failed(StageStatus.FAILED.value, "RESIDUAL_STRESS_ABOVE_TOLERANCE",
                          f"max |stress_ij| {parsed['max_stress_kbar']} > {stress_tol} kbar", diagnostics)
        final = {"lattice": parsed["final_lattice_A"],
                 "atoms": [{"element": e, "frac": tuple(f)} for e, f in zip(parsed["final_elements"], parsed["final_frac_coords"])]}
        inventory = dict(sorted(Counter(parsed["final_elements"]).items()))
        if inventory != geometry["inventory"]:
            return failed(StageStatus.FAILED.value, "COMPOSITION_CHANGED", f"{inventory} != {geometry['inventory']}", diagnostics)
        builder = self.spec.physical["protocols"]["builder"]
        mapping = map_to_parent_slots(final, self.spec.physical["builder_info"][decoration["topology_id"]],
                                      builder["bond_tolerance_A"], builder["slot_match_tolerance_A"])
        if not mapping["retained"]:
            # A transformation event: kept for audit, never a label for the original lattice model.
            return failed("TRANSFORMED", "TOPOLOGY_CHANGED_DURING_RELAXATION", mapping["reason"], diagnostics)
        # Reference phases (T09.2) have their own composition: their energy is normalised per cell and the
        # mixture balance uses the full cell inventory.
        per_cell = decoration.get("normalization") == "per_cell"
        n_fu = 1 if per_cell else formula_units(inventory, self.spec.physical["chemistry"]["target_elemental_inventory"])
        if n_fu is None:
            return failed(StageStatus.FAILED.value, "FORMULA_UNIT_MISMATCH", str(inventory), diagnostics)
        total = Fraction(parsed["total_energy"]) * self.adapter.energy_to_eV
        normalized = total / n_fu
        return make_record(
            "CalculationResult", self.spec.provenance, **common, status="SUCCEEDED", method=self.fidelity,
            energy_unit="eV", normalization="per_cell" if per_cell else "per_formula_unit", n_formula_units=n_fu,
            inventory=inventory,
            raw_energy={"value": parsed["total_energy"], "unit": parsed["energy_unit"],
                        "convention": self.protocol["zero_smearing_convention"]},
            total_energy_eV=float(total), total_energy_exact=str(total),
            base_energy_eV=float(normalized), base_energy_exact=str(normalized),
            relaxed_geometry={"lattice_A": parsed["final_lattice_A"], "elements": parsed["final_elements"],
                              "frac_coords": parsed["final_frac_coords"]},
            relaxed_geometry_hash=content_hash({"lattice": [[round(x, 4) for x in v] for v in parsed["final_lattice_A"]],
                                                "atoms": sorted((e, tuple(round(x, 4) % 1.0 for x in f)) for e, f in
                                                                zip(parsed["final_elements"], parsed["final_frac_coords"]))}),
            convergence=diagnostics, engine_details=parsed.get("engine_details"), slot_mapping="RETAINED",
            uncertainty={"numerical": "settings converged for energy differences (see DFTConvergenceReport)",
                         "magnetic": "lowest found among declared initial states; not a global certificate"})


def formula_units(inventory: dict, per_fu: dict):
    ratios = {Fraction(inventory.get(el, 0), n) for el, n in per_fu.items()}
    if set(inventory) - set(per_fu) or len(ratios) != 1:
        return None
    n = ratios.pop()
    return int(n) if n.denominator == 1 and n > 0 else None


def evaluate_geometry(backend: PeriodicDFTBackend, geometry: dict, decoration: dict, purpose: str) -> dict:
    """Relaxation (+ final single point) for every declared magnetic state; keeps each outcome separately."""
    outcomes = []
    for state in backend.magnetic_states:
        relax_id = backend.submit(geometry, backend.relaxation, state, purpose)
        relaxed = backend.collect(relax_id, decoration)
        result = relaxed
        if relaxed["status"] == "SUCCEEDED" and backend.protocol["final_single_point"] and backend.relaxation != "scf":
            sp_geometry = {**geometry, **relaxed["relaxed_geometry"], "geometry_hash": relaxed["relaxed_geometry_hash"],
                           "artifact_id": relaxed["artifact_id"]}
            result = backend.collect(backend.submit(sp_geometry, "scf", state, purpose + ":final_single_point"), decoration)
            result = {**result, "relaxation_result_id": relaxed["artifact_id"]}
        outcomes.append({"magnetic_state": state, "relaxation": relaxed, "result": result})
    succeeded = [o["result"] for o in outcomes if o["result"]["status"] == "SUCCEEDED"]
    if not succeeded:
        statuses = sorted({o["result"]["status"] for o in outcomes})
        return {"status": "TRANSFORMED" if statuses == ["TRANSFORMED"] else StageStatus.FAILED.value,
                "outcomes": outcomes, "selected": outcomes[0]["result"]}
    best = min(succeeded, key=lambda r: (Fraction(r["base_energy_exact"]), r["artifact_id"]))
    return {"status": "SUCCEEDED", "outcomes": outcomes, "selected": best,
            "state_selection": "lowest energy among converged declared initial states; not a global certificate"}


def difference_convergence_study(spec, backend: PeriodicDFTBackend, geometries: list) -> dict:
    """Single points over the declared cutoff series; is the energy DIFFERENCE converged at the protocol cutoff?"""
    protocol, adapter = backend.protocol, backend.adapter
    key = adapter.cutoff_key
    series = sorted(protocol["difference_convergence"][f"{key}_series"])
    state = backend.magnetic_states[0]
    rows = []
    for ecut in series:
        energies = []
        for geometry, decoration in geometries:
            job_id = backend.submit({**geometry}, "scf", state, "difference_convergence",
                                    adapter.cutoff_overrides(protocol, ecut), properties=("energy",))
            meta = backend.job_status(job_id)
            parsed = backend.read_output(job_id, geometry)
            if meta["state"] != JobState.SUCCEEDED.value or not parsed["scf_converged"] or parsed["total_energy"] is None:
                raise PipelineError("DFT_CONVERGENCE_STUDY_FAILED", f"{job_id} at {key}={ecut}")
            n_fu = formula_units(geometry["inventory"], spec.physical["chemistry"]["target_elemental_inventory"])
            energies.append(Fraction(parsed["total_energy"]) * adapter.energy_to_eV / n_fu)
        rows.append({key: ecut, "difference_eV_per_fu": energies[1] - energies[0]})
    reference = rows[-1]["difference_eV_per_fu"]
    tol = Fraction(str(spec.physical["tolerances"]["target_relative_energy_tolerance"]))
    at_protocol = next(r for r in rows if r[key] == protocol[key])
    deviation = abs(at_protocol["difference_eV_per_fu"] - reference)
    return make_record(
        "DFTConvergenceReport", spec.provenance, protocol_hash=backend.protocol_hash, engine=adapter.name,
        decorations=[d["configuration_id"] for _, d in geometries], quantity="E(b) - E(a) in eV per formula unit",
        parameter=key, series=[{key: r[key], "difference_eV_per_fu": float(r["difference_eV_per_fu"]),
                                "difference_exact": str(r["difference_eV_per_fu"])} for r in rows],
        protocol_cutoff=protocol[key], deviation_from_highest_cutoff_eV=float(deviation), tolerance_eV=float(tol),
        status=StageStatus.COMPLETED.value if deviation <= tol else "NOT_CONVERGED",
        scope=f"{key} only; k-point density fixed by kpoint_spacing_inv_A"
              + ("; Gaussian basis fixed by the protocol" if adapter.engine == "pyscf" else ""))
