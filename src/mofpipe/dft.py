"""Task 06: comparable periodic DFT evaluation through an asynchronous job interface.

``PeriodicDFTBackend`` mirrors the synthetic backend (submit -> job_status ->
collect). The engine adapter here is Quantum ESPRESSO ``pw.x``: the adapter
writes the input from a stored Geometry plus the registered protocol and parses
the raw output. Every scientific setting is part of the protocol hash; only the
declared numerical retry ladder may change between attempts. Unconverged,
composition-changed or topology-changed results never become usable energies.
"""
from __future__ import annotations

import hashlib
import json
import math
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
BOHR_TO_A = 0.529177210903
RY_PER_BOHR_TO_EV_PER_A = float(RY_TO_EV) / BOHR_TO_A
NUMERICAL_RETRY_KEYS = {"mixing_beta", "mixing_mode", "electron_maxstep"}  # anything else changes the science
ATOMIC_MASSES = {"H": 1.008, "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Mg": 24.305,
                 "Al": 26.982, "Si": 28.085, "P": 30.974, "S": 32.06, "Cl": 35.45, "Ti": 47.867, "Cr": 51.996,
                 "Mn": 54.938, "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38, "Br": 79.904,
                 "Zr": 91.224, "Cd": 112.41}
DFT_REQUIRED = ("protocol_id", "engine", "command", "xc_functional", "dispersion", "pseudo_dir", "pseudopotentials",
                "ecutwfc_Ry", "ecutrho_Ry", "kpoint_spacing_inv_A", "occupations", "zero_smearing_convention",
                "hubbard_u_eV", "conv_thr_Ry", "relaxation", "final_single_point", "numerical_retries",
                "difference_convergence", "n_cores", "timeout_s")


# -- protocol -----------------------------------------------------------------
def dft_protocol_errors(protocol: dict, root: Path, elements: set) -> list:
    """SCIENTIFIC_SPEC_MISSING / UNSUPPORTED errors for a DFT protocol; [] when it is complete."""
    errors = []

    def err(code, path, detail=""):
        errors.append({"code": code, "check": f"protocols.dft.{path}", "detail": detail})

    for key in DFT_REQUIRED:
        if protocol.get(key) is None:
            err("SCIENTIFIC_SPEC_MISSING", key)
    if errors:
        return errors
    if protocol["engine"] != "quantum_espresso":
        err("UNSUPPORTED_DFT_ENGINE", "engine", str(protocol["engine"]))
    if protocol["zero_smearing_convention"] != "total_energy_as_printed":
        err("UNSUPPORTED_ENERGY_CONVENTION", "zero_smearing_convention", str(protocol["zero_smearing_convention"]))
    for i, step in enumerate(protocol["numerical_retries"]):
        if set(step) - NUMERICAL_RETRY_KEYS:
            err("PROTOCOL_RETRY_CHANGES_SCIENCE", f"numerical_retries[{i}]", str(sorted(set(step) - NUMERICAL_RETRY_KEYS)))
    if not protocol["numerical_retries"]:
        err("SCIENTIFIC_SPEC_MISSING", "numerical_retries", "at least the default numerical settings")
    relax = protocol["relaxation"]
    if relax.get("calculation") not in ("vc-relax", "relax"):
        err("UNSUPPORTED_RELAXATION", "relaxation.calculation", str(relax.get("calculation")))
    series = protocol["difference_convergence"].get("ecutwfc_Ry_series") or []
    if protocol["ecutwfc_Ry"] not in series or max(series, default=0) <= protocol["ecutwfc_Ry"]:
        err("SCIENTIFIC_SPEC_MISSING", "difference_convergence.ecutwfc_Ry_series",
            "must contain the protocol cutoff and at least one higher cutoff")
    for el in sorted(elements):
        entry = protocol["pseudopotentials"].get(el)
        if not entry or not entry.get("file") or not entry.get("sha256"):
            err("SCIENTIFIC_SPEC_MISSING", f"pseudopotentials.{el}")
            continue
        path = (root / protocol["pseudo_dir"] / entry["file"])
        if not path.exists():
            err("MISSING_INPUT_FILE", f"pseudopotentials.{el}", str(path))
        elif hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            err("PSEUDOPOTENTIAL_HASH_MISMATCH", f"pseudopotentials.{el}", entry["file"])
        if el not in ATOMIC_MASSES:
            err("UNSUPPORTED_ELEMENT", f"pseudopotentials.{el}")
    return errors


def protocol_fingerprint(protocol: dict, magnetic_policy: dict, tolerances: dict) -> str:
    """Hash of every setting that defines the scientific calculation (retry ladder included)."""
    return content_hash({"dft": protocol, "magnetic_state_policy": magnetic_policy,
                         "force_tolerance": tolerances["force_tolerance_eV_per_angstrom"],
                         "stress_tolerance": tolerances["stress_tolerance_and_units"]})


# -- Quantum ESPRESSO adapter -------------------------------------------------
def _kmesh(lattice, spacing: float) -> list:
    a, b, c = lattice
    vol = abs(sum(a[i] * (b[(i + 1) % 3] * c[(i + 2) % 3] - b[(i + 2) % 3] * c[(i + 1) % 3]) for i in range(3)))
    mesh = []
    for u, v in ((b, c), (c, a), (a, b)):
        cross = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
        mesh.append(max(1, math.ceil(2 * math.pi * math.sqrt(sum(x * x for x in cross)) / vol / spacing)))
    return mesh


class QuantumEspressoAdapter:
    #TODO: Task 06 — adapter for licensed codes (VASP/CP2K) behind the same write_input/parse_output interface;
    #TODO: k-point convergence of energy differences (only the plane-wave cutoff is converged now).
    name = "quantum_espresso_pw"

    def write_input(self, geometry: dict, protocol: dict, tolerances: dict, calculation: str, magnetic_state: dict,
                    numerics: dict, ecutwfc=None, ecutrho=None, pseudo_dir: str = ".") -> str:
        lattice, elements = geometry["lattice_A"], geometry["elements"]
        species = sorted(set(elements))
        carts = [[sum(f[k] * lattice[k][c] for k in range(3)) for c in range(3)] for f in geometry["frac_coords"]]
        occ = protocol["occupations"]
        system = ["ibrav = 0", f"nat = {len(elements)}", f"ntyp = {len(species)}",
                  f"ecutwfc = {ecutwfc or protocol['ecutwfc_Ry']}", f"ecutrho = {ecutrho or protocol['ecutrho_Ry']}",
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
        force_ry = tolerances["force_tolerance_eV_per_angstrom"] / RY_PER_BOHR_TO_EV_PER_A
        control = [f"calculation = '{calculation}'", "prefix = 'mofpipe'", f"pseudo_dir = '{pseudo_dir}'",
                   "outdir = './tmp'", "tprnfor = .true.", "tstress = .true.", f"forc_conv_thr = {force_ry:.6e}"]
        electrons = [f"conv_thr = {protocol['conv_thr_Ry']}"] + [f"{k} = {v!r}" if isinstance(v, str) else f"{k} = {v}"
                                                                 for k, v in sorted(numerics.items())]
        blocks = [("CONTROL", control), ("SYSTEM", system), ("ELECTRONS", electrons)]
        if calculation in ("relax", "vc-relax"):
            blocks.append(("IONS", ["ion_dynamics = 'bfgs'"]))
        if calculation == "vc-relax":
            stress = tolerances["stress_tolerance_and_units"]
            blocks.append(("CELL", ["cell_dynamics = 'bfgs'", f"press_conv_thr = {stress['value']}",
                                    f"cell_dofree = '{protocol['relaxation'].get('cell_dofree', 'all')}'"]))
        out = []
        for name, lines in blocks:
            out += [f"&{name}"] + [f"  {l}" for l in lines] + ["/"]
        out.append("ATOMIC_SPECIES")
        out += [f"  {el} {ATOMIC_MASSES[el]} {protocol['pseudopotentials'][el]['file']}" for el in species]
        out.append("CELL_PARAMETERS angstrom")
        out += ["  " + " ".join(f"{x:.8f}" for x in v) for v in lattice]
        out.append("ATOMIC_POSITIONS angstrom")
        out += [f"  {el} " + " ".join(f"{x:.8f}" for x in r) for el, r in zip(elements, carts)]
        out.append("K_POINTS automatic")
        out.append("  " + " ".join(map(str, _kmesh(lattice, protocol["kpoint_spacing_inv_A"]))) + " 0 0 0")
        if protocol["hubbard_u_eV"]:
            out.append("HUBBARD (ortho-atomic)")
            out += [f"  U {orbital} {u}" for orbital, u in sorted(protocol["hubbard_u_eV"].items())]
        return "\n".join(out) + "\n"

    def parse_output(self, text: str, calculation: str, input_geometry: dict) -> dict:
        energies = re.findall(r"^!\s+total energy\s+=\s+(-?\d+\.\d+)\s+Ry", text, re.M)
        version = re.search(r"Program PWSCF (v\.\S+)", text)
        forces = [tuple(float(x) for x in m) for m in
                  re.findall(r"atom\s+\d+\s+type\s+\d+\s+force =\s+(\S+)\s+(\S+)\s+(\S+)", text)]
        pressure = re.findall(r"P=\s*(-?\d+\.\d+)", text)
        smearing = re.findall(r"smearing contrib\. \(-TS\)\s+=\s+(-?\d+\.\d+)\s+Ry", text)
        out = {
            "job_done": "JOB DONE." in text, "software_version": version.group(1) if version else None,
            "scf_converged": "convergence has been achieved" in text and "convergence NOT achieved" not in text,
            "total_energy_Ry": energies[-1] if energies else None,
            "smearing_contribution_Ry": smearing[-1] if smearing else None,
            "max_force_eV_per_A": None, "pressure_kbar": float(pressure[-1]) if pressure else None,
            "ionic_converged": True, "final_lattice_A": input_geometry["lattice_A"],
            "final_elements": input_geometry["elements"], "final_frac_coords": input_geometry["frac_coords"],
        }
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


# -- backend ------------------------------------------------------------------
class PeriodicDFTBackend:
    #TODO: Task 10 — submit through AiiDA CalcJobs on HPC; this backend runs pw.x locally and synchronously.
    name = "periodic_dft"

    def __init__(self, spec, store):
        physical = spec.physical
        self.spec, self.store = spec, store
        self.protocol = physical["protocols"]["dft"]
        self.tolerances = physical["tolerances"]
        self.magnetic_states = physical["chemistry"]["magnetic_state_policy"]["initial_states"]
        self.budgets = physical["budgets"]
        self.adapter = QuantumEspressoAdapter()
        self.root = spec.root
        command = list(self.protocol["command"])
        resolved = shutil.which(command[0]) or (str((self.root / command[0]).resolve())
                                                if (self.root / command[0]).exists() else None)
        if resolved is None:
            raise PipelineError("BACKEND_UNAVAILABLE", f"DFT executable not found: {command[0]}")
        command[0] = resolved
        self.command = command
        self.executable_sha256 = {c: hashlib.sha256(Path(c).read_bytes()).hexdigest() for c in command if Path(c).is_file()}
        self.protocol_hash = protocol_fingerprint(self.protocol, physical["chemistry"]["magnetic_state_policy"],
                                                  self.tolerances)
        self.fidelity = f"DFT:{self.protocol['protocol_id']}:{self.protocol_hash[:12]}"
        self.comparison_group = f"{physical['chemistry']['formula_unit']}|{self.fidelity}"

    # submit_periodic_dft -> job_id; job_status(job_id); collect_periodic_dft(job_id)
    def submit(self, geometry: dict, calculation: str, magnetic_state: dict, purpose: str, overrides=None) -> str:
        overrides = overrides or {}
        payload = {"backend": self.name, "protocol_hash": self.protocol_hash, "calculation": calculation,
                   "geometry_hash": geometry["geometry_hash"], "magnetic_state": magnetic_state, "overrides": overrides}
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
                "topology_id": geometry["topology_id"], "geometry_id": geometry["artifact_id"],
                "executable_sha256": self.executable_sha256, "attempts": []}
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
        self.store.bump("dft_submissions")
        self.store.log("job_submitted", job_id=job_id, purpose=purpose, calculation=calculation)
        pseudo_dir = str((self.root / self.protocol["pseudo_dir"]).resolve())
        for attempt, numerics in enumerate(self.protocol["numerical_retries"]):
            run_dir = job_dir / f"attempt_{attempt}"
            run_dir.mkdir(exist_ok=True)
            text = self.adapter.write_input(geometry, self.protocol, self.tolerances, calculation, magnetic_state,
                                            numerics, overrides.get("ecutwfc_Ry"), overrides.get("ecutrho_Ry"), pseudo_dir)
            (run_dir / "pw.in").write_text(text)
            start = time.perf_counter()
            try:
                proc = subprocess.run(self.command + ["-in", "pw.in"], cwd=run_dir, capture_output=True, text=True,
                                      timeout=self.protocol["timeout_s"])
                (run_dir / "pw.out").write_text(proc.stdout)
                outcome = {"returncode": proc.returncode, "stderr": proc.stderr[-2000:]}
                parsed = self.adapter.parse_output(proc.stdout, calculation, geometry)
            except subprocess.TimeoutExpired:
                outcome, parsed = {"returncode": None, "stderr": "TIMEOUT"}, None
            wall = time.perf_counter() - start
            self.store.bump("dft_core_seconds", int(math.ceil(wall * self.protocol["n_cores"])))
            scf_ok = bool(parsed and parsed["scf_converged"])
            meta["attempts"].append({"attempt": attempt, "numerics": numerics, **outcome, "wall_time_s": round(wall, 3),
                                     "scf_converged": scf_ok, "input_sha256": hashlib.sha256(text.encode()).hexdigest()})
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

    def collect(self, job_id: str, decoration: dict) -> dict:
        """collect_periodic_dft(job_id) -> CalculationResult. Any failed check gives a non-SUCCEEDED status."""
        meta = self.job_status(job_id)
        job_dir = self.store.jobs / job_id.replace(":", "__")
        geometry = json.loads((job_dir / "geometry.json").read_text())
        last = meta["attempts"][-1]
        common = {"job_id": job_id, "backend": self.name, "engine": self.adapter.name, "calculation": meta["calculation"],
                  "configuration_id": decoration["configuration_id"], "orbit_id": decoration["orbit_id"],
                  "topology_id": decoration["topology_id"], "bits": decoration["bits"],
                  "input_hash": content_hash({k: meta[k] for k in ("protocol_hash", "calculation", "geometry_hash",
                                                                   "magnetic_state", "overrides")}),
                  "geometry_id": meta["geometry_id"], "geometry_hash": meta["geometry_hash"],
                  "magnetic_state": meta["magnetic_state"], "protocol_hash": self.protocol_hash,
                  "base_fidelity": self.fidelity, "comparison_group": self.comparison_group,
                  "recovery_history": meta["attempts"], "executable_sha256": meta["executable_sha256"],
                  "pseudopotential_sha256": {el: p["sha256"] for el, p in sorted(self.protocol["pseudopotentials"].items())},
                  "raw_output_ref": f"{job_dir.name}/attempt_{last['attempt']}/pw.out"}

        def failed(status, code, detail="", parsed=None):
            return make_record("CalculationResult", self.spec.provenance, **common, status=status,
                               failure={"code": code, "detail": detail}, base_energy_exact=None, base_energy_eV=None,
                               diagnostics=parsed)

        if meta["state"] != JobState.SUCCEEDED.value:
            return failed(StageStatus.FAILED.value, "JOB_FAILED", str(last.get("stderr", ""))[:200])
        text = (job_dir / f"attempt_{last['attempt']}" / "pw.out").read_text()
        parsed = self.adapter.parse_output(text, meta["calculation"], geometry)
        diagnostics = {k: parsed[k] for k in ("software_version", "scf_converged", "ionic_converged", "max_force_eV_per_A",
                                              "pressure_kbar", "smearing_contribution_Ry", "job_done")}
        if not parsed["job_done"] or parsed["total_energy_Ry"] is None:
            return failed(StageStatus.FAILED.value, "INCOMPLETE_OUTPUT", parsed=diagnostics)
        if not parsed["scf_converged"]:
            return failed(StageStatus.FAILED.value, "SCF_NOT_CONVERGED", parsed=diagnostics)
        if not parsed["ionic_converged"]:
            return failed(StageStatus.FAILED.value, "IONIC_NOT_CONVERGED", parsed=diagnostics)
        force_tol = self.tolerances["force_tolerance_eV_per_angstrom"]
        if parsed["max_force_eV_per_A"] is None or parsed["max_force_eV_per_A"] > force_tol:
            return failed(StageStatus.FAILED.value, "RESIDUAL_FORCE_ABOVE_TOLERANCE",
                          f"{parsed['max_force_eV_per_A']} > {force_tol} eV/A", diagnostics)
        stress_tol = self.tolerances["stress_tolerance_and_units"]
        if parsed["pressure_kbar"] is None or abs(parsed["pressure_kbar"]) > stress_tol["value"]:
            return failed(StageStatus.FAILED.value, "RESIDUAL_STRESS_ABOVE_TOLERANCE",
                          f"|{parsed['pressure_kbar']}| > {stress_tol['value']} {stress_tol['units']}", diagnostics)
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
        n_fu = formula_units(inventory, self.spec.physical["chemistry"]["target_elemental_inventory"])
        if n_fu is None:
            return failed(StageStatus.FAILED.value, "FORMULA_UNIT_MISMATCH", str(inventory), diagnostics)
        total = Fraction(parsed["total_energy_Ry"]) * RY_TO_EV
        normalized = total / n_fu
        return make_record(
            "CalculationResult", self.spec.provenance, **common, status="SUCCEEDED", method=self.fidelity,
            energy_unit="eV", normalization="per_formula_unit", n_formula_units=n_fu,
            raw_energy={"value": parsed["total_energy_Ry"], "unit": "Ry", "convention": "total_energy_as_printed"},
            total_energy_eV=float(total), total_energy_exact=str(total),
            base_energy_eV=float(normalized), base_energy_exact=str(normalized),
            relaxed_geometry={"lattice_A": parsed["final_lattice_A"], "elements": parsed["final_elements"],
                              "frac_coords": parsed["final_frac_coords"]},
            relaxed_geometry_hash=content_hash({"lattice": [[round(x, 4) for x in v] for v in parsed["final_lattice_A"]],
                                                "atoms": sorted((e, tuple(round(x, 4) % 1.0 for x in f)) for e, f in
                                                                zip(parsed["final_elements"], parsed["final_frac_coords"]))}),
            convergence=diagnostics, slot_mapping="RETAINED",
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
        relax_id = backend.submit(geometry, backend.protocol["relaxation"]["calculation"], state, purpose)
        relaxed = backend.collect(relax_id, decoration)
        result = relaxed
        if relaxed["status"] == "SUCCEEDED" and backend.protocol["final_single_point"]:
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
    protocol = backend.protocol
    series = sorted(protocol["difference_convergence"]["ecutwfc_Ry_series"])
    ratio = Fraction(str(protocol["ecutrho_Ry"])) / Fraction(str(protocol["ecutwfc_Ry"]))
    state = backend.magnetic_states[0]
    rows = []
    for ecut in series:
        energies = []
        for geometry, decoration in geometries:
            overrides = {"ecutwfc_Ry": ecut, "ecutrho_Ry": float(ratio * Fraction(str(ecut)))}
            job_id = backend.submit({**geometry}, "scf", state, "difference_convergence", overrides)
            meta = backend.job_status(job_id)
            text = (backend.store.jobs / job_id.replace(":", "__") / f"attempt_{meta['attempts'][-1]['attempt']}" / "pw.out").read_text()
            parsed = backend.adapter.parse_output(text, "scf", geometry)
            if meta["state"] != JobState.SUCCEEDED.value or not parsed["scf_converged"] or parsed["total_energy_Ry"] is None:
                raise PipelineError("DFT_CONVERGENCE_STUDY_FAILED", f"{job_id} at ecutwfc={ecut}")
            n_fu = formula_units(geometry["inventory"], spec.physical["chemistry"]["target_elemental_inventory"])
            energies.append(Fraction(parsed["total_energy_Ry"]) * RY_TO_EV / n_fu)
        rows.append({"ecutwfc_Ry": ecut, "difference_eV_per_fu": energies[1] - energies[0]})
    reference = rows[-1]["difference_eV_per_fu"]
    tol = Fraction(str(spec.physical["tolerances"]["target_relative_energy_tolerance"]))
    at_protocol = next(r for r in rows if r["ecutwfc_Ry"] == protocol["ecutwfc_Ry"])
    deviation = abs(at_protocol["difference_eV_per_fu"] - reference)
    return make_record(
        "DFTConvergenceReport", spec.provenance, protocol_hash=backend.protocol_hash,
        decorations=[d["configuration_id"] for _, d in geometries], quantity="E(b) - E(a) in eV per formula unit",
        series=[{"ecutwfc_Ry": r["ecutwfc_Ry"], "difference_eV_per_fu": float(r["difference_eV_per_fu"]),
                 "difference_exact": str(r["difference_eV_per_fu"])} for r in rows],
        protocol_cutoff_Ry=protocol["ecutwfc_Ry"], deviation_from_highest_cutoff_eV=float(deviation), tolerance_eV=float(tol),
        status=StageStatus.COMPLETED.value if deviation <= tol else "NOT_CONVERGED",
        scope="plane-wave cutoff only; k-point density is fixed by kpoint_spacing_inv_A")
