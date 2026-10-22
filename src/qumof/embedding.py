"""Task 07 (T07.3): finite-cluster subtractive embedding and corrected-energy assembly (Milestone E).

Scheme ``finite_cluster_subtractive`` (mechanical embedding; labelled as a finite-cluster model, not WF-in-DFT).
Frozen total-energy expression for a candidate c with periodic DFT energy E_PBC[c] on its relaxed geometry G_c:

    E_corr[c] = E_PBC^low[c] + sum_k m_k (E_high[C_k] - E_low[C_k])

* Environment: the whole periodic cell at the DFT level (E_PBC^low). No embedding potential couples the regions.
* Regions: one per metal node of the cell. Region = node atoms, every atom within ``region.bond_hops`` bonds, and
  terminal atoms (a single bond) attached to those. Regions must be disjoint (no double counting) and must not
  wrap around the cell; every node of the compared cell must have one (coverage).
* Clusters C_k: the distinct regions (identical up to translation, coordinates to 1e-4 A) with multiplicity m_k.
  Each bond leaving a region is capped by ``capping.element`` on the bond line at the declared distance for the
  kept atom's element.
* E_low: the periodic low level (same functional and dispersion as protocols.dft) in the cluster basis;
  E_high: the declared high level in the same basis, charge and spin. Their difference is the contribution
  replaced at high level; the cluster basis is common to both, the periodic basis is that of protocols.dft.
* The correction is converted with CODATA 2018 Hartree -> eV and divided by the verified formula-unit count.

Corrected energies are emitted only when the scheme is VALIDATED for the protocol (``validate_scheme``): high=low
consistency, region balance across candidates, a boundary-convergence study of the relative correction and at
least one reproduced reference benchmark. Otherwise every correction is an ``EmbeddingBenchmarkResult`` with
``correction_status = UNVALIDATED`` and the ranking stays DFT-only.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from fractions import Fraction
from pathlib import Path

from .contract import JobState, PipelineError, StageStatus, content_hash, make_record
from .framework import _norm, _sub, bond_graph, deconstruct, frac_to_cart

SCHEME = "finite_cluster_subtractive"
HARTREE_TO_EV = Fraction("27.211386245988")  # CODATA 2018
ENERGY_EXPRESSION = ("E_corr[c] = E_PBC_low[c] + sum_k m_k (E_high[C_k] - E_low[C_k]); regions = node-centred "
                     "capped clusters, disjoint, covering every node of the cell")
HIGH_METHODS = ("HF", "MP2", "CCSD", "CCSD(T)", "active_space")
ACTIVE_SPACE_SOLVERS = ("exact_diagonalization", "pyscf_casci", "vqe_simulation", "sqd_simulation")
ENGINES = ("pyscf", "external")
CONSISTENCY_TOL_HA = Fraction(1, 10**8)  # high=low: the correction must vanish to numerical precision
CLUSTER_HASH_DECIMALS = 4  # translation-equivalent clusters: identical relative coordinates to 1e-4 A
REQUIRED = ("protocol_id", "scheme", "engine", "command", "region", "capping", "cluster_charge", "cluster_spin", "basis",
            "low_level", "high_level", "conv_tol_Ha", "n_cores", "timeout_s", "validation")


# -- protocol --------------------------------------------------------------------------------------------------
def protocol_errors(protocol: dict, dft_protocol: dict, budgets: dict, root: Path) -> list:
    """[(code, field, detail)] for protocols.embedding; nothing is defaulted."""
    errors = [("SCIENTIFIC_SPEC_MISSING", f"protocols.embedding.{k}", "") for k in REQUIRED if protocol.get(k) is None]
    if errors:
        return errors
    if protocol["scheme"] != SCHEME:
        errors.append(("UNSUPPORTED_EMBEDDING", "protocols.embedding.scheme", f"{protocol['scheme']}: only {SCHEME}"))
    if protocol["engine"] not in ENGINES:
        errors.append(("UNSUPPORTED_EMBEDDING_ENGINE", "protocols.embedding.engine", str(protocol["engine"])))
    hops = (protocol["region"] or {}).get("bond_hops")
    if type(hops) is not int or hops < 0:
        errors.append(("SCIENTIFIC_SPEC_MISSING", "protocols.embedding.region.bond_hops", "non-negative integer"))
    capping = protocol["capping"] or {}
    if not capping.get("element") or not capping.get("distance_A"):
        errors.append(("SCIENTIFIC_SPEC_MISSING", "protocols.embedding.capping", "element and distance_A by kept element"))
    low, high = protocol["low_level"] or {}, protocol["high_level"] or {}
    # The low level must be the periodic method itself, or the subtraction removes something the cell never had.
    same = (low.get("method") == "DFT" and str(low.get("xc", "")).upper() == str(dft_protocol.get("xc_functional", "")).upper()
            and low.get("dispersion") == dft_protocol.get("dispersion"))
    if not same:
        errors.append(("EMBEDDING_LOW_LEVEL_MISMATCH", "protocols.embedding.low_level",
                       f"{low} must be DFT with xc/dispersion of protocols.dft "
                       f"({dft_protocol.get('xc_functional')}, {dft_protocol.get('dispersion')})"))
    if high.get("method") not in HIGH_METHODS:
        errors.append(("UNSUPPORTED_HIGH_LEVEL", "protocols.embedding.high_level.method", str(high.get("method"))))
    elif high["method"] == "active_space" and not (high.get("active_space") and high.get("solver")):
        errors.append(("SCIENTIFIC_SPEC_MISSING", "protocols.embedding.high_level", "active_space and solver"))
    elif high["method"] == "active_space" and high["solver"] not in ACTIVE_SPACE_SOLVERS:
        errors.append(("UNSUPPORTED_HIGH_LEVEL", "protocols.embedding.high_level.solver",
                       f"{high['solver']}: one of {list(ACTIVE_SPACE_SOLVERS)}"))
    validation = protocol["validation"] or {}
    series = validation.get("bond_hops_series")
    if not (isinstance(series, list) and all(type(h) is int and h >= 0 for h in series) and type(hops) is int
            and hops in series and max(series) > hops):
        errors.append(("SCIENTIFIC_SPEC_MISSING", "protocols.embedding.validation.bond_hops_series",
                       "integers containing region.bond_hops and a larger value (boundary-convergence study)"))
    refs = validation.get("reference_benchmarks")
    if not isinstance(refs, list):
        errors.append(("SCIENTIFIC_SPEC_MISSING", "protocols.embedding.validation.reference_benchmarks",
                       "list of benchmark files ([] leaves every correction UNVALIDATED)"))
    else:
        errors += [("INVALID_REFERENCE_BENCHMARK", "protocols.embedding.validation.reference_benchmarks", str(p))
                   for p in refs if not (root / p).is_file()]
    max_jobs = (budgets or {}).get("max_active_space_jobs")
    if type(max_jobs) is not int or max_jobs < 1:
        errors.append(("SCIENTIFIC_SPEC_MISSING", "budgets.max_active_space_jobs", "positive cluster-job budget"))
    return errors


# -- regions and capped clusters -------------------------------------------------------------------------------
def cut_region(structure: dict, adjacency: list, seeds: dict, hops: int, capping: dict) -> dict:
    """Region around ``seeds`` ({atom: image}) and its capped cluster.

    Breadth-first over (atom, image) up to ``hops`` bonds, then terminal atoms bonded to the region. An atom
    reached under two different images means the region wraps around the cell (REGION_EXCEEDS_CELL). Every other
    bond leaving the region becomes a cap atom at the declared distance from the kept atom along the bond.
    """
    atoms, lattice = structure["atoms"], structure["lattice"]

    def neighbours(i, img):
        return [(j, tuple(img[k] + d[k] for k in range(3))) for j, d in adjacency[i]]

    def position(i, img):
        return frac_to_cart(lattice, [atoms[i]["frac"][k] + img[k] for k in range(3)])

    def add(j, img):
        if j in region:
            if region[j] != img:
                raise PipelineError("REGION_EXCEEDS_CELL", f"atom {j} reached under images {region[j]} and {img}")
            return False
        region[j] = img
        return True

    region = dict(seeds)
    frontier = list(seeds.items())
    for _ in range(hops):
        frontier = [(j, img) for i, i_img in frontier for j, img in neighbours(i, i_img) if add(j, img)]
    for i, img in list(region.items()):
        for j, j_img in neighbours(i, img):
            if len(adjacency[j]) == 1:
                add(j, j_img)
    caps = []
    for i, img in sorted(region.items()):
        for j, j_img in neighbours(i, img):
            if j in region:
                add(j, j_img)  # a bond to another image of a region atom also wraps the cell
                continue
            element = atoms[i]["element"]
            distance = capping["distance_A"].get(element)
            if distance is None:
                raise PipelineError("CAPPING_RULE_MISSING", f"no cap distance for a bond cut at {element} "
                                                            f"(to {atoms[j]['element']})")
            r_i, r_j = position(i, img), position(j, j_img)
            bond = _sub(r_j, r_i)
            caps.append({"element": capping["element"], "bonded_to": i, "replaces": j,
                         "position": tuple(r_i[k] + distance * bond[k] / _norm(bond) for k in range(3))})
    members = [(atoms[i]["element"], position(i, img)) for i, img in region.items()]
    members += [(c["element"], c["position"]) for c in caps]
    seed_xyz = [position(i, img) for i, img in seeds.items()]
    centre = tuple(sum(r[k] for r in seed_xyz) / len(seed_xyz) for k in range(3))
    rel = sorted((e, tuple(round(r[k] - centre[k], 6) + 0.0 for k in range(3))) for e, r in members)
    key = [(e, tuple(round(x, CLUSTER_HASH_DECIMALS) + 0.0 for x in r)) for e, r in rel]
    inventory = {}
    for e, _ in rel:
        inventory[e] = inventory.get(e, 0) + 1
    return {"atoms": sorted(region), "caps": [{k: c[k] for k in ("element", "bonded_to", "replaces")} for c in caps],
            "elements": [e for e, _ in rel], "coords_A": [list(r) for _, r in rel],
            "cluster_hash": content_hash(key)[:20], "inventory": dict(sorted(inventory.items()))}


def _structure(geometry: dict) -> dict:
    return {"lattice": geometry["lattice_A"],
            "atoms": [{"element": e, "frac": tuple(f)} for e, f in zip(geometry["elements"], geometry["frac_coords"])]}


def plan_regions(spec, calc: dict, hops: int, protocol: dict) -> dict:
    """EmbeddingPlan for one DFT result: one region per node of its relaxed cell, grouped into distinct clusters."""
    tid = calc["topology_id"]
    structure = _structure(calc["relaxed_geometry"])
    parts = deconstruct(structure, spec.physical["protocols"]["builder"]["bond_tolerance_A"])
    expected = len(spec.physical["lattices"][tid]["nodes"])
    if len(parts["nodes"]) != expected:
        raise PipelineError("INCOMPLETE_NODE_COVERAGE", f"{len(parts['nodes'])} nodes found, {expected} in the cell")
    regions, clusters, owner = [], {}, {}
    for n, node in enumerate(parts["nodes"]):
        cut = cut_region(structure, parts["adjacency"], {i: node["offsets"][i] for i in node["atoms"]}, hops,
                         protocol["capping"])
        clash = sorted({owner[i] for i in cut["atoms"] if i in owner})
        if clash:
            raise PipelineError("OVERLAPPING_REGIONS", f"region of node {n} shares atoms with nodes {clash}")
        owner.update({i: n for i in cut["atoms"]})
        regions.append({"node": n, "cluster_hash": cut["cluster_hash"], "atoms": cut["atoms"], "n_caps": len(cut["caps"])})
        entry = clusters.setdefault(cut["cluster_hash"], {k: cut[k] for k in ("elements", "coords_A", "inventory")}
                                    | {"caps": cut["caps"], "multiplicity": 0, "nodes": []})
        entry["multiplicity"] += 1
        entry["nodes"].append(n)
    inventory = {}
    for c in clusters.values():
        for e, k in c["inventory"].items():
            inventory[e] = inventory.get(e, 0) + k * c["multiplicity"]
    return make_record(
        "EmbeddingPlan", spec.provenance, scheme=SCHEME, protocol_id=protocol["protocol_id"], energy_expression=ENERGY_EXPRESSION,
        calculation_result_id=calc["artifact_id"], orbit_id=calc["orbit_id"], configuration_id=calc["configuration_id"],
        topology_id=tid, relaxed_geometry_hash=calc["relaxed_geometry_hash"], bond_hops=hops,
        n_nodes_expected=expected, n_formula_units=calc["n_formula_units"], regions=regions, clusters=clusters,
        region_inventory=dict(sorted(inventory.items())), overlap="NONE (disjoint regions verified)")


# -- cluster jobs ----------------------------------------------------------------------------------------------
class ClusterBackend:
    """submit -> job_status -> collect for cluster energies; idempotency key = scientific inputs."""
    name = "finite_cluster_engine"

    def __init__(self, spec, store, protocol: dict | None = None):
        self.spec, self.store = spec, store
        self.protocol = protocol or spec.physical["protocols"]["embedding"]
        self.budget = spec.physical["budgets"]["max_active_space_jobs"]
        command = list(self.protocol["command"])
        resolved = shutil.which(command[0]) or (os.path.abspath(spec.root / command[0])
                                                if (spec.root / command[0]).exists() else None)
        if resolved is None:
            raise PipelineError("BACKEND_UNAVAILABLE", f"cluster engine not found: {command[0]}")
        command[0] = resolved
        self.worker = Path(__file__).with_name("cluster_worker.py") if self.protocol["engine"] == "pyscf" else None
        self.args = command + ([str(self.worker)] if self.worker else [])
        files = [a for a in self.args if Path(a).is_file()]
        self.executable_sha256 = {a: hashlib.sha256(Path(a).read_bytes()).hexdigest() for a in files}
        science = {k: v for k, v in self.protocol.items() if k not in ("command", "n_cores", "timeout_s", "validation")}
        # The interpreter is an environment detail; scripts (worker, external engine) are part of the fidelity.
        self.protocol_hash = content_hash({"protocol": science, "scripts": {Path(a).name: h for a, h in
                                                                             self.executable_sha256.items() if a.endswith(".py")}})
        self.fidelity = f"EMB:{self.protocol['protocol_id']}:{self.protocol_hash[:12]}"

    def level(self, role: str) -> dict:
        """high, low, or high_as_low (the low level submitted as a separate high-level job, for the consistency test)."""
        return dict(self.protocol["high_level" if role == "high" else "low_level"])

    def submit(self, cluster_hash: str, cluster: dict, role: str, purpose: str) -> str:
        job = {"schema": "qumof_cluster_job_v1", "cluster_hash": cluster_hash, "elements": cluster["elements"],
               "coords_A": cluster["coords_A"], "charge": self.protocol["cluster_charge"], "spin": self.protocol["cluster_spin"],
               "basis": self.protocol["basis"], "ecp": self.protocol.get("ecp"), "level": self.level(role),
               "conv_tol_Ha": self.protocol["conv_tol_Ha"], "n_threads": self.protocol["n_cores"],
               "provenance": self.spec.provenance}
        payload = {"backend": self.name, "protocol_hash": self.protocol_hash, "role": role, "cluster_hash": cluster_hash,
                   "job": {k: v for k, v in job.items() if k not in ("n_threads",)}}
        job_id = "clusterjob:" + content_hash(payload)[:20]
        job_dir = self.store.jobs / job_id.replace(":", "__")
        meta_path = job_dir / "job.json"
        if meta_path.exists() and json.loads(meta_path.read_text())["state"] == JobState.SUCCEEDED.value:
            self.store.bump("cluster_cache_hits")
            return job_id
        if self.store.state["counters"].get("cluster_submissions", 0) >= self.budget:
            raise PipelineError("BUDGET_EXHAUSTED", f"max_active_space_jobs={self.budget}")
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "input.json").write_text(json.dumps(job, indent=1, sort_keys=True))
        meta = {"job_id": job_id, "purpose": purpose, "state": JobState.RUNNING.value, "role": role,
                "cluster_hash": cluster_hash, "protocol_hash": self.protocol_hash, "executable_sha256": self.executable_sha256}
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
        self.store.bump("cluster_submissions")
        self.store.log("job_submitted", job_id=job_id, purpose=purpose, role=role)
        start = time.perf_counter()
        env = {**os.environ, "OMP_NUM_THREADS": str(self.protocol["n_cores"]), "PYTHONHASHSEED": "0"}
        try:
            proc = subprocess.run(self.args + ["input.json"], cwd=job_dir, capture_output=True, text=True,
                                  timeout=self.protocol["timeout_s"], env=env)
            meta.update(returncode=proc.returncode, stderr=proc.stderr[-2000:])
        except subprocess.TimeoutExpired:
            meta.update(returncode=None, stderr="TIMEOUT")
        meta["wall_time_s"] = round(time.perf_counter() - start, 3)
        ok = meta["returncode"] == 0 and (job_dir / "result.json").exists()
        meta["state"] = JobState.SUCCEEDED.value if ok else JobState.FAILED.value
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
        self.store.log("job_finished", job_id=job_id, state=meta["state"])
        return job_id

    def job_status(self, job_id: str) -> dict:
        path = self.store.jobs / job_id.replace(":", "__") / "job.json"
        if not path.exists():
            raise PipelineError("UNKNOWN_JOB", job_id)
        return json.loads(path.read_text())

    def collect(self, job_id: str) -> dict:
        meta = self.job_status(job_id)
        job_dir = self.store.jobs / job_id.replace(":", "__")
        job = json.loads((job_dir / "input.json").read_text())
        result = json.loads((job_dir / "result.json").read_text()) if meta["state"] == JobState.SUCCEEDED.value else {}
        ok = result.get("status") == "SUCCEEDED" and isinstance(result.get("energy_Ha"), (int, float))
        return make_record(
            "ClusterCalculationResult", self.spec.provenance, job_id=job_id, backend=self.name, role=meta["role"],
            cluster_hash=meta["cluster_hash"], protocol_hash=self.protocol_hash, fidelity=self.fidelity,
            level=job["level"], basis=job["basis"], ecp=job["ecp"], charge=job["charge"], spin=job["spin"],
            status="SUCCEEDED" if ok else StageStatus.FAILED.value, energy_Ha=result.get("energy_Ha") if ok else None,
            failure=None if ok else (result.get("error") or {"code": "JOB_FAILED", "detail": meta.get("stderr", "")[-300:]}),
            details=result.get("details"), software=result.get("versions"), executable_sha256=meta["executable_sha256"],
            raw_output_ref=f"{job_dir.name}/result.json")

    def evaluate(self, plan: dict, roles=("high", "low"), purpose: str = "embedding_correction") -> dict:
        """{cluster_hash: {role: ClusterCalculationResult}} for every distinct cluster of the plan."""
        return {h: {role: self.collect(self.submit(h, c, role, purpose)) for role in roles}
                for h, c in sorted(plan["clusters"].items())}


# -- assembly --------------------------------------------------------------------------------------------------
def _blocked(spec, calc: dict, code: str, detail: str) -> dict:
    return make_record("BlockedResult", spec.provenance, orbit_id=calc.get("orbit_id"),
                       calculation_result_id=calc.get("artifact_id"), status=StageStatus.BLOCKED.value,
                       correction_status="BLOCKED", failure={"code": code, "detail": detail})


def assemble_corrected_energy(spec, calc: dict, plan: dict, results: dict, backend: ClusterBackend,
                              validation: dict | None = None, high_role: str = "high") -> dict:
    """assemble_corrected_energy(candidate, embedding_protocol, result_ids) -> CorrectedEnergy, an UNVALIDATED
    EmbeddingBenchmarkResult, or BlockedResult. ``results`` is ``ClusterBackend.evaluate`` output."""
    if calc.get("status") != "SUCCEEDED" or plan["calculation_result_id"] != calc["artifact_id"] \
            or plan["relaxed_geometry_hash"] != calc.get("relaxed_geometry_hash"):
        return _blocked(spec, calc, "INCOMPATIBLE_GEOMETRY", "the plan was not cut from this calculation's relaxed geometry")
    covered = sorted(n for c in plan["clusters"].values() for n in c["nodes"])
    if covered != list(range(plan["n_nodes_expected"])) or \
            sum(c["multiplicity"] for c in plan["clusters"].values()) != plan["n_nodes_expected"]:
        return _blocked(spec, calc, "INCOMPLETE_NODE_COVERAGE", f"nodes {covered} of {plan['n_nodes_expected']}")
    total, terms = Fraction(0), []
    for h, cluster in sorted(plan["clusters"].items()):
        pair = results.get(h) or {}
        high, low = pair.get(high_role), pair.get("low")
        if low is None or low["status"] != "SUCCEEDED":
            return _blocked(spec, calc, "MISSING_SUBTRACTION_TERM", f"E_low of cluster {h}: {(low or {}).get('failure')}")
        if high is None or high["status"] != "SUCCEEDED":
            return _blocked(spec, calc, "MISSING_HIGH_LEVEL_TERM", f"E_high of cluster {h}: {(high or {}).get('failure')}")
        if any(r["cluster_hash"] != h or r["protocol_hash"] != backend.protocol_hash for r in (high, low)) or \
                any(high[k] != low[k] for k in ("basis", "ecp", "charge", "spin")):
            return _blocked(spec, calc, "INCOMPATIBLE_CLUSTER_PAIR", f"cluster {h}: high/low inputs differ")
        delta = Fraction(repr(high["energy_Ha"])) - Fraction(repr(low["energy_Ha"]))
        total += cluster["multiplicity"] * delta
        terms.append({"cluster_hash": h, "multiplicity": cluster["multiplicity"], "high_result_id": high["artifact_id"],
                      "low_result_id": low["artifact_id"], "delta_Ha": float(delta), "delta_exact_Ha": str(delta)})
    per_fu = total * HARTREE_TO_EV / plan["n_formula_units"]
    common = dict(orbit_id=calc["orbit_id"], configuration_id=calc["configuration_id"], topology_id=calc["topology_id"],
                  calculation_result_id=calc["artifact_id"], plan_id=plan["artifact_id"], scheme=SCHEME,
                  energy_expression=ENERGY_EXPRESSION, embedding_fidelity=backend.fidelity, bond_hops=plan["bond_hops"],
                  terms=terms, correction_cell_Ha=float(total), correction_cell_exact_Ha=str(total),
                  n_formula_units=plan["n_formula_units"], high_level_role=high_role,
                  uncertainty={"numerical": "cluster SCF/correlation converged to protocols.embedding.conv_tol_Ha",
                               "embedding_systematic": "boundary-convergence deviation in the EmbeddingValidationReport",
                               "basis": "cluster basis fixed by the protocol; not assessed"})
    validated = (validation is not None and validation["status"] == "VALIDATED"
                 and validation["protocol_hash"] == backend.protocol_hash and high_role == "high")
    if not validated:
        return make_record("EmbeddingBenchmarkResult", spec.provenance, **common, correction_status="UNVALIDATED",
                           unvalidated_correction_eV_per_fu=float(per_fu), unvalidated_correction_exact=str(per_fu),
                           note="benchmark artifact only: not applied to any energy or ranking")
    base = Fraction(calc["base_energy_exact"])
    corrected_fidelity = f"{calc['base_fidelity']}+{backend.fidelity}"
    return make_record("CorrectedEnergy", spec.provenance, **common, correction_status="VALIDATED",
                       validation_id=validation["artifact_id"], status="SUCCEEDED",
                       base_energy_exact=str(base), correction_eV_per_fu=float(per_fu), correction_exact=str(per_fu),
                       corrected_energy_eV=float(base + per_fu), corrected_energy_exact=str(base + per_fu),
                       corrected_fidelity=corrected_fidelity,
                       comparison_group=f"{spec.physical['chemistry']['formula_unit']}|{corrected_fidelity}")


# -- scheme validation -----------------------------------------------------------------------------------------
def validate_scheme(spec, backend: ClusterBackend, planned: list, results: dict) -> dict:
    """{"report": EmbeddingValidationReport, "records": cluster results and benchmark reports it used}.

    VALIDATED only if every check passes. ``planned`` is [(calc, plan)] at the protocol's bond_hops; ``results``
    maps orbit_id -> ``ClusterBackend.evaluate`` output for those plans.
    """
    protocol, checks, records = backend.protocol, {}, {}
    hops = protocol["region"]["bond_hops"]
    if not planned:
        return {"records": {}, "report": make_record(
            "EmbeddingValidationReport", spec.provenance, scheme=SCHEME, protocol_id=protocol["protocol_id"],
            protocol_hash=backend.protocol_hash, status="UNVALIDATED", checks={},
            unresolved=["no candidate could be planned"])}
    # 1. high = low: the low level submitted as separate high-level jobs must give a zero correction.
    calc, plan = planned[0]
    as_low = backend.evaluate(plan, ("high_as_low",), "embedding_high_equals_low")
    merged = {h: {**results[calc["orbit_id"]][h], **as_low[h]} for h in as_low}
    records.update({r["artifact_id"]: r for pair in as_low.values() for r in pair.values()})
    test = assemble_corrected_energy(spec, calc, plan, merged, backend, high_role="high_as_low")
    residual = None if test["record_type"] == "BlockedResult" else abs(Fraction(test["correction_cell_exact_Ha"]))
    checks["high_equals_low"] = {"status": "PASSED" if residual is not None and residual <= CONSISTENCY_TOL_HA else "FAILED",
                                 "orbit_id": calc["orbit_id"], "residual_Ha": None if residual is None else float(residual),
                                 "tolerance_Ha": float(CONSISTENCY_TOL_HA), "failure": test.get("failure")}
    # 2. balance: every candidate's regions hold the same atoms per formula unit (no level imbalance between them).
    per_fu = {c["orbit_id"]: {e: str(Fraction(k, p["n_formula_units"])) for e, k in p["region_inventory"].items()}
              for c, p in planned}
    distinct = {json.dumps(v, sort_keys=True) for v in per_fu.values()}
    checks["balanced_regions"] = {"status": "PASSED" if len(distinct) == 1 else "FAILED",
                                  "region_inventory_per_fu": per_fu if len(distinct) > 1 else json.loads(distinct.pop())}
    # 3. boundary convergence of the RELATIVE correction between the two lowest DFT candidates.
    tol = Fraction(str(spec.physical["tolerances"]["target_relative_energy_tolerance"]))
    pair = sorted(planned, key=lambda cp: (Fraction(cp[0]["base_energy_exact"]), cp[0]["orbit_id"]))[:2]
    series = sorted(protocol["validation"]["bond_hops_series"])
    rows, study = [], {"status": "NOT_ASSESSABLE", "orbits": [c["orbit_id"] for c, _ in pair], "tolerance_eV": float(tol)}
    if len(pair) == 2:
        for h in series:
            corrections = []
            for c, p in pair:
                try:
                    p_h = p if h == hops else plan_regions(spec, c, h, protocol)
                except PipelineError as exc:
                    study["failure"] = {"bond_hops": h, **exc.as_dict()}
                    break
                res = results[c["orbit_id"]] if h == hops else backend.evaluate(p_h, purpose="embedding_boundary_study")
                records.update({r["artifact_id"]: r for v in res.values() for r in v.values()})
                out = assemble_corrected_energy(spec, c, p_h, res, backend)
                if out["record_type"] == "BlockedResult":
                    study["failure"] = {"bond_hops": h, **out["failure"]}
                    break
                corrections.append(Fraction(out["unvalidated_correction_exact"]))
            if "failure" in study:
                break
            rows.append({"bond_hops": h, "relative_correction_eV_per_fu": float(corrections[1] - corrections[0]),
                         "exact": str(corrections[1] - corrections[0])})
        if "failure" not in study:
            at = next(Fraction(r["exact"]) for r in rows if r["bond_hops"] == hops)
            deviation = abs(at - Fraction(rows[-1]["exact"]))
            study.update(status="CONVERGED" if deviation <= tol else "NOT_CONVERGED", deviation_eV=float(deviation))
    checks["boundary_convergence"] = {**study, "series": rows,
                                      "quantity": "correction(b) - correction(a) per formula unit vs bond_hops"}
    # 4. reference benchmarks for this scheme.
    benchmarks = []
    for path in protocol["validation"]["reference_benchmarks"]:
        bench = run_embedding_benchmark(json.loads((spec.root / path).read_text()))["report"]
        records[bench["artifact_id"]] = bench
        benchmarks.append({"file": path, "benchmark_id": bench["benchmark_id"], "status": bench["status"],
                           "report_id": bench["artifact_id"]})
    checks["reference_benchmarks"] = {"status": "PASSED" if benchmarks and all(b["status"] == "REPRODUCED" for b in benchmarks)
                                      else "FAILED" if benchmarks else "NONE_DECLARED", "benchmarks": benchmarks}
    passed = {"high_equals_low": "PASSED", "balanced_regions": "PASSED", "boundary_convergence": "CONVERGED",
              "reference_benchmarks": "PASSED"}
    unresolved = [f"{k}: {checks[k]['status']}" for k, want in passed.items() if checks[k]["status"] != want]
    report = make_record(
        "EmbeddingValidationReport", spec.provenance, scheme=SCHEME, protocol_id=protocol["protocol_id"],
        protocol_hash=backend.protocol_hash, embedding_fidelity=backend.fidelity, energy_expression=ENERGY_EXPRESSION,
        status="VALIDATED" if not unresolved else "UNVALIDATED", checks=checks, unresolved=unresolved,
        label="finite-cluster subtractive (mechanical) embedding; not WF-in-DFT",
        not_assessed=["alternative cluster spin states", "cluster basis-set convergence", "published Fe-MOF-74 reference"])
    return {"report": report, "records": records}


# -- embedding stage ---------------------------------------------------------------------------------------------
def correct_pool(spec, backend: ClusterBackend, pool: list) -> tuple:
    """(status, artifacts, detail) for the evaluated pool [(pool row, CalculationResult)].

    Plans, evaluates, validates and assembles every candidate with a DFT energy. Corrections become
    CorrectedEnergy records only when the scheme is VALIDATED; candidates that cannot be assembled keep a
    BlockedResult with the reason.
    """
    hops = backend.protocol["region"]["bond_hops"]
    arts, planned, results = {}, [], {}
    for row, calc in pool:
        if calc.get("status") != "SUCCEEDED":
            continue  # no DFT energy to correct; the ranking already reports it
        try:
            plan = plan_regions(spec, calc, hops, backend.protocol)
        except PipelineError as exc:
            arts[f"correction/{row['orbit_id']}"] = _blocked(spec, calc, exc.code, exc.detail)
            continue
        arts[f"embedding_plan/{row['orbit_id']}"] = plan
        results[calc["orbit_id"]] = backend.evaluate(plan)
        arts.update({f"cluster_result/{r['artifact_id']}": r for pair in results[calc["orbit_id"]].values()
                     for r in pair.values()})
        planned.append((calc, plan))
    validation = validate_scheme(spec, backend, planned, results)
    report = arts["embedding_validation"] = validation["report"]
    arts.update({f"embedding_evidence/{k}": v for k, v in validation["records"].items()})
    for calc, plan in planned:
        arts[f"correction/{calc['orbit_id']}"] = assemble_corrected_energy(spec, calc, plan, results[calc["orbit_id"]],
                                                                           backend, report)
    corrections = {k.split("/", 1)[1]: v for k, v in arts.items() if k.startswith("correction/")}
    applied = sorted(o for o, c in corrections.items() if c["record_type"] == "CorrectedEnergy")
    n_dft = sum(calc.get("status") == "SUCCEEDED" for _, calc in pool)
    if report["status"] != "VALIDATED":
        status = StageStatus.INCONCLUSIVE
    else:
        status = StageStatus.COMPLETED if len(applied) == n_dft else StageStatus.INCOMPLETE
    detail = {"scheme": SCHEME, "embedding_fidelity": backend.fidelity, "validation": report["status"],
              "unresolved": report["unresolved"], "checks": {k: v["status"] for k, v in report["checks"].items()},
              "n_with_dft_energy": n_dft, "n_corrected": len(applied),
              "blocked": [{"orbit_id": o, **c["failure"]} for o, c in sorted(corrections.items())
                          if c["record_type"] == "BlockedResult"],
              "ranking": ("corrected ranking emitted next to the DFT-only ranking" if report["status"] == "VALIDATED"
                          else "DFT-only: corrections are UNVALIDATED benchmark artifacts")}
    return status, arts, detail


# -- independently validated reference benchmark ----------------------------------------------------------------
def _molecule_structure(atoms: list, box_A: float = 100.0) -> dict:
    """A molecule as a periodic structure in a box far larger than any bond, so the periodic cutter applies."""
    return {"lattice": ((box_A, 0, 0), (0, box_A, 0), (0, 0, box_A)),
            "atoms": [{"element": a[0], "frac": tuple(0.5 + x / box_A for x in a[1:4])} for a in atoms]}


_BENCHMARK_REPORTS: dict = {}  # in-process memo: identical benchmark inputs and worker give the identical report


def run_embedding_benchmark(bench: dict, store=None) -> dict:
    """Subtractive embedding of a molecule against its own full high-level energy (independent reference).

    For every configuration: E_emb = E_low(full) + E_high(C) - E_low(C) with C the capped region of the declared
    seed atoms. Checks: (i) exact limit -- a region covering the whole molecule reproduces E_high(full);
    (ii) the embedded RELATIVE energy matches the full high-level one within ``tolerance_Ha``; (iii) the benchmark
    discriminates, i.e. the low level alone misses the reference by more than the tolerance.
    """
    from .cluster_worker import WORKER_VERSION, compute
    key = content_hash({"benchmark": bench, "worker": WORKER_VERSION})
    if key in _BENCHMARK_REPORTS:
        if store is not None:
            store.put(_BENCHMARK_REPORTS[key])
        return {"report": _BENCHMARK_REPORTS[key]}
    for key_ in ("benchmark_id", "scheme", "configurations", "region", "capping", "charge", "spin", "basis", "low_level",
                "high_level", "tolerance_Ha", "conv_tol_Ha", "source"):
        if bench.get(key_) is None:
            raise PipelineError("SCIENTIFIC_SPEC_MISSING", f"benchmark.{key_}")
    if bench["scheme"] != SCHEME:
        raise PipelineError("UNSUPPORTED_EMBEDDING", bench["scheme"])

    def energy(elements, coords, level):
        result = compute({"elements": elements, "coords_A": coords, "charge": bench["charge"], "spin": bench["spin"],
                          "basis": bench["basis"], "level": level, "conv_tol_Ha": bench["conv_tol_Ha"], "cluster_hash": ""})
        if result["status"] != "SUCCEEDED":
            raise PipelineError("BENCHMARK_CALCULATION_FAILED", str(result.get("error")))
        return Fraction(repr(result["energy_Ha"]))

    rows = {}
    for name, atoms in sorted(bench["configurations"].items()):
        structure = _molecule_structure(atoms)
        adjacency = bond_graph(structure, bench["region"]["bond_tolerance_A"])
        seeds = {i: (0, 0, 0) for i in bench["region"]["seed_atoms"]}
        cluster = cut_region(structure, adjacency, seeds, bench["region"]["bond_hops"], bench["capping"])
        elements, coords = [a[0] for a in atoms], [list(a[1:4]) for a in atoms]
        full_low, full_high = energy(elements, coords, bench["low_level"]), energy(elements, coords, bench["high_level"])
        c_low, c_high = (energy(cluster["elements"], cluster["coords_A"], bench[k]) for k in ("low_level", "high_level"))
        rows[name] = {"E_low_full": full_low, "E_high_full": full_high, "E_emb": full_low + c_high - c_low,
                      "cluster_inventory": cluster["inventory"], "n_caps": len(cluster["caps"])}
    # (i) exact limit on the first configuration: every atom in the region, nothing capped.
    first = sorted(bench["configurations"])[0]
    atoms = bench["configurations"][first]
    whole = cut_region(_molecule_structure(atoms), bond_graph(_molecule_structure(atoms), bench["region"]["bond_tolerance_A"]),
                       {i: (0, 0, 0) for i in range(len(atoms))}, 0, bench["capping"])
    limit_low = energy(whole["elements"], whole["coords_A"], bench["low_level"])
    limit_high = energy(whole["elements"], whole["coords_A"], bench["high_level"])
    limit = rows[first]["E_low_full"] + limit_high - limit_low - rows[first]["E_high_full"]
    a, b = sorted(rows)[:2]
    d_ref = rows[b]["E_high_full"] - rows[a]["E_high_full"]
    d_emb, d_low = rows[b]["E_emb"] - rows[a]["E_emb"], rows[b]["E_low_full"] - rows[a]["E_low_full"]
    tol = Fraction(str(bench["tolerance_Ha"]))
    balanced = rows[a]["cluster_inventory"] == rows[b]["cluster_inventory"]
    checks = {"exact_limit": {"passed": abs(limit) <= CONSISTENCY_TOL_HA, "residual_Ha": float(limit)},
              "balanced_regions": {"passed": balanced},
              "relative_energy": {"passed": abs(d_emb - d_ref) <= tol, "embedded_error_Ha": float(d_emb - d_ref)},
              "discriminating": {"passed": abs(d_low - d_ref) > tol, "low_level_error_Ha": float(d_low - d_ref)}}
    report = make_record(
        "EmbeddingBenchmarkReport", bench.get("provenance", "PHYSICAL"), benchmark_id=bench["benchmark_id"], scheme=SCHEME,
        energy_expression="E_emb = E_low(full) + E_high(C) - E_low(C)", source=bench["source"],
        reference_kind="independent full-system high-level calculation on the same geometries (not a published value)",
        levels={"low": bench["low_level"], "high": bench["high_level"], "basis": bench["basis"]},
        relative_energy={"of": f"{b} - {a}", "reference_Ha": float(d_ref), "embedded_Ha": float(d_emb), "low_level_Ha": float(d_low)},
        configurations={k: {"E_low_full_Ha": float(v["E_low_full"]), "E_high_full_Ha": float(v["E_high_full"]),
                            "E_embedded_Ha": float(v["E_emb"]), "cluster_inventory": v["cluster_inventory"],
                            "n_caps": v["n_caps"]} for k, v in rows.items()},
        tolerance_Ha=float(tol), checks=checks,
        status="REPRODUCED" if all(c["passed"] for c in checks.values()) else "NOT_REPRODUCED",
        note="validates the region cut, capping and assembly of the scheme against an independent calculation")
    _BENCHMARK_REPORTS[key] = report
    if store is not None:
        store.put(report)
    return {"report": report}
