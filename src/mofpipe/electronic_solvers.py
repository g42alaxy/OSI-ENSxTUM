"""Task 07 (T07.2): electronic solvers on one ActiveSpaceArtifact.

Every solver reads h1, h2, E_core and the electron counts from the same artifact and records its
``hamiltonian_hash``, so ED, CASCI, VQE and SQD always solve the identical electronic problem.

* ``exact_diagonalization``: independent determinant-basis FCI (this module's own second-quantised Hamiltonian
  builder; no PySCF FCI code). Exact within the active space.
* ``pyscf_casci``: PySCF ``fci.direct_spin1`` on the stored integrals (cross-check of the builder above).
* ``vqe_simulation``: Jordan-Wigner statevector simulation restricted to the (N_alpha, N_beta) sector, spin-orbital
  UCCSD ansatz from the Hartree-Fock determinant, BFGS. Qubit q = 2p (alpha) / 2p+1 (beta), interleaved; occupation
  bit q of the integer is qubit q (little-endian). No symmetry tapering.
* ``sqd_simulation``: sample-based quantum diagonalisation. Shots are drawn from a supplied state (the VQE state),
  optional declared bit-flip noise is applied, wrong-sector samples are recovered from average occupations, and
  H is diagonalised in the span of sampled alpha strings x beta strings. The samples are SIMULATED.
* ``sqd_hardware``: QPU sampling adapter. No backend is configured, so it is BLOCKED (credentials live only in
  backend services).
"""
from __future__ import annotations

import random
import time
from itertools import combinations

from .contract import PipelineError, StageStatus, make_record

ORDERING = "qubit 2p = alpha spin orbital p, qubit 2p+1 = beta spin orbital p; occupation bit q is qubit q"
EXACT_SOLVERS = ("exact_diagonalization", "pyscf_casci")
VARIATIONAL_SOLVERS = ("vqe_simulation", "sqd_simulation")
CONSISTENCY_TOL_HA = 1e-8  # numerical agreement between exact solvers; variational results may not go below


def _np():
    try:
        import numpy as np
        return np
    except ImportError as exc:
        raise PipelineError("BACKEND_UNAVAILABLE", f"numpy is required for electronic solvers: {exc}") from exc


# -- second quantisation in the occupation-number (Jordan-Wigner) basis ---------------------------------------
def _apply(occ: int, ops) -> tuple[int, int] | None:
    """Apply ladder operators right-to-left: ops = [(qubit, is_creator), ...]. Returns (sign, new_occ) or None."""
    sign = 1
    for q, creator in reversed(ops):
        occupied = occ >> q & 1
        if occupied == creator:
            return None
        if bin(occ & ((1 << q) - 1)).count("1") % 2:
            sign = -sign  # Jordan-Wigner string over lower qubits
        occ ^= 1 << q
    return sign, occ


def sector_determinants(norb: int, nelec) -> list:
    """Occupation integers of every determinant with (N_alpha, N_beta), alpha-major order."""
    na, nb = nelec
    out = []
    for a in combinations(range(norb), na):
        for b in combinations(range(norb), nb):
            out.append(sum(1 << (2 * p) for p in a) | sum(1 << (2 * p + 1) for p in b))
    return out


def hamiltonian_matrix(h1, h2, ecore: float, dets: list, norb: int):
    """<D'|H|D> on the given determinants: H = E_core + sum h_pq a+_ps a_qs + 1/2 sum (pq|rs) a+_ps a+_rt a_st a_qs."""
    np = _np()
    index = {d: k for k, d in enumerate(dets)}
    mat = np.zeros((len(dets), len(dets)))
    one = [(p, q, h1[p, q]) for p in range(norb) for q in range(norb) if abs(h1[p, q]) > 1e-14]
    two = [(p, q, r, s, h2[p, q, r, s]) for p in range(norb) for q in range(norb) for r in range(norb)
           for s in range(norb) if abs(h2[p, q, r, s]) > 1e-14]
    for k, d in enumerate(dets):
        mat[k, k] += ecore
        for p, q, v in one:
            for sp in (0, 1):
                res = _apply(d, [(2 * p + sp, True), (2 * q + sp, False)])
                if res and res[1] in index:
                    mat[index[res[1]], k] += v * res[0]
        for p, q, r, s, v in two:
            for sp in (0, 1):
                for st in (0, 1):
                    res = _apply(d, [(2 * p + sp, True), (2 * r + st, True), (2 * s + st, False), (2 * q + sp, False)])
                    if res and res[1] in index:
                        mat[index[res[1]], k] += 0.5 * v * res[0]
    return mat


def _problem(artifact: dict):
    from .active_space import problem_from_artifact
    return problem_from_artifact(artifact)


def _result(artifact: dict, solver: str, energy, status: str, **fields) -> dict:
    return make_record("ElectronicSolverResult", artifact["provenance"], solver=solver,
                       active_space_id=artifact["artifact_id"], hamiltonian_hash=artifact["hamiltonian_hash"],
                       energy_Ha=None if energy is None else float(energy), status=status, units="Hartree", **fields)


# -- solvers -----------------------------------------------------------------------------------------------
def exact_diagonalization(artifact: dict) -> dict:
    np = _np()
    h1, h2, ecore, nelec, norb = _problem(artifact)
    dets = sector_determinants(norb, nelec)
    t0 = time.perf_counter()
    values, vectors = np.linalg.eigh(hamiltonian_matrix(h1, h2, ecore, dets, norb))
    return _result(artifact, "exact_diagonalization", values[0], StageStatus.COMPLETED.value, exactness="EXACT_IN_ACTIVE_SPACE",
                   sector={"alpha": nelec[0], "beta": nelec[1]}, n_determinants=len(dets),
                   ground_state_weights={str(d): round(float(c) ** 2, 12) for d, c in zip(dets, vectors[:, 0]) if c ** 2 > 1e-8},
                   qubit_ordering=ORDERING, wall_time_s=round(time.perf_counter() - t0, 4))


def pyscf_casci(artifact: dict) -> dict:
    h1, h2, ecore, nelec, norb = _problem(artifact)
    try:
        from pyscf import fci
    except ImportError as exc:
        raise PipelineError("BACKEND_UNAVAILABLE", str(exc)) from exc
    solver = fci.direct_spin1.FCI()
    solver.conv_tol = 1e-12
    energy, _ = solver.kernel(h1, h2, norb, nelec, ecore=ecore)
    return _result(artifact, "pyscf_casci", energy, StageStatus.COMPLETED.value, exactness="EXACT_IN_ACTIVE_SPACE",
                   implementation="pyscf.fci.direct_spin1 on the stored integrals")


def _uccsd_generators(norb: int, nelec, dets: list):
    """Spin-conserving singles and doubles from the HF determinant as real antisymmetric sector matrices."""
    np = _np()
    index = {d: k for k, d in enumerate(dets)}
    occ_a, occ_b = list(range(nelec[0])), list(range(nelec[1]))
    occ = [2 * p for p in occ_a] + [2 * p + 1 for p in occ_b]
    vir = [2 * p for p in range(nelec[0], norb)] + [2 * p + 1 for p in range(nelec[1], norb)]
    excitations = [([a], [i]) for i in occ for a in vir if a % 2 == i % 2]
    excitations += [([a, b], [i, j]) for i, j in combinations(occ, 2) for a, b in combinations(vir, 2)
                    if (a % 2) + (b % 2) == (i % 2) + (j % 2)]
    gens = []
    for creators, annihilators in excitations:
        t = np.zeros((len(dets), len(dets)))
        ops = [(q, True) for q in creators] + [(q, False) for q in reversed(annihilators)]
        for k, d in enumerate(dets):
            res = _apply(d, ops)
            if res and res[1] in index:
                t[index[res[1]], k] = res[0]
        gens.append(t - t.T)
    return gens, excitations


def vqe_simulation(artifact: dict, seed: int = 0, max_iterations: int = 200) -> dict:
    np = _np()
    from scipy.linalg import expm
    from scipy.optimize import minimize
    h1, h2, ecore, nelec, norb = _problem(artifact)
    dets = sector_determinants(norb, nelec)
    ham = hamiltonian_matrix(h1, h2, ecore, dets, norb)
    hf = sum(1 << (2 * p) for p in range(nelec[0])) | sum(1 << (2 * p + 1) for p in range(nelec[1]))
    reference = np.zeros(len(dets))
    reference[dets.index(hf)] = 1.0
    gens, excitations = _uccsd_generators(norb, nelec, dets)
    calls = {"n": 0}

    def state(theta):
        psi = reference
        for g, t in zip(gens, theta):
            psi = expm(t * g) @ psi
        return psi

    def energy(theta):
        calls["n"] += 1
        psi = state(theta)
        return float(psi @ ham @ psi)

    t0 = time.perf_counter()
    x0 = np.random.default_rng(seed).normal(scale=1e-2, size=len(gens))
    opt = minimize(energy, x0, method="BFGS", options={"maxiter": max_iterations, "gtol": 1e-9})
    psi = state(opt.x)
    return _result(artifact, "vqe_simulation", opt.fun, StageStatus.COMPLETED.value, exactness="VARIATIONAL_UPPER_BOUND",
                   mapping="Jordan-Wigner", qubit_ordering=ORDERING, n_qubits=2 * norb, symmetry_tapering="none",
                   state_preparation_sector={"alpha": nelec[0], "beta": nelec[1], "reference": "Hartree-Fock determinant"},
                   ansatz="spin-orbital UCCSD, product of exp(theta_k (T_k - T_k^+)) in excitation order",
                   n_parameters=len(gens), parameters=[round(float(t), 10) for t in opt.x],
                   optimizer={"method": "BFGS (finite-difference gradient)", "seed": seed, "energy_evaluations": calls["n"],
                              "converged": bool(opt.success)},
                   simulator="exact statevector within the particle-number sector (no shot noise)",
                   final_state={str(d): float(c) for d, c in zip(dets, psi) if abs(c) > 1e-10},
                   wall_time_s=round(time.perf_counter() - t0, 4))


def _recover(sample: int, norb: int, nelec, occupancy: list) -> int | None:
    """Configuration recovery: flip the least (most) likely occupied (empty) spin orbitals until the sector holds."""
    for spin, target in ((0, nelec[0]), (1, nelec[1])):
        qubits = [2 * p + spin for p in range(norb)]
        while (n := sum(sample >> q & 1 for q in qubits)) != target:
            if n > target:
                q = min((q for q in qubits if sample >> q & 1), key=lambda q: (occupancy[q], q))
            else:
                q = max((q for q in qubits if not sample >> q & 1), key=lambda q: (occupancy[q], -q))
            sample ^= 1 << q
    return sample


def sqd_simulation(artifact: dict, state: dict, shots: int, seed: int, flip_probability: float = 0.0,
                   source_result_id: str | None = None) -> dict:
    """SQD from SIMULATED samples of ``state`` ({occupation int as str: amplitude}) produced by ``source_result_id``."""
    np = _np()
    h1, h2, ecore, nelec, norb = _problem(artifact)
    rng = random.Random(seed)
    support = [(int(d), float(a) ** 2) for d, a in state.items()]
    total = sum(w for _, w in support)
    raw = []
    for _ in range(shots):
        x, acc = rng.random() * total, 0.0
        for d, w in support:
            acc += w
            if x <= acc:
                break
        for q in range(2 * norb):
            if rng.random() < flip_probability:
                d ^= 1 << q  # declared readout-noise model of the simulation
        raw.append(d)

    def in_sector(d):
        return (sum(d >> (2 * p) & 1 for p in range(norb)), sum(d >> (2 * p + 1) & 1 for p in range(norb))) == tuple(nelec)

    valid = [d for d in raw if in_sector(d)]
    occupancy = [sum(d >> q & 1 for d in valid) / max(len(valid), 1) for q in range(2 * norb)]
    recovered = [d if in_sector(d) else _recover(d, norb, nelec, occupancy) for d in raw]
    alpha_mask = sum(1 << (2 * p) for p in range(norb))
    alphas = sorted({d & alpha_mask for d in recovered})
    betas = sorted({d & ~alpha_mask for d in recovered})
    subspace = [a | b for a in alphas for b in betas]
    values = np.linalg.eigvalsh(hamiltonian_matrix(h1, h2, ecore, subspace, norb))
    return _result(artifact, "sqd_simulation", values[0], StageStatus.COMPLETED.value, exactness="VARIATIONAL_UPPER_BOUND",
                   sample_source="SIMULATED (statevector sampling, declared bit-flip noise)",
                   source_result_id=source_result_id, shots=shots, seed=seed,
                   flip_probability=flip_probability, in_sector_fraction=len(valid) / shots,
                   recovered_samples=shots - len(valid), n_alpha_strings=len(alphas), n_beta_strings=len(betas),
                   subspace_dimension=len(subspace), full_sector_dimension=len(sector_determinants(norb, nelec)),
                   qubit_ordering=ORDERING)


def sqd_hardware(artifact: dict, backend: str | None = None) -> dict:
    """QPU sample collection for SQD. Hardware submission is scheduler-controlled and needs a configured backend."""
    #TODO: Task 07 — hardware SQD adapter: submit the VQE/LUCJ circuit to a configured QPU backend, record
    #TODO: hardware/circuit/shot provenance, then reuse sqd_simulation's recovery and subspace diagonalisation.
    return _result(artifact, "sqd_hardware", None, StageStatus.BLOCKED.value,
                   failure={"code": "BACKEND_UNAVAILABLE",
                            "detail": f"no QPU backend configured ({backend}); credentials live in backend services only"})


def solver_consistency(results: list) -> dict:
    """Exact solvers must agree with each other; variational ones may not go below the exact energy."""
    exact = [r["energy_Ha"] for r in results if r["solver"] in EXACT_SOLVERS and r["energy_Ha"] is not None]
    problems = []
    if exact and max(exact) - min(exact) > CONSISTENCY_TOL_HA:
        problems.append(f"exact solvers disagree by {max(exact) - min(exact):.3e} Ha")
    for r in results:
        if exact and r["solver"] in VARIATIONAL_SOLVERS and r["energy_Ha"] is not None \
                and r["energy_Ha"] < min(exact) - CONSISTENCY_TOL_HA:
            problems.append(f"{r['solver']} ({r.get('seed')}) is below the exact energy")
    return {"consistent": not problems, "problems": problems, "tolerance_Ha": CONSISTENCY_TOL_HA,
            "hamiltonian_hashes": sorted({r["hamiltonian_hash"] for r in results})}


def run_electronic_benchmark(spec: dict, store=None) -> dict:
    """Prepare the active space, run every requested solver on it, compare with the published reference."""
    from .active_space import prepare_active_space
    for key in ("benchmark_id", "system", "protocol", "solvers", "published_reference"):
        if spec.get(key) is None:
            raise PipelineError("SCIENTIFIC_SPEC_MISSING", f"benchmark.{key}")
    artifact = prepare_active_space(spec["system"], spec["protocol"])
    results = {}
    for name in spec["solvers"]:
        if name == "exact_diagonalization":
            results[name] = exact_diagonalization(artifact)
        elif name == "pyscf_casci":
            results[name] = pyscf_casci(artifact)
        elif name == "vqe_simulation":
            results[name] = vqe_simulation(artifact, **spec.get("vqe", {}))
        elif name == "sqd_simulation":
            if "vqe_simulation" not in results:  # the sampled state is a stored result, never an untracked one
                results["vqe_simulation"] = vqe_simulation(artifact, **spec.get("vqe", {}))
            source = results["vqe_simulation"]
            cfg = spec.get("sqd", {})
            results[name] = [sqd_simulation(artifact, source["final_state"], cfg.get("shots", 1000), s,
                                            cfg.get("flip_probability", 0.0), source["artifact_id"])
                             for s in cfg.get("seeds", [0])]
        elif name == "sqd_hardware":
            results[name] = sqd_hardware(artifact, spec.get("sqd", {}).get("backend"))
        else:
            raise PipelineError("UNSUPPORTED_SOLVER", name)
    ref = spec["published_reference"]
    exact = results.get("exact_diagonalization") or results.get("pyscf_casci")
    comparisons = {}
    if exact is not None:
        comparisons["exact_vs_published_Ha"] = exact["energy_Ha"] - ref["E_total_Ha"]
        for name, res in results.items():
            for r in (res if isinstance(res, list) else [res]):
                if r["energy_Ha"] is not None:
                    comparisons.setdefault(f"{name}_minus_exact_Ha", []).append(r["energy_Ha"] - exact["energy_Ha"])
    if "E_HF_Ha" in ref:
        comparisons["hf_vs_published_Ha"] = artifact["mean_field"]["energy_Ha"] - ref["E_HF_Ha"]
    flat = [r for res in results.values() for r in (res if isinstance(res, list) else [res])]
    consistency = solver_consistency(flat)
    reproduced = exact is not None and abs(comparisons["exact_vs_published_Ha"]) <= ref["tolerance_Ha"] and \
        abs(comparisons.get("hf_vs_published_Ha", 0.0)) <= ref["tolerance_Ha"] and consistency["consistent"]
    report = make_record(
        "ElectronicBenchmarkReport", artifact["provenance"], benchmark_id=spec["benchmark_id"],
        active_space_id=artifact["artifact_id"], hamiltonian_hash=artifact["hamiltonian_hash"],
        identical_problem=all(r["hamiltonian_hash"] == artifact["hamiltonian_hash"] for r in flat),
        published_reference=ref, comparisons=comparisons, solver_consistency=consistency,
        status="REPRODUCED" if reproduced else "NOT_REPRODUCED",
        solver_status={r["solver"] + (f":{r.get('seed')}" if r.get("seed") is not None else ""): r["status"] for r in flat},
        result_ids=[r["artifact_id"] for r in flat], correction_status="UNVALIDATED",
        note="reproduces a calculation from the literature; it is not agreement with experiment")
    if store is not None:
        for record in [artifact, report] + flat:
            store.put(record)
    return {"active_space": artifact, "results": results, "report": report}
