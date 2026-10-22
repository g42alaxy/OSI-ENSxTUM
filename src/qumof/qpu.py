"""Task 04 (T04.3): QPU hardware adapter, executed on Qiskit fake IBM devices (e.g. ``fake_brisbane``).

Accepted only after the simulator (qaoa.py): angles are optimised classically on the exact statevector simulation
of the same frozen Hamiltonian, then the scheduler submits one sampling job per seed to the device; no optimiser
iteration ever runs on the device and no agent submits. ``qpu_worker.py`` (an interpreter with qiskit,
qiskit-aer and qiskit-ibm-runtime) builds the circuit (exact Dicke-state preparation, physical-polynomial phase
separator, XY exchange mixer), checks its noiseless probabilities against the simulator, transpiles it for the
device and samples it with the device's noise model. Every sample is decoded, post-selected on the composition and
re-evaluated with the physical Hamiltonian. Shots count against ``budgets.max_qpu_shots`` when declared.

Fake devices are calibration snapshots simulated locally: the results measure noise robustness of the compiled
circuit, not hardware execution, and are never evidence of quantum speedup. Jobs are idempotent on their inputs.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

from .configuration_model import predict
from .contract import PipelineError, StageStatus, bit_string, content_hash, make_record
from .encoding import one_hot_states, physical_polynomial
from .structures import n_ones_of

KIND = "qpu_fake_device"
WORKER = Path(__file__).with_name("qpu_worker.py")
CIRCUIT_MATCH_TOL = 1e-6


def protocol_errors(protocol: dict) -> list:
    """(code, field) for the hardware fields; the shared QAOA fields are checked by qaoa.protocol_errors."""
    errors = []
    if not str(protocol.get("device", "")).startswith("fake_"):
        errors.append(("UNSUPPORTED_QUANTUM_BACKEND", "device (a Qiskit fake device such as fake_brisbane)"))
    if not protocol.get("python"):
        errors.append(("SCIENTIFIC_SPEC_MISSING", "python (interpreter with qiskit, qiskit-aer, qiskit-ibm-runtime)"))
    if protocol.get("optimization_level", 3) not in (0, 1, 2, 3):
        errors.append(("SCIENTIFIC_SPEC_MISSING", "optimization_level (0-3)"))
    return errors


def _python(spec, path: str) -> str:
    resolved = shutil.which(path) or (os.path.abspath(spec.root / path) if (spec.root / path).exists() else None)
    if resolved is None:
        raise PipelineError("BACKEND_UNAVAILABLE", f"QPU worker interpreter not found: {path}")
    return resolved


def hardware_benchmark(spec, store, topo, hamiltonian: dict, encoding: dict, exact: dict, protocol: dict) -> dict:
    from .qaoa import evolve, exchange_pairs, run_qaoa

    k = n_ones_of(spec, topo)
    states = [tuple(b) for b in one_hot_states(topo.n_sites, [topo.n_sites - k, k])]
    physical = [predict(hamiltonian, topo, s) for s in states]
    values = [float(e) for e in physical]
    mean = sum(values) / len(values)
    spread = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)) or 1.0
    # Scaled exactly as the simulator's phase separator: the polynomial equals the encoded cost on feasible states.
    terms = sorted([list(s), float(Fraction(c)) / spread] for s, c in physical_polynomial(hamiltonian, topo).items() if s)
    shots = protocol["shots"]
    budget = ((spec.physical or {}).get("budgets") or {}).get("max_qpu_shots")
    used = store.state["counters"].get("qpu_shots", 0)
    if budget is not None and used + shots * len(protocol["seeds"]) > budget:
        raise PipelineError("BUDGET_EXHAUSTED", f"max_qpu_shots={budget}")
    simulated = {seed: run_qaoa(spec, topo, hamiltonian, encoding, protocol, seed) for seed in protocol["seeds"]}
    job = {"device": protocol["device"], "n_qubits": topo.n_sites, "n_ones": k, "edges": [list(e) for e in topo.pairs_d1],
           "terms": terms, "optimization_level": protocol.get("optimization_level", 3),
           "runs": [{"seed": s, "params": simulated[s]["optimal_parameters"], "shots": shots} for s in protocol["seeds"]]}
    worker_sha = hashlib.sha256(WORKER.read_bytes()).hexdigest()
    job_id = "qpujob:" + content_hash({"job": job, "worker_sha256": worker_sha})[:20]
    job_dir = store.jobs / job_id.replace(":", "__")
    result_path = job_dir / "result.json"
    if result_path.exists():
        store.bump("qpu_cache_hits")
    else:
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "job.json").write_text(json.dumps(job))
        proc = subprocess.run([_python(spec, protocol["python"]), str(WORKER), "job.json"], cwd=job_dir,
                              capture_output=True, text=True, timeout=protocol.get("timeout_s", 3600))
        if proc.returncode != 0 or not result_path.exists():
            raise PipelineError("BACKEND_UNAVAILABLE", f"QPU worker failed: {proc.stderr[-400:]}")
        store.bump("qpu_shots", shots * len(protocol["seeds"]))
    result = json.loads(result_path.read_text())
    pairs = exchange_pairs(states, topo.pairs_d1)
    scaled = [(v - mean) / spread for v in values]
    minimum = min(physical)
    runs = []
    for run in result["runs"]:
        sim = simulated[run["seed"]]
        index = [sum(b << i for i, b in enumerate(s)) for s in states]
        mismatch = None
        if run["ideal_probabilities"] is not None:  # the compiled circuit implements the simulated evolution
            reference = [abs(a) ** 2 for a in evolve(sim["optimal_parameters"], scaled, pairs)]
            mismatch = max(abs(run["ideal_probabilities"][u] - p) for u, p in zip(index, reference))
        samples = [(tuple(int(c) for c in reversed(bits)), n) for bits, n in run["counts"].items()]
        feasible = [(s, n) for s, n in samples if sum(s) == k]
        n_shots = sum(n for _, n in samples)
        energies = sorted((predict(hamiltonian, topo, s), n) for s, n in feasible)  # re-evaluated, not read back
        best = energies[0][0] if energies else None
        median = None
        if energies:
            acc, half = 0, sum(n for _, n in energies) / 2
            for e, n in energies:
                acc += n
                if acc >= half:
                    median = e
                    break
        runs.append({"seed": run["seed"], "optimal_parameters": sim["optimal_parameters"],
                     "simulated_success_probability": sim["success_probability"],
                     "circuit_vs_simulator_max_abs_probability_difference": mismatch,
                     "shots": n_shots, "postselection_rate": sum(n for _, n in feasible) / n_shots,
                     "empirical_success_rate": sum(n for e, n in energies if e == minimum) / n_shots,
                     # per feasible sample, against drawing feasible states uniformly at random (no quantum signal)
                     "success_per_feasible_sample": (sum(n for e, n in energies if e == minimum) / sum(n for _, n in feasible)
                                                     if feasible else 0.0),
                     "uniform_random_baseline": sum(e == minimum for e in physical) / len(states),
                     "best_sample_energy_exact": None if best is None else str(best),
                     "median_feasible_sample_energy_exact": None if median is None else str(median),
                     "best_sample_bitstring": bit_string(min(feasible, key=lambda x: predict(hamiltonian, topo, x[0]))[0])
                     if feasible else None,
                     "found_optimum": best == minimum, "transpiled": run["transpiled"], "wall_time_s": run["wall_time_s"]})
    circuit_ok = all(r["circuit_vs_simulator_max_abs_probability_difference"] is None or
                     r["circuit_vs_simulator_max_abs_probability_difference"] <= CIRCUIT_MATCH_TOL for r in runs)
    if not circuit_ok:
        raise PipelineError("QPU_CIRCUIT_MISMATCH", f"{topo.topology_id}: compiled circuit differs from the simulated QAOA")
    return make_record(
        "QuantumHardwareRun", spec.provenance, solver="qaoa_sampling_on_fake_device", topology_id=topo.topology_id,
        hamiltonian_id=hamiltonian["artifact_id"], encoding_id=encoding["artifact_id"], exact_solver_id=exact["artifact_id"],
        exact_minimum_exact=exact["minimum_energy_exact"], job_id=job_id, device=result["identity"],
        status="AGREES_WITH_EXACT" if all(r["found_optimum"] for r in runs) else "DID_NOT_REACH_OPTIMUM",
        success_criterion="the best decoded feasible sample of every seed equals the exact minimum",
        configuration={"layers": protocol["layers"], "shots": shots, "seeds": protocol["seeds"],
                       "optimization_level": job["optimization_level"], "n_qubits": topo.n_sites,
                       "initial_state": "Dicke state (Baertschi-Eidenbenz, exact, compiled)",
                       "phase_separator": "physical polynomial on decision bits, scaled to zero mean and unit spread",
                       "mixer": "XX+YY exchange on pairs_distance_1 in list order", "angles": "statevector-optimised",
                       "bit_order": "qubit i = slot i; count strings reversed on decoding"},
        circuit_matches_simulator=circuit_ok, runs=runs,
        summary={"seeds_reaching_optimum": sum(r["found_optimum"] for r in runs),
                 "median_postselection_rate": sorted(r["postselection_rate"] for r in runs)[len(runs) // 2],
                 "median_empirical_success_rate": sorted(r["empirical_success_rate"] for r in runs)[len(runs) // 2],
                 "median_enrichment_over_random": sorted(r["success_per_feasible_sample"] / r["uniform_random_baseline"]
                                                         for r in runs)[len(runs) // 2],
                 "max_two_qubit_gates": max(r["transpiled"]["two_qubit_gates"] for r in runs)},
        note="fake-device noisy simulation of a calibration snapshot: not hardware execution, not evidence of speedup",
        stage_status=StageStatus.COMPLETED.value)
