"""Task 04 (T04.3): QAOA circuits on a Qiskit fake IBM device (one job per process).

Run by ``qpu.hardware_benchmark`` as ``PYTHON qpu_worker.py job.json`` in the job directory; writes ``result.json``.
For each seed the circuit is: Dicke state |D(n,k)> (Baertschi & Eidenbenz split-and-cyclic-shift construction,
exact), then per layer the phase separator exp(-i gamma C) with C the scaled physical polynomial on the decision
bits (P / CP / multi-controlled phase for 1-, 2-, 3-body terms) and the XY exchange mixer exp(-i beta (XX+YY)/2)
on every slot edge in list order. Angles come from the classical statevector optimisation (qaoa.run_qaoa); the
device only samples. The ideal (noiseless) probabilities are returned for the cross-check against the simulator,
then the circuit is transpiled for the device and sampled with its noise model (qiskit-aer, from_backend).

Bit order: qubit i is slot i; Qiskit count strings put qubit n-1 first. Fake devices are calibration snapshots of
real IBM devices simulated locally: noisy simulation, not hardware execution.
This file is self-contained (no qumof imports) so any interpreter with qiskit, qiskit-aer and
qiskit-ibm-runtime can run it.
"""
from __future__ import annotations

import json
import math
import sys
import time

WORKER_VERSION = "1"


def _scs(qc, q: list, n: int, k: int) -> None:
    from qiskit.circuit.library import RYGate
    qc.cx(q[n - 2], q[n - 1])
    qc.cry(2 * math.acos(math.sqrt(1 / n)), q[n - 1], q[n - 2])
    qc.cx(q[n - 2], q[n - 1])
    for l in range(2, k + 1):
        qc.cx(q[n - l - 1], q[n - 1])
        qc.append(RYGate(2 * math.acos(math.sqrt(l / n))).control(2), [q[n - 1], q[n - l], q[n - l - 1]])
        qc.cx(q[n - l - 1], q[n - 1])


def dicke(n: int, k: int):
    from qiskit import QuantumCircuit
    qc, q = QuantumCircuit(n, name=f"dicke_{n}_{k}"), list(range(n))
    qc.x(q[n - k:])
    for m in range(n, k, -1):
        _scs(qc, q[:m], m, k)
    for m in range(k, 1, -1):
        _scs(qc, q[:m], m, m - 1)
    return qc


def qaoa_circuit(job: dict, params: list):
    from qiskit.circuit.library import MCPhaseGate, XXPlusYYGate
    n = job["n_qubits"]
    qc = dicke(n, job["n_ones"])
    for gamma, beta in zip(params[0::2], params[1::2]):
        for sites, coeff in job["terms"]:
            angle = -gamma * coeff
            if len(sites) == 1:
                qc.p(angle, sites[0])
            elif len(sites) == 2:
                qc.cp(angle, sites[0], sites[1])
            else:
                qc.append(MCPhaseGate(angle, len(sites) - 1), list(sites))
        for i, j in job["edges"]:
            qc.append(XXPlusYYGate(2 * beta, 0.0), [i, j])
    return qc


def _device(name: str):
    from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2
    for backend in FakeProviderForBackendV2().backends():
        if backend.name == name:
            return backend
    raise ValueError(f"unknown fake device {name}")


def main(argv) -> int:
    import importlib.metadata as md

    from qiskit import transpile
    from qiskit.quantum_info import Statevector
    from qiskit_aer import AerSimulator

    job = json.loads(open(argv[0]).read())
    backend = _device(job["device"])
    simulator = AerSimulator.from_backend(backend)
    identity = {"device": backend.name, "device_qubits": backend.num_qubits, "basis_gates": sorted(backend.operation_names),
                "qiskit": md.version("qiskit"), "qiskit_aer": md.version("qiskit-aer"),
                "qiskit_ibm_runtime": md.version("qiskit-ibm-runtime"), "worker_version": WORKER_VERSION,
                "execution": "local noisy simulation of a fake-device calibration snapshot (not hardware)"}
    runs = []
    for run in job["runs"]:
        t0 = time.perf_counter()
        qc = qaoa_circuit(job, run["params"])
        ideal = Statevector(qc).probabilities().tolist() if job["n_qubits"] <= 16 else None
        measured = qc.copy()
        measured.measure_all()
        compiled = transpile(measured, backend, optimization_level=job["optimization_level"], seed_transpiler=run["seed"])
        ops = {k: int(v) for k, v in compiled.count_ops().items()}
        two_qubit = sum(v for k, v in ops.items() if k in ("ecr", "cx", "cz", "rzz"))
        counts = simulator.run(compiled, shots=run["shots"], seed_simulator=run["seed"]).result().get_counts()
        layout = compiled.layout.final_index_layout() if compiled.layout else None
        runs.append({"seed": run["seed"], "ideal_probabilities": ideal, "counts": counts,
                     "transpiled": {"depth": compiled.depth(), "ops": ops, "two_qubit_gates": two_qubit,
                                    "physical_qubits": layout, "logical_depth": qc.decompose().depth()},
                     "wall_time_s": round(time.perf_counter() - t0, 3)})
    json.dump({"identity": identity, "runs": runs}, open("result.json", "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
