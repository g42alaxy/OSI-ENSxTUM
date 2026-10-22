"""Task 04 (optimization): exact fixed-composition enumeration and a classical heuristic.

Every candidate is decoded and its physical cost re-evaluated from the
Hamiltonian, independently of the encoded objective. Quantum solvers are
disabled in this build.
"""
from __future__ import annotations

import math
import random
import time
from fractions import Fraction

from .configuration_model import predict
from .contract import PipelineError, StageStatus, bit_string, make_record
from .encoding import _IntPoly
from .structures import n_ones_of, states_of


def exact_enumeration(spec, topo, hamiltonian: dict, encoding: dict | None = None) -> dict:
    """solve_arrangements(..., solver_protocol=exact_enumeration) -> SolverResult (certified)."""
    #TODO: Task 04 — exact MILP/constraint solver with reported optimality gaps and timeouts for cells too large
    #TODO: to enumerate; async Job return for long solves.
    start = time.perf_counter()
    table = []
    encoded = _IntPoly(encoding) if encoding is not None else None
    for bits in states_of(spec, topo):
        energy = predict(hamiltonian, topo, bits)
        if encoded is not None and encoded(bits) != energy:
            # On feasible states the penalty vanishes, so the encoding must equal the physical energy.
            raise PipelineError("ENCODING_DECODE_MISMATCH", f"{topo.topology_id}:{bit_string(bits)}")
        table.append((bit_string(bits), energy))
    minimum = min(e for _, e in table)
    minimizers = sorted(b for b, e in table if e == minimum)
    return make_record(
        "SolverResult", spec.provenance, solver="exact_enumeration", topology_id=topo.topology_id,
        hamiltonian_id=hamiltonian["artifact_id"], encoding_id=encoding["artifact_id"] if encoding else None,
        optimality_status="OPTIMAL_CERTIFIED_EXHAUSTIVE", feasibility="all candidates satisfy composition",
        n_feasible=len(table), minimum_energy_eV=float(minimum), minimum_energy_exact=str(minimum),
        minimizing_bitstrings=minimizers, energy_basis="physical surrogate (no penalty)",
        encoded_objective_cross_check=encoding is not None,
        energies=[[b, str(e)] for b, e in table], wall_time_s=round(time.perf_counter() - start, 4),
    )


def simulated_annealing(spec, topo, hamiltonian: dict, seed: int, n_starts: int = 8, n_steps: int = 3000,
                        t_start: float = 0.05, t_end: float = 1e-4) -> dict:
    """Multi-start annealing with composition-preserving A/B swaps; uncertified heuristic."""
    rng = random.Random(seed)
    n, k = topo.n_sites, n_ones_of(spec, topo)
    per_start = []
    for _ in range(n_starts):
        ones = set(rng.sample(range(n), k))
        bits = [int(i in ones) for i in range(n)]
        energy = predict(hamiltonian, topo, bits)
        best_bits, best = list(bits), energy
        for step in range(n_steps):
            temp = t_start * (t_end / t_start) ** (step / max(1, n_steps - 1))
            i = rng.choice([s for s in range(n) if bits[s] == 1])
            j = rng.choice([s for s in range(n) if bits[s] == 0])
            bits[i], bits[j] = 0, 1
            trial = predict(hamiltonian, topo, bits)
            delta = float(trial - energy)
            if delta <= 0 or rng.random() < math.exp(-delta / temp):
                energy = trial
                if energy < best:
                    best, best_bits = energy, list(bits)
            else:
                bits[i], bits[j] = 1, 0
        # Decode and re-evaluate the physical cost independently of the search state.
        if sum(best_bits) != k:
            raise PipelineError("INFEASIBLE_SOLVER_OUTPUT", bit_string(best_bits))
        per_start.append((bit_string(best_bits), predict(hamiltonian, topo, best_bits)))
    best_energy = min(e for _, e in per_start)
    return make_record(
        "SolverResult", spec.provenance, solver="simulated_annealing", topology_id=topo.topology_id,
        hamiltonian_id=hamiltonian["artifact_id"], optimality_status="UNCERTIFIED_HEURISTIC",
        configuration={"seed": seed, "n_starts": n_starts, "n_steps": n_steps, "t_start": t_start, "t_end": t_end,
                       "move": "swap one A and one B site (composition preserving)"},
        minimum_energy_eV=float(best_energy), minimum_energy_exact=str(best_energy),
        candidates=sorted({b for b, _ in per_start}), per_start=[[b, str(e)] for b, e in per_start],
    )


def quantum_solver_status(spec) -> dict:
    #TODO: Task 04 — QAOA simulation: XY exchange mixer on the connected feasible graph with feasible initial
    #TODO: states, count-preserving one-hot exchanges for multicomponent sites, decode/re-evaluate every sample,
    #TODO: record depth/shots/seeds/optimizer calls; benchmark vs exact/SA across seeds.
    #TODO: Task 04 — QPU hardware adapter, only after simulator acceptance (scheduler controls budgets).
    enabled = spec.config["optimizer"].get("quantum_enabled", False)
    return {"qaoa_simulation": StageStatus.DISABLED_BY_CONFIG.value if not enabled else StageStatus.UNSUPPORTED.value,
            "qpu_hardware": StageStatus.DISABLED_BY_CONFIG.value,
            "note": "No circuits executed; fixed-count swap reachability is a graph check only."}


def table_from_result(result: dict) -> list:
    return [(tuple(int(c) for c in b), Fraction(e)) for b, e in result["energies"]]
