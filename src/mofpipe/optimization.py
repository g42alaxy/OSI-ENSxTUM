"""Task 04 (optimization): exact enumeration, exact branch-and-bound, and a classical heuristic.

Every candidate is decoded and its physical cost re-evaluated from the Hamiltonian, independently of the encoded
objective. ``branch_and_bound`` certifies optimality without enumeration and reports an honest gap when it stops
at its node or time limit. QAOA simulation lives in ``qaoa.py``.
"""
from __future__ import annotations

import math
import random
import time
from fractions import Fraction

from .configuration_model import predict
from .contract import PipelineError, StageStatus, bit_string, make_record
from .encoding import _IntPoly, complete_assignment
from .structures import n_ones_of, require_single_count, states_of


def exact_enumeration(spec, topo, hamiltonian: dict, encoding: dict | None = None) -> dict:
    """solve_arrangements(..., solver_protocol=exact_enumeration) -> SolverResult (certified)."""
    start = time.perf_counter()
    table = []
    encoded = _IntPoly(encoding) if encoding is not None else None
    for bits in states_of(spec, topo):
        energy = predict(hamiltonian, topo, bits)
        if encoded is not None and encoded(complete_assignment(encoding, bits)) != energy:
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


def branch_and_bound(spec, topo, hamiltonian: dict, node_limit: int = 1_000_000, time_limit_s: float = 60.0) -> dict:
    """Exact fixed-count minimisation of the physical polynomial (any degree) by depth-first branch and bound.

    Coefficients are scaled to integers by their exact common denominator, so rounding error is zero.
    Node bound: terms fixed by the assignment count exactly; every term still open contributes min(0, c) split
    equally over its free variables; the remaining r ones go to the r free variables with the smallest shares.
    This is a valid lower bound, so a completed search certifies optimality. At a limit the best feasible
    incumbent is returned with LB = min over unexplored node bounds and gap = incumbent - LB.
    """
    #TODO: Task 04 — return a Job for solves longer than the stage budget; LP/SDP bounds for large cells.
    from .encoding import physical_polynomial
    require_single_count(topo, "branch_and_bound")
    start = time.perf_counter()
    n, k = topo.n_sites, n_ones_of(spec, topo)
    poly = physical_polynomial(hamiltonian, topo)
    max_deg = max((len(key) for key in poly), default=1) or 1
    scale = math.lcm(*(Fraction(c).denominator for c in poly.values()), *range(1, max_deg + 1))
    terms = [(sum(1 << i for i in key), int(Fraction(c) * scale)) for key, c in poly.items()]
    full = (1 << n) - 1

    def bound(depth, ones):
        """(lower bound, exact value if complete) for variables < depth fixed with `ones` set."""
        zeros = ((1 << depth) - 1) & ~ones
        free = full & ~((1 << depth) - 1)
        fixed, share = 0, [0] * n
        for mask, c in terms:
            if mask & zeros:
                continue
            open_part = mask & free
            if not open_part:
                fixed += c * max_deg  # shares below are in units of 1/max_deg-free integers
            elif c < 0:
                members = [i for i in range(depth, n) if open_part >> i & 1]
                for i in members:
                    share[i] += c * max_deg // len(members)
        r = k - bin(ones).count("1")
        if r < 0 or r > n - depth:
            return None
        return fixed + sum(sorted(share[depth:])[:r])

    # Greedy feasible incumbent (add the cheapest one k times), so a limit always returns a feasible candidate.
    chosen = []
    for _ in range(k):
        options = [i for i in range(n) if i not in chosen]
        chosen.append(min(options, key=lambda i: predict(hamiltonian, topo, [int(j in chosen or j == i) for j in range(n)])))
    best_bits = sum(1 << i for i in chosen)
    best = bound(n, best_bits)
    stack, nodes, limit_hit = [(0, 0)], 0, None
    while stack:
        if nodes >= node_limit:
            limit_hit = "NODE_LIMIT"
            break
        if time.perf_counter() - start > time_limit_s:
            limit_hit = "TIME_LIMIT"
            break
        depth, ones = stack.pop()
        nodes += 1
        lb = bound(depth, ones)
        if lb is None or (best is not None and lb >= best):
            continue
        if depth == n:
            best, best_bits = lb, ones
            continue
        stack.append((depth + 1, ones))              # x_depth = 0 explored second
        stack.append((depth + 1, ones | 1 << depth))  # x_depth = 1 explored first
    open_bounds = [b for b in (bound(d, o) for d, o in stack) if b is not None] if limit_hit else []
    lower = min([best] + open_bounds)
    to_ev = lambda v: Fraction(v, scale * max_deg)  # noqa: E731
    bits = [best_bits >> i & 1 for i in range(n)]
    energy = predict(hamiltonian, topo, bits)  # decoded and re-evaluated independently of the search arithmetic
    if energy != to_ev(best):
        raise PipelineError("SOLVER_ENERGY_MISMATCH", f"{bit_string(bits)}: {energy} != {to_ev(best)}")
    return make_record(
        "SolverResult", spec.provenance, solver="branch_and_bound", topology_id=topo.topology_id,
        hamiltonian_id=hamiltonian["artifact_id"],
        optimality_status="OPTIMAL_CERTIFIED_BRANCH_AND_BOUND" if not limit_hit else f"UNCERTIFIED_{limit_hit}",
        stop_reason=limit_hit, nodes=nodes, node_limit=node_limit, time_limit_s=time_limit_s,
        minimum_energy_eV=float(energy), minimum_energy_exact=str(energy), minimizing_bitstrings=[bit_string(bits)],
        lower_bound_exact=str(to_ev(lower)), optimality_gap_eV=float(to_ev(best - lower)),
        coefficient_scaling={"integer_scale": scale * max_deg, "rounding_error_eV": 0},
        note="one minimiser is returned; degenerate minimisers are enumerated only by exact_enumeration",
        wall_time_s=round(time.perf_counter() - start, 4))


def quantum_solver_status(spec) -> dict:
    #TODO: Task 04 — QPU hardware adapter, only after simulator acceptance (scheduler controls budgets).
    enabled = spec.config["optimizer"].get("quantum_enabled", False)
    return {"qaoa_simulation": StageStatus.COMPLETED.value if enabled else StageStatus.DISABLED_BY_CONFIG.value,
            "qpu_hardware": StageStatus.DISABLED_BY_CONFIG.value,
            "note": ("QAOA statevector simulation benchmark on the frozen Hamiltonian; not evidence of quantum speedup."
                     if enabled else "No circuits executed; fixed-count swap reachability is a graph check only.")}


def table_from_result(result: dict) -> list:
    return [(tuple(int(c) for c in b), Fraction(e)) for b, e in result["energies"]]
