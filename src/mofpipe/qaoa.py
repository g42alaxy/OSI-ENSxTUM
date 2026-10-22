"""Task 04 (T04.2): QAOA statevector simulation with a constraint-preserving exchange mixer.

The register is simulated exactly inside its feasible subspace: every state assigns one species per slot with
the declared counts (binary: Hamming weight k; multicomponent: one-hot with species counts). The mixer is a
product over slot edges (pairs_distance_1, in list order) and species pairs of exchange rotations
exp(-i beta (|ab><ba| + h.c.)), the XY mixer for binary slots. Each rotation swaps species between two slots,
so it can never leave the feasible subspace; ``verify_mixer_preserves_constraints`` checks this on the full qubit
register. The initial state is the uniform superposition over feasible states (a Dicke state for binary slots;
its preparation circuit is not compiled). The phase separator is the encoding's diagonal cost on decision
variables, scaled to unit spread. Every sample is decoded and re-evaluated with the physical Hamiltonian,
and any mismatch with the encoded cost raises ENCODING_DECODE_MISMATCH (bit-order or ancilla errors).

This is a classical simulation: it is not evidence of quantum speedup.
"""
from __future__ import annotations

import bisect
import cmath
import math
import random
import time
from fractions import Fraction
from itertools import product

from .configuration_model import predict
from .contract import PipelineError, StageStatus, bit_string, make_record
from .encoding import evaluate_encoding, one_hot_states
from .structures import n_ones_of

PROTOCOL_REQUIRED = ("layers", "shots", "seeds", "restarts", "max_evaluations")


def protocol_errors(protocol: dict | None) -> list:
    if not protocol:
        return ["protocol"]
    missing = [k for k in PROTOCOL_REQUIRED if protocol.get(k) is None]
    if not missing and (type(protocol["layers"]) is not int or protocol["layers"] < 1 or not protocol["seeds"]):
        missing.append("layers/seeds")
    if protocol.get("kind", "qaoa_simulation") != "qaoa_simulation":
        missing.append(f"kind={protocol.get('kind')} (QPU hardware is T04.3)")
    return missing


# -- feasible subspace and mixer ---------------------------------------------------------------------------
def exchange_pairs(states: list, edges) -> list:
    """Per edge: (u, v) index pairs of states that differ by swapping the species of the two slots."""
    index = {s: u for u, s in enumerate(states)}
    out = []
    for i, j in edges:
        pairs = []
        for u, s in enumerate(states):
            if s[i] < s[j]:
                t = list(s)
                t[i], t[j] = t[j], t[i]
                pairs.append((u, index[tuple(t)]))
        out.append(pairs)
    return out


def apply_mixer(psi: list, pairs_per_edge: list, beta: float) -> None:
    c, s = math.cos(beta), -1j * math.sin(beta)
    for pairs in pairs_per_edge:
        for u, v in pairs:
            a, b = psi[u], psi[v]
            psi[u], psi[v] = c * a + s * b, s * a + c * b


def evolve(params: list, cost: list, pairs_per_edge: list) -> list:
    n = len(cost)
    psi = [complex(1 / math.sqrt(n))] * n
    for gamma, beta in zip(params[0::2], params[1::2]):
        for u in range(n):
            psi[u] *= cmath.exp(-1j * gamma * cost[u])
        apply_mixer(psi, pairs_per_edge, beta)
    return psi


def verify_mixer_preserves_constraints(n_slots: int, n_species: int, edges, one_hot: bool, mixer: str = "exchange") -> dict:
    """Full-register check: every nonzero mixer transition connects bit strings with equal constraint values.

    Binary slots use one qubit per slot (constraint: Hamming weight). One-hot slots use n_species qubits per slot
    (constraints: one bit per slot and the count of every species). ``mixer="x"`` is the unconstrained
    transverse-field mixer, which must fail this check.
    """
    width = n_slots * n_species if one_hot else n_slots

    def constraints(x):
        if not one_hot:
            return (sum(x),)
        per_slot = tuple(sum(x[i * n_species:(i + 1) * n_species]) for i in range(n_slots))
        per_species = tuple(sum(x[i * n_species + a] for i in range(n_slots)) for a in range(n_species))
        return per_slot + per_species

    def moves(x):
        if mixer == "x":
            return [tuple(b ^ (q == p) for q, b in enumerate(x)) for p in range(width)]
        out = []
        for i, j in edges:
            if not one_hot:
                if x[i] != x[j]:
                    y = list(x)
                    y[i], y[j] = y[j], y[i]
                    out.append(tuple(y))
                continue
            for a, b in product(range(n_species), repeat=2):
                ia, ib, ja, jb = i * n_species + a, i * n_species + b, j * n_species + a, j * n_species + b
                if a != b and x[ia] and x[jb] and not x[ib] and not x[ja]:
                    y = list(x)
                    y[ia], y[jb], y[ib], y[ja] = 0, 0, 1, 1
                    out.append(tuple(y))
        return out

    violations = sum(constraints(y) != constraints(x) for x in product((0, 1), repeat=width) for y in moves(x))
    return {"mixer": mixer, "register_qubits": width, "n_states_checked": 2 ** width,
            "constraint_violating_transitions": violations, "preserves_constraints": violations == 0}


# -- classical optimiser (seeded Nelder-Mead) ---------------------------------------------------------------
def nelder_mead(f, x0: list, step: float, max_evaluations: int) -> tuple[list, float, int]:
    simplex = [list(x0)] + [[x + (step if k == d else 0) for k, x in enumerate(x0)] for d in range(len(x0))]
    values = [f(x) for x in simplex]
    evals = len(simplex)
    while evals < max_evaluations:
        order = sorted(range(len(simplex)), key=values.__getitem__)
        simplex, values = [simplex[i] for i in order], [values[i] for i in order]
        centroid = [sum(p[k] for p in simplex[:-1]) / (len(simplex) - 1) for k in range(len(x0))]
        worst = simplex[-1]
        reflect = [c + (c - w) for c, w in zip(centroid, worst)]
        fr = f(reflect)
        evals += 1
        if fr < values[0]:
            expand = [c + 2 * (c - w) for c, w in zip(centroid, worst)]
            fe = f(expand)
            evals += 1
            simplex[-1], values[-1] = (expand, fe) if fe < fr else (reflect, fr)
        elif fr < values[-2]:
            simplex[-1], values[-1] = reflect, fr
        else:
            contract = [c + 0.5 * (w - c) for c, w in zip(centroid, worst)]
            fc = f(contract)
            evals += 1
            if fc < values[-1]:
                simplex[-1], values[-1] = contract, fc
            else:
                simplex = [simplex[0]] + [[b + 0.5 * (p - b) for p, b in zip(pt, simplex[0])] for pt in simplex[1:]]
                values = [values[0]] + [f(pt) for pt in simplex[1:]]
                evals += len(simplex) - 1
    best = min(range(len(simplex)), key=values.__getitem__)
    return simplex[best], values[best], evals


# -- QAOA on a pipeline Hamiltonian -------------------------------------------------------------------------
def run_qaoa(spec, topo, hamiltonian: dict, encoding: dict, protocol: dict, seed: int) -> dict:
    """One seeded QAOA simulation: optimise angles, sample shots, decode and re-evaluate every sample."""
    t0 = time.perf_counter()
    states = [tuple(b) for b in one_hot_states(topo.n_sites, [topo.n_sites - n_ones_of(spec, topo), n_ones_of(spec, topo)])]
    encoded = [evaluate_encoding(encoding, s) for s in states]  # what the device's phase separator implements
    physical = [predict(hamiltonian, topo, s) for s in states]
    for s, e, p in zip(states, encoded, physical):
        if e != p:  # the penalty vanishes on feasible states, so any difference is a bit-order/ancilla error
            raise PipelineError("ENCODING_DECODE_MISMATCH", f"{topo.topology_id}:{bit_string(s)}: {e} != {p}")
    values = [float(e) for e in encoded]
    mean = sum(values) / len(values)
    spread = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)) or 1.0
    cost = [(v - mean) / spread for v in values]
    pairs = exchange_pairs(states, topo.pairs_d1)
    rng = random.Random(seed)
    p = protocol["layers"]

    def expectation(params):
        return sum(abs(a) ** 2 * c for a, c in zip(evolve(params, cost, pairs), cost))

    best_params, best_value, calls = None, None, 0
    for _ in range(protocol["restarts"]):
        x0 = [rng.uniform(0, math.pi / 2) for _ in range(2 * p)]
        params, value, evals = nelder_mead(expectation, x0, 0.3, protocol["max_evaluations"])
        calls += evals
        if best_value is None or value < best_value:
            best_params, best_value = params, value
    t_solver = time.perf_counter()
    probs = [abs(a) ** 2 for a in evolve(best_params, cost, pairs)]
    cumulative, acc = [], 0.0
    for q in probs:
        acc += q
        cumulative.append(acc)
    samples = [min(bisect.bisect_left(cumulative, rng.random() * acc), len(states) - 1) for _ in range(protocol["shots"])]
    feasible = [u for u in samples if sum(states[u]) == n_ones_of(spec, topo)]  # decode: post-selection
    sampled = [predict(hamiltonian, topo, states[u]) for u in feasible]  # re-evaluated, not read from the encoding
    minimum = min(physical)
    optimal = [u for u, e in enumerate(physical) if e == minimum]
    sorted_sampled = sorted(sampled)
    return {
        "seed": seed, "optimal_parameters": [round(x, 6) for x in best_params], "expected_scaled_cost": best_value,
        "optimizer_calls": calls, "shots": protocol["shots"], "postselection_rate": len(feasible) / len(samples),
        "best_sample_energy_exact": str(sorted_sampled[0]), "median_sample_energy_exact": str(sorted_sampled[len(sorted_sampled) // 2]),
        "best_sample_bitstring": bit_string(states[min(feasible, key=lambda u: physical[u])]),
        "success_probability": sum(probs[u] for u in optimal),
        "empirical_success_rate": sum(physical[u] == minimum for u in feasible) / len(samples),
        "found_optimum": sorted_sampled[0] == minimum,
        "wall_time_end_to_end_s": round(time.perf_counter() - t0, 4), "wall_time_final_circuit_and_sampling_s":
            round(time.perf_counter() - t_solver, 4)}


def qaoa_benchmark(spec, topo, hamiltonian: dict, encoding: dict, exact: dict, protocol: dict) -> dict:
    """QAOA across seeds versus exact enumeration and simulated annealing on the same frozen Hamiltonian."""
    from .optimization import simulated_annealing
    from .structures import swap_graph_connected

    reach = swap_graph_connected(topo, topo.n_sites, n_ones_of(spec, topo))
    if not reach["connected"]:
        raise PipelineError("MIXER_GRAPH_DISCONNECTED", f"{topo.topology_id}: exchanges reach {reach['n_reached']}")
    check = verify_mixer_preserves_constraints(topo.n_sites, 2, topo.pairs_d1, one_hot=False) \
        if topo.n_sites <= 16 else {"preserves_constraints": None, "note": "register too large for the full check"}
    runs = [run_qaoa(spec, topo, hamiltonian, encoding, protocol, seed) for seed in protocol["seeds"]]
    annealing = []
    for seed in protocol["seeds"]:
        t0 = time.perf_counter()
        sa = simulated_annealing(spec, topo, hamiltonian, seed)
        annealing.append({"seed": seed, "best_energy_exact": sa["minimum_energy_exact"],
                          "found_optimum": Fraction(sa["minimum_energy_exact"]) == Fraction(exact["minimum_energy_exact"]),
                          "wall_time_s": round(time.perf_counter() - t0, 4)})
    agree = all(r["found_optimum"] for r in runs)
    success = sorted(r["success_probability"] for r in runs)
    return make_record(
        "QuantumBenchmark", spec.provenance, solver="qaoa_statevector_simulation", topology_id=topo.topology_id,
        hamiltonian_id=hamiltonian["artifact_id"], encoding_id=encoding["artifact_id"], exact_solver_id=exact["artifact_id"],
        exact_minimum_exact=exact["minimum_energy_exact"],
        status="AGREES_WITH_EXACT" if agree else "DID_NOT_REACH_OPTIMUM",
        success_criterion="the best decoded sample of every seed equals the exact minimum",
        configuration={"layers": protocol["layers"], "shots": protocol["shots"], "seeds": protocol["seeds"],
                       "restarts": protocol["restarts"], "max_evaluations": protocol["max_evaluations"],
                       "optimizer": "Nelder-Mead (seeded starts in [0, pi/2])",
                       "mixer": "product of XY exchange rotations over pairs_distance_1 in list order",
                       "initial_state": "uniform superposition over the feasible subspace (Dicke state, not compiled)",
                       "phase_separator": "encoding diagonal on decision variables, scaled to zero mean and unit spread",
                       "bit_order": encoding["bit_order"], "n_qubits": topo.n_sites + len(encoding.get("ancillas", [])),
                       "ancillas_in_circuit": False,
                       "exchange_gates_per_layer": len(topo.pairs_d1)},
        mixer_constraint_check=check, mixer_reachability=reach, runs=runs, simulated_annealing=annealing,
        summary={"median_success_probability": success[len(success) // 2],
                 "qaoa_seeds_reaching_optimum": sum(r["found_optimum"] for r in runs),
                 "annealing_seeds_reaching_optimum": sum(a["found_optimum"] for a in annealing),
                 "qaoa_end_to_end_s": sum(r["wall_time_end_to_end_s"] for r in runs),
                 "annealing_s": sum(a["wall_time_s"] for a in annealing)},
        note="classical statevector simulation; not evidence of quantum speedup",
        stage_status=StageStatus.COMPLETED.value)
