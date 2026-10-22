"""Task 08 (refinement) + Task 05 (acquisition): bounded, deterministic active learning.

Disabled for the initial fixture smoke test (project.json). When enabled it
acquires base-fidelity labels for training-split orbits only, refits against an
unchanged target definition and stops with an explicit StopReason. Budget
exhaustion is never reported as convergence.
"""
from __future__ import annotations

import random
from fractions import Fraction

from .configuration_model import feature_vector, fit_and_select
from .contract import PipelineError, StopReason, make_record
from .datasets import check_snapshot_against_split, ingest_labels, merge_snapshots


def plan_next_batch(spec, snapshot: dict, split: dict, batch_size: int, seed: int) -> list:
    """Diversity-driven acquisition in feature space, restricted to training-split orbits."""
    # Fixture active learning (feature-space novelty). Physical projects use acquisition.select_evaluation_batch.
    #TODO: Task 08 — revisit uncertain topology exclusions once a screening policy exists (T02.3).
    labelled_orbits = {e["orbit_id"] for e in snapshot["entries"]}
    rng = random.Random(seed)
    batch = []
    for tid, topo in spec.topologies.items():
        allowed = set(split["per_topology"][tid]["training"])  # never development or holdout orbits
        known = [tuple(feature_vector(topo, e["bits"], spec.feature_order))
                 for e in snapshot["entries"] if e["topology_id"] == tid]
        pool, seen_orbits = [], set()
        for row in spec.candidate_rows:
            if row["topology_id"] != tid or row["orbit_id"] not in allowed:
                continue
            if row["orbit_id"] in labelled_orbits or row["orbit_id"] in seen_orbits:
                continue
            seen_orbits.add(row["orbit_id"])
            pool.append(row)
        rng.shuffle(pool)
        chosen = []
        for _ in range(min(batch_size, len(pool))):
            def novelty(row):
                f = tuple(feature_vector(topo, row["bits"], spec.feature_order))
                return min((sum(abs(a - b) for a, b in zip(f, g)) for g in known), default=10**9)
            best = max(pool, key=novelty)  # max keeps the first (seeded) maximiser
            pool.remove(best)
            known.append(tuple(feature_vector(topo, best["bits"], spec.feature_order)))
            chosen.append({"topology_id": tid, "bits": best["bits"], "orbit_id": best["orbit_id"],
                           "reason": "max_feature_space_novelty"})
        batch.extend(chosen)
    return batch


def _diagnose(selection: dict) -> str:
    if selection["status"] != "MODEL_INADEQUATE":
        return "PASS"
    max_p = max(c["n_parameters"] for c in selection["candidates"])
    if any(c["status"] != "COMPLETED" for c in selection["candidates"]) or selection["n_training"] < 2 * max_p:
        return "NEEDS_LABELS"
    # Every candidate is well determined yet fails: more labels cannot add a missing interaction term.
    return "MODEL_FAMILY_INADEQUATE"


def _labels_from_results(results: list) -> list:
    rows = []
    for r in results:
        if r["status"] != "SUCCEEDED":
            continue  # a failed calculation never becomes a label
        rows.append({"bits": r["bits"], "comparison_group": r["comparison_group"], "configuration_id": r["configuration_id"],
                     "energy_eV": r["base_energy_eV"], "energy_unit": r["energy_unit"], "fidelity": r["base_fidelity"],
                     "n_formula_units": r["n_formula_units"], "normalization": r["normalization"],
                     "orbit_id": r["orbit_id"], "provenance": r["provenance"], "status": r["status"],
                     "topology_id": r["topology_id"]})
    return rows


def run_active_learning(spec, store, backend, train: dict, dev: dict, split: dict) -> tuple[dict, dict]:
    al = spec.config["active_learning"]
    max_iter = al["max_iterations_if_enabled"]
    if max_iter > 3:
        raise PipelineError("UNBOUNDED_ACTIVE_LEARNING", str(max_iter))
    batch_size = al.get("batch_size_per_topology")
    if type(batch_size) is not int or batch_size <= 0:
        raise PipelineError("SCIENTIFIC_SPEC_MISSING", "active_learning.batch_size_per_topology")
    history, snapshots, stop = [], [train["artifact_id"]], None
    for iteration in range(max_iter + 1):
        selections = {tid: fit_and_select(spec, topo, train, dev) for tid, topo in spec.topologies.items()}
        diagnosis = {tid: _diagnose(sel) for tid, sel in selections.items()}
        entry = {"iteration": iteration, "training_snapshot_id": train["artifact_id"], "n_labels": train["n_labels"],
                 "selected": {t: s["selected_model"] for t, s in selections.items()}, "diagnosis": diagnosis}
        # Fixture stopping rule; physical projects use assess_refinement (stable rounds, leading sets).
        #TODO: Task 08 — solver-gap accounting for non-exhaustive solvers, explicit model-family expansion on
        #TODO: demand, matched embedded-correction surrogate when justified.
        if all(d == "PASS" for d in diagnosis.values()):
            stop = StopReason.CONVERGED_WITHIN_SCOPE
        elif any(d == "MODEL_FAMILY_INADEQUATE" for d in diagnosis.values()):
            stop = StopReason.MODEL_INADEQUATE
        elif iteration == max_iter:
            stop = StopReason.BUDGET_EXHAUSTED
        if stop is not None:
            history.append(entry)
            break
        batch = plan_next_batch(spec, train, split, batch_size, seed=spec.config["random_seed"] + iteration)
        if not batch:
            stop = StopReason.INCONCLUSIVE
            history.append({**entry, "acquired": []})
            break
        job_id = backend.submit(batch, purpose=f"active_learning_iteration_{iteration}")
        results = backend.collect(job_id, train["fidelity"], train["comparison_group"])
        new = ingest_labels(spec, _labels_from_results(results), role="training", source=job_id)
        check_snapshot_against_split(new, split, "training")
        train = merge_snapshots(spec, train, new, role="training")
        store.put(new)
        store.put(train)
        snapshots.append(train["artifact_id"])
        history.append({**entry, "job_id": job_id, "acquired": [b["orbit_id"] for b in batch]})
    report = make_record("RefinementReport", spec.provenance, stop_reason=stop.value, max_iterations=max_iter,
                         batch_size_per_topology=batch_size, iterations=history, snapshot_ids=snapshots,
                         convergence_criterion="every topology selects a model with development RMSE <= threshold",
                         final_holdout_used=False)
    return report, train


# -- physical projects: DFT active learning with declared stopping rules (T08.1, T08.2) ----------------------
STOPPING_REQUIRED = ("max_rounds",)


def stopping_errors(stopping: dict, tolerances: dict) -> list:
    errors = [f"protocols.stopping.{k}" for k in STOPPING_REQUIRED if stopping.get(k) is None]
    if type(stopping.get("max_rounds")) is not int or not 0 < stopping.get("max_rounds", 0) <= 20:
        errors.append("protocols.stopping.max_rounds")
    rounds = tolerances.get("consecutive_stable_rounds")
    if type(rounds) is not int or rounds < 1:
        errors.append("tolerances.consecutive_stable_rounds")
    return sorted(set(errors))


def leading_sets(spec, hamiltonians: dict, spaces: dict, k: int, tol: Fraction) -> dict:
    """Orbits whose predicted energy is within tol of the k-th lowest one, per topology and pooled over topologies.

    Tolerance-aware equivalence: reordering inside the tolerance band does not change the set.
    """
    from .acquisition import bits_of
    from .configuration_model import predict

    energies = {}
    for tid, ham in hamiltonians.items():
        topo = spec.topologies[tid]
        for orbit in spaces[tid]["orbit_sizes"]:
            energies[orbit] = predict(ham, topo, bits_of(orbit))

    def lead(items: dict) -> list:
        values = sorted(items.values())
        if not values:
            return []
        cutoff = values[min(k, len(values)) - 1] + tol
        return sorted(o for o, e in items.items() if e <= cutoff)

    out = {tid: lead({o: e for o, e in energies.items() if o.split(":")[0] == tid}) for tid in hamiltonians}
    out["pooled"] = lead(energies)
    return out


def assess_refinement(round_index: int, selections: dict, lead: dict | None, previous: dict | None,
                      stable_rounds: int, required: int, max_rounds: int, acquirable: dict | None = None) -> tuple:
    """assess_refinement(snapshot_id, stopping_policy_id) -> (StopReason | None, stable_rounds, diagnosis)."""
    diagnosis = {}
    for tid, sel in selections.items():
        if sel["status"] == "COMPLETED":
            diagnosis[tid] = "PASS"
            continue
        done = [c for c in sel["candidates"] if c["status"] == "COMPLETED"]
        max_p = max((c.get("n_parameters", 0) for c in sel["candidates"]), default=0)
        well_determined = len(done) == len(sel["candidates"]) and sel["n_training"] >= 3 * max(max_p, 1)
        diagnosis[tid] = "MODEL_FAMILY_INADEQUATE" if well_determined else "NEEDS_LABELS"
    if all(d == "PASS" for d in diagnosis.values()) and lead is not None:
        stable_rounds = stable_rounds + 1 if previous is not None and previous == lead else 0
        if stable_rounds >= required:
            return StopReason.CONVERGED_WITHIN_SCOPE, stable_rounds, diagnosis
    else:
        stable_rounds = 0
    if any(d == "MODEL_FAMILY_INADEQUATE" for d in diagnosis.values()):
        return StopReason.MODEL_INADEQUATE, stable_rounds, diagnosis
    if acquirable is not None and any(d == "NEEDS_LABELS" and not acquirable.get(t) for t, d in diagnosis.items()):
        return StopReason.INCONCLUSIVE, stable_rounds, diagnosis  # more labels needed, none left to acquire
    if round_index >= max_rounds:
        return StopReason.BUDGET_EXHAUSTED, stable_rounds, diagnosis
    return None, stable_rounds, diagnosis
