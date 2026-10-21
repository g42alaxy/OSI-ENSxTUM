"""Task 08 (refinement) + Task 05 (acquisition): bounded, deterministic active learning.

Disabled for the initial fixture smoke test (project.json). When enabled it
acquires base-fidelity labels for training-split orbits only, refits against an
unchanged target definition and stops with an explicit StopReason. Budget
exhaustion is never reported as convergence.
"""
from __future__ import annotations

import random

from .configuration_model import feature_vector, fit_and_select
from .contract import PipelineError, StopReason, make_record
from .datasets import check_snapshot_against_split, ingest_labels, merge_snapshots


def plan_next_batch(spec, snapshot: dict, split: dict, batch_size: int, seed: int) -> list:
    """Diversity-driven acquisition in feature space, restricted to training-split orbits."""
    #TODO: Task 08 — registered energy/uncertainty/diversity acquisition policy, exploration outside the preferred
    #TODO: basin, revisiting uncertain topology exclusions and untested supercell patterns.
    labelled_orbits ={e["orbit_id"] for e in snapshot["entries"]}
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
        #TODO: Task 08 — stopping on stability over tolerances.consecutive_stable_rounds, tolerance-aware ranking
        #TODO: equivalence classes, coverage criteria and solver gaps; explicit model-family expansion on demand;
        #TODO: matched embedded-correction surrogate when justified.
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
