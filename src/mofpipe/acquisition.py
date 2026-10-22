"""Task 05 (T05.2): declared evaluation-batch acquisition policy for physical projects.

``select_evaluation_batch`` picks training labels for the next refinement round (Task 08 uses it; the final
finalist pool is still the top-k surrogate ranking of ``candidates.select_pool``). Policy
(``protocols.acquisition``):

  batch_size        labels per round across all topologies
  weights           {"energy", "uncertainty", "diversity"}: linear weights of the three normalised scores
  audit_fraction    share of each topology's quota drawn uniformly (seeded) from its deprioritised half
  min_per_topology  every topology with acquirable orbits gets at least this many
  seed              acquisition seed (with the round number)

Scores per candidate orbit, each normalised to [0, 1] within its topology:
  energy       (E_max - E) / (E_max - E_min) with the round's model prediction (eV per formula unit)
  uncertainty  committee spread: population std of the grouped-CV fold models' predictions (not a bound)
  diversity    Euclidean distance in the model's feature space to the nearest labelled or already chosen orbit
Remaining quota after the minimum goes to topologies by need = (CV RMSE / tolerance, or 2 + that when no model
passes) x (largest label count / own label count), D'Hondt-style, so an uncertain, under-sampled topology
receives candidates. Only unlabelled, never-attempted orbits outside the frozen holdout are eligible.
Constraint penalties never enter: predictions are physical energies of feasible decorations.
"""
from __future__ import annotations

import math
import random
from fractions import Fraction

from .configuration_model import feature_vector
from .contract import PipelineError, make_record

POLICY_REQUIRED = ("batch_size", "weights", "audit_fraction", "min_per_topology", "seed")


def policy_errors(policy: dict) -> list:
    errors = [k for k in POLICY_REQUIRED if policy.get(k) is None]
    weights = policy.get("weights") or {}
    if not errors and (set(weights) != {"energy", "uncertainty", "diversity"} or any(w < 0 for w in weights.values())):
        errors.append("weights")
    if not errors and not 0 <= policy["audit_fraction"] < 1:
        errors.append("audit_fraction")
    return errors


def bits_of(orbit_id: str) -> list:
    return [int(c) for c in orbit_id.split(":")[1]]


def acquisition_model(selection: dict) -> dict | None:
    """The selected model, else the completed candidate with the lowest CV RMSE (exploration still needs a model)."""
    done = [c for c in selection["candidates"] if c["status"] == "COMPLETED"]
    chosen = [c for c in done if c["model_id"] == selection.get("selected_model")]
    return chosen[0] if chosen else min(done, key=lambda c: (c["cv_rmse_eV"], c["model_id"]), default=None)


def _scores(topo, model: dict, pool: list, labelled: list) -> list:
    names = model["features"]
    coef = [Fraction(c) for c in model["coefficients_exact"]]
    folds = [[Fraction(c) for c in f] for f in model["fold_coefficients_exact"]]
    rows = []
    for orbit in pool:
        x = feature_vector(topo, bits_of(orbit), names)
        pred = sum((c * v for c, v in zip(coef, x)), Fraction(0))
        committee = [float(sum((c * v for c, v in zip(f, x)), Fraction(0))) for f in folds]
        mean = sum(committee) / len(committee)
        rows.append({"orbit_id": orbit, "x": [float(v) for v, n in zip(x, names) if n != "constant"],
                     "predicted_eV": float(pred), "uncertainty_eV": math.sqrt(sum((c - mean) ** 2 for c in committee) / len(committee))})
    known = [[float(v) for v, n in zip(feature_vector(topo, bits, names), names) if n != "constant"] for bits in labelled]
    return rows, known


def _normalise(values: list) -> list:
    lo, hi = min(values, default=0.0), max(values, default=0.0)
    return [0.0 if hi == lo else (v - lo) / (hi - lo) for v in values]


def _distance(a, b) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _allocate(quota: dict, need: dict, available: dict, total: int) -> dict:
    remaining = total - sum(quota.values())
    while remaining > 0:
        open_ = [t for t in quota if quota[t] < available[t]]
        if not open_:
            break
        best = max(open_, key=lambda t: (need[t] / (quota[t] + 1), t))
        quota[best] += 1
        remaining -= 1
    return quota


def select_evaluation_batch(spec, split: dict, selections: dict, snapshot: dict, attempted: dict, round_index: int) -> dict:
    """select_evaluation_batch(candidate_set_id, model_snapshot_id, acquisition_policy_id, budget_id) -> AcquisitionBatch."""
    policy = spec.physical["protocols"]["acquisition"]
    missing = policy_errors(policy)
    if missing:
        raise PipelineError("SCIENTIFIC_SPEC_MISSING", f"protocols.acquisition.{missing[0]}")
    tol = float(spec.physical["tolerances"]["target_relative_energy_tolerance"])
    weights = policy["weights"]
    rng = random.Random(f"{policy['seed']}:{round_index}")
    pools, labelled, need = {}, {}, {}
    for tid in spec.topologies:
        groups = split["per_topology"][tid]
        pools[tid] = sorted(o for o in groups["unlabelled"] if o not in attempted.get(tid, set()))
        labelled[tid] = [e["bits"] for e in snapshot["entries"] if e["topology_id"] == tid]
    most = max((len(v) for v in labelled.values()), default=1) or 1
    for tid in spec.topologies:
        model = acquisition_model(selections[tid])
        base = (2.0 if selections[tid]["status"] != "COMPLETED" else 0.0) + (model["cv_rmse_eV"] / tol if model else 2.0)
        need[tid] = max(base, 1e-9) * most / max(len(labelled[tid]), 1)
    quota = {t: min(policy["min_per_topology"], len(pools[t])) for t in spec.topologies}
    quota = _allocate(quota, need, {t: len(pools[t]) for t in spec.topologies}, policy["batch_size"])
    chosen = []
    for tid, topo in spec.topologies.items():
        q = quota[tid]
        if q == 0:
            continue
        model = acquisition_model(selections[tid])
        if model is None:  # no fitted model yet: seeded uniform exploration
            for orbit in rng.sample(pools[tid], q):
                chosen.append({"topology_id": tid, "orbit_id": orbit, "reason": "exploration_without_model", "score": None})
            continue
        rows, known = _scores(topo, model, pools[tid], labelled[tid])
        energy = _normalise([-r["predicted_eV"] for r in rows])
        uncert = _normalise([r["uncertainty_eV"] for r in rows])
        for r, e, u in zip(rows, energy, uncert):
            r["energy_score"], r["uncertainty_score"] = e, u
        n_audit = min(int(math.floor(policy["audit_fraction"] * q + 0.5)), q)
        picks, remaining = [], list(rows)
        for _ in range(q - n_audit):
            dist = [min((_distance(r["x"], k) for k in known), default=1.0) for r in remaining]
            div = _normalise(dist)
            scored = [(weights["energy"] * r["energy_score"] + weights["uncertainty"] * r["uncertainty_score"]
                       + weights["diversity"] * d, r["orbit_id"], r, d) for r, d in zip(remaining, div)]
            score, _, best, d = max(scored, key=lambda t: (t[0], [-ord(ch) for ch in t[1]]))
            picks.append({"topology_id": tid, "orbit_id": best["orbit_id"], "reason": "policy_score",
                          "score": score, "components": {"energy": best["energy_score"], "uncertainty": best["uncertainty_score"],
                                                         "diversity": d},
                          "predicted_eV_per_fu": best["predicted_eV"], "committee_std_eV": best["uncertainty_eV"]})
            known.append(best["x"])
            remaining.remove(best)
        if n_audit:
            # Deprioritised region: the lower half of the energy+uncertainty score among what is left.
            ranked = sorted(remaining, key=lambda r: (weights["energy"] * r["energy_score"]
                                                      + weights["uncertainty"] * r["uncertainty_score"], r["orbit_id"]))
            low = ranked[:max(n_audit, len(ranked) // 2)]
            for r in rng.sample(low, min(n_audit, len(low))):
                picks.append({"topology_id": tid, "orbit_id": r["orbit_id"], "reason": "audit_deprioritised_region",
                              "score": None, "predicted_eV_per_fu": r["predicted_eV"], "committee_std_eV": r["uncertainty_eV"]})
        for p in picks:
            p["model_id"], p["selection_report_id"] = model["model_id"], selections[tid]["artifact_id"]
        chosen += picks
    return make_record(
        "AcquisitionBatch", spec.provenance, round=round_index, snapshot_id=snapshot["artifact_id"],
        policy=policy, quota=quota, need=need, selected=chosen,
        eligible={t: len(pools[t]) for t in spec.topologies},
        holdout_excluded=True, note="committee spread guides exploration only; it is not a calibrated error bound")
