"""Task 09 (ranking/validation): compatible-energy ranking, competition coverage, structure checks."""
from __future__ import annotations

from fractions import Fraction

from .contract import PipelineError, StageStatus, make_record
from .datasets import check_energy_compatibility


def rank_candidates(spec, pool_rows: list, results: list, level: str) -> dict:
    """rank_candidates(candidate_ids, comparison_policy_id) -> RankingReport for one energy level."""
    if level not in ("base", "corrected"):
        raise PipelineError("UNKNOWN_ENERGY_LEVEL", level)
    by_cid = {}
    for r in results:
        by_cid.setdefault(r["orbit_id"], r)
    missing, failed, rows = [], [], []
    for cand in pool_rows:
        res = by_cid.get(cand["orbit_id"])
        if res is None:
            missing.append(cand["orbit_id"])
        elif res["status"] != "SUCCEEDED":
            failed.append({"orbit_id": cand["orbit_id"], "status": res["status"]})
        else:
            rows.append((cand, res))
    check_energy_compatibility([{"fidelity": res[f"{level}_fidelity"], "energy_unit": res["energy_unit"],
                                 "normalization": res["normalization"], "comparison_group": res["comparison_group"],
                                 "status": res["status"], "provenance": res["provenance"],
                                 "configuration_id": res["configuration_id"]} for _, res in rows], spec.provenance)
    table = []
    for cand, res in rows:
        energy = Fraction(res[f"{level}_energy_exact"])
        table.append({"orbit_id": cand["orbit_id"], "configuration_id": res["configuration_id"],
                      "topology_id": cand["topology_id"], "occupation": "".join(map(str, cand["bits"])),
                      "bits": cand["bits"], "surrogate_base_energy_eV": cand["predicted_base_energy_eV"],
                      "base_energy_eV": res["base_energy_eV"], "correction_eV": res["correction_eV"],
                      "corrected_energy_eV": res["corrected_energy_eV"], "ranking_energy_eV": float(energy),
                      "ranking_energy_exact": str(energy), "base_fidelity": res["base_fidelity"],
                      "corrected_fidelity": res["corrected_fidelity"], "provenance": res["provenance"],
                      "calculation_result_id": res["artifact_id"]})
    table.sort(key=lambda r: (Fraction(r["ranking_energy_exact"]), r["orbit_id"]))
    values = [Fraction(r["ranking_energy_exact"]) for r in table]
    for idx, row in enumerate(table):
        row["rank"] = idx + 1
        row["tied_with_previous"] = idx > 0 and values[idx] == values[idx - 1]
        row["n_degenerate"] = values.count(values[idx])
    minima = {}
    for row in table:
        minima.setdefault(row["topology_id"], row["ranking_energy_eV"])
    complete = not missing and not failed
    return make_record(
        "RankingReport", spec.provenance, level=level,
        fidelity=table[0][f"{level}_fidelity"] if table else None, comparison_group=rows[0][1]["comparison_group"] if rows else None,
        status=StageStatus.COMPLETED.value if complete else StageStatus.INCOMPLETE.value,
        missing_evaluations=missing, failed_evaluations=failed,
        tie_policy="exact rational equality is a true degeneracy; display order breaks ties by orbit_id",
        coverage="screened finalist pool (top-k unique orbits per graph), not a global certificate over all legal states",
        winning_topology=table[0]["topology_id"] if table else None, pool_minima_eV=minima, ranked=table,
    )


def ranking_order_changed(base: dict, corrected: dict) -> bool:
    return [r["orbit_id"] for r in base["ranked"]] != [r["orbit_id"] for r in corrected["ranked"]]


def evaluate_reference_mixtures(spec, ranking: dict) -> dict:
    """No phase-separated reference set is supplied, so no stability claim is possible."""
    #TODO: Task 09 — phase-separated reference set at the same overall composition and energy level; solve the
    #TODO: reference mixture balance explicitly and report incomplete coverage.
    return make_record("CompetitionReport", spec.provenance, ranking_id=ranking["artifact_id"],
                       status=StageStatus.INCOMPLETE.value, reference_set=None,
                       reason="no reference phases supplied; result is a relative ranking, not a stability certificate")


def validate_structures(spec) -> dict:
    #TODO: Task 09 — validate_structure wrappers (version, parameters, raw evidence): MOFChecker sanity checks,
    #TODO: CrystalNets/MOFid topology identity, Zeo++ pore descriptors, post-relaxation connectivity and
    #TODO: residual force/stress, phonon/elastic stability for finalists, PXRD (and validated NMR) comparison
    #TODO: with referenced measurements.
    skipped =StageStatus.SKIPPED_NO_ATOMIC_GEOMETRY.value
    return {name: skipped for name in ("atomic_construction", "mofchecker", "crystalnets_topology", "zeopp_pores",
                                       "post_relaxation_connectivity", "phonon_elastic_stability", "pxrd_comparison",
                                       "nmr_comparison")}
