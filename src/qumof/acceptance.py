"""Harness-side acceptance checks. Reads harness/ and reference_outputs/ AFTER a run is finalised.

Never imported by the learner modules (orchestrator and below).
"""
from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

from .configuration_model import predict_exact


def _close(a, b, tol=1e-10):
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


def check_run(spec, report: dict) -> dict:
    root = spec.root
    expected = json.loads((root / "harness/expected_results.json").read_text())[spec.case]
    truth = expected["correction_tests"][spec.correction_mode]
    reference = json.loads((root / f"reference_outputs/{spec.case}_{spec.correction_mode}.json").read_text())
    holdout = [json.loads(l) for l in (root / f"harness/{spec.case}_holdout.jsonl").read_text().splitlines() if l.strip()]
    checks = []

    def check(name, ok, detail=""):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    check("pipeline_completed", report.get("pipeline_status") == "COMPLETED", report.get("pipeline_status"))
    check("no_runtime_agents", report.get("runtime_agents_enabled") is False and report.get("llm_calls") == 0)
    check("winning_topology", report.get("winning_topology") == truth["winning_topology"],
          f"{report.get('winning_topology')} vs {truth['winning_topology']}")
    ref_winner_pool = reference["pooled_winner"]
    check("winner_matches_reference", report.get("winning_topology") == ref_winner_pool)
    for tid, exp in expected["topologies"].items():
        sec, ref = report["topologies"].get(tid, {}), reference["topologies"][tid]
        sol, ms = sec.get("solver", {}), sec.get("model_selection", {})
        check(f"{tid}:selected_model", ms.get("selected_model") == expected["expected_selected_model"], ms.get("selected_model"))
        check(f"{tid}:n_feasible", sol.get("n_feasible") == exp["n_feasible"])
        check(f"{tid}:n_symmetry_orbits", sec.get("n_symmetry_orbits") == exp["n_symmetry_orbits"])
        check(f"{tid}:base_minimum", _close(sol.get("base_minimum_energy_eV"), exp["minimum_energy_eV"]))
        check(f"{tid}:minimizing_bitstrings", sol.get("minimizing_bitstrings") == exp["minimizing_bitstrings"])
        check(f"{tid}:ground_state_orbits", sol.get("ground_state_orbits") == exp["ground_state_orbit_ids"])
        check(f"{tid}:corrected_minimum", _close((report.get("corrected_pool_minima_eV") or {}).get(tid), truth["minima_eV"][tid]))
        # Untouched holdout, evaluated only now with the frozen Hamiltonian.
        ham = sec.get("hamiltonian", {})
        topo = spec.topologies[tid]
        rows = [r for r in holdout if r["topology_id"] == tid]
        if ham:
            coef = [Fraction(c) for c in ham["coefficients_exact"]]
            errs = [predict_exact(ham["features"], coef, topo, r["bits"]) - Fraction(repr(r["energy_eV"])) for r in rows]
            value = math.sqrt(float(sum(e * e for e in errs) / len(errs)))
            check(f"{tid}:holdout_rmse", value <= 1e-10, f"{value:.3e} eV over {len(rows)} holdout orbits")
        else:
            check(f"{tid}:holdout_rmse", False, "no Hamiltonian")
        # Comparison with the bundled reference implementation.
        pool = [(r["orbit_id"], r["predicted_base_energy_eV"]) for r in sec.get("selected_pool", [])]
        ref_pool = [(r["orbit_id"], r["base_energy_eV"]) for r in ref["selected_pool"]]
        check(f"{tid}:pool_matches_reference", [p[0] for p in pool] == [p[0] for p in ref_pool]
              and all(_close(a[1], b[1]) for a, b in zip(pool, ref_pool)), f"{len(pool)} vs {len(ref_pool)}")
        enc = sec.get("encoding", {})
        check(f"{tid}:encoding_kind", enc.get("target") == ref["base_model_encoding"], enc.get("target"))
        check(f"{tid}:encoding_identity", enc.get("verification", {}).get("status") == "COMPLETED")
        if "qubo" in ref:
            art = json.loads((Path(report["_run_dir"]) / "artifacts" / (enc["artifact_id"].replace(":", "__") + ".json")).read_text()) \
                if "_run_dir" in report else None
            if art is not None:
                same = (_close(art["constant_eV"], ref["qubo"]["constant_eV"])
                        and all(_close(a, b) for a, b in zip(art["linear_eV"], ref["qubo"]["linear_eV"]))
                        and [(q["i"], q["j"]) for q in art["quadratic_eV"]] == [(q["i"], q["j"]) for q in ref["qubo"]["quadratic_eV"]]
                        and all(_close(a["value"], b["value"]) for a, b in zip(art["quadratic_eV"], ref["qubo"]["quadratic_eV"])))
                check(f"{tid}:qubo_coefficients_match_reference", same)
        if spec.case == "higher_order":
            q = report["stage_detail"]["encoding"][tid]["qubo_compilation"]
            check(f"{tid}:qubo_only_rejects_cubic", q.get("error", {}).get("code") == "UNSUPPORTED_HIGHER_ORDER_WITHOUT_QUADRATIZATION")
            check(f"{tid}:triplets_retained", enc.get("n_higher_order_terms") == exp["n_path_triplets"])
    ranked = [(r["orbit_id"], r["corrected_energy_eV"]) for r in report.get("corrected_ranking") or []]
    ref_ranked = [(r["orbit_id"], r["corrected_energy_eV"]) for r in reference["ranked_pool"]]
    check("ranked_pool_matches_reference", [r[0] for r in ranked] == [r[0] for r in ref_ranked]
          and all(_close(a[1], b[1]) for a, b in zip(ranked, ref_ranked)))
    changed = (report["stage_detail"].get("ranking") or {}).get("ranking_changed_by_correction")
    if spec.correction_mode in ("none", "constant"):
        check("correction_preserves_ranking", changed is False, str(changed))
    else:
        check("correction_changes_winner", changed is True and report["winning_topology"] != report["base_winning_topology"])
    electronic = (report["stage_detail"].get("evaluation") or {}).get("electronic", {})
    check("physical_stages_reported_skipped", report["stage_status"].get("validation") == "SKIPPED_NO_ATOMIC_GEOMETRY"
          and electronic.get("periodic_dft") == "DISABLED_BY_CONFIG")
    passed = all(c["passed"] for c in checks)
    return {"case": spec.case, "correction_mode": spec.correction_mode, "status": "PASS" if passed else "FAIL",
            "n_checks": len(checks), "n_failed": sum(not c["passed"] for c in checks), "checks": checks}
