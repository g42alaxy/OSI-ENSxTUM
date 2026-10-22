"""OSI scheme view of an agent run: scripted (offline) role policies and the scheme trace.

``scripted_policies()`` are deterministic, rule-based stand-ins for the four LLM roles, used with ``ReplayClient``
(``python -m mofpipe agents --scripted``) when no OpenRouter key is available. They read only their input packet,
answer in the same strict schemas and pass through the same application validation as a model's output; they are
labelled SCRIPTED in the run summary and are not evidence about LLM behaviour.

``scheme_trace(run_dir)`` maps every box of OSI_scheme.pdf onto the artifacts of a finished run:

    Text input -> A1 -> Topology search (builder + ML potential) -> A2 -> QUBO (classical + QAOA)
    -> Ground state (DFT; VQE and SQD) -> A3 = A67 confirmation -> A4 optimisation (loop to the QUBO step)
"""
from __future__ import annotations

import json
from pathlib import Path

from ..store import RunStore
from .validate import A4_FACT_REASONS, DISAGREEMENT_FACTS, _allowed_verdicts


# -- scripted role policies ---------------------------------------------------------------------------------
def scripted_a1(packet: dict) -> dict:
    """Select every feasible catalogue framework; exclude only on code-checked facts."""
    plans, excluded = [], []
    for o in packet["construction_options"]:
        if o["composition_feasible"] and o["cell_within_limit"]:
            plans.append({"framework_id": o["framework_id"], "construction_plan_id": o["construction_plan_id"],
                          "species_counts": o["species_counts"]})
        else:
            excluded.append({"framework_id": o["framework_id"],
                             "reason_code": "COMPOSITION_INFEASIBLE" if not o["composition_feasible"] else "CELL_TOO_LARGE"})
    return {"status": "OK", "plans": plans, "excluded": excluded, "missing_input_ids": [], "error_codes": []}


def _models_for(options: dict, tid: str, exclude=()) -> tuple[list, str] | None:
    """Up to three catalogue models (fewest parameters first) and the cheapest compiler holding their degree."""
    compilers = sorted(options["compilation_options"], key=lambda c: c["max_degree"])
    top = max(c["max_degree"] for c in compilers)
    models = sorted((m for m in options["model_catalogue"] if m["model_id"] not in exclude
                     and m["polynomial_degree"][tid] <= top), key=lambda m: (m["n_parameters"][tid], m["model_id"]))
    chosen = [m["model_id"] for m in models[:options["limits"]["max_models_per_topology"]]]
    if not chosen:
        return None
    degree = max(m["polynomial_degree"][tid] for m in models if m["model_id"] in chosen)
    compiler = next(c["compiler_id"] for c in compilers if c["max_degree"] >= degree)
    return chosen, compiler


def _row(options: dict, tid: str, models: list, compiler: str) -> dict:
    topo = next(t for t in options["topologies"] if t["topology_id"] == tid)
    return {"topology_id": tid, "encoding_id": options["encoding_catalogue"][0]["encoding_id"],
            "candidate_model_ids": models, "compiler_id": compiler, "constraint_policy_id": topo["constraint_policy_id"]}


def scripted_a2(packet: dict) -> dict:
    """Simplest models first: the three catalogue models with the fewest parameters per topology."""
    rows = []
    for t in packet["topologies"]:
        models, compiler = _models_for(packet, t["topology_id"])
        rows.append(_row(packet, t["topology_id"], models, compiler))
    return {"status": "OK", "topologies": rows, "missing_input_ids": [], "error_codes": []}


def scripted_a67(packet: dict) -> dict:
    """Verdicts limited to what the checks allow; issue codes from the packet; a templated, numbers-free report."""
    checks: dict = {}
    for c in packet["structural_checks"]:
        checks.setdefault(c["candidate_id"], []).append(c)
    rows = []
    for cand in packet["final_candidates"]:
        mine = checks.get(cand["candidate_id"], [])
        allowed = _allowed_verdicts(packet["mode"], mine)
        verdict = next(v for v in ("FAIL", "INCOMPLETE", "NOT_ASSESSED", "PASS_CHECKED_CRITERIA") if v in allowed)
        issues = ["SYNTHETIC_NOT_PHYSICAL"] if packet["mode"] == "synthetic_fixture" or "SYNTHETIC" in json.dumps(
            packet["coverage"].get("not_executed") or []) else []
        if any(c["status"] == "MISSING" and c["required"] for c in mine):
            issues.append("MISSING_STRUCTURAL_CHECK")
        if any(c["status"] == "FAILED" for c in mine):
            issues.append("FAILED_STRUCTURAL_CHECK")
        rows.append({"candidate_id": cand["candidate_id"], "structural_verdict": verdict,
                     "check_refs": [c["check_id"] for c in mine][:32], "issue_codes": issues})
    unresolved = sorted({i for r in rows for i in r["issue_codes"]})
    if packet["energy_results"].get("embedding") not in (None, "VALIDATED"):
        unresolved.append("UNVALIDATED_CORRECTION")
    if not packet["literature_evidence"]["passages"]:
        unresolved.append("MISSING_LITERATURE")
    if packet["numerical_verdicts"]["pipeline_status"] != "COMPLETED":
        unresolved.append("PIPELINE_INCOMPLETE")
    text = ("Scripted assessment (no LLM). Structural verdicts follow the code checks only: "
            + ", ".join(f"{r['candidate_id']} {r['structural_verdict']}" for r in rows)
            + ". Missing or failed required checks keep candidates below PASS_CHECKED_CRITERIA. No literature or dataset "
              "comparison is made without retrieved evidence. Open items: " + (", ".join(unresolved) or "none") + ".")
    return {"status": "INCOMPLETE", "candidates": rows, "comparisons": [], "report_markdown": text,
            "unresolved_codes": unresolved}


def scripted_a4(packet: dict) -> dict:
    """Revise when a disagreement fact holds and an untried model exists (rounds left); otherwise accept."""
    facts = packet["candidate_quality"]["facts"]
    true_reasons = [r for r, f in A4_FACT_REASONS.items() if facts.get(f)]
    disagreement = any(facts.get(f) for f in DISAGREEMENT_FACTS)
    options = packet["hamiltonian_options"]
    if disagreement and facts.get("untried_models_available") and packet["rounds_left"] > 0:
        failing = {t for t, d in packet["fit_results"].items() if d["status"] == "MODEL_INADEQUATE"}
        failing |= {t for t, p in packet["candidate_quality"]["per_topology"].items()
                    if not p["surrogate_best_is_evaluated_best_within_tolerance"]
                    or p["max_abs_surrogate_error_eV"] > packet["candidate_quality"]["tolerance_eV"]}
        rows = []
        for tid in sorted(failing & set(packet["current_model_grid"])):
            picked = _models_for(options, tid, exclude=packet["tried_models"].get(tid, []))
            if picked:
                rows.append(_row(options, tid, *picked))
        if rows:
            return {"decision": "REVISE_HAMILTONIAN", "reason_codes": [r for r in true_reasons if r != "ASSESSMENT_INCOMPLETE"]
                    or true_reasons, "topologies": rows, "missing_input_ids": [], "error_codes": []}
    reasons = true_reasons if disagreement else ["CANDIDATES_CONSISTENT"] + [r for r in true_reasons if r in
                                                                          ("ASSESSMENT_INCOMPLETE", "STRUCTURAL_CHECK_FAILED")]
    if disagreement and not facts.get("untried_models_available"):
        reasons.append("NO_UNTRIED_MODEL")
    return {"decision": "ACCEPT", "reason_codes": reasons[:8], "topologies": [], "missing_input_ids": [], "error_codes": []}


def scripted_policies(max_calls: int = 8) -> dict:
    """{role: [callable, ...]} for ReplayClient: enough answers for every optimisation round."""
    return {"A1": [scripted_a1] * 2, "A2": [scripted_a2] * 2, "A67": [scripted_a67] * max_calls,
            "A4": [scripted_a4] * max_calls}


# -- scheme trace -------------------------------------------------------------------------------------------
def _load(store: RunStore, report: dict, prefix: str) -> list:
    return [store.get(aid) for key, aid in sorted(report.get("artifact_index", {}).items()) if key.startswith(prefix)]


def scheme_trace(run_dir) -> dict:
    """Every scheme box with the evidence that it ran (or why it did not) for a finished ``agents`` run."""
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "agent_run.json").read_text())
    pipeline_dir = run_dir / "pipeline"
    report = json.loads((pipeline_dir / "report.json").read_text()) if (pipeline_dir / "report.json").exists() else {}
    store = RunStore(pipeline_dir)
    detail = report.get("stage_detail") or {}
    screening = summary.get("screening") or {}
    gen = (report.get("physical") or {}).get("topology_generation") or {}
    opt = detail.get("optimization") or {}
    ranking = report.get("base_ranking") or []
    clusters = [r for r in _load(store, report, "cluster_result/") if r.get("role") == "high" and r.get("status") == "SUCCEEDED"]
    cluster = clusters[0] if clusters else {}
    top = [{"configuration_id": r["configuration_id"], "topology_id": r["topology_id"], "dft_eV_per_fu": r["ranking_energy_eV"],
            "surrogate_eV_per_fu": r.get("surrogate_base_energy_eV"), "tied_with_previous": r.get("tied_with_previous")}
           for r in ranking[:5]]
    return {
        "status": summary["status"], "client": summary.get("client", "openrouter"),
        "text_input": summary.get("design_brief"),
        "A1_topology_selection": {"selected": [p["framework_id"] for p in summary.get("construction_plans") or []],
                                  "excluded": summary.get("excluded_topologies")},
        "topology_search": {"builder": gen.get("builder"), "versions": gen.get("versions"),
                            "generated": [a["net"] for a in gen.get("accepted", [])],
                            "rejected": len(gen.get("rejected", [])),
                            "ml_screen": {k: screening.get(k) for k in ("mode", "ranking_label", "retained")},
                            "ml_decisions": screening.get("decisions")},
        "A2_hamiltonian": {"model_grid_initial": (summary.get("pipeline_attempts") or [{}])[0].get("model_grid"),
                           "selected_models": {t: (s.get("model_selection") or {}).get("selected_model")
                                               for t, s in (report.get("topologies") or {}).items()}},
        "qubo_solved": {"encoding": {t: {k: v for k, v in d.items() if k in ("target", "n_ancillas", "verification")}
                                     for t, d in (detail.get("encoding") or {}).items() if isinstance(d, dict)},
                        "classical": {t: c for t, c in (opt.get("cross_checks") or {}).items()},
                        "qaoa_simulation": opt.get("quantum_benchmark"), "quantum_device": opt.get("quantum_hardware"),
                        "quantum_status": opt.get("quantum")},
        "ground_state_top5": {"dft": top,
                              "node_cluster_vqe_sqd": {k: (cluster.get("details") or {}).get(k) for k in (
                                  "vqe_energy_Ha", "exact_active_space_energy_Ha", "sqd_subspace_dimension",
                                  "sqd_full_sector_dimension", "mean_field_energy_Ha")} | {"sqd_energy_Ha": cluster.get("energy_Ha")},
                              "embedding": {k: (detail.get("embedding") or {}).get(k) for k in ("validation", "unresolved", "n_corrected")}},
        "A3_confirmation_A67": summary.get("assessment"),
        "A4_optimisation": summary.get("optimisation"),
        "pipeline_attempts": [{k: a.get(k) for k in ("pipeline_status", "model_selection_error", "model_grid")}
                              for a in summary.get("pipeline_attempts") or []],
        "llm_requests": summary.get("llm_requests"),
    }


def write_scheme_trace(run_dir) -> Path:
    trace = scheme_trace(run_dir)
    path = Path(run_dir) / "scheme_trace.json"
    path.write_text(json.dumps(trace, indent=2, default=str) + "\n")
    return path
