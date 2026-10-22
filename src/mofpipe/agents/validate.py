"""Application-side validation of A1/A2/A67 outputs and the deterministic assessment renderer.

validate_setup(proposal, packet) -> ValidatedSetup
validate_hamiltonian_proposal(proposal, packet) -> ValidatedModelGrid
validate_optimisation(proposal, packet) -> ValidatedRevision (A4)
validate_assessment(assessment, packet) -> AssessmentValidation
render_assessment(assessment, packet) -> Markdown with verified citations

IDs must resolve, coverage must be complete, counts must equal the selected option, compilers must support every
proposed model degree, and A67 can neither upgrade a failed/missing required check nor cite anything absent from
the evidence packet. An invalid output is rejected; it never changes numerical results.
"""
from __future__ import annotations

import re

CITATION = re.compile(r"\[\[([^\[\]]+)\]\]")
FAILED, PASSED = "FAILED", ("COMPLETED", "NOT_APPLICABLE_UNRELAXED_GEOMETRY")


def _result(errors: list, **fields) -> dict:
    return {"status": "VALID" if not errors else "REJECTED", "errors": errors, **fields}


def _error_state(proposal: dict) -> dict | None:
    """NEEDS_INPUT / UNSUPPORTED: a compact stop state, valid only with at least one code or missing ID."""
    if proposal["status"] == "OK":
        return None
    if not (proposal["error_codes"] or proposal["missing_input_ids"]):
        return _result([{"code": "EMPTY_ERROR_STATE", "detail": proposal["status"]}])
    return {"status": proposal["status"], "errors": [], "error_codes": proposal["error_codes"],
            "missing_input_ids": proposal["missing_input_ids"]}


def _ok_without_errors(proposal: dict, errors: list) -> None:
    if proposal["error_codes"] or proposal["missing_input_ids"]:
        errors.append({"code": "OK_WITH_ERROR_CODES", "detail": str(proposal["error_codes"] + proposal["missing_input_ids"])})


def _coverage(ids: list, expected: list, label: str, errors: list) -> None:
    for x in sorted({i for i in ids if ids.count(i) > 1}):
        errors.append({"code": f"DUPLICATE_{label}", "detail": x})
    for x in sorted(set(ids) - set(expected)):
        errors.append({"code": f"UNKNOWN_{label}", "detail": x})
    for x in sorted(set(expected) - set(ids)):
        errors.append({"code": f"OMITTED_{label}", "detail": x})


SCOPE_REASONS = {"OUT_OF_BRIEF_SCOPE", "INCOMPATIBLE_WITH_BRIEF_CHEMISTRY"}
FACT_REASONS = {"COMPOSITION_INFEASIBLE": "composition_feasible", "CELL_TOO_LARGE": "cell_within_limit"}


def validate_setup(proposal: dict, packet: dict) -> dict:
    """Every catalogue framework is selected (one valid option, exact counts) or excluded with a checkable reason."""
    stop = _error_state(proposal)
    if stop:
        return stop
    errors = []
    _ok_without_errors(proposal, errors)
    frameworks = [f["topology_id"] for f in packet["candidate_frameworks"]]
    selected = [p["framework_id"] for p in proposal["plans"]]
    _coverage(selected + [e["framework_id"] for e in proposal["excluded"]], frameworks, "FRAMEWORK", errors)
    if not selected:
        errors.append({"code": "NO_FRAMEWORK_SELECTED", "detail": "at least one framework must be analysed"})
    options = {o["construction_plan_id"]: o for o in packet["construction_options"]}
    by_framework = {o["framework_id"]: o for o in packet["construction_options"]}
    species = set(packet["target_composition"]["species_ids"])
    for plan in proposal["plans"]:
        option = options.get(plan["construction_plan_id"])
        if option is None:
            errors.append({"code": "UNKNOWN_CONSTRUCTION_PLAN", "detail": plan["construction_plan_id"]})
            continue
        if option["framework_id"] != plan["framework_id"]:
            errors.append({"code": "PLAN_FRAMEWORK_MISMATCH", "detail": f"{plan['construction_plan_id']} -> {plan['framework_id']}"})
        if not (option["composition_feasible"] and option["cell_within_limit"]):
            errors.append({"code": "SELECTED_INFEASIBLE_OPTION", "detail": plan["construction_plan_id"]})
        unknown = {c["species_id"] for c in plan["species_counts"]} - species
        if unknown:
            errors.append({"code": "UNKNOWN_SPECIES", "detail": f"{plan['framework_id']}: {sorted(unknown)}"})
        counts = sorted((c["species_id"], c["count"]) for c in plan["species_counts"])
        if counts != sorted((c["species_id"], c["count"]) for c in option["species_counts"]):
            errors.append({"code": "COUNT_MISMATCH", "detail": f"{plan['framework_id']}: {counts}"})
    for ex in proposal["excluded"]:
        reason, option = ex["reason_code"], by_framework.get(ex["framework_id"])
        if reason in SCOPE_REASONS and not packet["selection_policy"]["allow_scope_exclusions"]:
            errors.append({"code": "SCOPE_EXCLUSION_NOT_ALLOWED", "detail": ex["framework_id"]})
        if reason in FACT_REASONS and option is not None and option[FACT_REASONS[reason]]:
            errors.append({"code": "EXCLUSION_CONTRADICTS_FACTS", "detail": f"{ex['framework_id']}: {reason}"})
    plans = [{"framework_id": p["framework_id"], "construction_plan_id": p["construction_plan_id"],
              "constraint_policy_id": options[p["construction_plan_id"]]["constraint_policy_id"]}
             for p in proposal["plans"] if p["construction_plan_id"] in options]
    order = lambda fid: frameworks.index(fid) if fid in frameworks else len(frameworks)
    return _result(errors, plans=sorted(plans, key=lambda p: order(p["framework_id"])),
                   excluded=sorted(proposal["excluded"], key=lambda e: order(e["framework_id"])))


def _check_model_rows(rows: list, options: dict, errors: list, expansion: dict | None = None) -> dict:
    """Per-row Hamiltonian checks shared by A2 and A4; returns {topology_id: [model ids]} for known topologies."""
    topologies = {t["topology_id"]: t for t in options["topologies"]}
    encodings = {e["encoding_id"] for e in options["encoding_catalogue"]}
    models = {m["model_id"]: m for m in options["model_catalogue"]}
    compilers = {c["compiler_id"]: c for c in options["compilation_options"]}
    limit = options["limits"]["max_models_per_topology"]
    expansion = expansion or {}
    grid = {}
    for row in rows:
        tid = row["topology_id"]
        if tid not in topologies:
            continue
        ids = row["candidate_model_ids"]
        if row["encoding_id"] not in encodings:
            errors.append({"code": "UNKNOWN_ENCODING", "detail": f"{tid}: {row['encoding_id']}"})
        if len(ids) != len(set(ids)) or len(ids) > limit:
            errors.append({"code": "INVALID_MODEL_LIST", "detail": f"{tid}: {ids}"})
        if row["constraint_policy_id"] != topologies[tid]["constraint_policy_id"]:
            errors.append({"code": "CONSTRAINT_POLICY_CHANGED", "detail": f"{tid}: {row['constraint_policy_id']}"})
        compiler = compilers.get(row["compiler_id"])
        if compiler is None:
            errors.append({"code": "UNKNOWN_COMPILER", "detail": f"{tid}: {row['compiler_id']}"})
        for mid in ids:
            if mid not in models:
                errors.append({"code": "UNKNOWN_MODEL", "detail": f"{tid}: {mid}"})
            elif compiler and models[mid]["polynomial_degree"][tid] > compiler["max_degree"]:
                # Triplets need a higher-order path; they are never dropped silently.
                errors.append({"code": "COMPILER_DOES_NOT_SUPPORT_MODEL_DEGREE",
                               "detail": f"{tid}: {mid} degree {models[mid]['polynomial_degree'][tid]} > "
                                         f"{row['compiler_id']} max {compiler['max_degree']}"})
        if tid in expansion.get("failed_topologies", []) and set(ids) <= set(expansion["tried_model_ids"].get(tid, [])):
            errors.append({"code": "EXPANSION_WITHOUT_UNTRIED_MODEL", "detail": f"{tid}: {ids}"})
        grid[tid] = list(ids)
    return grid


def validate_hamiltonian_proposal(proposal: dict, packet: dict) -> dict:
    stop = _error_state(proposal)
    if stop:
        return stop
    errors = []
    _ok_without_errors(proposal, errors)
    _coverage([r["topology_id"] for r in proposal["topologies"]], [t["topology_id"] for t in packet["topologies"]],
              "TOPOLOGY", errors)
    grid = _check_model_rows(proposal["topologies"], packet, errors, packet.get("expansion_request"))
    rows = [{k: r[k] for k in ("topology_id", "encoding_id", "candidate_model_ids", "compiler_id", "constraint_policy_id")}
            for r in proposal["topologies"]]
    return _result(errors, model_grid=grid, rows=rows)


# Factual A4 reasons and the candidate_quality fact each one needs.
A4_FACT_REASONS = {"SURROGATE_RANKING_DISAGREES_WITH_DFT": "surrogate_ranking_disagrees_with_dft",
                   "SURROGATE_ERROR_ABOVE_TOLERANCE": "surrogate_error_above_tolerance",
                   "SURROGATE_BEST_NOT_DFT_BEST": "surrogate_best_not_dft_best",
                   "MODEL_INADEQUATE": "model_inadequate", "STRUCTURAL_CHECK_FAILED": "structural_check_failed",
                   "ASSESSMENT_INCOMPLETE": "assessment_incomplete"}
DISAGREEMENT_FACTS = ("surrogate_ranking_disagrees_with_dft", "surrogate_error_above_tolerance",
                      "surrogate_best_not_dft_best", "model_inadequate")


def validate_optimisation(proposal: dict, packet: dict) -> dict:
    """A4: reasons must match the code facts; a revision must be a new, executable, catalogue-valid grid."""
    errors = []
    facts = packet["candidate_quality"]["facts"]
    for reason in proposal["reason_codes"]:
        fact = A4_FACT_REASONS.get(reason)
        if fact and not facts.get(fact):
            errors.append({"code": "REASON_CONTRADICTS_FACTS", "detail": f"{reason}: {fact} is false"})
    if "CANDIDATES_CONSISTENT" in proposal["reason_codes"] and any(facts.get(f) for f in DISAGREEMENT_FACTS):
        errors.append({"code": "REASON_CONTRADICTS_FACTS",
                       "detail": "CANDIDATES_CONSISTENT with " + ", ".join(f for f in DISAGREEMENT_FACTS if facts.get(f))})
    if "NO_UNTRIED_MODEL" in proposal["reason_codes"] and facts.get("untried_models_available"):
        errors.append({"code": "REASON_CONTRADICTS_FACTS", "detail": "NO_UNTRIED_MODEL but untried catalogue models remain"})
    if proposal["error_codes"] or proposal["missing_input_ids"]:
        errors.append({"code": "UNEXPECTED_ERROR_STATE", "detail": str(proposal["error_codes"] + proposal["missing_input_ids"])})
    current, grid = packet["current_model_grid"], dict(packet["current_model_grid"])
    if proposal["decision"] == "ACCEPT":
        if proposal["topologies"]:
            errors.append({"code": "ACCEPT_WITH_REVISION_ROWS", "detail": str([r["topology_id"] for r in proposal["topologies"]])})
        return _result(errors, decision="ACCEPT", model_grid=current, executable=False)
    if not proposal["topologies"]:
        errors.append({"code": "REVISION_WITHOUT_ROWS", "detail": "REVISE_HAMILTONIAN needs at least one topology row"})
    ids = [r["topology_id"] for r in proposal["topologies"]]
    for tid in sorted({i for i in ids if ids.count(i) > 1}):
        errors.append({"code": "DUPLICATE_TOPOLOGY", "detail": tid})
    for tid in sorted(set(ids) - set(current)):
        errors.append({"code": "UNKNOWN_TOPOLOGY", "detail": tid})
    grid.update(_check_model_rows(proposal["topologies"], packet["hamiltonian_options"], errors))
    if any(grid == {t: list(g.get(t, [])) for t in grid} for g in packet["grid_history"]):
        errors.append({"code": "REVISION_REPEATS_A_TRIED_GRID", "detail": str(grid)})
    executable = packet["rounds_left"] > 0
    return _result(errors, decision="REVISE_HAMILTONIAN", model_grid=grid, executable=executable,
                   changed_topologies=sorted(t for t in grid if grid[t] != current.get(t)))


def _allowed_verdicts(mode: str, checks: list) -> set:
    if mode == "synthetic_fixture":
        return {"NOT_ASSESSED"}
    required = [c for c in checks if c["required"]]
    if any(c["status"] == FAILED for c in checks):
        return {"FAIL"}
    if all(c["status"] in PASSED for c in required):
        return {"PASS_CHECKED_CRITERIA", "INCOMPLETE"}
    return {"INCOMPLETE", "NOT_ASSESSED"}


def validate_assessment(assessment: dict, packet: dict) -> dict:
    errors = []
    candidates = [c["candidate_id"] for c in packet["final_candidates"]]
    _coverage([c["candidate_id"] for c in assessment["candidates"]], candidates, "CANDIDATE", errors)
    checks_by_candidate = {}
    for c in packet["structural_checks"]:
        checks_by_candidate.setdefault(c["candidate_id"], []).append(c)
    for row in assessment["candidates"]:
        cid, checks = row["candidate_id"], checks_by_candidate.get(row["candidate_id"], [])
        unknown = set(row["check_refs"]) - {c["check_id"] for c in checks}
        if unknown:
            errors.append({"code": "UNKNOWN_CHECK_REF", "detail": f"{cid}: {sorted(unknown)}"})
        allowed = _allowed_verdicts(packet["mode"], checks)
        if cid in candidates and row["structural_verdict"] not in allowed:
            errors.append({"code": "VERDICT_INCONSISTENT_WITH_CHECKS",
                           "detail": f"{cid}: {row['structural_verdict']} not in {sorted(allowed)}"})
    sources = {s["source_id"]: s for s in packet["literature_evidence"]["sources"]}
    passages = {p["passage_id"]: p for p in packet["literature_evidence"]["passages"]}
    records = {r["record_id"]: r for r in packet["dataset_evidence"]["records"]}
    metrics = {m["metric_id"]: m for m in packet["comparison_metrics"]}
    for comp in assessment["comparisons"]:
        tag = f"{comp['candidate_id']}/{comp['source_id']}"
        if comp["candidate_id"] not in candidates:
            errors.append({"code": "UNKNOWN_CANDIDATE", "detail": tag})
        source, record = sources.get(comp["source_id"]), records.get(comp["source_id"])
        if source is None and record is None:
            errors.append({"code": "UNKNOWN_SOURCE", "detail": tag})
        bad = [p for p in comp["passage_ids"] if p not in passages or passages[p]["source_id"] != comp["source_id"]]
        if bad:
            errors.append({"code": "UNKNOWN_OR_FOREIGN_PASSAGE", "detail": f"{tag}: {bad}"})
        if source is not None and not comp["passage_ids"]:
            errors.append({"code": "LITERATURE_COMPARISON_WITHOUT_PASSAGE", "detail": tag})
        foreign = [m for m in comp["metric_refs"] if m not in metrics or metrics[m]["candidate_id"] != comp["candidate_id"]
                   or metrics[m]["record_id"] != comp["source_id"]]
        if foreign:
            errors.append({"code": "UNKNOWN_OR_FOREIGN_METRIC_REF", "detail": f"{tag}: {foreign}"})
        if comp["comparability"] == "DIRECT":
            supported = (record is not None and comp["metric_refs"]) or (source is not None and source["access_level"] == "full_text")
            if packet["mode"] == "synthetic_fixture" or not supported:
                errors.append({"code": "DIRECT_COMPARISON_NOT_ALLOWED",
                               "detail": f"{tag}: DIRECT needs a code-computed metric (dataset) or full text (literature)"})
    cited = set(CITATION.findall(assessment["report_markdown"]))
    if cited - set(passages) - set(records):
        errors.append({"code": "UNKNOWN_CITATION", "detail": str(sorted(cited - set(passages) - set(records)))})
    if assessment["status"] == "COMPLETE":
        reasons = []
        if not passages and not records:
            reasons.append("no literature passages or dataset records were retrieved")
        if packet["numerical_verdicts"]["pipeline_status"] != "COMPLETED":
            reasons.append("the numerical pipeline did not complete")
        if any(r["structural_verdict"] == "INCOMPLETE" for r in assessment["candidates"]):
            reasons.append("a candidate verdict is INCOMPLETE")
        if reasons:
            errors.append({"code": "COMPLETE_STATUS_NOT_ALLOWED", "detail": "; ".join(reasons)})
    return _result(errors)


def _source_line(n: int, s: dict) -> str:
    authors = ", ".join(s.get("authors") or []) or "unknown authors"
    link = s.get("url") or (f"https://doi.org/{s['doi']}" if s.get("doi") else "")
    return (f"{n}. {authors}. *{s.get('title') or s['source_id']}*" + (f" ({s['year']})" if s.get("year") else "")
            + (f". [{link}]({link})" if link else "") + f" — {s['access_level']}, via {s['origin']}.")


def _record_line(n: int, r: dict) -> str:
    link = f" [https://doi.org/{r['doi']}](https://doi.org/{r['doi']})" if r.get("doi") and "doi.org" not in r["doi"] else (
        f" [{r['doi']}]({r['doi']})" if r.get("doi") else "")
    return (f"{n}. Dataset **{r['dataset_id']}** record {r.get('name') or r['row_id']} ({r['row_id']}; {r['match_level']}, "
            f"matched on {', '.join(r['matched_on'])}).{link}")


def render_assessment(assessment: dict, packet: dict, validation: dict) -> str:
    """Markdown with [[passage_id]] / [[record_id]] expanded to numbered references from the evidence packet only."""
    sources = {s["source_id"]: s for s in packet["literature_evidence"]["sources"]}
    passages = {p["passage_id"]: p for p in packet["literature_evidence"]["passages"]}
    records = {r["record_id"]: r for r in packet["dataset_evidence"]["records"]}
    metrics = {m["metric_id"]: m for m in packet["comparison_metrics"]}
    order: list = []

    def number(sid: str) -> int:
        if sid not in order:
            order.append(sid)
        return order.index(sid) + 1

    def ref(match):
        cid = match.group(1)
        sid = passages[cid]["source_id"] if cid in passages else (cid if cid in records else None)
        return f"[{number(sid)}]" if sid else f"[unverified citation {cid}]"

    body = CITATION.sub(ref, assessment["report_markdown"])
    comparisons = [c for c in assessment["comparisons"] if c["source_id"] in sources or c["source_id"] in records]
    lines = [f"# A67 final assessment — {packet['mode']}", "",
             f"Assessment status: **{assessment['status']}**. Application validation: **{validation['status']}**"
             + (f" ({'; '.join(e['code'] + ': ' + e['detail'] for e in validation['errors'])})" if validation["errors"] else "")
             + ".", "", "## Structural verdicts (performed checks only)", "",
             "| candidate | verdict | issues | checks cited |", "|---|---|---|---|"]
    lines += [f"| {c['candidate_id']} | {c['structural_verdict']} | {', '.join(c['issue_codes'])} | {len(c['check_refs'])} |"
              for c in assessment["candidates"]]
    if comparisons:
        lines += ["", "## Comparisons with papers and datasets", "",
                  "| candidate | source | axis | comparability | metrics | finding |", "|---|---|---|---|---|---|"]
        lines += [f"| {c['candidate_id']} | [{number(c['source_id'])}] | {c['axis']} | {c['comparability']} | "
                  + ", ".join(f"{metrics[m]['quantity']} {metrics[m]['difference']:+g} {metrics[m]['unit']}"
                              for m in c["metric_refs"] if m in metrics)
                  + f" | {c['finding'].replace('|', '/')} |" for c in comparisons]
    lines += ["", "## Assessment", "", body, "", "## Unresolved", "",
              ", ".join(assessment["unresolved_codes"]) or "none", "", "## References", ""]
    lines += [_source_line(i + 1, sources[sid]) if sid in sources else _record_line(i + 1, records[sid])
              for i, sid in enumerate(order)] or ["none cited"]
    return "\n".join(lines) + "\n"
