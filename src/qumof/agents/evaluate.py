"""Milestone F evaluation: agent-assisted runs against the deterministic baseline at equal scientific budget.

For each (case, correction) the deterministic pipeline and the agent run use the same project, evaluator and
solver settings. Measured per pair: completion, agreement of the final
ranking with the baseline, harness acceptance on the fixture (the ``no_runtime_agents`` check excluded for the
agent run), topologies kept by A1, invalid model outputs, LLM requests, tokens, cost, evaluator submissions
(duplicated jobs) and wall time. Fluent assessments are not a metric.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from ..intake import FIXTURE_PROJECT_TYPE, load_project
from ..orchestrator import run_pipeline
from ..store import RunStore
from .runner import AgentRun

AGREEMENT_KEYS = ("winning_topology", "base_winning_topology", "corrected_ranking", "base_ranking")


def _acceptance(spec, report: dict, run_dir) -> dict | None:
    if spec.project_type != FIXTURE_PROJECT_TYPE or not report:
        return None
    from ..acceptance import check_run  # harness side, after the run is finalised
    result = check_run(spec, {**report, "_run_dir": str(run_dir)})
    failed = [c["check"] for c in result["checks"] if not c["passed"] and c["check"] != "no_runtime_agents"]
    return {"n_checks": result["n_checks"], "failed": failed}


def _agent_metrics(run_dir: Path) -> dict:
    ledger = json.loads((run_dir / "agent" / "ledger.json").read_text())
    calls = [e for e in ledger["entries"] if not e.get("cache_hit")]
    usage = [e.get("usage") or {} for e in calls]
    store = RunStore(run_dir / "agent")
    rejected = sum(1 for p in store.artifacts.glob("AgentDecision__*.json")
                   if json.loads(p.read_text())["validation"]["status"] == "REJECTED")
    return {"llm_requests": len(calls), "transport_or_schema_errors": sum(e["outcome"] == "ERROR" for e in calls),
            "rejected_by_validation": rejected, "prompt_tokens": sum(u.get("prompt_tokens") or 0 for u in usage),
            "completion_tokens": sum(u.get("completion_tokens") or 0 for u in usage),
            "cost_usd": round(sum(u.get("cost") or 0 for u in usage), 5)}


def evaluate(project, controls, out_dir, combos, client_factory=None, brief=None, engine="auto") -> list:
    out_dir, rows = Path(out_dir), []
    for case, correction in combos:
        tag = f"{case or 'default'}_{correction or 'default'}"
        spec = load_project(project, case=case, correction_mode=correction)
        t0 = time.time()
        det_dir = out_dir / tag / "deterministic"
        det = run_pipeline(spec, det_dir)
        det_time = time.time() - t0
        t0 = time.time()
        run = AgentRun(project, controls, out_dir / tag / "agents", case=case, correction=correction,
                       client=client_factory() if client_factory else None, brief=brief)
        summary = run.run(engine)
        agent_time = time.time() - t0
        last = (summary["pipeline_attempts"] or [None])[-1]
        report = json.loads((Path(last["run_dir"]) / "report.json").read_text()) if last else {}
        agent_spec = run.agent_spec({"plans": summary["construction_plans"], "excluded": summary["excluded_topologies"]},
                                    summary["model_grid"]) if last else None
        rows.append({
            "case": case, "correction": correction,
            "deterministic": {"pipeline_status": det.get("pipeline_status"), "winning_topology": det.get("winning_topology"),
                              "acceptance": _acceptance(spec, det, det_dir), "wall_time_s": round(det_time, 1),
                              "evaluator_submissions": RunStore(det_dir).state["counters"].get("evaluator_submissions")},
            "agents": {"status": summary["status"], "stop": summary["stop"],
                       "pipeline_status": report.get("pipeline_status"), "winning_topology": report.get("winning_topology"),
                       "selected_topologies": [p["framework_id"] for p in summary["construction_plans"] or []],
                       "excluded_topologies": summary["excluded_topologies"], "model_grid": summary["model_grid"],
                       "selected_models": {t: (s.get("model_selection") or {}).get("selected_model")
                                           for t, s in (report.get("topologies") or {}).items()},
                       "agrees_with_deterministic": {k: report.get(k) == det.get(k) for k in AGREEMENT_KEYS} if report else None,
                       "acceptance": _acceptance(agent_spec, report, last["run_dir"]) if last else None,
                       "assessment": summary["assessment"], "wall_time_s": round(agent_time, 1),
                       "evaluator_submissions": sum(RunStore(d).state["counters"].get("evaluator_submissions", 0)
                                                    for d in {a["run_dir"] for a in summary["pipeline_attempts"]}),
                       **_agent_metrics(out_dir / tag / "agents")},
        })
    (out_dir / "evaluation.json").write_text(json.dumps(rows, indent=2, default=str) + "\n")
    (out_dir / "evaluation.md").write_text(render(rows))
    return rows


def render(rows: list) -> str:
    head = ("| case | correction | det. winner | agent winner | same ranking | acceptance det./agent | topologies | "
            "models | invalid outputs | requests | cost $ | evaluator jobs det./agent | A67 |")
    lines = ["# Milestone F: agent-assisted vs deterministic (equal scientific budget)", "", head,
             "|" + "---|" * (head.count("|") - 1)]
    for r in rows:
        d, a = r["deterministic"], r["agents"]
        acc = lambda x: "n/a" if not x else ("PASS" if not x["failed"] else f"FAIL {len(x['failed'])}")
        same = a["agrees_with_deterministic"] and a["agrees_with_deterministic"]["corrected_ranking"]
        lines.append(f"| {r['case']} | {r['correction']} | {d['winning_topology']} | {a['winning_topology']} | {same} | "
                     f"{acc(d['acceptance'])} / {acc(a['acceptance'])} | {', '.join(a['selected_topologies'])} | "
                     f"{json.dumps(a['selected_models'])} | {a['transport_or_schema_errors'] + a['rejected_by_validation']} | "
                     f"{a['llm_requests']} | {a['cost_usd']} | {d['evaluator_submissions']} / {a['evaluator_submissions']} | "
                     f"{(a['assessment'] or {}).get('status')} ({(a['assessment'] or {}).get('validation')}) |")
    return "\n".join(lines) + "\n"
