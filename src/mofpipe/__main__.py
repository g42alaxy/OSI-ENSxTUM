"""CLI: python -m mofpipe {run,accept,matrix} --project mof_no_agent_fixture/project.json

``run`` also accepts a ProjectSpec v0.1 file or a physical_case recipe (physical projects).
``electronic-benchmark --spec benchmarks/h2_sto3g_szabo_ostlund.json --run-dir DIR`` runs Task 07 solvers.
``embedding-benchmark --spec benchmarks/embedding_keto_enol_hf_ccsd.json --run-dir DIR`` checks the T07.3 scheme.
``agents --project P --controls controls.yaml`` runs A1 -> A2 -> pipeline -> A67 (OpenRouter, Valency MCP, local papers);
``--replay FILE`` replays scripted role outputs offline. ``agents-check --controls controls.yaml`` tests keys/endpoints.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .contract import PipelineError
from .intake import CORRECTION_MODES, load_project
from .orchestrator import run_pipeline

CASES = ("core_pairwise", "higher_order")


def _run_one(project, case, correction, run_dir, engine="local"):
    spec = load_project(project, case=case, correction_mode=correction)
    run_dir = Path(run_dir or f"runs/{spec.project_id}/{spec.case}_{spec.correction_mode}")
    if engine == "aiida":  # T10.1: same stages as an AiiDA WorkChain with stage provenance (aiida_engine.py)
        import os
        os.environ.setdefault("AIIDA_PATH", str(Path(__file__).resolve().parents[2] / ".aiida"))  # before aiida loads
        try:
            from .aiida_engine import run_pipeline_aiida
        except ImportError as exc:
            raise PipelineError("BACKEND_UNAVAILABLE", "aiida-core is not installed (pip install -e .[aiida])") from exc
        report = run_pipeline_aiida(project, run_dir, case, correction)
    else:
        report = run_pipeline(spec, run_dir)
    return spec, run_dir, report


def _accept(spec, run_dir, report):
    if spec.project_type != "synthetic_fixture":
        raise PipelineError("ACCEPTANCE_HARNESS_FIXTURE_ONLY", "physical projects have no harness/ answers")
    from .acceptance import check_run  # harness side, only after the run is finalised
    result = check_run(spec, {**report, "_run_dir": str(run_dir)})
    (Path(run_dir) / "acceptance.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _agents(args) -> int:
    from .agents.controls import load_controls  # agent dependencies load only for these commands
    from .agents.runner import AgentRun, check_connections

    controls = load_controls(args.controls)
    if args.command == "agents-design-bench":
        from .agents.design import load_materials, rescore, run_benchmark
        if args.rescore:
            rescore(args.out, load_materials(args.materials))
            print((Path(args.out) / "results.md").read_text())
            return 0
        models = [] if args.control_only else [m for m in args.models.split(",") if m]
        if models and controls.errors():
            print(json.dumps({"error": "INVALID_CONTROLS", "fields": [{"code": c, "field": f} for c, f in controls.errors()]}),
                  file=sys.stderr)
            return 2
        qpu = {"device": "fake_brisbane", "python": args.qpu_python, "shots": 4000} if args.qpu_python else None
        extra = {"reasoning": {"effort": args.reasoning_effort}} if args.reasoning_effort else {}
        run_benchmark(load_materials(args.materials), lambda m: load_controls(args.controls, {"openrouter": {"model": m, **extra}}),
                      models, args.repeats, args.out, qpu=qpu, workers=args.workers)
        print((Path(args.out) / "results.md").read_text())
        return 0
    if args.command == "datasets-fetch":
        from .agents.datasets import fetch
        result = {d["id"]: fetch(d) for d in controls["datasets"] if d.get("hf")}
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "agents-check":
        result = check_connections(controls)
        print(json.dumps(result, indent=2))
        return 0 if not result["config_errors"] else 1
    brief = getattr(args, "brief", None)
    if getattr(args, "brief_file", None):
        brief = Path(args.brief_file).read_text().strip()
    if args.command == "agents-eval":
        from .agents.evaluate import evaluate
        errors = controls.errors()
        if errors:
            print(json.dumps({"error": "INVALID_CONTROLS", "fields": [{"code": c, "field": f} for c, f in errors]}), file=sys.stderr)
            return 2
        combos = [tuple(None if x == "default" else x for x in (c.split(":") + ["default"])[:2]) for c in args.combos.split(",")]
        rows = evaluate(args.project, controls, args.out, combos, brief=brief, engine=controls["orchestration"]["engine"])
        print((Path(args.out) / "evaluation.md").read_text())
        return 0 if all(r["agents"]["status"] == "COMPLETED" for r in rows) else 1
    client = None
    if args.replay:
        from .agents.client import ReplayClient
        client = ReplayClient(json.loads(Path(args.replay).read_text()))
    errors = controls.errors(need_llm=client is None)
    if errors:
        print(json.dumps({"error": "INVALID_CONTROLS", "fields": [{"code": c, "field": f} for c, f in errors]}), file=sys.stderr)
        return 2
    project_id = json.loads(Path(args.project).read_text()).get("project_id", "project")
    run_dir = Path(args.run_dir or f"runs/{project_id}/agents_{args.case or 'default'}_{args.correction or 'default'}")
    summary = AgentRun(args.project, controls, run_dir, case=args.case, correction=args.correction,
                       client=client, brief=brief).run(args.engine or controls["orchestration"]["engine"])
    print(json.dumps({k: summary[k] for k in ("status", "engine", "llm_requests", "llm_request_ceiling", "cache_hits",
                                              "stop", "excluded_topologies", "model_grid", "expansion_used", "assessment")}
                     | {"run_dir": str(run_dir)}, indent=2, default=str))
    return 0 if summary["status"] == "COMPLETED" else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="mofpipe")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "accept"):
        p = sub.add_parser(name)
        p.add_argument("--project", required=True)
        p.add_argument("--case", choices=CASES)
        p.add_argument("--correction", choices=CORRECTION_MODES)
        p.add_argument("--run-dir")
        p.add_argument("--engine", choices=("local", "aiida"), default="local")
    m = sub.add_parser("matrix", help="run + accept every case x correction mode")
    m.add_argument("--project", required=True)
    m.add_argument("--out", default="runs")
    e = sub.add_parser("electronic-benchmark", help="active-space export + ED/CASCI/VQE/SQD vs a published value")
    e.add_argument("--spec", required=True)
    e.add_argument("--run-dir", required=True)
    b = sub.add_parser("embedding-benchmark", help="subtractive embedding vs an independent full high-level calculation")
    b.add_argument("--spec", required=True)
    b.add_argument("--run-dir", required=True)
    a = sub.add_parser("agents", help="agent-assisted run: A1 setup, A2 Hamiltonian proposal, pipeline, A67 assessment")
    a.add_argument("--project", required=True)
    a.add_argument("--case", choices=CASES)
    a.add_argument("--correction", choices=CORRECTION_MODES)
    a.add_argument("--controls", default="controls.yaml")
    a.add_argument("--run-dir")
    a.add_argument("--replay", help="JSON {A1: [...], A2: [...], A67: [...]} of scripted outputs; no API calls")
    a.add_argument("--engine", choices=("auto", "langgraph", "sequential"))
    a.add_argument("--brief", help="general description of the study for A1 (default: the project's purpose)")
    a.add_argument("--brief-file", help="file holding the design brief")
    v = sub.add_parser("agents-eval", help="Milestone F: deterministic vs agent-assisted runs at equal budget")
    v.add_argument("--project", required=True)
    v.add_argument("--controls", default="controls.yaml")
    v.add_argument("--out", default="runs/agents_eval")
    v.add_argument("--combos", default="core_pairwise:none,higher_order:topology_dependent",
                   help="comma-separated case:correction pairs (fixture); 'default' for physical projects")
    v.add_argument("--brief")
    f = sub.add_parser("datasets-fetch", help="download the reference datasets declared in controls.yaml (hf: repo/revision)")
    f.add_argument("--controls", default="controls.yaml")
    d = sub.add_parser("agents-design-bench", help="Kang et al. (2025) benchmark: A2 designs the Hamiltonian, code solves")
    d.add_argument("--controls", default="controls.yaml")
    d.add_argument("--materials", default="benchmarks/kang2025")
    d.add_argument("--models", default="qwen/qwen3.8-flash", help="comma-separated OpenRouter model IDs")
    d.add_argument("--repeats", type=int, default=3)
    d.add_argument("--out", default="runs/kang2025_design")
    d.add_argument("--workers", type=int, default=4)
    d.add_argument("--qpu-python", help="interpreter with qiskit: also sample each design on fake_brisbane")
    d.add_argument("--control-only", action="store_true", help="only the paper's weighting through the same code (no LLM)")
    d.add_argument("--reasoning-effort", choices=("low", "medium", "high"), help="OpenRouter reasoning effort for A2")
    d.add_argument("--rescore", action="store_true", help="recompute metrics of the runs in --out from their saved designs")
    c = sub.add_parser("agents-check", help="check controls.yaml: OpenRouter key/model and Valency MCP access")
    c.add_argument("--controls", default="controls.yaml")
    args = parser.parse_args(argv)
    try:
        if args.command in ("agents", "agents-check", "agents-eval", "datasets-fetch", "agents-design-bench"):
            return _agents(args)
        if args.command == "electronic-benchmark":
            from .electronic_solvers import run_electronic_benchmark
            from .store import RunStore
            report = run_electronic_benchmark(json.loads(Path(args.spec).read_text()), RunStore(args.run_dir))["report"]
            print(json.dumps({k: report[k] for k in ("benchmark_id", "status", "identical_problem", "comparisons",
                                                     "solver_status")}, indent=2))
            return 0 if report["status"] == "REPRODUCED" else 1
        if args.command == "embedding-benchmark":
            from .embedding import run_embedding_benchmark
            from .store import RunStore
            report = run_embedding_benchmark(json.loads(Path(args.spec).read_text()), RunStore(args.run_dir))["report"]
            print(json.dumps({k: report[k] for k in ("benchmark_id", "status", "relative_energy", "checks")}, indent=2))
            return 0 if report["status"] == "REPRODUCED" else 1
        if args.command == "matrix":
            rows = []
            for case in CASES:
                for mode in CORRECTION_MODES:
                    spec, run_dir, report = _run_one(args.project, case, mode, Path(args.out) / f"{case}_{mode}")
                    result = _accept(spec, run_dir, report)
                    rows.append({"case": case, "correction": mode, "pipeline": report.get("pipeline_status"),
                                 "winner": report.get("winning_topology"),
                                 "selected": {t: s.get("model_selection", {}).get("selected_model") for t, s in report["topologies"].items()},
                                 "acceptance": result["status"], "checks": f"{result['n_checks'] - result['n_failed']}/{result['n_checks']}"})
            print(json.dumps(rows, indent=2))
            return 0 if all(r["acceptance"] == "PASS" for r in rows) else 1
        spec, run_dir, report = _run_one(args.project, args.case, args.correction, args.run_dir, args.engine)
        summary = {"run_dir": str(run_dir), "engine": args.engine, **({"aiida": report["_aiida"]} if "_aiida" in report else {}),
                   "pipeline_status": report.get("pipeline_status"),
                   "winning_topology": report.get("winning_topology"), "stage_status": report.get("stage_status")}
        if args.command == "accept":
            result = _accept(spec, run_dir, report)
            summary["acceptance"] = {k: result[k] for k in ("status", "n_checks", "n_failed")}
            summary["failed_checks"] = [c for c in result["checks"] if not c["passed"]]
        print(json.dumps(summary, indent=2))
        return 0 if report.get("pipeline_status") == "COMPLETED" and summary.get("acceptance", {}).get("status", "PASS") == "PASS" else 1
    except PipelineError as exc:
        print(json.dumps({"error": exc.as_dict()}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
