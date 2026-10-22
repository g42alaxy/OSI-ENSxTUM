"""CLI: python -m mofpipe {run,accept,matrix} --project mof_no_agent_fixture/project.json

``run`` also accepts a ProjectSpec v0.1 file or a physical_case recipe (physical projects).
``electronic-benchmark --spec benchmarks/h2_sto3g_szabo_ostlund.json --run-dir DIR`` runs Task 07 solvers.
``embedding-benchmark --spec benchmarks/embedding_keto_enol_hf_ccsd.json --run-dir DIR`` checks the T07.3 scheme.
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


def _run_one(project, case, correction, run_dir):
    spec = load_project(project, case=case, correction_mode=correction)
    run_dir = Path(run_dir or f"runs/{spec.project_id}/{spec.case}_{spec.correction_mode}")
    report = run_pipeline(spec, run_dir)
    return spec, run_dir, report


def _accept(spec, run_dir, report):
    if spec.project_type != "synthetic_fixture":
        raise PipelineError("ACCEPTANCE_HARNESS_FIXTURE_ONLY", "physical projects have no harness/ answers")
    from .acceptance import check_run  # harness side, only after the run is finalised
    result = check_run(spec, {**report, "_run_dir": str(run_dir)})
    (Path(run_dir) / "acceptance.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="mofpipe")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "accept"):
        p = sub.add_parser(name)
        p.add_argument("--project", required=True)
        p.add_argument("--case", choices=CASES)
        p.add_argument("--correction", choices=CORRECTION_MODES)
        p.add_argument("--run-dir")
    m = sub.add_parser("matrix", help="run + accept every case x correction mode")
    m.add_argument("--project", required=True)
    m.add_argument("--out", default="runs")
    e = sub.add_parser("electronic-benchmark", help="active-space export + ED/CASCI/VQE/SQD vs a published value")
    e.add_argument("--spec", required=True)
    e.add_argument("--run-dir", required=True)
    b = sub.add_parser("embedding-benchmark", help="subtractive embedding vs an independent full high-level calculation")
    b.add_argument("--spec", required=True)
    b.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    try:
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
        spec, run_dir, report = _run_one(args.project, args.case, args.correction, args.run_dir)
        summary = {"run_dir": str(run_dir), "pipeline_status": report.get("pipeline_status"),
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
