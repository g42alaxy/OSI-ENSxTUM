"""Offline/oracle isolation, resume without duplicate submissions, and run-dir integrity."""
import json
import os
import sys

import pytest
from conftest import FIXTURE, PROJECT

from qumof.contract import PipelineError
from qumof.intake import load_project
from qumof.orchestrator import STAGES, run_pipeline

_events = None


def _hook(event, args):
    if _events is not None and event in ("open", "import", "subprocess.Popen") or (
            _events is not None and event.startswith("socket.")):
        _events.append((event, args))


sys.addaudithook(_hook)


def test_learner_never_touches_harness_oracle_or_network(tmp_path):
    global _events
    _events = []
    try:
        spec = load_project(PROJECT, case="higher_order", correction_mode="topology_dependent")
        report = run_pipeline(spec, tmp_path / "run")
    finally:
        events, _events = _events, None
    assert report["pipeline_status"] == "COMPLETED"
    opened = [os.fsdecode(a[0]) for e, a in events if e == "open" and a and isinstance(a[0], (str, bytes, os.PathLike))]
    assert any("/data/" in p for p in opened)  # the hook really observed learner reads
    forbidden = [p for p in opened if "/harness/" in p or "/reference_outputs/" in p
                 or p.endswith(("fixture_core.py", "generate_fixture.py", "oracle_definition.json"))]
    assert not forbidden, forbidden
    imported = {a[0] for e, a in events if e == "import"}
    assert not imported & {"fixture_core", "generate_fixture", "mock_evaluator", "socket", "urllib.request", "http.client"}
    assert not [e for e, _ in events if e.startswith("socket.")]
    popen = [a for e, a in events if e == "subprocess.Popen"]
    assert len(popen) == 1 and str(FIXTURE / "tools/mock_evaluator.py") in [str(x) for x in popen[0][1]]


def _state(run_dir):
    return json.loads((run_dir / "state.json").read_text())


def test_rerun_reuses_every_stage_and_never_resubmits(tmp_path):
    spec = load_project(PROJECT, correction_mode="topology_dependent")
    first = run_pipeline(spec, tmp_path / "run")
    second = run_pipeline(load_project(PROJECT, correction_mode="topology_dependent"), tmp_path / "run")
    assert first["artifact_id"] == second["artifact_id"]
    assert _state(tmp_path / "run")["counters"]["evaluator_submissions"] == 1
    events = [json.loads(l) for l in (tmp_path / "run/events.jsonl").read_text().splitlines()]
    assert sum(e["event"] == "stage_reused" for e in events) == len(STAGES)


def test_resume_after_interruption_during_evaluation_uses_job_cache(tmp_path):
    spec = load_project(PROJECT)
    first = run_pipeline(spec, tmp_path / "run")
    state = _state(tmp_path / "run")
    for stage in ("evaluation", "ranking", "validation", "report"):  # crash after the job finished, before commit
        state["stages"].pop(stage)
    (tmp_path / "run/state.json").write_text(json.dumps(state))
    second = run_pipeline(load_project(PROJECT), tmp_path / "run")
    counters = _state(tmp_path / "run")["counters"]
    assert counters["evaluator_submissions"] == 1 and counters["evaluator_cache_hits"] == 1
    assert second["artifact_id"] == first["artifact_id"]


def test_failed_evaluator_job_never_becomes_energy(tmp_path):
    from conftest import copy_fixture
    root = copy_fixture(tmp_path / "fx")
    (root / "tools/mock_evaluator.py").write_text("import sys\nsys.exit('backend crashed')\n")
    report = run_pipeline(load_project(root / "project.json"), tmp_path / "run")
    assert report["stage_status"]["evaluation"] == "FAILED"
    assert report["stage_status"]["ranking"] == "NOT_RUN"
    assert report["corrected_ranking"] is None and report["winning_topology"] is None


def test_relative_run_dir(tmp_path, monkeypatch):
    # The evaluator subprocess runs with cwd=fixture root; job paths must not be cwd-relative.
    monkeypatch.chdir(tmp_path)
    report = run_pipeline(load_project(PROJECT), "relative/run")
    assert report["pipeline_status"] == "COMPLETED"
    assert (tmp_path / "relative/run/report.json").exists()
    assert not (FIXTURE / "relative").exists()


def test_acceptance_reports_failure_instead_of_crashing(tmp_path):
    from conftest import copy_fixture
    from qumof.acceptance import check_run
    root = copy_fixture(tmp_path / "fx", with_harness=True)
    (root / "tools/mock_evaluator.py").write_text("import sys\nsys.exit('backend crashed')\n")
    spec = load_project(root / "project.json")
    result = check_run(spec, {**run_pipeline(spec, tmp_path / "run"), "_run_dir": str(tmp_path / "run")})
    assert result["status"] == "FAIL" and result["n_failed"] > 0


def test_run_dir_bound_to_inputs(tmp_path):
    run_pipeline(load_project(PROJECT), tmp_path / "run")
    with pytest.raises(PipelineError, match="RUN_DIR_BELONGS_TO_DIFFERENT_INPUTS"):
        run_pipeline(load_project(PROJECT, case="higher_order"), tmp_path / "run")
