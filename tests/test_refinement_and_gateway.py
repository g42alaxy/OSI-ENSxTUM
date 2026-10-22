"""Task 08 bounded active learning (mock evaluator) and Task 10 proposal validation (no LLM)."""
import json

import pytest
from conftest import PROJECT, copy_fixture, read_jsonl

from qumof.gateway import review_proposal
from qumof.intake import load_project
from qumof.orchestrator import Pipeline, run_pipeline


def _small_training_fixture(tmp_path, case, per_topology, models=None):
    root = copy_fixture(tmp_path / "fx")
    rows = read_jsonl(root / f"data/{case}_training.jsonl")
    kept, counts = [], {}
    for r in rows:
        if counts.get(r["topology_id"], 0) < per_topology:
            kept.append(r)
            counts[r["topology_id"]] = counts.get(r["topology_id"], 0) + 1
    (root / f"data/{case}_training.jsonl").write_text("".join(json.dumps(r) + "\n" for r in kept))
    if models is not None:
        path = root / "data/model_candidates.json"
        m = json.loads(path.read_text())
        m["candidates"] = [c for c in m["candidates"] if c["id"] in models]
        path.write_text(json.dumps(m))
    return root / "project.json"


def _al(max_iter, batch):
    return {"active_learning": {"mode": "enabled", "max_iterations_if_enabled": max_iter, "batch_size_per_topology": batch}}


def _refinement(run_dir):
    state = json.loads((run_dir / "state.json").read_text())
    aid = state["stages"]["refinement"]["artifact_ids"]["refinement"]
    return json.loads((run_dir / "artifacts" / (aid.replace(":", "__") + ".json")).read_text()), state


def test_active_learning_recovers_cubic_model_from_sparse_start(tmp_path):
    project = _small_training_fixture(tmp_path, "higher_order", per_topology=2)
    spec = load_project(project, case="higher_order", overrides=_al(3, 2))
    report = run_pipeline(spec, tmp_path / "run")
    refinement, state = _refinement(tmp_path / "run")
    assert refinement["stop_reason"] == "CONVERGED_WITHIN_SCOPE"
    assert 1 <= len(refinement["iterations"]) - 1 <= 3
    acquired = [o for it in refinement["iterations"] for o in it.get("acquired", [])]
    split = spec.split_manifest
    for orbit in acquired:  # never development or final-holdout orbits
        tid = orbit.split(":")[0]
        assert orbit in split[tid]["training"] and orbit not in split[tid]["holdout"] + split[tid]["development"]
    assert {s["model_selection"]["selected_model"] for s in report["topologies"].values()} == {"M2_pairs_and_triplets"}
    assert report["winning_topology"] == "prism6" and report["pipeline_status"] == "COMPLETED"
    assert state["counters"]["evaluator_submissions"] == len(refinement["iterations"])  # AL batches + final pool


def test_more_labels_cannot_fix_missing_interaction_term(tmp_path):
    project = _small_training_fixture(tmp_path, "higher_order", per_topology=30, models={"M0_nearest_pairs", "M1_two_shell_pairs"})
    report = run_pipeline(load_project(project, case="higher_order", overrides=_al(3, 2)), tmp_path / "run")
    refinement, state = _refinement(tmp_path / "run")
    assert refinement["stop_reason"] == "MODEL_INADEQUATE"
    assert state["counters"]["evaluator_submissions"] == 0  # no labels wasted
    assert report["stage_status"]["model_selection"] == "FAILED"


def test_budget_exhaustion_is_not_convergence(tmp_path):
    project = _small_training_fixture(tmp_path, "higher_order", per_topology=1)
    report = run_pipeline(load_project(project, case="higher_order", overrides=_al(1, 1)), tmp_path / "run")
    refinement, _ = _refinement(tmp_path / "run")
    assert refinement["stop_reason"] == "BUDGET_EXHAUSTED"
    assert report["stage_status"]["refinement"] == "INCONCLUSIVE"
    assert report["pipeline_status"] != "COMPLETED"


@pytest.fixture
def pipeline(tmp_path):
    p = Pipeline(load_project(PROJECT), tmp_path / "run")
    p.run()
    return p


def _proposal(p, **changes):
    fit = p.ctx["selection/ring12"]["artifact_id"]
    base = {"project_id": p.spec.project_id, "expected_state_version": p.store.version, "role": "A2",
            "evidence_artifact_ids": [fit],
            "proposed_action": {"tool": "propose_model_candidates",
                                "arguments": {"topologies": [{"topology_id": "ring12",
                                                              "candidate_model_ids": ["M1_two_shell_pairs"]}]}}}
    base.update(changes)
    return base


@pytest.mark.parametrize("changes,code", [
    ({}, "AGENTS_DISABLED"),
    ({"expected_state_version": 0}, "STALE_STATE_VERSION"),
    ({"role": "model_search"}, "UNKNOWN_ROLE"),
    ({"proposed_action": {"tool": "submit_periodic_dft"}}, "TOOL_NOT_PERMITTED_FOR_ROLE"),
    ({"proposed_action": {"tool": "advance_workflow"}}, "TOOL_NOT_PERMITTED_FOR_ROLE"),
    ({"evidence_artifact_ids": ["FIT_REPORT_ID"]}, "UNKNOWN_EVIDENCE_ARTIFACT"),
    ({"project_id": "PROJECT_ID"}, "WRONG_PROJECT"),
])
def test_replayed_proposals_are_validated_and_never_executed(pipeline, changes, code):
    decision = review_proposal(pipeline.store, pipeline.spec.project_id, _proposal(pipeline, **changes), agents_enabled=False)
    assert decision["outcome"] == "REJECTED" and decision["error"]["code"] == code and decision["executed"] is False


def test_valid_proposal_is_accepted_but_not_executed_when_agents_enabled(pipeline):
    decision = review_proposal(pipeline.store, pipeline.spec.project_id, _proposal(pipeline), agents_enabled=True)
    assert decision["outcome"] == "ACCEPTED" and decision["error"] is None and decision["executed"] is False
    wrong_role = review_proposal(pipeline.store, pipeline.spec.project_id,
                                 _proposal(pipeline, proposed_action={"tool": "write_final_assessment"}), agents_enabled=True)
    assert wrong_role["error"]["code"] == "TOOL_NOT_PERMITTED_FOR_ROLE"


def test_malformed_proposal(pipeline):
    decision = review_proposal(pipeline.store, pipeline.spec.project_id, {"role": "A2"}, agents_enabled=False)
    assert decision["error"]["code"] == "MALFORMED_PROPOSAL"
