"""OSI scheme end to end (OSI_scheme.pdf): every box of the diagram on one run, with the A4 optimisation loop.

Text input -> A1 -> topology search (ML-potential screen) -> A2 -> QUBO (exact + branch and bound + QAOA) ->
ground state of the top candidates (periodic DFT + node-cluster embedding) -> A67 (= A3) -> A4.

SYNTHETIC: the two-topology toy (tests/toy_framework.py), mock pw.x, mock ML worker (mirroring the mock DFT so the
screen is calibrated), mock cluster engine; the four roles are the scripted stand-ins of agents/scheme.py through
ReplayClient (no network, no LLM). The test checks the wiring and the evidence of each box, not chemistry.
"""
import json
import shutil
import sys

import pytest

pytest.importorskip("jsonschema")

from conftest import REPO  # noqa: E402
from toy_framework import embedding_protocol, write_toy_milestone_c_project  # noqa: E402

from mofpipe.agents.client import ReplayClient  # noqa: E402
from mofpipe.agents.controls import load_controls  # noqa: E402
from mofpipe.agents.runner import AgentRun  # noqa: E402
from mofpipe.agents.scheme import scheme_trace, scripted_policies  # noqa: E402

ML = {"engine": "orb", "model": "orb-v3-conservative-inf-omat", "dispersion": "D3BJ", "python": sys.executable,
      "worker": "tools/mock_mlip.py", "relaxation": {"optimizer": "FIRE", "fmax_eV_per_A": 0.05, "max_steps": 300,
                                                     "cell": "variable"}}
QAOA = {"kind": "qaoa_simulation", "layers": 1, "shots": 256, "seeds": [0], "restarts": 1, "max_evaluations": 30}


def scheme_project(tmp_path):
    dst = tmp_path / "toy"
    (dst / "tools").mkdir(parents=True, exist_ok=True)
    project = write_toy_milestone_c_project(dst, {
        "protocols": {"ml_potential_checkpoint": ML,
                      "ml_calibration": {"n_structures_per_topology": 2, "tolerance_eV_per_formula_unit": 0.01},
                      "screening": {"energetic_policy": {"deprioritise_above_eV_per_formula_unit": 0.5}},
                      "classical_solvers": ["exact_enumeration", "branch_and_bound"],
                      "quantum_solvers": [QAOA],
                      "embedding": embedding_protocol(dst, bond_hops=1, series=(1, 2))},
        "budgets": {"max_active_space_jobs": 200}})
    for tool in ("mock_mlip.py", "mock_cluster.py"):
        shutil.copy(REPO / "tests" / tool, dst / "tools" / tool)
    return project


@pytest.fixture(scope="module")
def scheme_run(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("scheme")
    mp = pytest.MonkeyPatch()
    mp.setenv("MOCK_MLIP_MIRROR_PW", "1")  # ML agrees with the mock DFT, so the screen is calibrated
    controls = load_controls(None, {"cache": {"enabled": False}, "mcp": {"valency": {"enabled": False}},
                                    "literature": {"references": []}, "budget": {"max_optimisation_rounds": 1}})
    run = AgentRun(scheme_project(tmp_path), controls, tmp_path / "run", client=ReplayClient(scripted_policies()),
                   brief="Zinc toy frameworks with L and LN linkers 1:1; which net and arrangement are most stable?")
    summary = run.run("sequential")
    mp.undo()
    return tmp_path / "run", summary, scheme_trace(tmp_path / "run")


def test_every_scheme_box_runs_and_leaves_evidence(scheme_run):
    run_dir, summary, trace = scheme_run
    report = json.loads((run_dir / "pipeline/report.json").read_text())
    # Every stage ran; the only open one is the embedding correction, UNVALIDATED with the mock cluster engine and no
    # reference benchmark, so the run is honestly INCOMPLETE rather than COMPLETED.
    assert {k: v for k, v in report["stage_status"].items() if v != "COMPLETED"} == {"embedding": "INCONCLUSIVE"}
    assert summary["status"] == "INCOMPLETE" and not summary.get("stop")
    # Text input -> A1: the brief reaches A1, which selects among the catalogue topologies.
    assert trace["text_input"].startswith("Zinc toy frameworks")
    assert set(trace["A1_topology_selection"]["selected"]) <= {"toy_pcu", "toy_pcu_staggered"}
    assert trace["A1_topology_selection"]["selected"]
    # Topology search: ML-potential screen, calibrated against DFT, with a decision per topology.
    screen = trace["topology_search"]["ml_screen"]
    assert screen["mode"] == "ml_potential" and screen["ranking_label"] == "CALIBRATED" and screen["retained"]
    # A2 -> QUBO: a model per kept topology; the QUBO solved exactly, by branch and bound and by QAOA.
    assert all(trace["A2_hamiltonian"]["selected_models"].values())
    qubo = trace["qubo_solved"]
    assert qubo["classical"] and all(c["agrees_with_exact"] for c in qubo["classical"].values())
    assert qubo["qaoa_simulation"] and qubo["quantum_status"]["qaoa_simulation"] == "COMPLETED"
    # Ground state of the top candidates: periodic DFT energies and the node-cluster embedding stage.
    top = trace["ground_state_top5"]["dft"]
    assert 1 <= len(top) <= 5 and all(r["dft_eV_per_fu"] is not None for r in top)
    assert trace["ground_state_top5"]["embedding"]["validation"] == "UNVALIDATED"
    # A3 = A67 confirmation, then A4: one optimisation decision recorded with its facts.
    assert trace["A3_confirmation_A67"]["validation"] == "VALID"
    decisions = trace["A4_optimisation"]
    assert decisions and decisions[0]["decision"] in ("ACCEPT", "REVISE_HAMILTONIAN") and decisions[0]["validation"] == "VALID"
    assert "facts" in decisions[0]


def test_scripted_run_uses_one_request_per_role_call_and_records_it(scheme_run):
    run_dir, summary, trace = scheme_run
    ledger = json.loads((run_dir / "agent/ledger.json").read_text())
    roles = [e["role"] for e in ledger["entries"]]
    assert roles[:4] == ["A1", "A2", "A67", "A4"]  # in diagram order
    revisions = summary["hamiltonian_revisions"]
    assert len(roles) == 4 + 2 * revisions and summary["llm_requests"] == len(roles)  # a revision adds A67 + A4
    assert summary["llm_requests"] <= summary["llm_request_ceiling"]


def test_a4_revises_an_inadequate_hamiltonian_and_the_second_pass_succeeds(tmp_path):
    """Fixture higher_order: A2 proposes pair models only, fitting fails (MODEL_INADEQUATE); A4 revises to the
    untried triplet model; the pipeline re-runs from fitting and completes with it."""
    from conftest import PROJECT
    from mofpipe.agents.scheme import scripted_a1, scripted_a4, scripted_a67

    def pairs_only(packet):
        return {"status": "OK", "missing_input_ids": [], "error_codes": [], "topologies": [
            {"topology_id": t["topology_id"], "encoding_id": packet["encoding_catalogue"][0]["encoding_id"],
             "candidate_model_ids": ["M0_nearest_pairs", "M1_two_shell_pairs"], "compiler_id": "direct_quadratic",
             "constraint_policy_id": t["constraint_policy_id"]} for t in packet["topologies"]]}

    controls = load_controls(None, {"cache": {"enabled": False}, "mcp": {"valency": {"enabled": False}},
                                    "literature": {"references": []}, "budget": {"max_optimisation_rounds": 1}})
    script = {"A1": [scripted_a1], "A2": [pairs_only], "A67": [scripted_a67] * 2, "A4": [scripted_a4] * 2}
    summary = AgentRun(PROJECT, controls, tmp_path / "run", case="higher_order", client=ReplayClient(script)).run("sequential")
    first, second = summary["optimisation"]
    assert first["decision"] == "REVISE_HAMILTONIAN" and first["executed"]
    assert all("M2_pairs_and_triplets" in models for models in first["model_grid_after"].values())
    assert second["decision"] == "ACCEPT" and summary["hamiltonian_revisions"] == 1
    report = json.loads((tmp_path / "run/pipeline/report.json").read_text())
    assert report["pipeline_status"] == "COMPLETED"
    assert {s["model_selection"]["selected_model"] for s in report["topologies"].values()} == {"M2_pairs_and_triplets"}
