"""Task 10 controller side: envelope validation of agent proposals (A1 / A2 / A67).

Three runtime roles only (MOF_AI_Agent_Implementation_Guide-2): A1 initial setup, A2 Hamiltonian proposal and
A67 final assessment. Each role answers one structured request; its logical functions below are the only
actions a proposal from that role may name. With agents disabled every proposal is recorded and rejected with
AGENTS_DISABLED after the checks run. With agents enabled a proposal that passes is ACCEPTED but not executed:
the controller (agents/validate.py) checks its arguments semantically and applies it to the next pipeline input.
"""
from __future__ import annotations

from .contract import make_record

ROLE_TOOLS = {
    "A1": {"select_construction_options", "map_species_ids", "flag_missing_inputs"},
    "A2": {"propose_encoding", "propose_model_candidates", "select_compilation_path"},
    "A67": {"assess_structural_correctness", "compare_existing_results", "write_final_assessment"},
}
CONTROLLER_ONLY = {"advance_workflow"}  # state commits are never delegated to an LLM role
REQUIRED = ("project_id", "expected_state_version", "role", "evidence_artifact_ids", "proposed_action")


def review_proposal(store, project_id: str, proposal: dict, agents_enabled: bool,
                    provenance: str = "SYNTHETIC") -> dict:
    """advance_workflow(project_id, expected_state_version, action) -> TransitionResult (DecisionRecord)."""

    def decide(outcome, code=None, detail=""):
        error = {"code": code, "detail": detail} if code else None
        record = make_record("DecisionRecord", provenance, state_version=store.version, proposal=proposal,
                             outcome=outcome, error=error, executed=False)
        store.log("proposal_rejected" if outcome == "REJECTED" else "proposal_accepted", code=code)
        return record

    def reject(code, detail=""):
        return decide("REJECTED", code, detail)

    missing = [k for k in REQUIRED if k not in proposal]
    if missing:
        return reject("MALFORMED_PROPOSAL", f"missing {missing}")
    if proposal["project_id"] != project_id:
        return reject("WRONG_PROJECT", proposal["project_id"])
    if proposal["expected_state_version"] != store.version:
        return reject("STALE_STATE_VERSION", f"{proposal['expected_state_version']} != {store.version}")
    role = proposal["role"]
    if role not in ROLE_TOOLS:
        return reject("UNKNOWN_ROLE", role)
    action = proposal["proposed_action"]
    tool = action.get("tool") if isinstance(action, dict) else None
    if tool in CONTROLLER_ONLY or tool not in ROLE_TOOLS[role]:
        return reject("TOOL_NOT_PERMITTED_FOR_ROLE", f"{role}:{tool}")
    absent = [a for a in proposal["evidence_artifact_ids"] if not store.has(a)]
    if absent:
        return reject("UNKNOWN_EVIDENCE_ARTIFACT", ", ".join(absent))
    if not agents_enabled:
        return reject("AGENTS_DISABLED", "runtime_agents.enabled is false; deterministic controller only")
    return decide("ACCEPTED")
