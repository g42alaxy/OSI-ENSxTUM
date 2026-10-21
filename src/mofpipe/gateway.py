"""Task 10 controller side: validation of (replayed) agent proposals.

No LLM is called anywhere in this package. A proposal - from a human or a
future agent - enters the same validated interface; with agents disabled every
proposal is recorded and rejected with AGENTS_DISABLED after the checks run.
"""
from __future__ import annotations

from .contract import make_record

ROLE_TOOLS = {
    "structure": {"validate_project", "build_framework", "generate_seed_decorations", "submit_relaxations",
                  "screen_topologies", "validate_structure"},
    "model_search": {"fit_hamiltonian", "validate_hamiltonian", "compile_encoding", "solve_arrangements",
                     "deduplicate_candidates", "select_evaluation_batch", "assess_refinement", "plan_next_batch"},
    "electronic": {"submit_periodic_dft", "collect_periodic_dft", "prepare_active_space", "solve_active_space",
                   "assemble_corrected_energy"},
    "reviewer": {"validate_structure", "rank_candidates", "evaluate_reference_mixtures", "generate_report"},
}
CONTROLLER_ONLY = {"advance_workflow"}  # state commits are never delegated to an LLM role
REQUIRED = ("project_id", "expected_state_version", "role", "evidence_artifact_ids", "proposed_action")


def review_proposal(store, project_id: str, proposal: dict, agents_enabled: bool) -> dict:
    """advance_workflow(project_id, expected_state_version, action) -> TransitionResult (DecisionRecord)."""

    def reject(code, detail=""):
        record = make_record("DecisionRecord", "SYNTHETIC", state_version=store.version, proposal=proposal,
                             outcome="REJECTED", error={"code": code, "detail": detail}, executed=False)
        store.log("proposal_rejected", code=code)
        return record

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
    #TODO: Task 10 — execute accepted proposals: validate arguments against policy, reserve budget atomically,
    #TODO: submit idempotently, record job/artifact IDs, bounded repair on invalid output; expose the typed tool
    #TODO: gateway to the LLM provider (roles, prompts, per-role permissions from runtime.allowed_tools_by_role).
    return reject("UNSUPPORTED", "proposal execution not implemented")
