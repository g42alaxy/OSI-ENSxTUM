Task 08 — Active learning and stopping criteria

Assignment
Implement src/refinement as a versioned deterministic controller with an optional LLM advisor.

Inputs
Hamiltonian snapshots, candidate acquisition history, compatible DFT/corrected energies, calibration reports, independent validation policy and remaining budget.

Implement
1. Maintain immutable data/model versions. Ingest only validated labels with correct fidelity, geometry and comparison group. Build a DFT surrogate and, when justified, a matched embedded-correction surrogate or consistent high-level target.
2. Refit with the predeclared validation policy. Distinguish low-energy error, ranking error, coverage and calibration failure; request additional labels or a richer model as appropriate.
3. Acquire a batch using the registered energy/uncertainty/diversity policy. Revisit uncertain topology exclusions and untested supercell patterns. Include exploration outside the current model's preferred basin.
4. Preserve an untouched final holdout. Use development validation for iterative decisions; a holdout inspected for tuning becomes development data and requires a new independent final test.
5. Re-solve and compare leading candidates and gaps using tolerance-aware equivalence classes. Require declared coverage, numerical/model criteria and stability over a configured number of rounds, not exact agreement of all ranks.
6. Distinguish a screened finalist set from a global optimum certificate. Account for solver gaps and finite search coverage.
7. Stop with CONVERGED_WITHIN_SCOPE, BUDGET_EXHAUSTED, MODEL_INADEQUATE, SCIENTIFIC_SPEC_MISSING or INCONCLUSIVE. Budget exhaustion is never convergence.
8. Persist an event log and resume from the last completed state. Prevent duplicate batches and stale agent proposals through input hashes and state-version checks.

Proposed tools
assess_refinement(snapshot_id, stopping_policy_id) -> RefinementReport
plan_next_batch(snapshot_id, acquisition_policy_id, budget_id) -> AcquisitionBatch
advance_workflow(project_id, expected_state_version, action) -> TransitionResult

Acceptance
A deliberately biased initial training set can recover a missed basin on a controlled enumerable fixture. No final holdout leaks into acquisition. Near-degenerate candidates remain tied within tolerance. Interrupted iterations resume without repeated expensive work. Stale proposals are rejected.

Agent/API use
A modelling agent can diagnose why the model fails and propose an approved next experiment. The controller evaluates stopping rules and commits state; the LLM cannot pronounce convergence.

