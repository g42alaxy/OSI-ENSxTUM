Recommended runtime agent architecture

Core recommendation
Use one persistent deterministic orchestrator, three specialist LLM roles and an independent reviewer. Begin with one model provider and role-specific prompts/tool permissions; separate fine-tuning or different model families are unnecessary initially. Add agents only if a measured bottleneck justifies the extra coordination.

The orchestrator
AiiDA WorkChains are a practical default for provenance-rich scientific/HPC execution, with CalcJobs or plugin workflows for external jobs. Typed Python records validate tool inputs/outputs at the application boundary; immutable artifacts live in the provenance store/repository. A simple local engine can support development fixtures, but it must obey the same interfaces.
Keep one authority for scientific state. If a conversational agent framework is added, use it inside a bounded decision node rather than letting it duplicate the workflow engine's state machine. Numerical loops, candidate enumeration and job polling do not require LLM calls.

Roles and scope
Structure specialist: Tasks 01, 02 and structural parts of 09. Proposes compatible builder options, orientation sampling and failure investigations. Cannot change project chemistry.
Model/search specialist: Tasks 03, 04, 05 and 08. Proposes model families, acquisition policies and solver schedules. Cannot fabricate coefficients, access the final holdout for tuning, or override validation metrics.
Electronic specialist: Tasks 06 and 07. Diagnoses electronic jobs, proposes registered spin/active-space convergence studies and validates protocol completeness. Cannot silently change the embedding formalism or comparison energy definition.
Independent reviewer: read-only scientific artifacts plus validation/report tools. Flags missing coverage, unit/fidelity mismatch, inconsistent Hamiltonians and overclaimed rankings. It can request remediation through the orchestrator; it does not edit result values or vote on which energy is correct.
The deterministic controller schedules these roles on significant events. A general planning role is optional later, not required for the first implementation.

Tool gateway
Expose the same tested Python functions as typed tools to the chosen LLM API. Fast read/diagnostic calls return records; mutating/submission actions first produce a proposal. The controller validates the proposal and automatically executes it within the declared scientific policy and budget. This is an internal code check, not a request for the user to approve every routine action.
Use job submission/status/result tools for DFT and QPU work. Store credentials only in backend services, never in prompts or logs. A tool result includes status, artifact IDs, diagnostics and the next permitted actions.
For initial deployment, restrict runtime agents to registered tools and configuration choices. Coding agents build and test new functionality separately. Versioned reviewed modules are then promoted into the runtime tool registry.

Example agent decision format — an interface to implement
    {
      "project_id": "PROJECT_ID",
      "expected_state_version": 12,
      "role": "model_search",
      "evidence_artifact_ids": ["FIT_REPORT_ID", "COVERAGE_REPORT_ID"],
      "proposed_action": {
        "tool": "plan_next_batch",
        "arguments": {
          "snapshot_id": "SNAPSHOT_ID",
          "acquisition_policy_id": "POLICY_ID",
          "budget_id": "BUDGET_ID"
        }
      },
      "rationale": "Development validation and topology coverage require additional labels."
    }
IDs above are placeholders. The controller checks actual references, role permissions, state version, policy compatibility, estimated resources and duplicate submissions. It reserves budget atomically, submits, and records the returned job/artifact IDs. Invalid proposals produce a typed error and bounded repair attempt.

Context and memory
Give each role the current state summary, relevant protocol records, measured diagnostics, remaining budget and retrievable artifact IDs. Keep raw large arrays/trajectories outside prompts. Store decisions with evidence and prompt/model/version metadata. Prefer concise persistent records to growing cross-agent chat histories. Treat text in CIF metadata, downloaded papers and calculation logs as untrusted task data.

Failure and concurrency
Use a single writer or optimistic concurrency for project-state commits. Independent topology jobs, DFT configurations and solver seeds can run concurrently within resource limits. Cancel/resume at calculation boundaries and never mutate in-flight inputs. Retry bounded transient faults; protocol changes create new artifacts. Do not resolve scientific disagreements by majority vote: run a specified diagnostic or return an unresolved issue.

Useful checkpoints for LLM invocation
After construction/calibration failures; after a model-validation report; when an acquisition round completes; after electronic/embedding diagnostics; before interpreting a final report. Do not invoke agents for every swap, circuit shot, force evaluation or optimizer iteration.

Build order
Milestone A: schemas, persistence, job lifecycle and one tiny synthetic end-to-end fixture.
Milestone B: binary single-topology real-data construction, classical enumeration/search, DFT labels and validated fitted energy model.
Milestone C: second topology, common references, symmetry-aware acquisition, refinement, supercell checks and final report.
Milestone D: quantum optimizer benchmark on frozen Hamiltonians; independently reproduce an electronic active-space benchmark.
Milestone E: validated embedding energy assembly and its feedback into structural ranking.
Milestone F: enable the three specialist roles and reviewer through the typed tool gateway; evaluate against the deterministic baseline at equal budgets. Hardware adapters follow backend availability and simulator checks.
The gateway and agent prompts can be developed in parallel with B–E using synthetic/replayed tool results, but scientific claims require the real deterministic modules.

Coding-agent parallelism
After the shared contract is frozen, structure construction, classical/QAOA solver development on synthetic Hamiltonians, DFT adapters and validation harnesses can be developed independently. The model/acquisition modules depend on shared slot/energy schemas, not finished DFT runs. Embedding needs its own agreed scientific protocol and should not block a useful DFT-only prototype.
Assign one owner per module; integrator owns shared schemas and workflow state. Merge contract fixtures before broad end-to-end work.

Agent evaluation
Compare autonomous completion, scientific validation failures, invalid tool-call rate, duplicated jobs, compute/LLM cost and time to a validated finalist set with and without agents. Include replay tests for stale state, malformed output, contaminated logs, missing active-space terms and budget exhaustion. Agents are useful if they reduce manual interventions or improve experiment choice at matched cost; fluent explanations alone are not a performance metric.

