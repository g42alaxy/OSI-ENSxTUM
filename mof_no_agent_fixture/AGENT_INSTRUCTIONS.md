# Instructions for the coding agent integrating this fixture

The user is building a MOF arrangement/topology pipeline. Runtime AI agents have **not** been added. Your task is to integrate this pack as a deterministic development fixture in the existing codebase and demonstrate that the implemented modules produce the known results.

You are an external coding assistant implementing the software. Do not confuse that role with a deployed LLM agent inside the pipeline. No runtime LLM calls, agent SDKs, credentials or training are needed for this task.

## 1. Inspect the repository and add a fixture adapter

Read the existing repository instructions and identify its input schema, graph representation, energy records, model-fitting interface, solver interface and test conventions. Add an adapter from `project.json` and `data/` into those interfaces. Preserve the existing architecture rather than replacing it with this reference script.

The fixture schema is a proposed interchange format, not a claim about the user's current implementation. Resolve path references relative to the fixture root. Record provenance as `SYNTHETIC` in downstream artifacts. Do not rename these graphs `pcu`, `tbo`, `fmj` or another physical net.

Configure all runtime LLM/agent features as disabled. The fixture must run offline with the pipeline's ordinary local dependencies. Avoid adding an LLM dependency merely to choose between fixed configurations.

## 2. Load structures, labels and target composition

Load 12 occupation sites from each finite substitution graph. Map A to 0 and B to 1. Require exactly six of each. Validate length, integer binary values, site identifiers and species counts before scoring.

Use `finite_substitution_interaction_graph` as an explicit input mode. Bypass CIF parsing and atomic assembly in this mode. Report geometric screening and atomic validation as `SKIPPED_NO_ATOMIC_GEOMETRY`, not as passed checks. Retain both supplied graphs; there is no physical screening evidence for discarding either one.

If the repository accepts only CIFs, introduce the smallest internal graph-fixture entry point at the output of structure preprocessing. Do not fabricate a CIF or pretend placeholder coordinates represent a valid framework.

## 3. Implement deterministic Hamiltonian selection

Read the candidate feature sets from `data/model_candidates.json`. Fit coefficients from training labels only. Use the supplied symmetry-orbit split so equivalent configurations cannot appear in training and validation. Select the least-complex candidate satisfying the configured development error threshold. Report each model's errors and the selection reason as structured data.

For this fixture, select M1 on `core_pairwise` and M2 on `higher_order`. Never hard-code those winners. If no model passes, emit `MODEL_INADEQUATE`; do not substitute an LLM decision or claim validation succeeded. Treat a singular feature matrix as a failed candidate with a recorded reason.

Do not inspect `harness/` or `reference_outputs/` while fitting or selecting. The oracle formula in the fixture generator is also out of bounds for the learner. A test harness may inspect these files after the run is finalized. If integrating a controlled evaluator, the learner can request labels for named configurations through that interface.

The initial fixture uses a fixed development split. Production model selection should use a predeclared grouped validation procedure suited to the actual data, plus a genuinely untouched test set. A runtime LLM can later propose model families, but numerical validation should remain authoritative.

## 4. Compile and solve with a classical baseline

Implement exact fixed-composition enumeration first: 924 states per graph. Score the learned model directly, keep degeneracies, and record solver status and tolerances. Preserve each graph's energy offset for cross-graph comparison.

For M1, compile to binary QUBO with the documented coefficient convention and a separate constant. Check QUBO energy against the learned polynomial on every one of the 4,096 binary strings, including invalid compositions. Check the 1.0 eV ratio penalty in this fixture rather than assuming any positive penalty is sufficient in general.

For M2, direct enumeration of the cubic polynomial is supported. A QUBO-only backend must either reject it with `UNSUPPORTED_HIGHER_ORDER_WITHOUT_QUADRATIZATION`, or explicitly quadratize it and validate the ancilla constraints and energy equivalence. Do not silently omit triplets or call a cubic model a QUBO.

Keep quantum execution disabled for this acceptance run. Later QAOA tests must reproduce the classical objective and respect the specified bit ordering. Fixed-count exchange reachability is checked here, but this does not validate quantum gates, noise models or ancilla handling.

## 5. Deduplicate and pool candidates

Canonicalize an occupation under the supplied site permutations and prefix the key with `topology_id`. Expect 50 ring orbits and 55 prism orbits. Do not add species-label interchange as an extra equivalence operation. Deduplicate separately within each graph, then retain the best five unique orbits per graph, including cutoff ties.

Pool those representatives using the common composition and normalization. Output a ranked table containing configuration/orbit IDs, occupations, topology ID, base energy, correction, corrected energy, fidelity and provenance. Use deterministic tie-breaking while preserving a record of true degeneracy.

The known base minima are −0.120 eV for the ring and −0.160 eV for the prism. Both have two raw minimizing strings but one minimizing orbit. Assert the identities and orbit counts, not just the winning graph.

## 6. Exercise the expensive-evaluation boundary with an explicit mock

Use the provided labelled files for the first run. If the application expects jobs submitted to an energy backend, adapt `tools/mock_evaluator.py` as an explicitly named synthetic backend. Requests identify graph and occupation; responses preserve identity, units, status, normalization and fidelity.

Test the no-correction, constant-correction and topology-dependent-correction modes. The constant correction must preserve rankings. The topology-dependent mode must change the winner from prism to ring. This tests energy bookkeeping only; report DFT, embedding, VQE, SQD and FCI as not executed.

Never append corrected labels to the base-energy training set without changing the target definition and fidelity. Failed/nonconverged calculations must not become usable numeric energies. Check equal composition, units, reference conventions and normalization before pooling physical energies in a future case.

In a later real electronic-structure backend, define the embedding correction with a consistent subtraction/reference and double-counting treatment. A bare active-space FCI or VQE energy is not by itself an additive DFT correlation correction. This fixture deliberately makes no claim to implement that scientific method.

## 7. Keep the initial active-learning loop disabled

The starting fixture is sufficient for fitting the correct candidate model. Do not add an unbounded refit-until-agreement loop just to make it resemble the final workflow.

If the repository already has an active-learning loop, add a separate test using the mock evaluator: keep an untouched evaluation set, request genuinely new configurations, refit against a consistently defined target, and enforce a maximum of three iterations. If starting from an inadequate model family, allow an explicit model-family expansion or return `MODEL_INADEQUATE`; more labels alone cannot repair a missing interaction term. Record convergence based on predeclared error/ranking tolerances and budget, not unconditional exact ranking agreement.

Do not claim iterative active learning was verified by the bundled reference; it was not.

## 8. Generate a static provenance report

Use `data/literature_catalog.json` to list sources and the limited claims they support. No web search or LLM literature review is needed during this test. Set live literature review to `NOT_RUN`.

Do not compare the synthetic graph energies numerically with published Kang, Cai or Fe–MOF-74 results. Those are separate scientific benchmarks requiring their own structures, compositions, Hamiltonians and calculation protocols. Do not fabricate experimental agreement or citations.

Report structural stability, topology confirmation from atomic coordinates, MOFChecker, Zeo++, PXRD and NMR comparison as skipped in this fixture. No finite-temperature stage should be added.

## 9. Acceptance criteria and deliverables

Run the bundled verifier first to establish that the fixture is intact. Then run the same acceptance conditions through the user's actual pipeline adapters; a passing bundled reference alone is not proof of integration.

Required conditions:

- The pipeline runs without network access, API keys or runtime agents.
- It loads both graphs and rejects malformed occupations and incompatible energy records.
- Train/development/holdout symmetry orbits stay disjoint.
- It selects the expected model by the fixed numerical policy and passes the untouched holdout tolerance.
- Enumeration, deduplication, QUBO identity and ground-state results match the reference.
- It retains graph-specific offsets and common normalization when pooling.
- Both correction-routing tests produce the specified behavior.
- Higher-order terms are retained or explicitly rejected by a QUBO-only stage.
- The output clearly distinguishes completed software tests from skipped physical stages.

Deliver the adapter/configuration changes, a reproducible run command, the actual pipeline's result report, a concise comparison with the reference, and a list of any stages that remain unimplemented. Do not claim the user's pipeline passed unless you executed it.

## 10. Route to physical inputs and future runtime agents

After the software checks pass, use `physical_case/recipe.json` as a separate next-stage specification. Obtain and validate real CIF/building-block files and periodic energies before enabling physical conclusions. Do not reinterpret the synthetic labels as molecular-potential energies.

Keep these policy boundaries explicit so runtime agents can be added later:

| Boundary | Current deterministic implementation | Possible later agent contribution |
|---|---|---|
| Model proposals | Fixed feature grid in JSON | Propose additional physically motivated candidate models |
| Model acceptance | Numerical validation and complexity policy | Explain diagnostics; acceptance remains numerical |
| New calculations | Explicit candidate IDs and a fixed budget | Suggest informative configurations within that budget |
| Literature reporting | Static verified metadata | Retrieve papers and map claims to cited passages |
| Workflow progression | Stage states, data contracts and error codes | Summarize failures and propose permitted recovery actions |

An LLM proposal should enter the same validated configuration interface as a human proposal. The deterministic fixture must remain runnable with the agent layer switched off.
