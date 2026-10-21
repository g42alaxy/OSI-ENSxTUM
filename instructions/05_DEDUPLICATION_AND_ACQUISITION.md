Task 05 — Symmetry deduplication and candidate selection

Assignment
Implement src/candidates and src/acquisition. Produce an auditable high-level evaluation batch.

Inputs
Decoded solver candidates, configuration predictions/uncertainties, permitted symmetry group, current dataset, topology coverage policy and evaluation budget.

Implement
1. Canonicalize occupation/orientation assignments using transformations that preserve the parent net, slot chemistry and composition. Account for periodic translations and record orbit membership.
2. Deduplicate discrete decorations separately from relaxed-geometry equivalence. Avoid merging distinct nets, conformers or magnetic states merely because energies match.
3. Select candidates using a declared combination of low predicted energy, uncertainty and feature diversity. Allocate coverage across surviving topologies and a controlled audit fraction to deprioritised regions.
4. Normalize physical energies consistently; exclude constraint penalties from acquisition. Compare cross-topology predictions only if references and model fidelities are compatible.
5. Label candidates with the model version and evidence used at selection time. Ensure existing pending/successful calculations are not submitted twice.
6. Include larger-supercell candidates when needed to test ordering-period dependence. Retain the underlying unsymmetrised mappings for construction.

Proposed tools
deduplicate_candidates(candidate_ids, symmetry_policy_id) -> CandidateSet
select_evaluation_batch(candidate_set_id, model_snapshot_id, acquisition_policy_id, budget_id) -> AcquisitionBatch

Acceptance
Known symmetry copies collapse; deliberately inequivalent arrangements remain separate. An uncertain under-sampled topology receives candidates under the declared policy. Candidate selection is reproducible with a seed, and every selected candidate has a machine-readable reason.

Agent/API use
A planning agent may propose changing the acquisition policy for a future version. It cannot quietly suppress a topology or use held-out labels to select training candidates.

