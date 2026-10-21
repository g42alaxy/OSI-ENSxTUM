Task 03 — Fit and compile configurational Hamiltonians

Assignment
Implement src/configuration_model and src/encoding. Separate the physical surrogate from the optimization encoding.

Inputs
Slot lattice and chemical symmetry operations; versioned relaxed energy labels; target composition; reference/normalization policy; candidate cluster orders/ranges and model-selection policy.

Implement
1. Define symmetry orbits of site, pair and selected triplet features, including periodic images and any explicit orientation variables. Tie equivalent interactions and preserve physically inequivalent slots.
2. Fit one topology model with an explicit constant/reference term. Start with a transparent sparse linear cluster model. icet may supply lattice/orbit machinery only if a faithful slot-lattice representation is demonstrated; do not treat whole linkers as ordinary atomic substitutions without a tested mapping.
3. Use grouped/nested validation: equivalent structures and near-duplicate relaxations stay in one group. Freeze an independent final holdout. Under fixed composition, detect rank-deficient features and choose a documented gauge or remove redundant columns.
4. Evaluate physical energy errors, low-energy recall/ranking, ground-state recovery on a tiny enumerable case and larger-supercell transfer. Report ties within uncertainty. Record data requirements when fit quality is inadequate.
5. Treat ML, DFT and embedded-corrected energies as separate fidelities. Fit calibrated differences between matched calculations if using a multifidelity model; never concatenate them as interchangeable targets.
6. Calibrate predictive uncertainty with held-out residuals, documenting its limited domain. Bootstrap/committee disagreement alone is not a certified error bound.
7. Compile binary or multicomponent encodings. Retain the physical model unchanged when adding constraint penalties. Support native higher-order diagonal Hamiltonians or explicit quadratization; do not label a cubic polynomial QUBO.
8. Preserve offsets, multiplicities, cell-size scaling and all penalty/auxiliary mappings. Select penalties by an auditable bound or small-instance certification. Constant composition does not make constants irrelevant to cross-topology ranking.

Proposed tools
fit_hamiltonian(topology_id, dataset_snapshot_id, fit_policy_id) -> HamiltonianArtifact
validate_hamiltonian(hamiltonian_id, split_id) -> ModelValidationReport
compile_encoding(hamiltonian_id, decoration_space_id, encoding_policy_id) -> EncodingArtifact

Acceptance
Symmetry-related arrangements have identical features/energies. Periodic multiplicity and replication tests pass. No symmetry orbit crosses train/holdout boundaries. On exhaustive tiny fixtures, minimizing over ancillas reproduces the original feasible cost up to the documented constant, with no infeasible ground state. Report failed scientific validation explicitly.

Agent/API use
The modelling agent may propose a candidate feature family or acquisition request. Numerical validation selects among eligible models; the LLM does not choose coefficients or override failed validation.

