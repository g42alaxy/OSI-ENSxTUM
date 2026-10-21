Task 02 — Conservative topology screening and training seeds

Assignment
Implement src/screening and calculator adapters. Produce prioritised candidate structures and an auditable exclusion ledger.

Inputs
Validated topologies and decorations from Task 01; ML-potential checkpoint/version; associated DFT reference protocol; screening and calibration budgets.

Implement
1. Generate representative pure-linker endpoints where chemically meaningful, ordered mixed configurations, feasible random configurations and optional SQS-like configurations in the slot representation. Include orientation diversity.
2. Evaluate all configurations with the same documented relaxation policy. Use a registered ASE-compatible calculator adapter where available; verify compatibility with the elements, charge and magnetic assumptions.
3. Calibrate the ML potential against DFT for representative structures and energy DIFFERENCES relevant to screening. Record the calibration domain. Until calibration is available, label ML ranking exploratory.
4. Check relaxation convergence, topology retention, cell distortions and atom/block mapping. Failed relaxations remain failures. Connectivity-changing structures are routed as transformed candidates, not silently used as labels for their original lattice model.
5. Compare energies only within compatible chemical inventories; pure endpoints of different formulae cannot be compared by raw total energy. Use matched references for mixing energies.
6. Output retain, uncertain, deprioritise or reject states. Hard rejection is reserved for independently established incompatibility; energetic rejection requires a declared policy and audit sampling. Sampled high energies are not a lower bound.
7. Retain a reproducible audit sample of excluded/deprioritised structures and allow later reinstatement. Produce seed training data with provenance and fidelity tags.

Proposed tools
generate_seed_decorations(topology_id, sampling_policy_id) -> DecorationSet
submit_relaxations(decoration_ids, calculator_protocol_id) -> JobBatch
screen_topologies(result_ids, screening_policy_id) -> ScreeningReport

Acceptance
Composition survives every sampling method; pure/SQS/ordered states are distinguishable in metadata; out-of-domain potential use is flagged; a failed relaxation is never ranked as high energy; an uncertain topology survives for later exploration. A tiny known ordering fixture tests whether screening would discard its favourable mixed state.

Agent/API use
A chemistry agent may request alternative orientations or targeted DFT calibration based on diagnostics. The rejection policy is enforced by code and every exclusion cites evidence.

