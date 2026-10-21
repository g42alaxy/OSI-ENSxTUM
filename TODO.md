# TODO — unimplemented parts of `mofpipe`

This file explains every `#TODO` marker in `src/mofpipe/`. The pipeline runs end-to-end on the synthetic graph fixture (`mof_no_agent_fixture/`, Milestone A) and, since Milestone B, on physical projects: CIF input → slot lattice → built frameworks → periodic DFT labels → grouped-validated model → DFT-ranked candidates. Milestone B is verified only on a SYNTHETIC toy framework with a mock `pw.x` (`tests/toy_framework.py`, `tests/mock_pw.py`). Real MOF-5 runs still need the inputs listed under **Needs**.

- Find all markers: `grep -rn '#TODO' src/mofpipe`. Line numbers below are as of this commit; the function names are the stable anchors.
- Task numbers refer to `instructions/NN_*.md`. Each item's "Done when" criteria come from that brief's *Acceptance* section.
- **Now** says what the code does today, and which status it reports instead of a result. **Needs** lists inputs or scientific decisions that are not in the repository. Per `instructions/README.md`, these must not be invented. Until they exist, the stage returns `BLOCKED` / `SCIENTIFIC_SPEC_MISSING`.

## Rules that apply to every item

From `instructions/00_SHARED_CONTRACT.md`, as already implemented in `contract.py`, `store.py` and `electronic.py`:

- Return typed records built with `make_record`. Long jobs go through `submit → job_status → collect`, with idempotency keys derived from the scientific inputs.
- Failures stay `FAILED` / `BLOCKED` / `UNSUPPORTED` / `INCONCLUSIVE`. They are never turned into numeric penalty energies.
- Keep raw outputs, software/pseudopotential/checkpoint hashes and seeds in fingerprints. A scientific setting change creates a new protocol and new artifacts.
- Never pool energies of different fidelity, units, normalization or comparison group (`datasets.check_energy_compatibility`).
- Keep SYNTHETIC and physical results in separate projects and run directories.

## Index

| ID | Task | What | Marker | Status |
|---|---|---|---|---|
| T01.1 | 01 | Full ProjectSpec and physical-case loader | — | **Done (B)** |
| T01.2 | 01 | CIF / periodic-net input mode | — | **Done (B)** |
| T01.3 | 01 | Atomic framework construction | [framework.py:569](src/mofpipe/framework.py#L569) | **Done (B)**; PORMAKE wiring open |
| T01.4 | 01 | Linker orientation states | [structures.py:54](src/mofpipe/structures.py#L54) | open |
| T01.5 | 01/05 | Supercells and periodic translations | [structures.py:67](src/mofpipe/structures.py#L67) | partly (translations in CIF mode) |
| T02.1 | 02 | Seed decoration generation | [structures.py:136](src/mofpipe/structures.py#L136) | open |
| T02.2 | 02 | ML-potential relaxation | [structures.py:137](src/mofpipe/structures.py#L137) | open |
| T02.3 | 02 | ML calibration against DFT and screening decisions | [structures.py:139](src/mofpipe/structures.py#L139) | open |
| T03.1 | 03 | Cluster-feature generation from the slot lattice | [configuration_model.py:15](src/mofpipe/configuration_model.py#L15) | partly (site orbits, pair multiplicity) |
| T03.2 | 03 | Model validation for real (noisy) data | — | **Done (B)** |
| T03.3 | 03 | Uncertainty calibration and multifidelity models | [configuration_model.py:278](src/mofpipe/configuration_model.py#L278) | open |
| T03.4 | 03 | Quadratization of higher-order models | [encoding.py:56](src/mofpipe/encoding.py#L56) | open |
| T03.5 | 03/04 | Multicomponent encodings and penalty bounds | [encoding.py:63](src/mofpipe/encoding.py#L63), [intake.py:281](src/mofpipe/intake.py#L281) | open |
| T04.1 | 04 | Exact MILP / constraint solver for large cells | [optimization.py:22](src/mofpipe/optimization.py#L22) | open |
| T04.2 | 04 | QAOA simulation | [optimization.py:86](src/mofpipe/optimization.py#L86), [intake.py:323](src/mofpipe/intake.py#L323) | open |
| T04.3 | 04 | QPU hardware adapter | [optimization.py:89](src/mofpipe/optimization.py#L89) | open |
| T05.1 | 05 | Geometry-level deduplication | [candidates.py:18](src/mofpipe/candidates.py#L18) | open |
| T05.2 | 05 | Evaluation-batch acquisition policy | [candidates.py:45](src/mofpipe/candidates.py#L45) | open |
| T06.1 | 06 | Periodic DFT backend | [dft.py:103](src/mofpipe/dft.py#L103), [dft.py:200](src/mofpipe/dft.py#L200) | **Done (B)**; licensed codes, k-point study open |
| T06.2 | 01/06/07 | Real evaluation path in the orchestrator | [orchestrator.py:349](src/mofpipe/orchestrator.py#L349) | **Done (B)**; embedding hook open (T07.3) |
| T07.1 | 07 | Embedding scheme and active-space export | [electronic.py:21](src/mofpipe/electronic.py#L21), [electronic.py:137](src/mofpipe/electronic.py#L137) | open |
| T07.2 | 07 | Electronic solvers: CASCI/FCI, VQE, SQD | [electronic.py:138](src/mofpipe/electronic.py#L138), [electronic.py:139](src/mofpipe/electronic.py#L139) | open |
| T07.3 | 07 | Corrected-energy assembly | [electronic.py:21](src/mofpipe/electronic.py#L21), [intake.py:343](src/mofpipe/intake.py#L343) | open |
| T08.1 | 08 | Active-learning acquisition policy | [refinement.py:19](src/mofpipe/refinement.py#L19) | open |
| T08.2 | 08 | Stopping criteria | [refinement.py:88](src/mofpipe/refinement.py#L88) | open |
| T09.1 | 09 | Structural validation tool wrappers | [ranking.py:83](src/mofpipe/ranking.py#L83) | partly (builder sanity, slot mapping, force/stress) |
| T09.2 | 09 | Reference-phase competition | [ranking.py:75](src/mofpipe/ranking.py#L75) | open |
| T09.3 | 09 | Benchmarks, ablations, literature review | [report.py:101](src/mofpipe/report.py#L101), [intake.py:351](src/mofpipe/intake.py#L351) | open |
| T10.1 | 10 | AiiDA workflow engine and concurrency | [orchestrator.py:17](src/mofpipe/orchestrator.py#L17), [dft.py:200](src/mofpipe/dft.py#L200) | open |
| T10.2 | 10 | Runtime agent layer | [gateway.py:52](src/mofpipe/gateway.py#L52), [intake.py:265](src/mofpipe/intake.py#L265) | open |

---

## Task 01 — Inputs and framework construction

### T01.1 Full ProjectSpec and physical-case loader — done (Milestone B)
- **Where:** `project_spec.load_physical_project`, `project_spec.validate_physical_project`; dispatched from `intake.load_project` / `intake.validate_project`.
- **Implemented:**
  - ProjectSpec v0.1 (`instructions/project_spec.template.json` shape) loads into the same `ProjectSpec` dataclass with `project_type = "physical_structure"`. Protocols may be inline objects or project-relative JSON files.
  - Every required field that is null, empty or absent gives one `SCIENTIFIC_SPEC_MISSING` error with its exact path (`REQUIRED_FIELDS`). The untouched template gives exactly 28. Later-milestone fields (orientation catalogue, ML potential, embedding, quantum solvers, stopping, LLM provider) must stay null and are rejected as unsupported otherwise.
  - Nested protocol fields are checked too (`protocols.dft.*`, `protocols.builder.*`, `protocols.model_selection.*`, `building_blocks[i].*`).
  - `physical_case/recipe.json` is adapted into the same schema (`spec_from_recipe`), so it loads as a separate PHYSICAL project and blocks on exactly the inputs it lacks (parent CIF, 3D linker geometry/substitution rule, protocols, tolerances, budgets).
  - Physical projects declare their split policy explicitly (`protocols.model_selection`: `label_orbits_per_topology`, `holdout_fraction`, `n_folds`, `seed`). The checksummed fixture `project.json` keeps `DEFAULT_SPLIT_MANIFEST`.
  - A run directory records its `project_type` and `provenance`; reusing it for the other kind raises `RUN_DIR_PROJECT_TYPE_MISMATCH`. Comparison groups embed the DFT protocol hash, so physical and synthetic labels never pool (`INCOMPATIBLE_FIDELITY`).
- **Still open:** multiple topologies and transfer cells (Milestone C) are rejected at load time.
- **Needs:** A team decision on the real project's structures, tolerances and budgets.

### T01.2 CIF / periodic-net input mode — done (Milestone B)
- **Where:** `cif.py` (parser), `framework.build_slot_lattice`, `project_spec._load_source`.
- **Implemented:**
  - Standard-library CIF reader: cell, symmetry operations (`_space_group_symop_operation_xyz` or `_symmetry_equiv_pos_as_xyz`), atom-site loop, values with standard uncertainties. Expands to P1. The original text, its SHA-256 (verified against the spec) and its source are kept.
  - Partial occupancy and disorder groups are reported, never filled. `chemistry.disorder_resolution_policy` must be declared; only `require_fully_ordered` is implemented, so disordered input gives `DISORDER_UNRESOLVED`.
  - Unbound guests: `require_desolvated` gives `UNRESOLVED_GUEST_SPECIES`; `remove_unbound_guests` removes them and records what was removed.
  - Deconstruction: periodic covalent-radius bond graph → metal-oxo nodes and organic linkers, with unwrap offsets. Each linker is one slot. Linker and node species are identified by formula against the declared building blocks.
  - Checks before any calculation: connector-atom count and node connectivity per block, species per slot, integral counts per sublattice in the declared supercell (`NONINTEGRAL_COMPOSITION`), elemental inventory as an integer multiple of `target_elemental_inventory` (`INVENTORY_MISMATCH`), and charge balance under the supplied formal charges (`CHARGE_IMBALANCE`; reported as `SUPPLIED_NOT_INFERRED`).
  - Slot IDs are ordered by wrapped centroid, so they are stable under atom reordering (tested with a reversed and a randomly ordered P1 CIF).
  - The result is emitted as a `TopologySpec`: slot-graph pairs at distance 1 (with periodic contact multiplicity `pair_weights_d1`) and 2, connected triplets, and slot permutations from the CIF space-group operations combined with supercell translations. `topology_checks` verifies group closure and graph/weight invariance as for the fixture.
- **Still open:** linker-on-node net representations (only linker-on-edge is derived), net identification (`declared_net` is recorded but not verified; CrystalNets/MOFid is T09.1), non-diagonal supercell matrices.
- **Needs:** A verified desolvated parent CIF, e.g. MOF-5 for the physical case, plus a disorder/solvent policy.

### T01.3 Atomic framework construction — done (Milestone B), PORMAKE wiring open
- **Where:** `framework.build_framework`, `framework.ParentSubstitutionBuilder`, `framework.PormakeBuilder`.
- **Implemented:**
  - Builder adapter interface selected by `protocols.builder.builder_id`. `parent_substitution_v1` decorates the parsed parent supercell. Each substituted slot replaces one declared atom of its parent linker (e.g. H) with a declared group geometry (e.g. NH₂) in a local frame. The first such atom in Cartesian order about the linker centroid is the deterministic initial orientation.
  - Output `Geometry` record: lattice, coordinates, atom→block→slot map, slot adjacency, inventory, ordered and order-independent geometry hashes, and an exported P1 CIF.
  - Early sanity checks give `BuildFailure` records (`ATOM_OVERLAP`, `INVENTORY_MISMATCH`, `CONNECTIVITY_CHANGED`, `NO_CONSTRUCTION_RULE`), never energies. The connectivity check maps the built structure back onto the parent slots and nodes (`map_to_parent_slots`).
  - The structures stage rebuilds the known parent and requires an identical canonical geometry hash. It also builds one legal decoration. Tests build every symmetry-distinct legal decoration and check its counts.
- **To do:**
  - Wire PORMAKE (`PormakeBuilder` currently returns `BUILDER_UNAVAILABLE` / `BUILDER_NOT_VERIFIED`) after verifying the installed API and block database. Put ToBaCCo behind the same adapter.
  - Orientation states as search variables (T01.4).
- **Needs:** Validated 3D building blocks with attachment points, or a substitution rule per functionalised linker. The SMILES in `recipe.json` are not enough.

### T01.4 Linker orientation states
- **Where:** `structures.make_decoration`
- **Now:** A decoration is a binary occupation of slots; orientation is not represented.
- **To do:**
  - Add orientation variables from a finite declared catalogue, e.g. for asymmetric 2-amino-BDC.
  - Propagate them to symmetry canonicalization (`canonical_bits`), features (T03.1) and encodings (T03.5).
  - Fix a deterministic initial orientation convention for reproducibility.
- **Done when:** Orientation-distinct arrangements are not merged. Symmetry-equivalent ones get identical features.

### T01.5 Supercells and periodic translations
- **Where:** `structures.define_decoration_space`; also `candidates.deduplicate_candidates`
- **Now:** One working cell per topology. Fixture graphs have an explicit point-group-like permutation list. CIF-derived slot lattices use one diagonal `search.supercell_matrices` entry, and their permutations already combine the space-group operations with the supercell translations. A second matrix is rejected (`UNSUPPORTED_SUPERCELL_MATRIX`). Exact enumeration limits a cell to 20 slots.
- **To do:**
  - Enumerate the decoration space for each matrix in `search.supercell_matrices`, including non-diagonal ones.
  - Add lattice translations to the symmetry operations.
  - Keep a second cell for transfer tests, and verify integral counts per cell.
- **Done when:** Periodic replication tests pass: a replicated configuration has the same normalized energy, and its orbit maps consistently.

## Task 02 — Topology prescreen and ML potentials

### T02.1 Seed decoration generation
- **Where:** `structures.screen_topologies`
- **Now:** Identity screen: all supplied graphs are retained with evidence `NO_PHYSICAL_SCREENING_EVIDENCE`. Training labels come from fixture files.
- **To do:** Implement `generate_seed_decorations(topology_id, sampling_policy_id)`. It should produce pure endpoints (where chemically meaningful), ordered mixed states, feasible random states and optional SQS-like states, with orientation diversity, all tagged with their sampling method.
- **Done when:** Composition survives every sampling method, and the pure/SQS/ordered origin is distinguishable in the metadata.

### T02.2 ML-potential relaxation
- **Where:** same function
- **Now:** `ml_potential: SKIPPED_NO_ATOMIC_GEOMETRY`
- **To do:**
  - Add `submit_relaxations(decoration_ids, calculator_protocol_id)` through a registered ASE-compatible calculator adapter, using the job interface.
  - Verify the potential supports the elements, charges and magnetic assumptions, and flag out-of-domain use.
  - Use one documented relaxation policy for all configurations.
  - Check convergence, topology retention, cell distortion and mapping. Route connectivity-changing structures as transformed candidates, never as labels for the original lattice.
- **Done when:** A failed relaxation is never ranked as high energy, and out-of-domain use is flagged.
- **Needs:** The ML checkpoint and version.

### T02.3 Calibration and screening decisions
- **Where:** same function
- **To do:**
  - Calibrate ML energy *differences* against DFT (T06.1) on representative structures, and record the calibration domain. Label ML rankings exploratory until calibrated.
  - Compare only matched inventories, using mixing energies.
  - Emit retain / uncertain / deprioritise / reject with cited evidence. Hard rejection needs independent incompatibility evidence; energetic rejection needs a declared policy.
  - Keep a reproducible audit sample of excluded structures, with reinstatement possible.
- **Done when:** An uncertain topology survives for later exploration. A tiny known ordering fixture shows that screening does not discard its favourable mixed state.
- **Needs:** A screening/rejection policy and a sampling policy. The template has no fields for either (see *Open specification decisions*).

## Task 03 — Configurational Hamiltonians and encodings

### T03.1 Cluster-feature generation
- **Where:** `configuration_model.FEATURE_DEGREE` (the fixed feature library)
- **Now:**
  - Four fixed features from the fixture: `constant`, `unlike_distance_1`, `unlike_distance_2`, `BBB_paths`.
  - Milestone B added `site_orbit_k` point features, generated from the slot orbits of the topology's symmetry permutations. They are needed when the slot graph is not regular, e.g. a tetragonal supercell of a cubic net. Model candidates request them with `"site_orbits"`.
  - Milestone B also added periodic contact multiplicity on distance-1 pairs (`TopologySpec.pair_weights_d1`).
- **To do:**
  - Generate symmetry orbits of pair and selected triplet clusters from the slot lattice and symmetry operations, including periodic images and orientation variables.
  - Tie equivalent interactions while keeping physically inequivalent slots separate.
  - Generate feature polynomials automatically. `encoding.physical_polynomial` already accepts any feature polynomial.
  - icet may supply orbit machinery, but only after a tested slot-lattice mapping.
- **Done when:** Symmetry-related arrangements get identical features (extend `test_feature_symmetry_and_swap_reachability`), and periodic multiplicity and replication tests pass.

### T03.2 Model validation for real data — done (Milestone B)
- **Where:** `configuration_model.fit_and_select_grouped`, `evaluate_holdout`, `datasets.freeze_physical_split`.
- **Implemented:**
  - The split is frozen before any label exists: seeded content-hash order of symmetry orbits gives labelled orbits, a final holdout fraction and an unlabelled rest. The holdout snapshot is stored separately and read exactly once, after selection (`HoldoutReport`). `HOLDOUT_VALIDATION_FAILED` halts the run.
  - Grouped K-fold cross-validation over orbit groups. Orbits whose relaxed geometries coincide share a group.
  - Exact rank check with a documented gauge. `rank_deficiency: drop_dependent_columns` keeps a column only if it raises the rank (feature order decides); `fail` records `RANK_DEFICIENT_FEATURES`. At fixed composition the site orbits sum to a constant, so one is always dropped.
  - Optional ridge regularisation (`regularization: {kind: ridge, alpha}`); the constant is not penalised.
  - Metrics: CV RMSE and max error, tie-aware low-energy recall at `tolerances.low_energy_validation_criterion.k`, Spearman rank correlation, and recovery of the lowest labelled state. Pass = CV RMSE ≤ `target_relative_energy_tolerance` and recall ≥ `min_recall`. Selection takes the fewest effective parameters, then the lowest CV RMSE, then the model ID. No passing model gives `MODEL_INADEQUATE` with recorded data requirements.
  - The ranking uses tolerance-equivalence ties (`tie_tolerance`) instead of exact equality.
- **Still open:** nested CV for hyper-parameters (alpha is fixed by the protocol); larger-supercell transfer metric (needs T01.5); ground-state recovery over all states (only over the labelled set).
- **Needs:** The real model-selection policy and physical accuracy tolerance. The toy's 5 meV per formula unit is a test value.

### T03.3 Uncertainty and multifidelity
- **Where:** same function
- **Now:** `HamiltonianArtifact.uncertainty` stores only the development RMSE, with a note that it is not a certified bound.
- **To do:**
  - Calibrate predictive uncertainty on held-out residuals and document its domain. Bootstrap or committee spread alone is not a bound.
  - Treat ML, DFT and embedded-corrected energies as separate fidelities. A multifidelity model must fit calibrated differences between matched calculations and never concatenate fidelities as one target. `datasets.merge_snapshots` already refuses mixed fidelity.

### T03.4 Quadratization
- **Where:** `encoding.compile_encoding`
- **Now:** A QUBO target with degree > 2 raises `UNSUPPORTED_HIGHER_ORDER_WITHOUT_QUADRATIZATION`. The cubic model is kept as a HUBO for direct enumeration.
- **To do:**
  - Add an explicit reduction, e.g. substitution with ancilla variables, recording ancillas in `EncodingArtifact.ancillas` and the decode rules.
  - Choose penalty strengths by a stated bound.
  - Extend `verify_encoding` to minimize over ancillas.
- **Done when:** On exhaustive small fixtures, minimizing over ancillas reproduces the original feasible cost up to the documented constant, with no infeasible ground state.

### T03.5 Multicomponent encodings and penalty bounds
- **Where:** `encoding.compile_encoding`; `intake.validate_project` (the `binary_encoding` check)
- **Now:** Binary species only; the composition is a single global count `(Σx − k)²`. Penalty sufficiency is certified only by checking all 2ⁿ states, which works for n = 12 only.
- **To do:**
  - One-hot (or domain-wall) encodings for more than two species per slot.
  - Per-sublattice count constraints.
  - An auditable penalty bound for instances too large to enumerate.
- **Done when:** On small instances, the encoded minimum is feasible and equals the physical minimum. The penalty choice is recorded with its justification.

## Task 04 — Arrangement solvers

### T04.1 Exact solver for large cells
- **Where:** `optimization.exact_enumeration`
- **Now:** Full enumeration of C(n, k) states (924 here), certified `OPTIMAL_CERTIFIED_EXHAUSTIVE`. Simulated annealing exists as an uncertified heuristic.
- **To do:**
  - Add a mixed-integer or constraint-programming solver with honest optimality gaps.
  - On timeout, return the best feasible candidate with an uncertified status.
  - Bound the coefficient-rounding error if integer scaling is used.
  - Return a `Job` for long solves.
- **Done when:** It agrees with enumeration on small fixtures and reports gaps correctly on larger ones.

### T04.2 QAOA simulation
- **Where:** `optimization.quantum_solver_status`; the `quantum_disabled` gate in `intake.validate_project`
- **Now:** Reports `DISABLED_BY_CONFIG`. Only the classical prerequisite exists: `structures.swap_graph_connected` shows that fixed-count exchanges along edges reach all 924 states.
- **To do:**
  - Simulate QAOA with an XY exchange mixer on that connected graph, starting from feasible states.
  - For one-hot states, use exchanges that preserve both one-species-per-site and the global counts.
  - Verify domain-wall/ancilla dynamics separately, if used.
  - Decode every sample and re-evaluate it with the physical Hamiltonian.
  - Record bit order, depth, shots, optimizer calls, seeds, post-selection rate and wall time.
  - Benchmark best/median quality and success probability against exact enumeration and simulated annealing across seeds, for both solver-only and end-to-end cost.
- **Done when:** It agrees with exact minima on small fixtures within its stated success criteria. Mixer evolution preserves every declared constraint, and bit-order or ancilla errors are caught. Simulation is not reported as evidence of quantum speedup.

### T04.3 QPU hardware adapter
- **Where:** `optimization.quantum_solver_status`
- **To do:** Add a hardware adapter only after the simulator is accepted. Submission and budgets are controlled by the scheduler, never per optimizer iteration by an agent. Keep credentials in backend services only.

## Task 05 — Deduplication and acquisition

### T05.1 Geometry-level deduplication
- **Where:** `candidates.deduplicate_candidates`
- **Now:** Deduplicates discrete occupations by symmetry orbit (lexicographically smallest image, no A/B exchange).
- **To do:**
  - Add relaxed-geometry equivalence as a separate step. Never merge distinct nets, conformers or magnetic states just because their energies match.
  - Add periodic-translation canonicalization (T01.5).
  - Keep the unsymmetrized mapping for construction.
- **Done when:** Known symmetry copies collapse and deliberately inequivalent arrangements stay separate.

### T05.2 Evaluation-batch acquisition policy
- **Where:** `candidates.select_pool`
- **Now:** Takes the top k orbits per topology by surrogate base energy, keeping all ties at the cutoff, with a machine-readable `selection_reason`.
- **To do:**
  - Implement `select_evaluation_batch` with a declared mix of low predicted energy, uncertainty and feature diversity.
  - Allocate coverage across surviving topologies, plus a controlled audit fraction of deprioritized regions.
  - Respect an evaluation budget, and label each candidate with the model version used at selection.
  - Skip individual candidates that are already pending or successful. Today the job cache only deduplicates identical whole batches.
  - Include larger-supercell candidates to test ordering-period dependence.
- **Done when:**
  - An uncertain, under-sampled topology receives candidates.
  - Selection is reproducible from a seed.
  - Held-out labels are never used to select training candidates.

## Task 06 — Periodic DFT

### T06.1 Periodic DFT backend — done (Milestone B); licensed codes, k-points and HPC open
- **Where:** `dft.py` (`PeriodicDFTBackend`, `QuantumEspressoAdapter`, `evaluate_geometry`, `difference_convergence_study`).
- **Implemented:**
  - Same `submit → job_status → collect` interface as the synthetic backend. The idempotency key is the protocol hash plus geometry hash, calculation, magnetic state and overrides, so a restart reuses successful jobs.
  - The Quantum ESPRESSO `pw.x` adapter writes inputs from a stored `Geometry`: XC functional, dispersion, pseudopotential files (SHA-256 verified at intake), cutoffs, k-mesh from a declared spacing, occupations/smearing, spin, Hubbard U, convergence thresholds, force/stress tolerances. It parses the raw output: energy, SCF/ionic convergence, forces, pressure, final cell and coordinates, version.
  - Consistent `vc-relax`/`relax`, then a final single point on the relaxed geometry. Every declared initial magnetic state runs separately and every outcome is stored. The lowest converged state is selected and labelled "not a global certificate".
  - Before any label, a plane-wave cutoff convergence study of an energy difference (two decorations) runs against `target_relative_energy_tolerance` in eV per formula unit. If it is not converged the run stops with `BLOCKED: DFT_SETTINGS_NOT_CONVERGED`.
  - Post-run checks give non-energy results: `SCF_NOT_CONVERGED`, `IONIC_NOT_CONVERGED`, `RESIDUAL_FORCE_ABOVE_TOLERANCE`, `RESIDUAL_STRESS_ABOVE_TOLERANCE`, `COMPOSITION_CHANGED`, `FORMULA_UNIT_MISMATCH`, `JOB_FAILED`, `INCOMPLETE_OUTPUT`. A slot-mapping failure after relaxation becomes `TRANSFORMED` (a transformation event, never a label).
  - Normalisation is to the formula-unit count verified from the relaxed inventory. The fidelity is `DFT:<protocol_id>:<hash>`; the comparison group is formula unit + fidelity.
  - Retries follow `numerical_retries` and may only change `mixing_beta`, `mixing_mode` and `electron_maxstep` (`PROTOCOL_RETRY_CHANGES_SCIENCE` otherwise). Each attempt is recorded in `recovery_history`.
  - Budgets: `max_dft_jobs` and `max_cpu_core_hours` are checked before every submission (`BUDGET_EXHAUSTED` → stage `BLOCKED`). A missing executable gives `BACKEND_UNAVAILABLE` → `BLOCKED`.
- **To do:**
  - Adapters for licensed codes behind the same interface.
  - k-point convergence of energy differences (only the cutoff is studied).
  - Zero-smearing extrapolation conventions (only `total_energy_as_printed` is accepted).
  - Remote/HPC submission through AiiDA CalcJobs (T10.1). Jobs currently run locally and synchronously.
- **Needs:** The DFT protocol, magnetic-state policy, pseudopotentials, HPC resources and budgets.

### T06.2 Real evaluation path in the orchestrator — done (Milestone B)
- **Where:** `Pipeline._physical_labels`, `_physical_evaluation`, `_physical_ranking`, `_physical_validation` (selected automatically for physical projects).
- **Implemented:**
  - Labels: `build_framework` → DFT relaxation and single point for every labelled orbit representative.
  - Pool evaluation: the same chain for the candidate pool. Already-labelled candidates come from the job cache.
  - `BuildFailure`, failed and transformed calculations are recorded and become missing evaluations; the ranking is then `INCOMPLETE`.
  - The ranking is DFT-only, at base level, with tolerance ties. The corrected ranking is `DISABLED_BY_CONFIG`.
- **To do:** Optional embedding correction per candidate geometry once T07.3 exists ([orchestrator.py:349](src/mofpipe/orchestrator.py#L349)).

## Task 07 — Embedding and electronic solvers

This is a research milestone, not API wiring. The embedding formalism must be specified by the team before coding.

### T07.1 Embedding scheme and active-space export
- **Where:** `electronic.py` header; `electronic_stage_status` (`embedding_correction`)
- **Now:** `DISABLED_BY_CONFIG`. Corrections in fixture runs are labelled `ARTIFICIAL_BOOKKEEPING_TEST`.
- **To do:**
  - Freeze a mathematically specified total-energy expression: environment, region(s), coupling terms, frozen/core contributions, boundary/capping terms, and the part replaced at high level.
  - Implement `prepare_active_space(candidate_id, embedding_protocol_id)`. It must take a specific geometry and calculation ID, because one candidate can have several metastable or magnetic states. It exports h1, h2, scalar constants, electron counts by spin, orbital ordering, integral convention and geometry/basis/embedding hashes, as FCIDUMP plus a metadata record.
  - Choose a balanced active space that preserves metal, ligand and neighbouring-node character. Do not force identical orbital counts.
- **Done when:** Identical inputs reproduce identical integrals and constants.

### T07.2 Electronic solvers
- **Where:** `electronic_stage_status` (`active_space_fci`, `vqe`, `sqd_hardware`)
- **Now:** `NOT_RUN` / `DISABLED_BY_CONFIG`.
- **To do:**
  - Run PySCF CASCI/FCI where feasible, and simulated VQE on exactly the same Hamiltonian, orbitals and constants. Account for the fermion-to-qubit mapping, bit order, symmetry tapering and state-preparation sector.
  - Add hardware SQD as a separate adapter: sample collection, configuration recovery, subspace diagonalization, plus shot/hardware provenance and sample-sensitivity data.
  - Never call approximate classical references exact.
- **Done when:** Tiny Hamiltonians match an independent exact diagonalization, and VQE/SQD comparisons use identical electronic problems.

### T07.3 Corrected-energy assembly
- **Where:** `electronic.py` header
- **To do:**
  - Implement the scheme's own `assemble_corrected_energy`, with no overlapping-region double counting and every inequivalent node in the compared cell included.
  - Emit corrected energies only when the assembler and comparison group are validated. Otherwise return a benchmark artifact with `correction_status=UNVALIDATED` and keep a DFT-only ranking.
- **Done when:**
  - The high = low consistency test passes where the formalism requires it.
  - An unchanged correction leaves the ranking unchanged. This is already tested for the synthetic constant correction.
  - Missing subtraction terms or incomplete node coverage block assembly.
  - At least one published embedding reference (the Fe–MOF-74 workflow) is reproduced.

## Task 08 — Adaptive refinement

### T08.1 Active-learning acquisition policy
- **Where:** `refinement.plan_next_batch`
- **Now:** Picks configurations by feature-space novelty (L1 distance to labelled feature vectors), restricted to training-split orbits. It never picks development or holdout orbits.
- **To do:**
  - Use the registered energy/uncertainty/diversity policy, shared with T05.2.
  - Explore outside the current model's preferred basin.
  - Revisit uncertain topology exclusions (T02.3) and untested supercell patterns (T01.5).
- **Done when:** On a controlled enumerable fixture, a deliberately biased initial training set recovers a missed basin.

### T08.2 Stopping criteria
- **Where:** `refinement.run_active_learning`
- **Now:**
  - `CONVERGED_WITHIN_SCOPE` when every topology selects a model passing the development threshold in one round.
  - `MODEL_INADEQUATE` when every candidate is well determined but fails.
  - `BUDGET_EXHAUSTED` at the iteration cap (at most 3).
  - `INCONCLUSIVE` when nothing is acquirable.
- **To do:**
  - Require stability over `tolerances.consecutive_stable_rounds`.
  - Re-solve leading candidates every round and compare gaps using tolerance-aware equivalence classes.
  - Add coverage criteria, and account for solver gaps and finite search coverage.
  - Allow explicit model-family expansion when requested.
  - Add a matched embedded-correction surrogate when justified.
  - Keep a persistent event log, and resume mid-loop from the last completed iteration.
- **Done when:**
  - Near-degenerate candidates stay tied within tolerance.
  - Interrupted iterations resume without repeating expensive work.
  - Budget exhaustion is never reported as convergence (already tested).

## Task 09 — Validation, competition and report

### T09.1 Structural validation tool wrappers
- **Where:** `ranking.validate_structures`
- **Now:**
  - Fixture runs report every check as `SKIPPED_NO_ATOMIC_GEOMETRY`.
  - Physical runs already report builder sanity, post-relaxation slot mapping, and residual force/stress (from `framework.py` and `dft.py`).
  - External tools are still skipped.
- **To do:** Implement `validate_structure(geometry_id, validation_protocol_id)`, wrapping each tool with its version, parameters, parseable status and raw evidence:
  - MOFChecker sanity checks
  - CrystalNets / MOFid topology identity
  - Zeo++ pore descriptors
  - post-relaxation connectivity, and residual force/stress checks
  - targeted phonon/elastic stability for finalists
  - PXRD comparison, and NMR only with a validated protocol, against referenced measurements with stated instrumental tolerances

  Tool success is not proof of synthesizability.
- **Done when:** Simulated observations are labelled simulated, and every reported number traces to a stored artifact.

### T09.2 Reference-phase competition
- **Where:** `ranking.evaluate_reference_mixtures`
- **Now:** Always `INCOMPLETE`: no reference set exists, so the result is a relative ranking, not a stability certificate.
- **To do:**
  - Accept a reference set of phase-separated combinations at the same overall composition and compatible energy level.
  - Solve the mixture balance explicitly.
  - Report incomplete reference coverage instead of claiming absolute stability.
- **Done when:** A missing reference gives an incomplete result, never a stability certificate.

### T09.3 Benchmarks, ablations and literature review
- **Where:** `report.build_report`; the literature check in `intake.validate_project`
- **Now:** The report lists static literature entries with `live_review: NOT_RUN`. No benchmark or ablation sections.
- **To do:**
  - Benchmark sections that reproduce the Kang (arrangements), Cai (topology competition) and Fe–MOF-74 (electronic workflow) studies, with exact paper/data versions. Keep reproduction of a paper separate from agreement with experiment.
  - Ablations that test actual claims: fixed topology vs joint search, heuristic vs fitted costs, no vs validated embedding, classical vs quantum optimizer, deterministic vs agent-assisted runs.
  - Optionally, a live literature review that maps claims to cited passages.

## Task 10 — Agent architecture

### T10.1 AiiDA workflow engine and concurrency
- **Where:** `orchestrator.py` (module level, before `STAGES`)
- **Now:** A local sequential engine (`Pipeline`) with a JSON run store: authoritative `state.json`, an event log, content-addressed artifacts and stage reuse on re-run.
- **To do:**
  - Port the stages to AiiDA WorkChains, with CalcJobs for external DFT/QPU jobs, keeping the same interfaces.
  - Add single-writer or optimistic-concurrency state commits.
  - Run independent topologies, DFT configurations and solver seeds concurrently within resource limits.
  - Support cancel/resume at calculation boundaries.

### T10.2 Runtime agent layer
- **Where:** `gateway.review_proposal`; the `runtime_agents_disabled` check in `intake.validate_project`
- **Now:** Proposals are fully validated: schema, project ID, state version, role permissions, controller-only tools and evidence artifact existence. They are then always rejected (`AGENTS_DISABLED`, or `UNSUPPORTED` if agents were enabled). Projects with agents enabled are `BLOCKED`.
- **To do:**
  - Execute accepted proposals:
    - validate the arguments against policy
    - reserve budget atomically
    - submit idempotently
    - record job and artifact IDs
    - attempt a bounded repair on invalid output
  - Expose the same tested functions as typed tools to the chosen LLM provider, with role prompts and per-role permissions from `runtime.allowed_tools_by_role`.
  - Invoke roles only at the checkpoints listed in instructions/10.
  - Store each decision with its evidence plus prompt/model/version metadata.
  - Treat CIF metadata, papers and logs as untrusted data.
- **Done when (Milestone F):**
  - Compared against the deterministic baseline at equal budget: completion rate, invalid tool-call rate, duplicated jobs and cost.
  - Replay tests cover stale state, malformed output, contaminated logs, missing active-space terms and budget exhaustion.

---

## Open specification decisions (no code marker)

These came out of the instruction-set review and must be settled in `instructions/` or the project spec. Code cannot resolve them.

- **Template fields missing from `project_spec.template.json`:**
  - ~~disorder resolution policy (T01.2)~~ — the loader requires `chemistry.disorder_resolution_policy`; it should be added to the template.
  - ~~topology definitions themselves~~ — physical projects derive `TopologySpec` from `source_structures[i]` (CIF + `topology_id` + checksum + source); the template's `source_structures: []` needs this entry shape documented.
  - ~~holdout/split policy (T03.2)~~ — carried in `protocols.model_selection`; should be documented in the template.
  - project `provenance` (`SYNTHETIC` | `PHYSICAL`) — required by the loader, absent from the template.
  - reference phase sets (T09.2)
  - seed sampling and screening policies (T02.1, T02.3)
  - ranking comparison policy (tie tolerance currently reuses `target_relative_energy_tolerance`)
  - electronic solver protocols for CASCI, VQE and SQD (T07.2)
  - split manifest path in the fixture `project.json` (T01.1) — unchanged because the fixture is checksummed.
- **Signature ambiguities:**
  - `prepare_active_space` should take a geometry/calculation ID, not a candidate ID (T07.1).
  - `assemble_corrected_energy` needs a third outcome, the `UNVALIDATED` benchmark artifact (T07.3).
  - `select_evaluation_batch` (05) and `plan_next_batch` (08) both produce `AcquisitionBatch`; decide which owns it.
  - `advance_workflow` must stay controller-only. `gateway.CONTROLLER_ONLY` already enforces this.
- **Physical tolerances:** energy tie tolerance, force/stress tolerances, low-energy validation criterion and stable-round count are `null` in the template and must be chosen per study.

Already resolved in code, and worth proposing back into `00_SHARED_CONTRACT.md`:
- one status vocabulary (`contract.StageStatus`, `JobState`, `StopReason`);
- the glue producers `define_decoration_space`, `ingest_labels` / `merge_snapshots` (DatasetSnapshot), `freeze_split` (ValidationSplit) and candidate registration via `CandidateSet`.

## Suggested order

Following the milestones in `instructions/10_AGENT_ARCHITECTURE.md`. Milestone A, the synthetic end-to-end fixture, is done. Milestone B (T01.1, T01.2, T01.3, T06.1, T06.2, T03.2) is done in software and verified on the SYNTHETIC toy; running it on MOF-5 BDC/NH₂-BDC needs the verified CIF, a substitution rule or 3D blocks, a DFT protocol with pseudopotentials, and tolerances/budgets.

1. **Milestone C:** T01.5, T03.1, T05.1, T05.2, T08.1, T08.2, T09.1, T09.2, plus T02.1–T02.3 if ML screening is used. Second topology, supercells, acquisition, refinement and the full report.
2. **Milestone D:** T04.1, T04.2, T03.4, T03.5, then T07.1–T07.2 on a published benchmark.
3. **Milestone E:** T07.3, embedding corrections fed back into ranking.
4. **Milestone F:** T10.1, T10.2, T04.3, T09.3.
