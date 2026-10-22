# TODO — unimplemented parts of `mofpipe`

This file explains every `#TODO` marker in `src/mofpipe/`. The pipeline runs end to end on two kinds of input:
- the synthetic graph fixture (`mof_no_agent_fixture/`, Milestone A);
- physical projects: CIF input → slot lattices → built frameworks → periodic DFT labels → grouped-validated models → DFT-ranked candidates.
  - Milestone B covers a single topology.
  - Milestone C adds several topologies, transferable cluster features, supercell transfer checks, acquisition and refinement, relaxed-geometry dedup, references, validation and the report.
  - Milestone D adds quadratization, sublattice constraints, branch and bound, a QAOA simulation benchmark, active-space export and ED/CASCI/VQE/SQD solvers.
  - Milestone E adds the finite-cluster subtractive embedding: corrected-energy assembly, scheme validation and a corrected ranking next to the DFT one.
  - Milestone F adds ML screening, fake-device QPU sampling, the AiiDA engine and the runtime agents (A1/A2/A67).
  - After F (this update, uncommitted until reviewed):
    - topology generation with PORMAKE (`protocols.topology_generation`);
    - an exploratory ML prioritisation opt-in;
    - a drop-instead-of-halt policy for topologies no model can fit;
    - SQD as an embedding high level, with declared SCF numerics for transition-metal clusters;
    - the A4 optimisation role (in progress).

    The MOF-5 scheme project is `tests/mof5_scheme.py`; environment setup is in the README, "Installation on a new machine".

Periodic DFT runs through PySCF (GPW, CPU/GPU) or a Quantum ESPRESSO adapter. Physical results so far come from SYNTHETIC toys (`tests/toy_framework.py` with a mock `pw.x`; `tests/pyscf_toy.py` with real PySCF). Real MOF-5 runs still need the inputs listed under **Needs**.

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
| T01.1 | 01 | Full ProjectSpec and physical-case loader | — | **Done (B, C: several topologies)** |
| T01.2 | 01 | CIF / periodic-net input mode | — | **Done (B)** |
| T01.3 | 01 | Atomic framework construction | [framework.py:571](src/mofpipe/framework.py#L571), [topology_builders.py](src/mofpipe/topology_builders.py) | **Done (B)**; **PORMAKE topology generation done (post-F)**; ToBaCCo, per-slot PORMAKE assembly open |
| T01.4 | 01 | Linker orientation states | [structures.py:77](src/mofpipe/structures.py#L77) | open |
| T01.5 | 01/05 | Supercells and periodic translations | [structures.py:91](src/mofpipe/structures.py#L91) | **Done (C)** for diagonal supercells; non-diagonal open |
| T02.1 | 02 | Seed decoration generation | [screening.py](src/mofpipe/screening.py) | **Done (F)**; orientation diversity needs T01.4 |
| T02.2 | 02 | ML-potential relaxation | [mlip.py](src/mofpipe/mlip.py), [mlip_worker.py](src/mofpipe/mlip_worker.py) | **Done (F)**: orb-v3 + D3(BJ) |
| T02.3 | 02 | ML calibration against DFT and screening decisions | [screening.py](src/mofpipe/screening.py) | **Done (F)**; real-DFT calibration of orb on MOF-5 not run |
| T03.1 | 03 | Cluster-feature generation from the slot lattice | [configuration_model.py:17](src/mofpipe/configuration_model.py#L17) | **Done (C)** for point/pair orbits; triplets, orientation open |
| T03.2 | 03 | Model validation for real (noisy) data | — | **Done (B)**; transfer metric added (C) |
| T03.3 | 03 | Uncertainty calibration and multifidelity models | [configuration_model.py:325](src/mofpipe/configuration_model.py#L325) | open |
| T03.4 | 03 | Quadratization of higher-order models | — | **Done (D)** for cubic models |
| T03.5 | 03/04 | Multicomponent encodings and penalty bounds | [intake.py:292](src/mofpipe/intake.py#L292) | **Done (D)** at the encoding layer; pipeline-wide multicomponent open |
| T04.1 | 04 | Exact solver for large cells | [optimization.py:92](src/mofpipe/optimization.py#L92) | **Done (D)**: branch and bound with gaps; Job return, stronger bounds open |
| T04.2 | 04 | QAOA simulation | — | **Done (D)** |
| T04.3 | 04 | QPU hardware adapter | [qpu.py](src/mofpipe/qpu.py), [qpu_worker.py](src/mofpipe/qpu_worker.py) | **Done (F)** on Qiskit fake devices; real IBM devices open |
| T05.1 | 05 | Geometry-level deduplication | [candidates.py:19](src/mofpipe/candidates.py#L19) | **Done (C)**; non-diagonal canonicalisation open |
| T05.2 | 05 | Evaluation-batch acquisition policy | [candidates.py:46](src/mofpipe/candidates.py#L46) | **Done (C)** for training acquisition; finalist-pool budget open |
| T06.1 | 06 | Periodic DFT backend | [dft.py:75](src/mofpipe/dft.py#L75), [dft.py:229](src/mofpipe/dft.py#L229), [dft.py:429](src/mofpipe/dft.py#L429) | **Done (B, C: PySCF GPW engine, CPU/GPU)**; DFT+U/hybrids in PySCF, k-point study, HPC open |
| T06.2 | 01/06/07 | Real evaluation path in the orchestrator | — | **Done (B)**; embedding stage added (E) |
| T06.3 | 06 | GPU path validation | [dft.py:229](src/mofpipe/dft.py#L229) | written against gpu4pyscf 1.8, **not exercised** (no CUDA device) |
| T07.1 | 07 | Embedding scheme and active-space export | [active_space.py:45](src/mofpipe/active_space.py#L45) | **Done (D)** for bare systems; **(E)** capped node clusters for the subtractive scheme; WF-in-DFT open |
| T07.2 | 07 | Electronic solvers: CASCI/FCI, VQE, SQD | [electronic_solvers.py:242](src/mofpipe/electronic_solvers.py#L242) | **Done (D)** in simulation; hardware SQD open |
| T07.3 | 07 | Corrected-energy assembly | [electronic.py:24](src/mofpipe/electronic.py#L24) | **Done (E)** for the finite-cluster subtractive scheme; WF-in-DFT and Fe–MOF-74 reference open |
| T08.1 | 08 | Active-learning acquisition policy | [refinement.py:21](src/mofpipe/refinement.py#L21) | **Done (C)** for physical projects |
| T08.2 | 08 | Stopping criteria | [refinement.py:90](src/mofpipe/refinement.py#L90) | **Done (C)**; solver-gap accounting, model-family expansion open |
| T09.1 | 09 | Structural validation tool wrappers | [ranking.py:83](src/mofpipe/ranking.py#L83) | **Done (C)**: in-house checks + optional MOFChecker/Zeo++; CrystalNets, phonons, PXRD open |
| T09.2 | 09 | Reference-phase competition | — | **Done (C)** for pure-endpoint references |
| T09.3 | 09 | Benchmarks, ablations, literature review | [report.py:171](src/mofpipe/report.py#L171), [intake.py:369](src/mofpipe/intake.py#L369) | partial: live literature/dataset review (A67) and the deterministic-vs-agent ablation; paper reproductions open |
| T10.1 | 10 | AiiDA workflow engine and concurrency | [aiida_engine.py](src/mofpipe/aiida_engine.py), [store.py](src/mofpipe/store.py) | **Done (F)**: WorkChain + provenance, optimistic commits; HPC CalcJobs and concurrency open |
| T10.2 | 10 | Runtime agent layer | [agents/runner.py](src/mofpipe/agents/runner.py), [gateway.py](src/mofpipe/gateway.py) | **Done (F)** on the fixture; physical runs replay-tested only |

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
- **Milestone C:**
  - Several `candidate_topology_ids` are accepted, each with its own `source_structures` entry; the IDs must match (`TOPOLOGY_SOURCE_MISMATCH`).
  - Per-source `supercell_matrices` override `search.supercell_matrices`. The first matrix is the working cell, any further ones are transfer cells.
  - New protocol fields are validated at intake: `protocols.stopping`, the acquisition policy, `protocols.validation.{reference_phases, reference_set_complete, geometry_equivalence_tolerance_A, supercell_transfer}`.
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
- **Topology generation with PORMAKE (post-F)** — `topology_builders.py`, `pormake_worker.py`, CLI `mofpipe generate-topologies`:
  - **Declaration and wiring.** `protocols.topology_generation` replaces `source_structures` (declaring both is `TOPOLOGY_SOURCE_CONFLICT`). The loader runs generation before validation; accepted nets become source structures with their own `supercell_matrices`, so everything downstream treats them like supplied CIFs.
  - **Worker.** Runs PORMAKE 0.2.3 in the declared interpreter:
    - net search: a `.cgd` connectivity pre-filter, then `Locator` RMSD of the node block on every unique node, limited by `max_node_rmsd_A` and `max_nodes_per_cell`;
    - `Builder.build_by_type` with the node block on every node type and the parent edge block on every edge type, written as a P1 CIF;
    - substituent frame coordinates derived from a declared PORMAKE edge block (e.g. `E72` for NH₂). It refuses blocks whose root atom is on the connection path, e.g. the –NH– bridge `E201`.
  - **Main-process checks for each built net:**
    - smallest diagonal supercell with integral composition within [min, max] slots;
    - the ordinary slot-lattice checks;
    - a build with every slot substituted (catches substituent clashes in strained nets).

    Failing nets are rejected with a reason and never repaired.
  - **Cache.** Results go to `<project>/generated/<key>/`, keyed by protocol, blocks, composition and worker hash. The report (accepted/rejected, versions, selection rule) is in `spec.physical["topology_generation"]` and is shown to A1 as `topology_search`.
  - **MOF-5 result** (Zn₄O `N16` + p-phenylene `E14`, NH₂ from `E72`, automatic scan):
    - accepted: `pcu` (RMSD 0, 2×2×1, 12 slots) and `bsn` (RMSD 0.206 Å, 12 slots);
    - rejected: `snp` (substituent overlap 0.15 Å), `sxb` (ambiguous amino anchor), `kag`/`sxc` (9 slots per cell), `acs` (RMSD 0.42 Å), `btu` (beyond `max_candidates`), plus connectivity/size rejections and four unreadable `.cgd` files.
  - **Interpreters.** Uses `.venv` (pormake, jax, pymatgen); orb-v3 needs a separate Python ≤ 3.13 environment `.venv-mlip`. See the README, "Installation on a new machine".
  - **Not yet:** automated tests for generation, and the full ML screen of the generated nets (an orb-v3 run was in progress when this was written).
- **To do:**
  - ToBaCCo (refused as `UNSUPPORTED_TOPOLOGY_BUILDER`: not an installable package; needs a pinned checkout behind the same interface).
  - Per-decoration PORMAKE assembly (`PormakeBuilder`, an edge block per slot) instead of parent substitution.
  - Mixed node blocks or several edge types per net; non-diagonal supercells.
  - Point-group symmetry for generated nets. They keep translations only, which is consistent with the substituent orientation convention (T01.4).
  - Orientation states as search variables (T01.4).
- **Needs:** Validated 3D building blocks with attachment points, or a substitution rule per functionalised linker. The SMILES in `recipe.json` are not enough; PORMAKE's block library now supplies both for the MOF-5 case.

### T01.4 Linker orientation states
- **Where:** `structures.make_decoration`
- **Now:** A decoration is a binary occupation of slots; orientation is not represented.
- **To do:**
  - Add orientation variables from a finite declared catalogue, e.g. for asymmetric 2-amino-BDC.
  - Propagate them to symmetry canonicalization (`canonical_bits`), features (T03.1) and encodings (T03.5).
  - Fix a deterministic initial orientation convention for reproducibility.
- **Done when:** Orientation-distinct arrangements are not merged. Symmetry-equivalent ones get identical features.

### T01.5 Supercells and periodic translations — done for diagonal supercells (Milestone C)
- **Where:** `project_spec._load_source` (working and transfer cells), `transfer.py`, `orchestrator._physical_supercell_checks`.
- **Implemented:**
  - Every further diagonal matrix is a transfer cell with its own TopologySpec (`<topology>@AxBxC`), slot permutations (space group × translations), integral counts per cell and formula units.
  - The structures stage checks exactly that every working-cell orbit, replicated, has identical per-formula-unit features and maps to one transfer orbit.
  - The `supercell_checks` stage compares DFT on replicated labels (same normalised energy within tolerance) and on seeded native transfer-cell orderings (model prediction error), giving a `SupercellTransferReport`.
- **To do:** non-diagonal supercell matrices (the builder and the cluster enumeration assume diagonal cells).
- **Done when (remaining):** the same replication tests pass for a non-diagonal transfer cell.

## Task 02 — Topology prescreen and ML potentials

### T02.1 Seed decoration generation
- **Where:** `screening.generate_seed_decorations`
- **Now:** Per topology:
  - pure-linker endpoints, used as matched references;
  - the most symmetric ordered states, seeded random states and SQS-like states (unlike-pair count closest to random), all at the fixed composition.
  - Every seed is tagged with its sampling method. The policy is `protocols.screening.sampling`.
  - The fixture keeps the identity screen (no atoms).
- **To do:** Orientation diversity, once linker orientation states exist (T01.4).

### T02.2 ML-potential relaxation
- **Where:** `mlip.MLPotentialBackend`, `mlip_worker.py`; protocol `protocols.ml_potential_checkpoint`
- **Now:** **orb-v3-conservative-inf-omat + DFT-D3(BJ)**, chosen from MOFSimBench: orb-v3-omat and eSEN-OAM gave the most successful and accurate MOF optimizations, and orb's weights are open (Apache-2.0).
  - Relaxations run in a worker subprocess with any interpreter that has orb-models, so the core never imports torch.
  - One documented relaxation policy (FIRE/BFGS, variable or fixed cell, fmax, max steps).
  - Jobs are idempotent on the protocol and geometry hashes. The engine identity, including the checkpoint sha256, is verified.
  - Domain flags: net charge, open-shell initial states, declared elements.
  - Checks: convergence, slot/node topology retention (TRANSFORMED candidates) and cell distortion. Failures are never energies.
  - Verified on MOF-5 from CoRE MOF 2024: relaxed in 33 s on CPU, volume +1.2% vs experiment.
- **To do:** GPU/MPS device and batching for large screens. Alternative engines (eSEN/UMA via fairchem, gated weights) behind the same protocol.

### T02.3 Calibration and screening decisions
- **Where:** `screening.calibrate`, `screening.screen`; `protocols.ml_calibration`, `protocols.screening`; `Pipeline._physical_screening`
- **Now:**
  - ML-vs-DFT errors of energy differences are measured on representative seeds (lowest/highest ML energy per topology, cross-topology pairs included). The ranking is EXPLORATORY until within tolerance.
  - Comparisons use the common formula unit at fixed composition; mixing energies are taken against each topology's own endpoints.
  - Decisions:
    - reject only on structural incompatibility;
    - deprioritise only when calibrated, with a declared `energetic_policy`, and only when the gap exceeds threshold + 2 × calibration MAE;
    - otherwise uncertain or retain;
    - `reinstate` brings a topology back. The audit sample is the excluded topologies' seed results.
  - Later stages see only kept topologies.
  - Tested with consistent mock ML/DFT oracles: deprioritised when the gap is real. When calibration fails or relaxations fail, every topology survives.
- **Post-F:** `screening.energetic_policy.allow_exploratory: true` is an explicit opt-in to deprioritise on the *uncalibrated* ML gap. It applies only when every seed relaxed and there are no domain flags. The decision is labelled `EXPLORATORY_ML_GAP`, stays in the audit sample and can be reinstated; without the opt-in, behaviour is unchanged.
  - First real numbers on PORMAKE frameworks (orb-v3-conservative-20-omat + D3(BJ), 150 FIRE steps, not converged):
    - `pcu` −1521.79 eV per cell;
    - `bsn` −1519.90 eV (gap about 0.95 eV/f.u., still relaxing).

    About 1 s per step for 224 atoms on a laptop CPU.
- **To do:** A real-DFT calibration of orb on the MOF-5 BDC/NH₂-BDC case (needs the Milestone B inputs).

## Task 03 — Configurational Hamiltonians and encodings

### T03.1 Cluster-feature generation — done for point and pair orbits (Milestone C)
- **Where:** `clusters.py`; tokens are expanded per topology in `configuration_model.model_features`.
- **Implemented:**
  - Point orbits are parent-lattice slot sites under the space group. Pair orbits are pairs of slot sites within a centroid-distance cutoff, classified by rotation, fractional translation and lattice translation, so physically inequivalent pairs at equal distance stay separate.
  - Values are counts per formula unit with every periodic image, which makes them identical between supercells.
  - The polynomial form feeds `encoding.physical_polynomial` unchanged.
  - Candidate tokens: `point_orbits`, `pairs<=R`. The fixed fixture features remain available, but they are not transferable.
- **To do:**
  - Triplet cluster orbits (the only higher-order feature is still the fixture's `BBB_paths`).
  - Orientation variables (T01.4).
  - Graph-distance (node-mediated) shells as an alternative to centroid distance.

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
- **Post-F:** `protocols.model_selection.on_inadequate: drop_topology` (default `halt`).
  - A topology no catalogue model fits leaves the search with its selection report, and later stages see the rest. The stage is `INCOMPLETE`, so the pipeline is never reported as covering every topology.
  - It needs at least one fitted topology, otherwise the run halts as before.
  - On PORMAKE MOF-5 with the mock `pw.x`:
    - the fixed features (`unlike_distance_1/2`, `BBB_paths`) miss tolerance (CV RMSE about 25 meV/f.u.) because the mock energy depends on which linker end carries NH₂;
    - the translation-symmetric pair-orbit model (`point_orbits` + `pairs<=9.5`) fits `pcu` with CV RMSE 0.05 meV/f.u.;
    - it is rank-deficient on the 1×1×1 `bsn` cell (28 parameters), so `bsn` is dropped.
- **Needs:** The real model-selection policy and physical accuracy tolerance. The toy's 5 meV per formula unit is a test value.

### T03.3 Uncertainty and multifidelity
- **Where:** same function
- **Now:** `HamiltonianArtifact.uncertainty` stores only the development RMSE, with a note that it is not a certified bound.
- **To do:**
  - Calibrate predictive uncertainty on held-out residuals and document its domain. Bootstrap or committee spread alone is not a bound.
  - Treat ML, DFT and embedded-corrected energies as separate fidelities. A multifidelity model must fit calibrated differences between matched calculations and never concatenate fidelities as one target. `datasets.merge_snapshots` already refuses mixed fidelity.

### T03.4 Quadratization — done for cubic models (Milestone D)
- **Where:** `encoding.quadratize`, `encoding.compile_encoding(..., quadratize_higher_order=True)`, `encoding.verify_encoding`.
- **Implemented:**
  - Encoding policy `higher_order: "quadratize"` (fixture `qubo.higher_order`, physical `protocols.encoding.higher_order`). The default `retain_HUBO_or_explicitly_reject` still rejects a cubic QUBO.
  - Rosenberg substitution: one ancilla per distinct substituted pair, penalty M·(x_i x_j − 2 x_i y − 2 x_j y + 3y). The stated bound M = 2·Σ|c| over the terms that use the ancilla is strictly above Σ|c|, so a wrong ancilla always costs more than it can gain.
  - `EncodingArtifact.ancillas` records index, pair, M and the rule. `complete_assignment` decodes ancillas; the decode rule drops them.
  - `verify_encoding` minimizes over ancillas exactly (they couple only to decision variables, so the minimum is separable) on all 2ⁿ decision strings.
- **Result:** fixture `higher_order` gives ring12 11 ancillas and prism6 23. The identity is exact on all 4096 strings, the minimum is feasible, and pipeline results are unchanged.
- **Still open:** degree > 3 (raises `UNSUPPORTED_ENCODING_FORMAT`).

### T03.5 Multicomponent encodings and penalty bounds — done at the encoding layer (Milestone D)
- **Where:** `encoding.count_constraints`, `encoding.single_flip_bound`, `encoding.compile_one_hot` / `verify_one_hot` / `one_hot_states`.
- **Implemented:**
  - One count constraint per substituted sublattice (`TopologySpec.sublattice_counts`, otherwise one global count). Exact enumeration visits only states meeting every count; branch and bound and the QAOA exchange mixer assume one global count and refuse declared sublattices (`UNSUPPORTED_SUBLATTICE_CONSTRAINTS`). No loader populates the field yet.
  - Auditable penalty bound, for cells too large to enumerate: λ > L = max_i Σ_{terms ∋ i}|c| guarantees a feasible penalised minimum, since a violation d costs λd² ≥ λd > dL. Every `EncodingArtifact.penalty_bound` records L and whether λ is certified. The exhaustive check states when the bound alone was insufficient.
  - One-hot QUBO for more than two species per slot: one-per-slot plus per-species count constraints, with an exhaustive certification for ≤ 20 variables.
- **Still open:** multicomponent decorations, features and enumeration across the pipeline (intake is still binary; [intake.py:292](src/mofpipe/intake.py#L292)); sublattice counts from the project spec, in branch and bound and in the mixer; a one-hot penalty bound (only exhaustive certification exists).

## Task 04 — Arrangement solvers

### T04.1 Exact solver for large cells — done (Milestone D): branch and bound
- **Where:** `optimization.branch_and_bound`; optional cross-check through `optimizer.cross_check_solvers` (fixture) or `protocols.classical_solvers: ["exact_enumeration", "branch_and_bound"]` (physical).
- **Implemented:**
  - Depth-first branch and bound over the fixed-count physical polynomial of any degree.
  - Exact integer arithmetic: coefficients are scaled by their common denominator, so the rounding error is zero.
  - Valid node bound: fixed terms are exact; each open negative term is split over its free variables, and the r smallest shares are taken.
  - A greedy feasible incumbent is computed first, so a node or time limit always returns a feasible candidate with `UNCERTIFIED_NODE_LIMIT`/`UNCERTIFIED_TIME_LIMIT`, a lower bound and the gap.
  - The returned minimizer is re-evaluated with the Hamiltonian; a disagreement with enumeration halts the stage (`SOLVER_DISAGREEMENT`).
- **Still open:** `Job` return for long solves; LP/SDP bounds for cells much larger than the fixtures; candidate pools for cells too large to enumerate (dedup still uses the enumeration table).

### T04.2 QAOA simulation — done (Milestone D)
- **Where:** `qaoa.py`; enabled by `optimizer.quantum_enabled` + `optimizer.quantum_protocol` (fixture) or `protocols.quantum_solvers` (physical). It runs in the optimization stage as a benchmark on the frozen Hamiltonian and never changes the candidate pool.
- **Implemented:**
  - Exact statevector simulation inside the feasible subspace. The XY exchange mixer is a product of rotations over `pairs_d1` edges, generalised to species exchanges for one-hot slots. The initial state is uniform over the feasible states (a Dicke state for binary slots).
  - The phase separator is the encoding's diagonal. Every sample is decoded and re-evaluated with the physical Hamiltonian. A mismatch raises `ENCODING_DECODE_MISMATCH`, which catches bit-order and ancilla errors.
  - `verify_mixer_preserves_constraints` checks the full qubit register (binary and one-hot); the X mixer fails it. Mixer-graph reachability is checked too.
  - Records bit order, qubits (decision variables only; quadratization ancillas are recorded but not in the circuit), depth and gates per layer, shots, seeded Nelder–Mead calls, post-selection rate and wall times (solver-only and end-to-end).
  - The protocol needs positive integer `layers`, `shots`, `restarts`, `max_evaluations` and a non-empty integer `seeds` list (`SCIENTIFIC_SPEC_MISSING`); a `kind` other than `qaoa_simulation` is `UNSUPPORTED_QUANTUM_BACKEND`.
  - Benchmark across seeds against exact enumeration and simulated annealing: best and median sample, success probability. Labelled "not evidence of quantum speedup".
- **Result:** on the fixture (p = 2, 200 shots) every seed's best sample is the exact minimum, with median success probability 0.11–0.23 against 0.002 for uniform sampling.
- **Still open:** compiled Dicke-state preparation and a circuit-level (ancilla-qubit) simulation of quadratized QUBOs; noise models.

### T04.3 QPU hardware adapter
- **Where:** `qpu.hardware_benchmark`, `qpu_worker.py`; `optimizer.quantum_protocol.kind = qpu_fake_device`
- **Now:** Angles are optimised on the accepted statevector simulator, then the scheduler submits one sampling job per seed to a Qiskit fake IBM device (e.g. `fake_brisbane`), with no optimiser iterations on the device.
  - The circuit is: exact Dicke-state preparation (Bärtschi–Eidenbenz), the physical-polynomial phase separator (1-, 2- and 3-body phases) and the XX+YY exchange mixer.
  - Its noiseless probabilities match the simulator to 1e-6 or the stage fails.
  - Samples are decoded, post-selected and re-evaluated, then reported against a uniform-random baseline.
  - Shots count against `budgets.max_qpu_shots`. Jobs are idempotent.
  - Fixture result on FakeBrisbane: about 850–1,050 ECR gates; about 22% of shots keep the composition.
- **To do:** Real IBM devices through qiskit-ibm-runtime (account credentials in backend services only); error mitigation; shallower mixers.

## Task 05 — Deduplication and acquisition

### T05.1 Geometry-level deduplication — done (Milestone C)
- **Where:** `geometry_match.py`; used in `orchestrator._geometry_classes`.
- **Implemented:** relaxed structures are equivalent when all of these hold:
  - same inventory and magnetic state;
  - lattices within 1 %;
  - one parent space-group operation combined with a supercell translation maps every atom onto a same-element atom within `protocols.validation.geometry_equivalence_tolerance_A`.

  A bond-count fingerprint pre-filters candidates. Energies are never used. The classes define the CV groups and are reported for the evaluated pool. Without a declared tolerance, only identical relaxed geometries are grouped (the Milestone B rule).
- **To do:** canonicalisation under non-diagonal supercell translations (T01.5).

### T05.2 Evaluation-batch acquisition policy — done for training acquisition (Milestone C)
- **Where:** `acquisition.select_evaluation_batch`; called by the physical refinement stage.
- **Implemented:**
  - A declared policy combines low predicted energy, committee spread of the grouped-CV fold models (exploration only, not a bound) and feature-space diversity, with a seeded audit share from the deprioritised half.
  - Every topology gets a minimum quota. The rest goes by need (CV error, failing model, few labels), so an uncertain, under-sampled topology receives candidates.
  - Only unlabelled, never-attempted orbits outside the frozen holdout are eligible, so nothing is submitted twice.
  - Each pick records its reason, score components and the model and selection report it used.
- **To do:**
  - The finalist pool (`candidates.select_pool`) is still the surrogate top-k plus ties, with no evaluation budget across topologies.
  - Larger-supercell candidates enter only through the transfer check, not through acquisition.

## Task 06 — Periodic DFT

### T06.1 Periodic DFT backend — done (B; C adds the PySCF GPW engine); licensed codes, k-points and HPC open
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
- **Milestone C: PySCF engine** (`dft.PySCFAdapter`, `pyscf_worker.py`):
  - **Method.** Gaussian and plane waves with GTH basis sets and pseudopotentials, which may be mixed per element. `multigrid` (Gamma-point GPW) or `fftdf` (any k-mesh); DFT-D3/D4 from pyscf-dispersion; ASE relaxations; smearing conventions (`internal_energy`, `free_energy`, `zero_temperature_extrapolated`); CPU or GPU (gpu4pyscf).
  - **Fidelity.** The engine identity (versions plus basis and pseudopotential data hashes) is part of the protocol hash.
  - **Guards.** A stress from the FFTDF integrator is refused when it disagrees with the multigrid energy. Multigrid with k-points, DFT+U, hybrids and meta-GGAs are refused before any job.
  - **Verification.** Finite-difference checks of forces and stress, and k-point folding against a replicated cell.
- **Milestone C: result-check fixes** (from the Milestone B review):
  - Force and stress gates apply only to what the protocol relaxed; `relaxation.calculation: "scf"` declares rigid geometries.
  - The stress check uses the full tensor.
  - QE's `forc_conv_thr` is set consistent with the norm check.
  - A `vc-relax` output without a cell block is a parse error.
- **To do:**
  - Adapters for licensed codes behind the same interface.
  - k-point convergence of energy differences, and Gaussian-basis convergence for PySCF (only the plane-wave cutoff is studied).
  - In PySCF: DFT+U (k-point DFT+U exists without periodic stress or multigrid gradients), hybrid functionals (no periodic stress), user-supplied basis and pseudopotential files, per-site (antiferromagnetic) initial moments.
  - Remote/HPC submission through AiiDA CalcJobs (T10.1). Jobs currently run locally and synchronously.
- **Needs:** The DFT protocol, magnetic-state policy, pseudopotentials or basis sets, HPC resources and budgets.

### T06.3 GPU path validation
- **Where:** `pyscf_worker._scf_object` and `Evaluator` (`device: gpu`).
- **Now:**
  - Written against the gpu4pyscf 1.8.1 API: `to_gpu`, `multigrid_numint`, `Gradients().get_stress()` and GPU smearing.
  - The probe refuses `device: gpu` with `BACKEND_UNAVAILABLE` when gpu4pyscf/cupy or a CUDA device is missing.
  - It has never been run, because this machine has no CUDA device.
- **To do:** Run `tests/test_pyscf_backend.py` with `device="gpu"` on a CUDA host. Compare energies, forces and stress with the CPU path to about 1e-6 Ha.

### T06.2 Real evaluation path in the orchestrator — done (Milestone B)
- **Where:** `Pipeline._physical_labels`, `_physical_evaluation`, `_physical_ranking`, `_physical_validation` (selected automatically for physical projects).
- **Implemented:**
  - Labels: `build_framework` → DFT relaxation and single point for every labelled orbit representative.
  - Pool evaluation: the same chain for the candidate pool. Already-labelled candidates come from the job cache.
  - `BuildFailure`, failed and transformed calculations are recorded and become missing evaluations; the ranking is then `INCOMPLETE`.
  - The ranking is DFT-only, at base level, with tolerance ties. The corrected ranking is `DISABLED_BY_CONFIG`.
- **Milestone E:** the `embedding` stage (T07.3) corrects the evaluated pool; the ranking stage adds a corrected ranking only when the scheme is validated.

## Task 07 — Embedding and electronic solvers

This is a research milestone, not API wiring. The embedding formalism must be specified by the team before coding.

### T07.1 Embedding scheme and active-space export — done for bare systems (Milestone D)
- **Where:** `active_space.prepare_active_space`, `read_fcidump`, `problem_from_artifact`.
- **Implemented:**
  - `prepare_active_space(system, protocol)` takes one specific geometry with its calculation ID and runs the declared RHF/ROHF reference.
  - It exports h1 (including the frozen core), h2 (chemists' notation), E_core, electron counts by spin, orbital ordering, the integral convention, geometry/basis/protocol hashes, and per-atom Mulliken orbital character, as FCIDUMP plus an `ActiveSpaceArtifact`.
  - Identical inputs reproduce identical integrals, FCIDUMP SHA-256 and artifact ID. The reference and integral steps run on one OpenMP thread: threaded reductions differed by ~10⁻¹⁴ Ha between runs and changed the stored integrals and artifact ID (fixed after the Milestone D review). The FCIDUMP round-trips to 10⁻¹².
  - Milestone E: node-centred capped clusters of MOF geometries (`embedding.cut_region`) can be solved through an active space (`high_level.method: active_space`) by any T07.2 solver.
- **Still open:** active spaces in an embedding potential (WF-in-DFT; `embedding` other than `none` gives `UNSUPPORTED_EMBEDDING`); a balanced metal/ligand/neighbour-node orbital selection policy (only frontier windows or explicit orbital lists exist).
- **Needs:** the team's choice of a WF-in-DFT formalism if one is wanted beyond the subtractive scheme.

### T07.2 Electronic solvers — done in simulation (Milestone D); hardware SQD open
- **Where:** `electronic_solvers.py`; benchmark runner `run_electronic_benchmark`, CLI `python -m mofpipe electronic-benchmark`.
- **Implemented:**
  - Every solver reads the same artifact and records its `hamiltonian_hash`.
  - Independent determinant-basis exact diagonalization, written here; it matches PySCF FCI (`pyscf_casci` on the stored integrals) to 10⁻⁹ Ha or better.
  - Simulated VQE: Jordan–Wigner, interleaved α/β qubit order, no tapering, (N_α, N_β) sector from the HF determinant, spin-orbital UCCSD, BFGS. It is a variational upper bound and is never called exact.
  - SQD from simulated samples, with declared bit-flip noise, configuration recovery and subspace diagonalisation. Shots, recovery count, subspace size and seeds are recorded.
  - `sqd_hardware` reports `BLOCKED`: no QPU backend is configured.
- **Consistency gate** (added after the Milestone D review): `REPRODUCED` also requires ED and CASCI to agree within 10⁻⁸ Ha and VQE/SQD not to fall below them. SQD records the result whose state it sampled; the VQE it samples from is always stored.
- **Published benchmark:** `benchmarks/h2_sto3g_szabo_ostlund.json`, minimal-basis H₂ at R = 1.4 a₀. HF is −1.11671 Ha against the published −1.1167, and full CI −1.13728 against −1.1373, within the published 4-decimal precision. ED, CASCI, VQE and SQD agree. Status `REPRODUCED`; this is reproduction of a calculation, not agreement with experiment.
- **Still open:** hardware SQD adapter ([electronic_solvers.py:237](src/mofpipe/electronic_solvers.py#L237)); benchmarks with strong correlation (open-shell Fe–MOF-74); approximate references (DMRG, SHCI) for larger spaces.

- **Post-F, the ground-state step of the OSI scheme:**
  - `protocols.embedding.high_level.solver: sqd_simulation` runs the simulated VQE on the cluster active space and samples SQD from that VQE state. The SQD energy is the level energy; the VQE energy, exact ED (≤ 8 orbitals), mean-field energy and SQD subspace size are recorded in the cluster result.
  - Unknown solver names are refused at intake. A failed reference SCF is a typed failure, not a worker crash. The cluster worker is now version 3.
  - Declared SCF numerics per level (`scf_solver: diis | newton`, `max_cycle`, `level_shift_Ha`) and for the active-space reference (`scf_solver`, `max_cycle`). They are part of the embedding fidelity.
  - Reason: on the PORMAKE MOF-5 node cluster Zn₄O(HCO₂)₆ (STO-3G, 173 AOs), DIIS did not converge, even with level shift and 200 cycles; second-order SCF converged in about 60 s.
  - Full MOF-5 VQE/SQD run: in progress when this was written.

### T07.3 Corrected-energy assembly — done for the finite-cluster subtractive scheme (Milestone E)
- **Where:** `embedding.py` (`plan_regions`, `cut_region`, `ClusterBackend`, `assemble_corrected_energy`, `validate_scheme`, `correct_pool`, `run_embedding_benchmark`), `cluster_worker.py`; the `embedding` stage between `evaluation` and `ranking`; CLI `python -m mofpipe embedding-benchmark`.
- **Scheme** (`protocols.embedding.scheme: finite_cluster_subtractive`, labelled a finite-cluster mechanical embedding, not WF-in-DFT). The frozen expression is stored in every plan, correction and validation record:
  - E_corr[c] = E_PBC^low[c] + Σ_k m_k (E_high[C_k] − E_low[C_k]).
  - Environment: the periodic DFT cell. No embedding potential couples the regions.
  - Regions: one per metal node of the relaxed cell. A region is the node atoms, every atom within `region.bond_hops` bonds, and terminal atoms bonded to those. Each bond leaving a region is capped by `capping.element` at `capping.distance_A[kept element]`. Translation-equivalent clusters (1e-4 Å) share one calculation with multiplicity m_k.
  - E_low is the periodic functional and dispersion in the cluster basis; E_high is HF, MP2, CCSD, CCSD(T), or an active space solved by a T07.2 solver. Both use the same basis, ECP, charge and spin.
- **Assembly** returns `CorrectedEnergy`, an `EmbeddingBenchmarkResult` (`correction_status=UNVALIDATED`, never applied), or a `BlockedResult`. Blocking codes:
  - `INCOMPATIBLE_GEOMETRY`: the plan was not cut from this calculation's relaxed geometry.
  - `INCOMPLETE_NODE_COVERAGE`: some node of the cell has no region, or the node count changed.
  - `MISSING_SUBTRACTION_TERM` / `MISSING_HIGH_LEVEL_TERM`; `INCOMPATIBLE_CLUSTER_PAIR` (basis, charge, spin or protocol differ).
  - `OVERLAPPING_REGIONS` (would double count), `REGION_EXCEEDS_CELL` (the region wraps around the periodic cell), `CAPPING_RULE_MISSING`.
- **Intake:** every field is required, nothing is defaulted. The low level must equal `protocols.dft` (`EMBEDDING_LOW_LEVEL_MISMATCH`). `validation.bond_hops_series` must contain `region.bond_hops` and a larger value. `validation.reference_benchmarks` must name existing files (`[]` is allowed and leaves every correction UNVALIDATED). Cluster jobs count against `budgets.max_active_space_jobs`.
- **Cluster jobs** use `submit → job_status → collect` with an idempotency key from the protocol hash, role and canonical cluster. The engine is the PySCF worker or an external command, and the worker or engine script hash is part of the embedding fidelity.
- **Validation** (`EmbeddingValidationReport`; VALIDATED only if all four pass):
  - high = low: the low level resubmitted as the high level gives |Δ| ≤ 10⁻⁸ Ha.
  - balanced regions: identical region inventory per formula unit in every candidate.
  - boundary convergence: the relative correction between the two lowest DFT candidates, over `bond_hops_series`, within `target_relative_energy_tolerance`.
  - reference benchmarks: every declared benchmark is `REPRODUCED`.
- **Ranking feedback:** when VALIDATED, corrected energies (fidelity `DFT:…+EMB:…`) give a corrected pooled ranking next to the DFT one, with `ranking_changed_by_correction`. The reference-phase competition stays at the DFT level. A blocked or unvalidated embedding never stops the DFT-only ranking.
- **Reference benchmark** (`benchmarks/embedding_keto_enol_hf_ccsd.json`): propanal vs (E)-prop-1-en-1-ol, CCSD/6-31G. The region is the C–C(=O)/C=C–OH fragment and the methyl group is capped.
  - The reference is an independent full-molecule CCSD on the same idealised geometries, not a published value.
  - Embedded relative-energy error: 0.25 mHa. HF alone is off by 2.36 mHa, against a 1.6 mHa (1 kcal/mol) tolerance, so the benchmark discriminates.
  - Exact limit: a region covering the whole molecule reproduces the full CCSD energy to 10⁻¹⁴ Ha.
  - HF is the low level because PBE already matches CCSD for this pair within 0.2 mHa.
- **Toy result (SYNTHETIC, mock engine):**
  - With `bond_hops: 1`, the two-hop region of the one-cell-thick toy wraps around c. The boundary study is then `NOT_ASSESSABLE` and the run stays DFT-only.
  - With `bond_hops: 0` and the reference benchmark, the scheme is VALIDATED. Every node cluster is identical, so the correction per formula unit is a constant, and the corrected order, ties and winner equal the DFT ones (the "unchanged correction" acceptance test).
- **Still open:**
  - The published Fe–MOF-74 embedding workflow. It needs the paper's data and an HF-embedding mode; HF-embedded correlation cannot be added to DFT.
  - WF-in-DFT (projection-based) embedding.
  - Overlapping regions through inclusion–exclusion (now blocked).
  - Alternative cluster spin states, cluster basis-set convergence, corrected reference phases.
- **Needs:** the real high level, cluster charge and spin, capping distances, region size, and a reference benchmark for the chosen chemistry.

## Task 08 — Adaptive refinement

### T08.1 Active-learning acquisition policy — done for physical projects (Milestone C)
- **Where:** `orchestrator._physical_refinement`, `acquisition.py`.
- **Implemented:** rounds of the declared acquisition policy (T05.2) with DFT labels, never from the holdout. Construction or DFT failures and transformations are recorded and never become labels. The CV groups are updated from relaxed-geometry equivalence.
- **Now (fixture):** feature-space novelty, as before.
- **To do:**
  - Revisit uncertain topology exclusions once a screening policy exists (T02.3).
  - A controlled enumerable fixture with a deliberately biased initial set that recovers a missed basin (the brief's acceptance test) still has to be built.

### T08.2 Stopping criteria — done for physical projects (Milestone C)
- **Where:** `refinement.assess_refinement`, `refinement.leading_sets`.
- **Implemented:**
  - `CONVERGED_WITHIN_SCOPE` when every topology passes grouped CV and the leading set is unchanged for `tolerances.consecutive_stable_rounds` rounds. The leading set is the orbits within tolerance of the k-th lowest prediction, per topology and pooled, so reordering inside the tolerance band does not count as instability.
  - `MODEL_INADEQUATE` when every candidate is well determined but fails.
  - `INCONCLUSIVE` when a topology needs labels and none are left.
  - `BUDGET_EXHAUSTED` at `protocols.stopping.max_rounds` or the DFT budget, never reported as convergence.
  - An interrupted refinement resumes from the job cache.
- **To do:**
  - Account for solver gaps once a non-exhaustive solver exists (T04.1).
  - Explicit model-family expansion on demand.
  - A matched embedded-correction surrogate (T07).
  - Coverage criteria beyond the leading set.

## Task 09 — Validation, competition and report

### T09.1 Structural validation tool wrappers — done for in-house checks + optional MOFChecker/Zeo++ (Milestone C)
- **Where:** `validation.validate_structure`; run on every ranked finalist by `orchestrator._physical_validation`.
- **Implemented:**
  - Checks: post-relaxation slot/node mapping, residual force and stress (or `NOT_APPLICABLE` for rigid geometries), minimum interatomic distance, metal coordination numbers against the parent, cell distortion.
  - MOFChecker (`mofchecker` package) and Zeo++ (`network` executable) run when installed, with their raw output; otherwise `SKIPPED_TOOL_UNAVAILABLE`. Neither is a dependency.
- **To do:**
  - CrystalNets/MOFid net identity. This would also verify that candidate CIFs are distinct nets; `declared_net` is still unverified.
  - Targeted phonon/elastic stability for finalists.
  - PXRD comparison, and NMR only with a validated protocol.
- **Done when (remaining):** simulated observations are labelled simulated, and every reported number traces to a stored artifact (already true for the implemented checks).

### T09.2 Reference-phase competition — done for pure-endpoint references (Milestone C)
- **Where:** `references.py`; `orchestrator._reference_phases`.
- **Implemented:**
  - `protocols.validation.reference_phases: "pure_endpoints"` evaluates every topology with one linker species on every slot, using the same DFT protocol and normalising per cell.
  - The mixture balance (min Σcᵢ Eᵢ subject to the inventory constraint, c ≥ 0) is solved exactly over basic feasible solutions, and mixing energies are reported per finalist.
  - The competition is `INCOMPLETE` unless `reference_set_complete: true` and every reference was evaluated.
- **To do:**
  - User-declared reference phases (other CIFs, e.g. dense polymorphs or linker/oxide phases), with their own formula units.
  - Mixing energies for the whole enumerated space via the surrogate.

### T09.3 Benchmarks, ablations and literature review
- **Where:** `report.build_report`; the literature check in `intake.validate_project`
- **Now:**
  - The deterministic report lists static literature entries with `live_review: NOT_RUN`.
  - Agent runs add the A67 review: claims are mapped to cited passages and dataset records, with code-computed metrics.
  - `mofpipe agents-eval` runs the deterministic-vs-agent ablation.
- **To do:**
  - Benchmark sections that reproduce the Kang (arrangements), Cai (topology competition) and Fe–MOF-74 (electronic workflow) studies, with exact paper/data versions. Keep reproduction of a paper separate from agreement with experiment.
  - The remaining ablations: fixed topology vs joint search, heuristic vs fitted costs, no vs validated embedding, classical vs quantum optimizer.

## Task 10 — Agent architecture

### T10.1 AiiDA workflow engine and concurrency
- **Where:** `aiida_engine.py` (`mofpipe run --engine aiida`); `store.RunStore` (optimistic commits)
- **Now:**
  - `MofPipeWorkChain` steps through the stages with the same interfaces (`Pipeline.run(stop_after=...)`). Each committed stage is recorded as a provenance `Dict` (status, fingerprint, state version, artifact sha256), and it resumes at stage boundaries.
  - Runs in-process on a project-local `verdi presto` profile (`.aiida`, SQLite, no RabbitMQ). Reproduces the local engine's report exactly.
  - Every state write is locked and checks the on-disk revision; a second writer gets `STATE_VERSION_CONFLICT`.
- **To do:** CalcJobs for DFT/ML/QPU on HPC (aiida-quantumespresso, a PySCF code plugin), and concurrent topologies/configurations/seeds with a daemon and broker. Cancel within a calculation.

### T10.2 Runtime agent layer
- **Where:** `src/mofpipe/agents/` (packets, contracts, client, validate, literature, datasets, runner, evaluate); `gateway.review_proposal`; `intake.runtime_agent_errors`, `intake.select_topologies`
- **Now:** Implemented per `MOF_AI_Agent_Implementation_Guide-2`:
  - A1 selects topologies from a design brief with explicit, checked exclusions. A2 proposes the Hamiltonian form. A67 runs the final assessment against checks, papers (Valency MCP, local) and datasets (local CoRE MOF 2024 / QMOF copies with code-computed metrics).
  - Strict contracts, the request budget (3, plus 1 optional), cache, ledger and decision records, LangGraph orchestration.
  - `agents-eval` compares agent-assisted with deterministic runs (Milestone F).
  - The A2 model-family expansion exists but is off by default.
- **To do:**
  - Several construction options per framework (alternative cells/orientations). A1's catalogue can now be generated by PORMAKE (`topology_search` in its packet), but A1 still cannot request nets outside it.
  - **A4 optimisation role (post-F, in progress):**
    - code: `contracts.A4_SCHEMA`, `packets.build_a4_packet` / `candidate_quality`, `validate.validate_optimisation`, `runner.node_a4`;
    - budget: `budget.max_optimisation_rounds`; `max_llm_requests` = 3 without A4, otherwise 4 + 2 × rounds;
    - Ledger slots `round_k` and the gateway role A4;
    - scripted stand-ins and the scheme trace in `agents/scheme.py`.

    Validated once on the fixture with scripted answers (MODEL_INADEQUATE → revised to the triplet model → accepted, 6 requests).

    The existing agent tests now pin `max_optimisation_rounds: 0`. Still to do:
    - A4 tests;
    - `--scripted` CLI wiring;
    - the scheme trace in the run summary;
    - a live LLM run.
  - Split the pipeline before A2 once a real screening stage exists (T02.3).
  - A live physical run, which needs the MOF-5 inputs.
  - Full-text retrieval for Valency hits (abstract-only now).

---

## Open specification decisions (no code marker)

These came out of the instruction-set review and must be settled in `instructions/` or the project spec. Code cannot resolve them.

- **Template fields missing from `project_spec.template.json`:**
  - ~~disorder resolution policy (T01.2)~~ — the loader requires `chemistry.disorder_resolution_policy`; it should be added to the template.
  - ~~topology definitions themselves~~ — physical projects derive `TopologySpec` from `source_structures[i]` (CIF + `topology_id` + checksum + source); the template's `source_structures: []` needs this entry shape documented.
  - ~~holdout/split policy (T03.2)~~ — carried in `protocols.model_selection`; should be documented in the template.
  - project `provenance` (`SYNTHETIC` | `PHYSICAL`) — required by the loader, absent from the template.
  - reference phase sets (T09.2): carried as `protocols.validation.reference_phases` / `reference_set_complete`; only `pure_endpoints` exists, and the template should document both.
  - acquisition and stopping policies (T05.2, T08): `protocols.acquisition` (`batch_size`, `weights`, `audit_fraction`, `min_per_topology`, `seed`) and `protocols.stopping.max_rounds` should be documented in the template.
  - transfer-test policy (T01.5): `protocols.validation.supercell_transfer` (`n_replicated`, `n_native`, `seed`) and per-source `supercell_matrices`.
  - DFT engine fields: the PySCF engine needs `basis`, `pseudo`, `ke_cutoff_Ha`, `integral_scheme`, `device`, `precision`, `max_memory_MB`, `conv_tol_Ha`; QE needs `pseudo_dir`, `pseudopotentials`, `ecutwfc_Ry`, `ecutrho_Ry`, `conv_thr_Ry`. The template has a single `protocols.dft` slot.
  - seed sampling and screening policies (T02.1, T02.3)
  - ranking comparison policy (tie tolerance currently reuses `target_relative_energy_tolerance`)
  - electronic solver protocols for CASCI, VQE and SQD (T07.2)
  - the embedding protocol shape (T07.3): `protocols.embedding` (scheme, engine, region, capping, cluster charge/spin, basis, low/high levels, validation) and `budgets.max_active_space_jobs` as the cluster-job budget
  - split manifest path in the fixture `project.json` (T01.1) — unchanged because the fixture is checksummed.
- **Signature ambiguities:**
  - `prepare_active_space` should take a geometry/calculation ID, not a candidate ID (T07.1).
  - ~~`assemble_corrected_energy` needs a third outcome~~ — it returns `CorrectedEnergy`, an `UNVALIDATED` `EmbeddingBenchmarkResult` or a `BlockedResult` (T07.3).
  - `select_evaluation_batch` (05) and `plan_next_batch` (08) both produce `AcquisitionBatch`; decide which owns it.
  - `advance_workflow` must stay controller-only. `gateway.CONTROLLER_ONLY` already enforces this.
- **Physical tolerances:** energy tie tolerance, force/stress tolerances, low-energy validation criterion and stable-round count are `null` in the template and must be chosen per study.

Already resolved in code, and worth proposing back into `00_SHARED_CONTRACT.md`:
- one status vocabulary (`contract.StageStatus`, `JobState`, `StopReason`);
- the glue producers `define_decoration_space`, `ingest_labels` / `merge_snapshots` (DatasetSnapshot), `freeze_split` (ValidationSplit) and candidate registration via `CandidateSet`.

## Suggested order

Following the milestones in `instructions/10_AGENT_ARCHITECTURE.md`. Milestones A–E are done in software:
- A: the synthetic end-to-end fixture;
- B: T01.1–T01.3, T06.1, T06.2, T03.2;
- C: T01.5, T03.1, T05.1, T05.2, T08.1, T08.2, T09.1 (in-house + optional tools), T09.2 (pure endpoints), plus the PySCF engine.

They are verified on SYNTHETIC toys. Running MOF-5 BDC/NH₂-BDC needs: the verified CIF, a substitution rule or 3D blocks, a converged DFT protocol (basis/cutoff/k-points for PySCF), tolerances and budgets. T02.1–T02.3 (ML screening) were done in Milestone F (orb-v3).

After F, PORMAKE now supplies the frameworks and the amino geometry. Real MOF-5 periodic energies remain the gap: a 200-atom cell is beyond laptop DFT, so `tests/mof5_scheme.py` uses the mock `pw.x` and is labelled SYNTHETIC.

1. **Before real runs:** T06.3 (GPU validation on a CUDA host), Gaussian-basis and k-point convergence studies, non-diagonal supercells if the MOF-5 primitive/conventional relation needs them.
2. **Milestone D — done:** T03.4 (cubic quadratization), T03.5 (sublattice constraints, penalty bound, one-hot at the encoding layer), T04.1 (branch and bound), T04.2 (QAOA simulation benchmark), T07.1–T07.2 (active-space export and ED/CASCI/VQE/SQD, H₂/STO-3G published benchmark reproduced).
3. **Milestone E — done:** T07.3 for the finite-cluster subtractive scheme (assembly, validation, corrected ranking), checked against an independent full-CCSD benchmark. Still open: the published Fe–MOF-74 reference and WF-in-DFT.
4. **Milestone F — done in software:** T10.1 (AiiDA engine, optimistic commits), T10.2 (agents), T04.3 (fake-device QPU), T02.1–T02.3 (ML screening), T09.3 (A67 literature/dataset review, agent ablation). Open: HPC CalcJobs, real QPU devices, paper reproductions.
5. **After F, OSI scheme (OSI_scheme.pdf), in progress:**
   - **Done:** PORMAKE topology generation (T01.3); exploratory ML prioritisation opt-in (T02.3); `model_selection.on_inadequate: drop_topology` (T03.2); SQD high level and declared SCF numerics for the cluster embedding (T07); the A4 optimisation loop (T10.2, fixture-validated).
   - **Next:**
     - tests for generation and A4;
     - finish the orb screen and the full MOF-5 scheme run (scripted roles), then a live LLM run;
     - ToBaCCo.
