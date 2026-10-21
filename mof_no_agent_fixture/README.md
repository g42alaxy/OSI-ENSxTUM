# MOF pipeline test input — no runtime AI agents

Start with `project.json`. This pack supplies two small, fully synthetic occupation problems with known answers. It tests the software path from a substitution graph and composition through Hamiltonian fitting, optimization, symmetry deduplication and cross-graph ranking. Every executable uses only the Python standard library. No LLM API key, network connection, DFT package or quantum service is required.

**These graphs are not atomic MOFs or crystallographic nets.** A/B are abstract linker labels; the metal block is a placeholder. The data are appropriate for the first deterministic integration test. They do not validate synthesis, stability, an ML potential, DFT accuracy or a quantum advantage. The optional `physical_case/` directory supplies a later chemistry recipe, not a ready-to-run CIF dataset.

## Quick start

Use Python 3.10 or newer. From the extracted `mof_no_agent_fixture` directory:

```bash
python3 tools/verify_fixture.py
python3 tools/reference_run.py
python3 tools/reference_run.py --case higher_order
python3 tools/reference_run.py --correction constant
python3 tools/reference_run.py --correction topology_dependent
```

The first command should print `"status": "PASS"`. Reports are saved in `reference_outputs/`. This verifies the fixture and its reference implementation; it does not execute your pipeline. Hand `AGENT_INSTRUCTIONS.md` and the entire pack to the coding agent that will integrate it into your repository.

The reference runner is deliberately compact. It uses the fixed policies described in `project.json`; it is not a general manifest loader or a replacement for your pipeline.

## Input specification

| Input | Value |
|---|---|
| Candidate graph 1 | `ring12`: cycle with 12 occupation sites |
| Candidate graph 2 | `prism6`: hexagonal prism with 12 occupation sites |
| Species | A = 0; B = 1 |
| Composition | Exactly 6 A and 6 B in each graph |
| Metal block | Fixed placeholder; no metal chemistry is simulated |
| Variable meaning | Site occupation, not atom identity or Cartesian coordinates |
| Bit order | Leftmost character is site 0 |
| Solver | Exact enumeration over fixed-composition states |
| Symmetry | Explicit site permutations; no extra A/B interchange |
| Energy convention | eV per one abstract 12-slot formula unit |
| Runtime AI agents | Disabled |
| Finite temperature | Disabled |

The composition gives `C(12,6) = 924` arrangements per graph. Each graph has 24 supplied symmetry operations. The occupation states form 50 symmetry orbits for `ring12` and 55 for `prism6`.

For example, `010101010101` means alternating A/B around the 12-cycle. Its core-fixture energy is −0.120 eV. For the prism, `010101101010` gives alternating occupation on both rings with opposite occupations on the rungs; its energy is −0.160 eV.

## Files your pipeline should read

| File | Purpose |
|---|---|
| `project.json` | Fixed, agent-free configuration and stage switches |
| `data/topologies.json` | Sites, interaction pairs/triplets and explicit symmetry permutations |
| `data/building_blocks.json` | Abstract A/B labels and their binary encoding |
| `data/candidates_unlabelled.jsonl` | All 1,848 legal configurations without energies |
| `data/model_candidates.json` | Fixed candidate feature sets |
| `data/core_pairwise_training.jsonl` | 60 labelled training arrangements across both graphs |
| `data/core_pairwise_development.jsonl` | 20 labelled model-selection arrangements |
| `data/higher_order_training.jsonl` | Alternative training set requiring a triplet feature |
| `data/higher_order_development.jsonl` | Corresponding development set |
| `data/split_manifest.json` | Explicit symmetry-orbit partition |
| `data/literature_catalog.json` | Static references; no automated literature review |

Each graph contributes 30 training orbits and 10 development orbits, with one representative per orbit. The remaining 10/15 orbits are withheld in `harness/`. No symmetry orbit crosses a split. The intentionally tiny fixture uses a fixed development split; use grouped cross-validation and a separate final test set for a real scientific dataset.

**Do not expose `harness/`, `reference_outputs/`, or the oracle source to the learner or model selector.** They are available to the test harness only. The source for the synthetic evaluator necessarily contains the truth; access it through the evaluator boundary if the pipeline requests new labels.

## Hamiltonian-selection test

The fixed candidate models are:

| Model | Features | Parameters | Polynomial degree |
|---|---|---:|---:|
| M0 | Constant + unlike nearest-neighbour pairs | 2 | 2 |
| M1 | M0 + unlike graph-distance-two pairs | 3 | 2 |
| M2 | M1 + all-B length-two-path triples | 4 | 3 |

Fit each model to training energies and measure its development RMSE. Select the fewest parameters among models with RMSE ≤ `1e-10` eV, breaking a remaining tie by model ID. This near-zero tolerance is appropriate only because the labels are exact and noiseless. Do not reuse it as a universal physical accuracy threshold.

The expected choice is M1 for `core_pairwise` and M2 for `higher_order`. Both should predict their untouched holdout energies to the stated tolerance. This exercises deterministic Hamiltonian selection without an LLM.

M2 is cubic. It must remain a higher-order binary model for direct enumeration, or undergo an explicitly validated quadratization before use with a QUBO-only solver. Silently dropping its triplets is an error.

## Ground truth and expected results

The synthetic core energy is:

`E = offset − 0.020 × unlike_distance_1 + 0.006 × unlike_distance_2`

The offsets are 0.120 eV for the ring and 0.200 eV for the prism. The higher-order case additionally includes `+0.015 × BBB_paths`. Both cases have the same minimum energies and occupations; they require different model expressiveness.

| Quantity | ring12 | prism6 |
|---|---:|---:|
| Legal arrangements | 924 | 924 |
| Unique symmetry orbits | 50 | 55 |
| Nearest-neighbour pairs | 12 | 18 |
| Distance-two pairs | 12 | 24 |
| Path triples | 12 | 36 |
| Base minimum, eV | −0.120 | −0.160 |
| Raw minimizing arrangements | 2 | 2 |
| Symmetry-distinct minimizing arrangements | 1 | 1 |

The base winner is `prism6`. The constant term must be retained when comparing graphs; dropping per-graph offsets can invalidate cross-topology comparisons in a real calculation.

For QUBO compilation, the convention is a separate constant, linear terms and **upper-triangle off-diagonal terms counted once**. The reference adds `1.0 eV × (sum(bits) − 6)^2`. The verifier checks the energy identity on all 4,096 bit strings for each graph and confirms that the penalized minimum is feasible. The selected-state energy used for ranking must exclude any nonzero constraint penalty, and infeasible states must be rejected.

## Correction and pooling tests

The reference retains the five lowest-energy unique orbits per graph, including all ties at the cutoff, and pools them. It also computes an exhaustive corrected reference so the selected pool can be checked.

| Mode | Ring minimum, eV | Prism minimum, eV | Winner |
|---|---:|---:|---|
| `none` | −0.120 | −0.160 | prism6 |
| `constant`: add 0.050 eV everywhere | −0.070 | −0.110 | prism6 |
| `topology_dependent` | −0.120 | −0.100 | ring12 |

The last mode adds 0.060 eV to prism configurations and 0.001 eV per all-B path triple to either graph. **These are artificial corrections for testing bookkeeping and ranking.** They are not DFT embedding, correlation energies or quantum results. Base and corrected energies have separate fields and fidelity labels. A constant correction should leave rankings unchanged; the other correction deliberately changes the winning graph.

For a mock evaluation boundary, provide JSONL requests containing `topology_id` and `bits`:

```bash
python3 tools/mock_evaluator.py --input data/core_pairwise_development.jsonl --out reference_outputs/mock_labels.jsonl --correction topology_dependent
```

The evaluator returns explicitly synthetic base and corrected labels. It is a test backend; the learner must not import its generating formula.

## Coverage and limits

Covered by the included verifier: dataset integrity, legal counts, orbit splitting, graph symmetry, fixed-count swap reachability, model selection, holdout prediction, pairwise QUBO compilation, exact minima, pooled ranking, correction routing, normalization arithmetic and malformed-input rejection.

Not covered: atomic construction, geometric fit, ML-potential transferability, periodic DFT, electronic embedding, actual QAOA/VQE/SQD execution, iterative active learning, structural stability, PXRD/NMR agreement, or live literature review. Swap reachability is a graph check, not a quantum-circuit simulation.

`physical_case/recipe.json` specifies a later MOF-5-family BDC/amino-BDC test. It requires verified atomic input files and calculated energies before it can run. Keep synthetic and physical results in separate projects.

## Reproducibility

`tools/generate_fixture.py` regenerates the numeric datasets using a fixed hash-based symmetry-orbit partition. It is a fixture-author tool, not part of model fitting. The exact-arithmetic reference uses Python fractions; production code may use floating-point regression and satisfy the stated tolerance. `reference_outputs/verification.json` records the bundled checks, while `SHA256SUMS` records delivered-file hashes.
