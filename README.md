Quantum-Enhanced completed pipeline for automatic MTV-MOF structure modeling

## Deterministic pipeline (`src/mofpipe`)

Implementation of the specifications in `instructions/` with runtime LLM agents disabled. Python ≥ 3.10. The pipeline core is standard-library Python; the periodic DFT engine uses PySCF, pyscf-dispersion and ASE (`pyproject.toml`), with an optional GPU switch through gpu4pyscf (`pip install -e .[gpu]`). It runs on:

- the synthetic fixture in `mof_no_agent_fixture/` (abstract substitution graphs, not atomistic MOFs; Milestone A);
- physical projects: a ProjectSpec v0.1 file with parent CIFs, building blocks and a DFT protocol. Milestone B covers a single topology from CIF through DFT labels to a validated model. Milestone C adds several candidate topologies on one common formula unit, transferable cluster features, supercell transfer checks, acquisition and refinement rounds, relaxed-geometry deduplication, reference-phase competition, structural validation and the full report.

DFT engines (`protocols.dft.engine`):
- `pyscf` (primary): Gaussian-and-plane-waves DFT with GTH basis sets and pseudopotentials. It offers multigrid GPW at the Gamma point or FFT GPW with any k-mesh, DFT-D3/D4, ASE relaxations (`relax`, `vc-relax`), and `device: cpu | gpu`.
- `quantum_espresso`: a `pw.x` adapter, exercised in tests only through a mock `pw.x`.

All physical results so far are on **SYNTHETIC** toy frameworks. Do not read them as MOF predictions.

```bash
# fixture (Milestone A): one run, run + harness acceptance, all 6 case x correction combinations
PYTHONPATH=src python3 -m mofpipe run    --project mof_no_agent_fixture/project.json --case higher_order --correction topology_dependent
PYTHONPATH=src python3 -m mofpipe accept --project mof_no_agent_fixture/project.json
PYTHONPATH=src python3 -m mofpipe matrix --project mof_no_agent_fixture/project.json --out runs
# Task 07 benchmark: active-space export + ED/CASCI/VQE/SQD vs the published minimal-basis H2 values
PYTHONPATH=src python3 -m mofpipe electronic-benchmark --spec benchmarks/h2_sto3g_szabo_ostlund.json --run-dir runs/h2_benchmark
# Task 07 embedding: finite-cluster subtractive scheme vs an independent full CCSD calculation (keto/enol)
PYTHONPATH=src python3 -m mofpipe embedding-benchmark --spec benchmarks/embedding_keto_enol_hf_ccsd.json --run-dir runs/embedding_benchmark
# physical plumbing toys with the mock pw.x: Milestone B (one topology) and Milestone C (two topologies, transfer cell, refinement)
python3 -c "import sys; sys.path[:0] = ['tests']; import toy_framework as t; print(t.write_toy_project('runs/toy_b'), t.write_toy_milestone_c_project('runs/toy_c'))"
PYTHONPATH=src python3 -m mofpipe run    --project runs/toy_c/project.json
# real PySCF GPW calculations on the SYNTHETIC aluminium-formate toy: at the shipped 25 Ha cutoff the convergence
# gate stops it (DFT_SETTINGS_NOT_CONVERGED, ~10 min); raise ke_cutoff_Ha and the series (~100 Ha) for a full run
.venv/bin/python tests/pyscf_toy.py runs/toy_pyscf && PYTHONPATH=src .venv/bin/python -m mofpipe run --project runs/toy_pyscf/project.json
# the physical_case recipe and the untouched template load as physical projects and stop at intake with their missing fields
PYTHONPATH=src python3 -m mofpipe run    --project mof_no_agent_fixture/physical_case/recipe.json
# tests: everything except the real-DFT checks; then the PySCF checks (minutes); the PySCF end-to-end run is opt-in
.venv/bin/python -m pytest -m "not pyscf"
.venv/bin/python -m pytest -m pyscf                 # MOFPIPE_SLOW_TESTS=1 adds the end-to-end formate run
```

Each run directory holds `state.json` (authoritative workflow state), `events.jsonl`, content-addressed `artifacts/`, evaluator and DFT `jobs/` (inputs, raw outputs, PySCF logs), `report.json` and `report.md`. Re-running with identical inputs reuses committed stages and never resubmits a successful job.

| Stage | Module | Instruction task |
|---|---|---|
| intake, project validation | `intake.py`, `project_spec.py` | 01 |
| CIF parsing, slot lattice, framework builder, working and transfer cells | `cif.py`, `framework.py`, `project_spec.py` | 01 |
| topology search: PORMAKE net scan, parent frameworks, substituent geometry (instead of supplied CIFs) | `topology_builders.py`, `pormake_worker.py` | 01, 02 |
| decoration space, symmetry, swap reachability | `structures.py` | 01, 04 |
| topology screen: identity (fixture) or ML potential (orb-v3 + D3) with seeds, DFT calibration and conservative decisions | `structures.py`, `screening.py`, `mlip.py` | 02 |
| label ingest, dataset snapshots, frozen split | `datasets.py` | integrator glue |
| parent-lattice cluster orbits (point/pair, periodic, per formula unit) | `clusters.py` | 03 |
| features, exact fit, model selection (fixture split; grouped CV + frozen holdout for physical projects) | `configuration_model.py` | 03 |
| supercell replication and transfer checks | `transfer.py` | 01, 03 |
| QUBO / HUBO compilation, cubic quadratization, sublattice constraints, penalty bound, one-hot; 2ⁿ identity check | `encoding.py` | 03 |
| exact enumeration, branch and bound (certified or with gap), simulated annealing | `optimization.py` | 04 |
| QAOA statevector simulation with exchange mixer; benchmark vs exact and annealing | `qaoa.py` | 04 |
| QAOA sampling on Qiskit fake IBM devices (FakeBrisbane noise), circuit checked against the simulator | `qpu.py`, `qpu_worker.py` | 04 |
| symmetry dedup, top-k pool; relaxed-geometry equivalence | `candidates.py`, `geometry_match.py` | 05 |
| acquisition policy (energy / uncertainty / diversity / audit) | `acquisition.py` | 05 |
| evaluator boundary (fixture mock backend as subprocess) | `electronic.py` | 06/07 boundary |
| active-space export (FCIDUMP + metadata); ED, PySCF CASCI, simulated VQE and SQD on one Hamiltonian | `active_space.py`, `electronic_solvers.py` | 07 |
| finite-cluster subtractive embedding: capped node clusters, cluster jobs, corrected-energy assembly, scheme validation, corrected ranking | `embedding.py`, `cluster_worker.py` | 07 |
| periodic DFT: engine adapters, jobs, cutoff convergence of energy differences | `dft.py`, `pyscf_worker.py` | 06 |
| refinement: fixture active learning; physical rounds with declared stopping rules | `refinement.py`, `orchestrator.py` | 08 |
| ranking, reference-phase competition, structural validation, report | `ranking.py`, `references.py`, `validation.py`, `report.py` | 09 |
| proposal validation (A1/A2/A67 envelopes) | `gateway.py` | 10 |
| AiiDA WorkChain engine with stage provenance; optimistic state commits | `aiida_engine.py`, `store.py` | 10 |
| runtime agents: packets, strict contracts, OpenRouter client, validation, literature (Valency MCP, local papers), LangGraph run | `agents/` | 10 |

## Installation on a new machine

The core pipeline is standard-library Python. Every heavy engine runs as a subprocess worker in an interpreter the
project names (`protocols.*.python`), so each engine can live in its own environment. Two environments are needed:
`.venv` for the main code and most engines, and `.venv-mlip` for the ML potential. Both are gitignored, so create
them on every machine.

**1. Main environment (`.venv`).** Python ≥ 3.10; developed on 3.14.

```bash
python3 -m venv .venv
.venv/bin/pip install -e . --no-deps                       # mofpipe itself
.venv/bin/pip install "pyscf>=2.14" "ase>=3.23" pytest     # PySCF DFT engine, cluster/active-space solvers, tests
.venv/bin/pip install "pormake==0.2.3"                     # topology generation (pulls jax[cpu], pymatgen, networkx)
.venv/bin/pip install "qiskit>=2.0" "qiskit-aer>=0.17" "qiskit-ibm-runtime>=0.40"   # QAOA on fake IBM devices
.venv/bin/pip install "jsonschema>=4.21" "pyyaml>=6" "langgraph>=0.4" "pypdf>=4"     # runtime agents
# optional: .venv/bin/pip install "mcp>=1.9" pandas pyarrow h5py huggingface_hub     # Valency MCP, reference datasets
# optional: .venv/bin/pip install "aiida-core>=2.6" && AIIDA_PATH=$PWD/.aiida .venv/bin/verdi presto --profile-name mofpipe
```

`pyscf-dispersion>=1.5` (a core dependency) has no macOS arm64 wheel on PyPI. Without it the DFT-D3/D4 path of the
PySCF engine is unavailable; use `"dispersion": "none"` there (the cluster low level must then match).

**2. ML-potential environment (`.venv-mlip`).** Python ≤ 3.13: `orb-models` needs `dm-tree`, which has no Python 3.14
wheel and does not compile there.

```bash
python3.12 -m venv .venv-mlip
.venv-mlip/bin/pip install "orb-models>=0.7" "ase>=3.23"   # torch, orb v3, nvalchemi D3(BJ) kernels
```

The first relaxation downloads the Orb checkpoint (about 100 MB) into `~/.cache/cached_path`. Projects then point
`protocols.ml_potential_checkpoint.python` at `.venv-mlip/bin/python` (absolute path, or relative to the project).

**3. Check the installation.**

```bash
.venv/bin/python -m pytest -m "not pyscf"                    # core suite, no real DFT
.venv/bin/python tests/mof5_scheme.py runs/mof5_scheme       # MOF-5 project with generated topologies
PYTHONPATH=src .venv/bin/python -m mofpipe generate-topologies --project runs/mof5_scheme/project.json
```

The last command should accept `pcu` and `bsn` and list every rejected net with its reason, in a few seconds.
`tests/mof5_scheme.py` writes absolute interpreter paths for this checkout (`.venv` and `.venv-mlip`); regenerate the
project after moving the repository.

Interpreters a physical project can name:

| Field | Needs | Typical value |
|---|---|---|
| `protocols.topology_generation.python` | `pormake` | `.venv/bin/python` |
| `protocols.ml_potential_checkpoint.python` | `orb-models` | `.venv-mlip/bin/python` |
| `protocols.quantum_solvers[0].python` (`kind: qpu_fake_device`) | `qiskit`, `qiskit-aer`, `qiskit-ibm-runtime` | `.venv/bin/python` |
| `protocols.embedding.command` (`engine: pyscf`) | `pyscf` | `[".venv/bin/python"]` |
| `protocols.dft.command` (`engine: quantum_espresso`) | `pw.x` | `["pw.x"]` (or the mock `tests/mock_pw.py`) |

## Topology generation (PORMAKE)

A physical project may declare `protocols.topology_generation` instead of `source_structures`. The candidate
frameworks then come from a topology builder rather than from supplied CIFs (`topology_builders.py`, worker
`pormake_worker.py`):

1. **Net search.** Every RCSR net in PORMAKE's database whose nodes all have the node block's connectivity (or the
   declared `nets` list) is scored by PORMAKE's `Locator`: the worst node RMSD of the node block. Nets above
   `max_node_rmsd_A` or with more than `max_nodes_per_cell` nodes are rejected.
2. **Construction.** PORMAKE builds the parent framework (node block on every node, parent edge block on every edge,
   cell rescaled), written as a P1 CIF.
3. **Working cell.** The smallest diagonal supercell (multipliers ≤ `max_multiplier`) with integral linker counts
   and between `min_substitution_sites` and `max_substitution_sites` slots.
4. **Checks.** Each built CIF goes through the ordinary slot-lattice checks (nodes, linkers, connectors, inventory,
   charge). The framework with every slot substituted must also pass the builder's sanity checks; this catches
   substituent clashes in strained nets.
5. **Substituent geometry.** Taken from a PORMAKE block (`substituents`, e.g. `E72` = 2-aminophenylene). It
   replaces a hand-written `substitution.group`. A block whose root atom lies on the connection path (an –NH–
   bridge such as `E201`) is refused.

Accepted nets become ordinary source structures: A1's catalogue, ML screening, labels, fitting and solving treat
them like supplied CIFs. Rejected nets are recorded with their reason. Results are cached under
`<project>/generated/<key>/`, so reloading a project never re-runs the builder.

```json
"topology_generation": {
  "builder": "pormake", "python": ".venv/bin/python",
  "node_block": {"block_id": "Zn4O", "pormake_id": "N16"},
  "edge_block": {"block_id": "BDC", "pormake_id": "E14"},
  "substituents": [{"block_id": "NH2_BDC", "pormake_id": "E72", "root_element": "N", "replace_element": "H"}],
  "nets": "auto", "max_node_rmsd_A": 0.25, "max_nodes_per_cell": 4, "max_candidates": 6,
  "min_substitution_sites": 12, "max_substitution_sites": 12, "max_multiplier": 2
}
```

**MOF-5 result** (Zn₄O + BDC/NH₂-BDC 1:1, automatic scan of PORMAKE 0.2.3):

| Net | Outcome | Reason |
|---|---|---|
| `pcu` | accepted | node RMSD 0.000 Å, 2×2×1 cell, 12 slots |
| `bsn` | accepted | node RMSD 0.206 Å, 1×1×1 cell, 12 slots |
| `snp` | rejected | amino groups overlap (0.15 Å) when every slot is substituted |
| `sxb` | rejected | amino H bonds to two atoms (ambiguous anchor) |
| `kag`, `sxc` | rejected | 9 slots per cell: 1:1 not integral within 12 slots |
| `acs` | rejected | trigonal-prismatic node: RMSD 0.42 Å |
| `btu` | rejected | beyond `max_candidates` |
| others | rejected | node connectivity, too many nodes per cell, or unreadable `.cgd` files |

ToBaCCo is refused (`UNSUPPORTED_TOPOLOGY_BUILDER`): it is a script repository, not an installable package.

The rest of `tests/mof5_scheme.py` exercises the other engines on these frameworks:
- orb-v3 + D3(BJ) relaxations in `.venv-mlip`;
- QAOA on the exact simulator and on `fake_brisbane`;
- PySCF Zn₄O(HCO₂)₆ node clusters with VQE/SQD;
- the mock `pw.x` as periodic DFT, so the project is labelled SYNTHETIC.

## Milestone F extras (ML screening, fake-device QPU, AiiDA)

- **ML screening:** physical projects declare `protocols.ml_potential_checkpoint`, for example:
  `{"engine": "orb", "model": "orb-v3-conservative-20-omat", "dispersion": "D3BJ", "python": ".venv-mlip/bin/python", "relaxation": {"optimizer": "FIRE", "fmax_eV_per_A": 0.1, "max_steps": 400, "cell": "variable"}}`.
  Optional `protocols.ml_calibration` and `protocols.screening` set the calibration and decision policy.
  `screening.energetic_policy.allow_exploratory: true` lets the uncalibrated ML gap deprioritise a topology. It is
  an explicit opt-in, labelled EXPLORATORY, kept in the audit sample and reinstatable. Without it, only a
  DFT-calibrated potential can deprioritise.
- **QPU:** `optimizer.quantum_protocol` with `"kind": "qpu_fake_device", "device": "fake_brisbane", "python": ".venv/bin/python"`.
- **AiiDA:** `python -m mofpipe run --engine aiida ...`.

## Design-Hamiltonian benchmark (Kang et al. 2025)

`benchmarks/kang2025/` holds the inputs Kang et al. started from: the net-derived site graph with edge distances,
the linker characteristic lengths and the composition. There are three materials: Cu-THQ-HHTP, Py-MV-DBA-COF and MUF-7.

In design mode A2 (the LLM) composes the Hamiltonian from a term catalogue: length balance and like/unlike pair
mixing per edge kind, with distance, distance^α or uniform weights. Code computes the coefficients, solves exactly,
runs QAOA and optionally samples on FakeBrisbane. The result is scored against the experimental arrangement, which
the model never sees.

`kang_control` runs the paper's own weighting through the same code.

```bash
PYTHONPATH=src .venv/bin/python -m mofpipe agents-design-bench --models qwen/qwen3.8-flash,qwen/qwen3.8-max-0902 \
    --repeats 3 --reasoning-effort low --qpu-python .venv/bin/python
```

## Runtime agents (`src/mofpipe/agents`, optional)

Three LLM roles from `MOF_AI_Agent_Implementation_Guide-2` (it supersedes the role list in `instructions/10`). Each role
makes one strict-JSON request to OpenRouter (Qwen 3.8 by default). Code validates every output and does all the
numerics.

- **A1** reads a design brief (`--brief`, default: the project's purpose). It decides for every catalogue topology,
  meaning every topology with validated inputs, whether to analyse it. An exclusion carries a reason code. Code
  checks the factual reasons, and brief-scope exclusions are allowed only by policy. The project is then narrowed
  to the selected topologies.
- **A2** proposes, per selected topology, the Hamiltonian form: at most three catalogue models (feature families),
  the encoding and the compilation path. A path that can't hold a model's degree is rejected. Code fits the
  coefficients and selects the model with the frozen validation rule. Code also sets and verifies the
  penalty-constrained QUBO/HUBO.
- **A67** assesses the final candidates against the structural checks, literature passages and reference-dataset
  records. Literature comes from Valency MCP and local papers. Dataset matches (local CoRE MOF 2024 / QMOF copies)
  carry code-computed metrics (density, LCD, PLD). A67 can't pass a failed or missing check, and it can't cite
  anything outside the evidence packet.
- **A4** (optimisation loop of OSI_scheme.pdf; **in progress, not yet covered by tests**) runs after every A67
  assessment.
  - It reads code-computed facts on the evaluated candidates: surrogate vs DFT rank correlation, surrogate error
    against tolerance, whether the surrogate's best is the DFT best, model inadequacy, failed checks.
  - It answers `ACCEPT`, or `REVISE_HAMILTONIAN` with new catalogue model rows in the A2 format.
  - Code rejects reasons the facts don't support, repeated grids and unsupported compilers.
  - A revision re-runs the pipeline from model fitting onward, followed by A67 and A4 again, at most
    `budget.max_optimisation_rounds` times (default 1, so 6 normal requests). `max_optimisation_rounds: 0` restores
    the three-role run.
  - So far it has been run once on the fixture with scripted answers: inadequate pairwise models → revised to the
    triplet model → accepted.
  - `agents/scheme.py` holds deterministic scripted stand-ins for all four roles (offline demos) and a trace that
    maps each scheme box to run artifacts. Neither is wired to the CLI yet.

Scheme mapping (OSI_scheme.pdf):

| Scheme box | Implementation |
|---|---|
| A1 | A1 |
| Topology search | builder + ML screening |
| A2 | A2 |
| QUBO solved | exact, B&B, QAOA, fake device |
| Ground state | DFT + node-cluster VQE/SQD in the embedding stage |
| A3 | A67 |
| A4 | A4 |

```bash
pip install -e '.[agents]'            # requests, jsonschema, pyyaml, langgraph, mcp, pypdf, pandas, pyarrow, h5py, huggingface_hub
# put your OpenRouter key and Valency token in controls.yaml (gitignored)
PYTHONPATH=src python3 -m mofpipe agents-check
PYTHONPATH=src python3 -m mofpipe datasets-fetch                 # pinned CoRE MOF 2024 / QMOF copies into data/reference/
PYTHONPATH=src python3 -m mofpipe agents --project mof_no_agent_fixture/project.json --case higher_order \
    --correction topology_dependent --brief "..."
PYTHONPATH=src python3 -m mofpipe agents-eval --project mof_no_agent_fixture/project.json   # Milestone F comparison
```

The graph is fixed: A1 → construction and screening (ML potential when declared) → A2 → the rest of the pipeline (screened stages reused) → evidence → A67. It runs on LangGraph, or on a sequential runner with the
same routes. A normal run makes three requests. One optional extra request repairs an invalid output. Re-asking A2
after `MODEL_INADEQUATE` is off by default (`budget.allow_model_expansion`).

A run directory holds:
- `agent/`: packets, gateway `DecisionRecord`s, `AgentDecision`s, `ledger.json`, `evidence.json` and `datasets.json`;
- `pipeline_N/`: ordinary pipeline runs, with `agent_layer` in the report;
- `assessment.md`;
- `agent_run.json`.

The deterministic commands never import this package. A project with agents enabled is `BLOCKED` unless it is run
through `mofpipe agents`. Tests (`tests/test_agents.py`) replay scripted outputs and use no network.

See [IMPLEMENTATION.md](IMPLEMENTATION.md) for what each stage does and what every test checks, and [TODO.md](TODO.md) for every unimplemented part (`#TODO` markers in code) and what it requires.

Not implemented: ToBaCCo (PORMAKE is wired), multi-node / multi-edge-block assignments in topology generation, linker orientation search, non-diagonal supercells, DFT+U and hybrid functionals in the PySCF engine, WF-in-DFT embedding and the published Fe–MOF-74 embedding reference, hardware SQD and real-QPU execution (fake IBM devices only), HPC CalcJobs and concurrent execution under AiiDA, multicomponent decorations beyond the encoding layer, quartic quadratization, triplet cluster orbits, CrystalNets/MOFid, phonons, PXRD/NMR, paper reproductions (Kang, Cai, Fe–MOF-74). MOFChecker and Zeo++ run only when installed. Anything not run is reported as `SKIPPED_*`, `DISABLED_BY_CONFIG`, `NOT_RUN` or `INCOMPLETE`, never as passed. The GPU path (`device: gpu`) follows the gpu4pyscf 1.8 API but has not been exercised: no CUDA device was available.
