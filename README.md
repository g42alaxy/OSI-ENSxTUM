# Quantum-enhanced pipeline for automatic MTV-MOF structure modelling

`mofpipe` predicts how linkers arrange in a multivariate metal–organic framework (MTV-MOF). It works from a description of
the material, picks candidate topologies, writes the linker arrangement problem as a QUBO and solves it with classical
and quantum (QAOA) methods. It then evaluates the best arrangements with DFT and quantum electronic-structure solvers,
and checks the result against structural tests, papers and reference datasets.

![Pipeline overview](docs/pipeline.png)

Four LLM agents (A1–A4) make the decisions that need judgement. Code does all the numerics and validates every agent
output. The pipeline also runs without agents, in a deterministic mode where the project file fixes those decisions.

> [!WARNING]
> All physical results so far are on **SYNTHETIC** toy frameworks. Do not read them as MOF predictions.

## Contents

- [How the pipeline works](#how-the-pipeline-works)
- [Installation](#installation)
- [Examples](#examples)
- [Run outputs](#run-outputs)
- [Module map](#module-map)
- [Limitations](#limitations)

## How the pipeline works

The steps below follow the diagram from left to right. Each step lists what it does and where the code is.

### 1. Text input: MOF modelling characteristics

The user describes the material in two parts:

- **A project file** (ProjectSpec v0.1, `project.json`). It declares the parent CIFs, building blocks (metals and
  linkers), the target composition, the DFT protocol and the run policies. The template is
  `instructions/project_spec.template.json`.
- **A design brief**: free text describing what the study is about (`--brief` or `--brief-file`). The default is the
  project's `purpose` field.

Intake (`intake.py`, `project_spec.py`) validates the project before anything runs. A project with missing scientific
fields stops here with a named error such as `SCIENTIFIC_SPEC_MISSING`.

### 2. A1 agent: select MOF structures (topologies)

A1 reads the brief and the catalogue of topologies whose inputs passed validation. For each topology it decides whether
to analyse it. Each exclusion carries a reason code. Code checks the factual reasons. Exclusions based only on the
brief's scope are allowed only when `selection.allow_scope_exclusions` is on. The project is then narrowed to the
selected topologies.

The pipeline builds the selected frameworks from their CIFs (`cif.py`, `framework.py`). It then enumerates the decoration
space, meaning the linker arrangements at the fixed composition, and its symmetry (`structures.py`).

### 3. Topology search with an ML potential

Each selected topology gets a set of seed arrangements: pure-linker endpoints, the most symmetric ordered states, random
states and SQS-like states. An ML potential relaxes every seed (orb-v3 + D3(BJ), declared in
`protocols.ml_potential_checkpoint`). Energies are compared per common formula unit.

The decisions are conservative:

- A topology is rejected only for structural incompatibility: no seed builds, or every relaxed seed changes topology.
- A topology is deprioritised only when the potential has been calibrated against DFT (`protocols.ml_calibration`) and
  an energetic policy is declared.
- Uncertain topologies are kept.

Code: `screening.py`, `mlip.py`, `mlip_worker.py`. The synthetic fixture uses an identity screen that keeps every
topology.

### 4. A2 agent: construct the QUBO Hamiltonian

A2 sees only the topologies that screening kept, together with the screening evidence. For each topology it proposes the
form of the Hamiltonian:

- up to three catalogue models, which are families of cluster features such as point and pair orbits;
- the encoding;
- the compilation path (QUBO, HUBO or quadratized QUBO).

A path that cannot represent a model's polynomial degree is rejected. Code then does the numerics:

1. Builds cluster features on the parent lattice (`clusters.py`).
2. Fits the coefficients to DFT labels and selects a model with grouped cross-validation and a frozen holdout
   (`configuration_model.py`).
3. Compiles the QUBO/HUBO with the composition penalty λ(Σx − N)², sublattice constraints and one-hot encodings. Cubic
   terms are quadratized. Where the cell is small enough, code checks the energy identity on all 2ⁿ bit strings
   (`encoding.py`).

### 5. QUBO: solve with classical and quantum methods

| Method | Kind | Code |
|---|---|---|
| Exact enumeration | classical, certified | `optimization.py` |
| Branch and bound (certified, or returns a gap at a limit) | classical | `optimization.py` |
| Simulated annealing | classical, heuristic | `optimization.py` |
| QAOA, statevector simulation (Dicke initial state, XY exchange mixer that preserves composition) | quantum, simulated | `qaoa.py` |
| QAOA sampling on Qiskit fake IBM devices (FakeBrisbane noise model) | quantum, noisy simulation | `qpu.py`, `qpu_worker.py` |

QAOA is benchmarked against the exact solution. The candidate pool comes from the certified classical solver, so QAOA
never changes it. The solutions are deduplicated by symmetry, and the top unique arrangements per topology go on, plus
every tie at the cutoff (`candidates.py`). The fixture keeps 5 per topology.

### 6. Ground-state evaluation of the top candidates

The candidates are evaluated at two levels:

- **Classical, periodic DFT** (`dft.py`, `pyscf_worker.py`). The main engine is PySCF Gaussian-and-plane-waves DFT
  with GTH basis sets and pseudopotentials, DFT-D3/D4 dispersion, ASE relaxations (`relax`, `vc-relax`), any k-mesh, and
  `device: cpu | gpu`. A Quantum ESPRESSO `pw.x` adapter is also available. A cutoff-convergence gate checks energy
  differences before any ranking is trusted.
- **Quantum, VQE and SQD** (`embedding.py`, `cluster_worker.py`, `active_space.py`, `electronic_solvers.py`). Each metal
  node is cut out as a capped cluster. A high-level correction is computed on the cluster, either with CCSD(T) or with
  an active space solved by exact diagonalisation, CASCI, VQE or SQD (VQE and SQD are simulated). The corrections are
  added to the periodic DFT energy in a finite-cluster subtractive scheme. A corrected ranking is reported only after
  the scheme is validated.

The relaxed structures are deduplicated again by geometry (`geometry_match.py`). They are ranked against reference phases
(`ranking.py`, `references.py`). Optional refinement rounds (`refinement.py`, `acquisition.py`) add new DFT labels, refit
the model and solve again until a declared stopping rule applies.

### 7. A3 agent: confirmation

A3 checks that the top candidates are physically correct and compares them with known results. In the code this role is
called `A67`. Before A3 runs, code collects the evidence:

- structural checks (`validation.py`): post-relaxation slot mapping, residual forces and stress, minimum interatomic
  distances, metal coordination, MOFChecker and CrystalNets;
- literature passages from Valency MCP and local papers;
- matching records from local copies of CoRE MOF 2024 and QMOF, with metrics that code computes (density, LCD, PLD).

A3 writes `assessment.md` for the top 5 candidates (`assessment.top_k_candidates`). It cannot pass a candidate whose
required check failed or is missing. It cannot cite anything outside the evidence packet.

### 8. A4 agent: optimisation feedback loop

A4 revises the Hamiltonian choice when the pipeline produced the wrong candidates, and sends the run back to step 4. In
the code this is the A2 expansion branch (`node_a2_expand` in `agents/runner.py`):

1. Model selection fails with `MODEL_INADEQUATE`.
2. A2 is asked again, with the fit diagnostics and the list of models already tried.
3. A2 proposes a new model family, and the pipeline reruns from fitting onward.

The loop is off by default (`budget.allow_model_expansion: false`) and uses the run's single extra LLM request. The
trigger is the model-validation failure. A3's verdict does not start the loop yet.

### Agent budget and orchestration

The agents call OpenRouter with strict JSON schemas. The default model is Qwen 3.8. The graph is fixed:

```text
A1 → construction and screening → A2 → rest of the pipeline [→ A4/A2 expansion] → evidence → A3
```

It runs on LangGraph, or on a sequential runner with the same routes. A normal run makes three LLM requests, one per
role. One optional extra request either repairs an invalid output or drives the A4 loop.

### Deterministic mode (agents off)

`mofpipe run` executes the same stages without any LLM. The project file fixes the topologies and model candidates.
These commands never import the agent package. A project with agents enabled is `BLOCKED` unless it is run through
`mofpipe agents`.

## Installation

Python ≥ 3.10. The pipeline core is standard-library Python. The periodic DFT engine needs PySCF, pyscf-dispersion and
ASE.

```bash
# Create a virtual environment and install the package itself
python3 -m venv .venv && .venv/bin/pip install -e . --no-deps

# DFT engine and test runner
.venv/bin/pip install "pyscf>=2.14" "pyscf-dispersion>=1.5" "ase>=3.23" pytest
```

`pyscf-dispersion` has no macOS arm64 wheel on PyPI. Leave it out there; the DFT-D3/D4 path is then unavailable.

The optional extras run in the same `.venv`. The core stays importable without them: workers run as subprocesses, and
the AiiDA module loads only with `--engine aiida`.

```bash
# Register the extras (no dependency resolution), then install the packages each extra lists in pyproject.toml:
#   mlip   -> orb-models                                   (step 3, ML-potential screening)
#   qpu    -> qiskit, qiskit-aer, qiskit-ibm-runtime       (step 5, fake-device QAOA)
#   agents -> requests, jsonschema, langgraph, mcp, ...    (agents A1-A4)
#   aiida  -> aiida-core                                   (AiiDA engine)
.venv/bin/pip install -e '.[agents,mlip,qpu,aiida]' --no-deps

# Optional GPU DFT: gpu4pyscf (pick the wheel matching your CUDA toolkit)
.venv/bin/pip install -e '.[gpu]'

# One-time AiiDA profile; needs no services
AIIDA_PATH=$PWD/.aiida .venv/bin/verdi presto --profile-name mofpipe
```

## Examples

All commands run from the repository root. Run directories go to `runs/`, which is gitignored.

### Synthetic fixture (deterministic mode)

`mof_no_agent_fixture/` holds two abstract 12-site substitution graphs, `ring12` and `prism6`. They are test problems,
not atomistic MOFs. The fixture has a known answer, so it checks the whole pipeline end to end.

```bash
# One run: the case with cubic (higher-order) interactions and a topology-dependent correction
PYTHONPATH=src python3 -m mofpipe run    --project mof_no_agent_fixture/project.json \
    --case higher_order --correction topology_dependent

# Run, then compare the report with the harness's known answers
PYTHONPATH=src python3 -m mofpipe accept --project mof_no_agent_fixture/project.json

# All 6 combinations of case x correction mode, each run and accepted
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

`benchmarks/kang2025/` holds the inputs that Kang et al. started from for three materials: Cu-THQ-HHTP, Py-MV-DBA-COF and
MUF-7. The inputs are the site graph derived from the net, with edge distances, the linkers' characteristic lengths and
the composition.

In design mode, A2 composes the Hamiltonian from a catalogue of terms: length balance, and like/unlike pair mixing per
edge kind with distance, distance^α or uniform weights. Code computes the coefficients, solves exactly, runs QAOA and
optionally samples on FakeBrisbane. The result is scored against the experimental arrangement, which the model never
sees. `kang_control` runs the paper's own weighting through the same code.

```bash
# Two Qwen models, 3 repeats each, low reasoning effort; also sample each design on FakeBrisbane
PYTHONPATH=src .venv/bin/python -m mofpipe agents-design-bench \
    --models qwen/qwen3.8-flash,qwen/qwen3.8-max-0902 \
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
# Everything except the real-DFT checks (~3 min)
.venv/bin/python -m pytest -m "not pyscf"

# Real PySCF checks (minutes)
.venv/bin/python -m pytest -m pyscf

# Also the end-to-end PySCF pipeline run on the formate toy (opt-in)
MOFPIPE_SLOW_TESTS=1 .venv/bin/python -m pytest -m pyscf -k end_to_end
```

The agent tests (`tests/test_agents.py`) replay scripted outputs and use no network.

## Run outputs

A pipeline run directory holds:

- `state.json`: the authoritative workflow state;
- `events.jsonl`: the event log;
- `artifacts/`: content-addressed, immutable artifacts;
- `jobs/`: evaluator and DFT inputs, raw outputs and PySCF logs;
- `report.json` and `report.md`.

A re-run with identical inputs reuses the committed stages and never resubmits a successful job.

An agent run directory adds:

- `agent/`: packets, gateway `DecisionRecord`s, `AgentDecision`s, `ledger.json`, `evidence.json` and `datasets.json`;
- `pipeline/`: the ordinary pipeline run, with `agent_layer` in the report;
- `assessment.md`: the A3 assessment;
- `agent_run.json`: a summary of status, request counts, decisions and attempts.

## Module map

| Diagram step | Stage | Module | Instruction task |
|---|---|---|---|
| 1 | intake, project validation | `intake.py`, `project_spec.py` | 01 |
| 2 | CIF parsing, slot lattice, framework builder, working and transfer cells | `cif.py`, `framework.py`, `project_spec.py` | 01 |
| 2 | decoration space, symmetry, swap reachability | `structures.py` | 01, 04 |
| 3 | topology screen: identity (fixture) or ML potential (orb-v3 + D3) with seeds, DFT calibration and conservative decisions | `structures.py`, `screening.py`, `mlip.py` | 02 |
| 4 | label ingest, dataset snapshots, frozen split | `datasets.py` | integrator glue |
| 4 | parent-lattice cluster orbits (point/pair, periodic, per formula unit) | `clusters.py` | 03 |
| 4 | features, exact fit, model selection (fixture split; grouped CV + frozen holdout for physical projects) | `configuration_model.py` | 03 |
| 4 | supercell replication and transfer checks | `transfer.py` | 01, 03 |
| 4 | QUBO / HUBO compilation, cubic quadratization, sublattice constraints, penalty bound, one-hot; 2ⁿ identity check | `encoding.py` | 03 |
| 5 | exact enumeration, branch and bound (certified or with gap), simulated annealing | `optimization.py` | 04 |
| 5 | QAOA statevector simulation with exchange mixer; benchmark vs exact and annealing | `qaoa.py` | 04 |
| 5 | QAOA sampling on Qiskit fake IBM devices (FakeBrisbane noise), circuit checked against the simulator | `qpu.py`, `qpu_worker.py` | 04 |
| 5 | symmetry dedup, top-k pool; relaxed-geometry equivalence | `candidates.py`, `geometry_match.py` | 05 |
| 6 | acquisition policy (energy / uncertainty / diversity / audit) | `acquisition.py` | 05 |
| 6 | evaluator boundary (fixture mock backend as subprocess) | `electronic.py` | 06/07 boundary |
| 6 | periodic DFT: engine adapters, jobs, cutoff convergence of energy differences | `dft.py`, `pyscf_worker.py` | 06 |
| 6 | active-space export (FCIDUMP + metadata); ED, PySCF CASCI, simulated VQE and SQD on one Hamiltonian | `active_space.py`, `electronic_solvers.py` | 07 |
| 6 | finite-cluster subtractive embedding: capped node clusters, cluster jobs, corrected-energy assembly, scheme validation, corrected ranking | `embedding.py`, `cluster_worker.py` | 07 |
| 6 | refinement: fixture active learning; physical rounds with declared stopping rules | `refinement.py`, `orchestrator.py` | 08 |
| 6–7 | ranking, reference-phase competition, structural validation, report | `ranking.py`, `references.py`, `validation.py`, `report.py` | 09 |
| A1–A4 | proposal validation (A1/A2/A67 envelopes) | `gateway.py` | 10 |
| A1–A4 | runtime agents: packets, strict contracts, OpenRouter client, validation, literature (Valency MCP, local papers), LangGraph run | `agents/` | 10 |
| — | AiiDA WorkChain engine with stage provenance; optimistic state commits | `aiida_engine.py`, `store.py` | 10 |

[IMPLEMENTATION.md](IMPLEMENTATION.md) describes what each stage does and what every test checks.
[TODO.md](TODO.md) lists every unimplemented part (`#TODO` markers in the code) and what it needs.

Not implemented: ToBaCCo (PORMAKE is wired), multi-node / multi-edge-block assignments in topology generation, linker orientation search, non-diagonal supercells, DFT+U and hybrid functionals in the PySCF engine, WF-in-DFT embedding and the published Fe–MOF-74 embedding reference, hardware SQD and real-QPU execution (fake IBM devices only), HPC CalcJobs and concurrent execution under AiiDA, multicomponent decorations beyond the encoding layer, quartic quadratization, triplet cluster orbits, CrystalNets/MOFid, phonons, PXRD/NMR, paper reproductions (Kang, Cai, Fe–MOF-74). MOFChecker and Zeo++ run only when installed. Anything not run is reported as `SKIPPED_*`, `DISABLED_BY_CONFIG`, `NOT_RUN` or `INCOMPLETE`, never as passed. The GPU path (`device: gpu`) follows the gpu4pyscf 1.8 API but has not been exercised: no CUDA device was available.
