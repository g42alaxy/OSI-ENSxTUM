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
```

### Physical toy projects with a mock `pw.x`

These exercise the physical path (CIF → frameworks → DFT labels → model → ranking) with a mock DFT engine. They test
the bookkeeping, not the physics.

```bash
# Write two toy projects:
#   runs/toy_b: one topology (Milestone B)
#   runs/toy_c: two topologies, a transfer supercell and refinement rounds (Milestone C)
python3 -c "import sys; sys.path[:0] = ['tests']; import toy_framework as t; \
print(t.write_toy_project('runs/toy_b'), t.write_toy_milestone_c_project('runs/toy_c'))"

# Run the two-topology toy end to end
PYTHONPATH=src python3 -m mofpipe run --project runs/toy_c/project.json
```

### Real PySCF DFT on a synthetic toy

```bash
# Write a SYNTHETIC aluminium-formate toy project that uses the real PySCF GPW engine
.venv/bin/python tests/pyscf_toy.py runs/toy_pyscf

# Run it. At the shipped 25 Ha cutoff, the convergence gate stops the run with DFT_SETTINGS_NOT_CONVERGED
# (~10 min). For a full run, raise ke_cutoff_Ha and the cutoff series (~100 Ha) in the project file.
PYTHONPATH=src .venv/bin/python -m mofpipe run --project runs/toy_pyscf/project.json
```

### Incomplete physical projects

```bash
# The physical_case recipe loads as a physical project and stops at intake, listing its missing fields
PYTHONPATH=src python3 -m mofpipe run --project mof_no_agent_fixture/physical_case/recipe.json
```

### Quantum electronic-structure benchmarks (step 6)

```bash
# Active-space export, then ED, CASCI, VQE and SQD on one Hamiltonian,
# compared with the published minimal-basis H2 values (Szabo & Ostlund)
PYTHONPATH=src python3 -m mofpipe electronic-benchmark \
    --spec benchmarks/h2_sto3g_szabo_ostlund.json --run-dir runs/h2_benchmark

# Finite-cluster subtractive embedding (HF low level, CCSD high level) on keto/enol tautomers,
# compared with an independent full-molecule CCSD calculation
PYTHONPATH=src python3 -m mofpipe embedding-benchmark \
    --spec benchmarks/embedding_keto_enol_hf_ccsd.json --run-dir runs/embedding_benchmark
```

### Optional features in the project file

```jsonc
// Step 3: ML-potential screening (protocols.ml_potential_checkpoint).
// Optional protocols.ml_calibration and protocols.screening set the calibration and the decision policy.
{"engine": "orb", "model": "orb-v3-conservative-inf-omat", "dispersion": "D3BJ", "python": ".venv/bin/python",
 "relaxation": {"optimizer": "FIRE", "fmax_eV_per_A": 0.05, "max_steps": 400, "cell": "variable"}}

// Step 5: QAOA sampling on a fake IBM device (optimizer.quantum_protocol)
{"kind": "qpu_fake_device", "device": "fake_brisbane", "python": ".venv/bin/python"}
```

```bash
# Run any project on the AiiDA WorkChain engine, with stage provenance
PYTHONPATH=src .venv/bin/python -m mofpipe run --engine aiida --project <project.json>
```

### Agent-assisted run (A1–A4)

```bash
# 1. Install the agent extras: requests, jsonschema, pyyaml, langgraph, mcp, pypdf, pandas, pyarrow, h5py, huggingface_hub
pip install -e '.[agents]'

# 2. Copy the example controls and add your OpenRouter key and Valency token (controls.yaml is gitignored)
cp controls_example.yaml controls.yaml

# 3. Check the key, the model's structured-output support and Valency MCP access
PYTHONPATH=src python3 -m mofpipe agents-check

# 4. Download pinned copies of CoRE MOF 2024 and QMOF into data/reference/ (used by A3)
PYTHONPATH=src python3 -m mofpipe datasets-fetch
```

```bash
# Full agent run: A1 selects topologies, A2 designs the Hamiltonian, the pipeline runs, A3 assesses.
# Add --replay <file.json> to use scripted agent outputs and make no API calls.
PYTHONPATH=src python3 -m mofpipe agents --project mof_no_agent_fixture/project.json \
    --case higher_order --correction topology_dependent --brief "..."

# Compare deterministic and agent-assisted runs at equal budget (Milestone F evaluation)
PYTHONPATH=src python3 -m mofpipe agents-eval --project mof_no_agent_fixture/project.json
```

To enable the A4 loop, set `budget.allow_model_expansion: true` in `controls.yaml`.

### Design-Hamiltonian benchmark (Kang et al. 2025)

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

# The paper's own weighting only, with no LLM calls
PYTHONPATH=src .venv/bin/python -m mofpipe agents-design-bench --control-only
```

### Tests

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

## Limitations

These are not implemented:

- **Structure building:** PORMAKE/ToBaCCo builders, linker orientation search, non-diagonal supercells, CrystalNets/MOFid,
  and multicomponent decorations beyond the encoding layer.
- **Hamiltonian:** quartic quadratization and triplet cluster orbits.
- **DFT and embedding:** DFT+U and hybrid functionals in the PySCF engine, WF-in-DFT embedding and the published
  Fe–MOF-74 embedding reference.
- **Quantum hardware:** hardware SQD and real-QPU execution. Only fake IBM devices are supported.
- **Workflow:** HPC CalcJobs and concurrent execution under AiiDA.
- **Validation:** phonons, PXRD/NMR.
- **Paper reproductions:** Kang, Cai, Fe–MOF-74.

MOFChecker and Zeo++ run only when they are installed. Anything not run is reported as `SKIPPED_*`,
`DISABLED_BY_CONFIG`, `NOT_RUN` or `INCOMPLETE`, never as passed. The GPU path (`device: gpu`) follows the gpu4pyscf
1.8 API but has not been run, because no CUDA device was available.
