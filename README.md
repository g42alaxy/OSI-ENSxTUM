# mofpipe: quantum-enhanced modelling of multivariate MOFs

A multivariate metal–organic framework (MTV-MOF) mixes two or more kinds of organic linker in one crystal. Knowing the
composition, say half linker A and half linker B, is not enough to know the material. What matters is *where* each
linker sits. `mofpipe` predicts that arrangement.

It turns the question "which arrangement has the lowest energy?" into a binary optimisation problem (a QUBO). It solves
that problem with classical and quantum algorithms, checks the best answers with DFT and quantum chemistry, and has an
AI reviewer compare them with published results.

![Pipeline overview](docs/pipeline.png)

> [!WARNING]
> **Status (v0.2.0): research prototype.** Every physical result so far comes from **synthetic** toy frameworks. None
> of them is a prediction for a real MOF. Quantum algorithms run on simulators and on fake (simulated, noisy) IBM
> devices, not on real quantum hardware.

## Contents

- [Quick start](#quick-start)
- [What happens in a run](#what-happens-in-a-run)
- [Installation](#installation)
- [Usage](#usage)
- [Tests](#tests)
- [What a run produces](#what-a-run-produces)
- [Repository layout](#repository-layout)
- [Limitations](#limitations)
- [License](#license)

## Quick start

You only need Python 3.10 or newer. No other packages are required for this step.

```bash
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

This runs the whole pipeline on a small synthetic test problem whose answer is known. It runs 6 times (2 problems × 3
correction modes) and checks each run against the answer. It takes a few seconds, and every row should end with
`"acceptance": "PASS"`.

## What happens in a run

The steps follow the diagram above from left to right.

1. **Describe the material.** You write a project file, `project.json`. It lists the parent crystal structures (CIFs),
   the metals and linkers, the target composition and the DFT settings. The template is
   [instructions/project_spec.template.json](instructions/project_spec.template.json). You can also give a short
   free-text *design brief* that says what the study is about. The pipeline validates the project first. If a
   scientific input is missing, it stops with a named error such as `SCIENTIFIC_SPEC_MISSING`.

2. **Choose topologies (agent A1).** A topology is the underlying net of the framework. A1 reads the brief and decides
   which candidate topologies are worth analysing. Every exclusion needs a reason, and code checks the factual ones.

3. **Screen with an ML potential.** For each topology, a set of starting arrangements is relaxed with a machine-learned
   interatomic potential (orb-v3 + D3). A topology is dropped only if it cannot be built or if every relaxed structure
   changes topology. Uncertain cases are kept.

4. **Build the energy model (agent A2).** A2 chooses which interactions the model should contain (single sites,
   neighbouring pairs, triplets) and how to encode them as bits. Code then:
   - fits the coefficients to DFT energies;
   - picks the best model by cross-validation, with a held-out test set;
   - compiles it into a QUBO, a quadratic function of 0/1 variables with one variable per linker site. A penalty term
     enforces the composition. On small cells, code checks that the QUBO gives the same energy as the model for every
     bit string.

5. **Solve the QUBO.** Classical solvers give the certified answer: exact enumeration, branch and bound, and simulated
   annealing. QAOA, a quantum optimisation algorithm, runs on a statevector simulator and on a noisy fake IBM device
   (FakeBrisbane). It is benchmarked against the classical answer, but the candidates always come from the classical
   solver, so QAOA never changes them. Arrangements that are the same up to symmetry are merged. The 5 best distinct
   arrangements per topology go on to the next step.

6. **Evaluate ground states.**
   - *Classical:* periodic DFT with PySCF, which relaxes and scores the top candidates. A Quantum ESPRESSO adapter also
     exists. Before any ranking is trusted, a convergence check confirms that energy differences are stable with
     respect to the basis cutoff.
   - *Quantum:* optionally, each metal node is cut out as a small cluster and corrected with a higher-level method. This
     is either CCSD(T) or an active-space solver: exact diagonalisation, CASCI, or simulated VQE or SQD.

   The candidates are then ranked against competing reference phases. Optional refinement rounds add new DFT data,
   refit the model and solve again.

7. **Confirm (agent A3).** Before A3 runs, code collects the evidence:
   - structural checks: residual forces, interatomic distances, metal coordination, MOFChecker and CrystalNets;
   - passages from papers, found through Valency MCP or in local PDFs;
   - matching entries from the CoRE MOF 2024 and QMOF databases.

   A3 reads this evidence and writes an assessment of the top 5 candidates. It cannot pass a candidate that failed a
   required check, and it cannot cite anything outside the evidence.

8. **Feedback loop (agent A4).** If the energy model cannot reproduce the DFT data (`MODEL_INADEQUATE`), A2 is asked once
   more for a different model family, and the pipeline reruns from the fitting step. This loop is off by default.

### How the AI agents are used

- The agents make judgement calls only. All numbers come from code, and every agent answer is checked against a
  strict JSON schema before it is used.
- A normal run makes three LLM requests (A1, A2, A3) through OpenRouter. At most one extra request is allowed, either
  to repair an invalid answer or to run the A4 loop.
- **The agents are optional.** `mofpipe run` executes the same stages without any LLM. The project file then fixes
  which topologies and models to use.
- Two roles have different names in the code. A3 is called `A67`, and A4 is the "A2 expansion" step
  (`node_a2_expand` in [agents/runner.py](src/mofpipe/agents/runner.py)).

## Installation

Python 3.10 or newer is required. On Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"        # core package + PySCF, pyscf-dispersion, ASE, pytest
```

Install the optional extras you need:

| Extra | Enables | Main packages |
|---|---|---|
| `agents` | AI agents A1–A4, literature and database lookup | requests, jsonschema, langgraph, mcp, pandas, h5py |
| `mlip` | ML-potential screening (step 3) | orb-models |
| `qpu` | QAOA on fake IBM devices (step 5) | qiskit, qiskit-aer, qiskit-ibm-runtime |
| `aiida` | AiiDA workflow engine with provenance | aiida-core |
| `gpu` | DFT on a GPU (untested, see [Limitations](#limitations)) | gpu4pyscf-cuda12x |

```bash
pip install -e ".[agents,mlip,qpu]"

# AiiDA only: create a local profile once (no database server or broker needed)
pip install -e ".[aiida]"
AIIDA_PATH=$PWD/.aiida verdi presto --profile-name mofpipe
```

The package still imports without the extras. Each optional component loads only when it is used.

> [!NOTE]
> **macOS on Apple silicon:** `pyscf-dispersion` has no wheel for this platform, so a plain `pip install -e .` fails.
> Install the package with `--no-deps`, then add the dependencies yourself, leaving out `pyscf-dispersion`:
>
> ```bash
> pip install -e . --no-deps
> pip install "pyscf>=2.14" "ase>=3.23" pytest
> ```
>
> Do the same for an extra: register it with `pip install -e ".[agents]" --no-deps`, then install the packages it lists
> in [pyproject.toml](pyproject.toml). Without `pyscf-dispersion`, DFT-D3/D4 dispersion corrections are unavailable.

## Usage

Run every command from the repository root. Run directories go to `runs/`, which git ignores. Commands written with
`python3` need no extra packages. Commands written with `python` expect the virtual environment from
[Installation](#installation) to be active.

### 1. Synthetic test problem (no dependencies)

[mof_no_agent_fixture/](mof_no_agent_fixture/) contains two abstract 12-site graphs, `ring12` and `prism6`. They are
test problems with a known answer, not real MOFs.

```bash
# One run: the problem with three-body interactions and a topology-dependent correction
PYTHONPATH=src python3 -m mofpipe run --project mof_no_agent_fixture/project.json \
    --case higher_order --correction topology_dependent

# One run, then compare the result with the known answer
PYTHONPATH=src python3 -m mofpipe accept --project mof_no_agent_fixture/project.json

# All 6 combinations, each run and checked
PYTHONPATH=src python3 -m mofpipe matrix --project mof_no_agent_fixture/project.json --out runs
```

### 2. Toy frameworks with a mock DFT code

These toys run the physical path (CIF → framework → DFT energies → model → ranking) with a fake DFT program. They test
the plumbing, not the physics. The second toy takes about 30 seconds.

```bash
# Write two toy projects: runs/toy_b (one topology) and runs/toy_c (two topologies, supercell, refinement)
python3 -c "import sys; sys.path[:0] = ['tests']; import toy_framework as t; \
print(t.write_toy_project('runs/toy_b'), t.write_toy_milestone_c_project('runs/toy_c'))"

PYTHONPATH=src python3 -m mofpipe run --project runs/toy_c/project.json
```

### 3. Real DFT on a toy (PySCF)

```bash
# Write a synthetic aluminium-formate project that uses the real PySCF engine
python tests/pyscf_toy.py runs/toy_pyscf

PYTHONPATH=src python -m mofpipe run --project runs/toy_pyscf/project.json
```

With the shipped 25 Ha cutoff, the convergence check stops this run on purpose with `DFT_SETTINGS_NOT_CONVERGED`
(about 10 minutes). For a complete run, raise `ke_cutoff_Ha` and the cutoff series to about 100 Ha in the project file.

### 4. An incomplete project

```bash
# Loads as a physical project, then stops at intake and lists the missing fields
PYTHONPATH=src python3 -m mofpipe run --project mof_no_agent_fixture/physical_case/recipe.json
```

### 5. Quantum chemistry benchmarks

```bash
# H2 in a minimal basis: exact diagonalisation, CASCI, VQE and SQD on the same Hamiltonian,
# compared with the textbook values (Szabo & Ostlund)
PYTHONPATH=src python -m mofpipe electronic-benchmark \
    --spec benchmarks/h2_sto3g_szabo_ostlund.json --run-dir runs/h2_benchmark

# Cluster embedding (HF corrected to CCSD) on keto/enol tautomers,
# compared with an independent CCSD calculation on the whole molecule
PYTHONPATH=src python -m mofpipe embedding-benchmark \
    --spec benchmarks/embedding_keto_enol_hf_ccsd.json --run-dir runs/embedding_benchmark
```

### 6. With the AI agents

Set up once:

```bash
pip install -e ".[agents]"

# Copy the example settings and fill in your OpenRouter key and Valency token.
# controls.yaml is gitignored. Alternatively, set OPENROUTER_API_KEY, OPENROUTER_MODEL and VALENCY_API_TOKEN.
cp controls_example.yaml controls.yaml

# Check the key, the model and the Valency connection
PYTHONPATH=src python -m mofpipe agents-check

# Download pinned copies of the CoRE MOF 2024 and QMOF databases into data/reference/ (used by A3)
PYTHONPATH=src python -m mofpipe datasets-fetch
```

The example settings use `qwen/qwen3.8-flash`. The code has no default model, so `openrouter.model` (or
`OPENROUTER_MODEL`) must be set.

Then run:

```bash
# Full run: A1 picks topologies, A2 designs the model, the pipeline runs, A3 writes the assessment
PYTHONPATH=src python -m mofpipe agents --project mof_no_agent_fixture/project.json \
    --case higher_order --correction topology_dependent --brief "What the study is about"

# Same, but replay saved agent answers instead of calling the API
PYTHONPATH=src python -m mofpipe agents --project mof_no_agent_fixture/project.json --replay answers.json

# Compare runs with and without agents at the same budget
PYTHONPATH=src python -m mofpipe agents-eval --project mof_no_agent_fixture/project.json
```

To turn on the A4 feedback loop, set `budget.allow_model_expansion: true` in `controls.yaml`.

### 7. Benchmark: Kang et al. (2025)

Kang, Kim & Kim (*ACS Cent. Sci.* 2025, [doi:10.1021/acscentsci.5c00918](https://doi.org/10.1021/acscentsci.5c00918))
used a hand-built Hamiltonian and VQE to predict linker arrangements in three materials: Cu-THQ-HHTP, Py-MV-DBA-COF
and MUF-7. [benchmarks/kang2025/](benchmarks/kang2025/) contains their inputs: the site graph, edge distances, linker
lengths and composition.

In this benchmark, A2 builds the Hamiltonian itself from a small catalogue of terms. Code computes the coefficients and
solves the problem exactly and with QAOA. The result is scored against the experimental arrangement, which the model
never sees.

```bash
# The paper's own weighting through the same code; no LLM calls (still reads a controls file)
PYTHONPATH=src python -m mofpipe agents-design-bench --control-only --controls controls_example.yaml

# Two Qwen models, 3 repeats each; also sample every design on the fake IBM device
PYTHONPATH=src python -m mofpipe agents-design-bench \
    --models qwen/qwen3.8-flash,qwen/qwen3.8-max-0902 \
    --repeats 3 --reasoning-effort low --qpu-python .venv/bin/python
```

The control run reproduces the paper's energy spectrum for all three materials (Spearman ρ = 1.000). With that
Hamiltonian, QAOA samples the experimental arrangement with a median probability of 0.25–0.49. A uniform random pick
among arrangements with the right composition gives 0.03–0.10. Results land in `runs/kang2025_design/results.md`. The
fourth material in the paper, SIOC-COF2, is not included, because its graph could not be reconstructed.

### Optional features in the project file

```jsonc
// Step 3: ML-potential screening (protocols.ml_potential_checkpoint).
// protocols.ml_calibration and protocols.screening set the calibration and the decision policy.
{"engine": "orb", "model": "orb-v3-conservative-inf-omat", "dispersion": "D3BJ", "python": ".venv/bin/python",
 "relaxation": {"optimizer": "FIRE", "fmax_eV_per_A": 0.05, "max_steps": 400, "cell": "variable"}}

ToBaCCo is refused (`UNSUPPORTED_TOPOLOGY_BUILDER`): it is a script repository, not an installable package.

To run any project on the AiiDA engine and record the provenance of every stage:

```bash
PYTHONPATH=src python -m mofpipe run --engine aiida --project <project.json>
```

## Tests

```bash
python -m pytest -m "not pyscf"                                  # everything except real DFT (~4 minutes)
python -m pytest -m pyscf                                        # real PySCF calculations (minutes)
MOFPIPE_SLOW_TESTS=1 python -m pytest -m pyscf -k end_to_end     # full PySCF pipeline run on the toy (opt-in)
```

The agent tests need the `agents` extra. They replay saved answers and do not use the network.

## What a run produces

A pipeline run directory contains:

| Path | Contents |
|---|---|
| `report.md`, `report.json` | the result: ranking, chosen models, the status of every stage |
| `state.json` | the workflow state |
| `events.jsonl` | the event log |
| `artifacts/` | intermediate results, stored by content hash and never overwritten |
| `jobs/` | DFT and evaluator inputs, raw outputs and logs |

Rerunning with the same inputs reuses finished stages and never resubmits a successful job.

An agent run adds `agent/` (the agents' inputs, answers and request ledger), `pipeline/` (the ordinary run),
`assessment.md` (A3's verdict) and `agent_run.json` (a summary).

## Repository layout

| Path | Contents |
|---|---|
| [src/mofpipe/](src/mofpipe/) | the pipeline |
| [src/mofpipe/agents/](src/mofpipe/agents/) | the AI agents: prompts, schemas, OpenRouter client, literature and database lookup |
| [tests/](tests/) | test suite, mock DFT and ML programs, toy project generators |
| [mof_no_agent_fixture/](mof_no_agent_fixture/) | synthetic test problem with known answers |
| [benchmarks/](benchmarks/) | H2, embedding and Kang et al. benchmark inputs |
| [instructions/](instructions/) | the original design specification, one brief per module (tasks 01–10) |
| [controls_example.yaml](controls_example.yaml) | example settings for the agents |

<details>
<summary>Module map: which file implements which step</summary>

| Step | What | Module | Spec |
|---|---|---|---|
| 1 | project validation | `intake.py`, `project_spec.py` | 01 |
| 2 | CIF parsing, framework building, supercells | `cif.py`, `framework.py` | 01 |
| 2 | arrangement space and symmetry | `structures.py` | 01, 04 |
| 3 | topology screening with an ML potential | `screening.py`, `mlip.py`, `mlip_worker.py` | 02 |
| 4 | DFT labels, dataset snapshots, frozen train/test split | `datasets.py` | — |
| 4 | cluster features on the parent lattice | `clusters.py` | 03 |
| 4 | model fitting and selection | `configuration_model.py` | 03 |
| 4 | transfer to larger supercells | `transfer.py` | 01, 03 |
| 4 | QUBO/HUBO compilation, cubic-to-quadratic reduction, constraints | `encoding.py` | 03 |
| 5 | exact enumeration, branch and bound, simulated annealing | `optimization.py` | 04 |
| 5 | QAOA (statevector simulation) | `qaoa.py` | 04 |
| 5 | symmetry deduplication, candidate pool, geometry matching | `candidates.py`, `geometry_match.py` | 05 |
| 6 | choosing which structures get new DFT data | `acquisition.py` | 05 |
| 6 | periodic DFT and cutoff convergence | `dft.py`, `pyscf_worker.py` | 06 |
| 6 | active spaces; ED, CASCI, VQE, SQD | `active_space.py`, `electronic_solvers.py`, `electronic.py` | 07 |
| 6 | cluster embedding | `embedding.py`, `cluster_worker.py` | 07 |
| 6 | refinement rounds | `refinement.py`, `orchestrator.py` | 08 |
| 6–7 | ranking, reference phases, structural checks, report | `ranking.py`, `references.py`, `validation.py`, `report.py` | 09 |
| A1–A4 | validation of agent answers | `gateway.py` | 10 |
| A1–A4 | agent runtime | `agents/` | 10 |
| — | AiiDA engine, run storage | `aiida_engine.py`, `store.py` | 10 |

The Spec column points to the matching brief in [instructions/](instructions/).

</details>

## Limitations
A check that did not run is reported as `SKIPPED_*`, `DISABLED_BY_CONFIG`, `NOT_RUN` or `INCOMPLETE`, never as passed.

## License

[CC0 1.0](LICENSE) (public domain dedication).
