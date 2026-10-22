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

## Milestone F extras (ML screening, fake-device QPU, AiiDA)

These run in the project `.venv`. The core stays importable without them: workers run as subprocesses, and the
AiiDA module loads only with `--engine aiida`.

```bash
python3 -m venv .venv && .venv/bin/pip install -e . --no-deps
.venv/bin/pip install "pyscf>=2.10" "ase>=3.23" pytest
.venv/bin/pip install -e '.[agents,mlip,qpu,aiida]' --no-deps   # then install the listed extras' packages
AIIDA_PATH=$PWD/.aiida .venv/bin/verdi presto --profile-name mofpipe   # one-time, no services needed
```

`pyscf-dispersion>=1.5` (a core dependency) has no macOS arm64 wheel on PyPI. Without it, the DFT-D3/D4 path of the
PySCF engine is unavailable on that platform.

- **ML screening:** physical projects declare `protocols.ml_potential_checkpoint`, for example:
  `{"engine": "orb", "model": "orb-v3-conservative-inf-omat", "dispersion": "D3BJ", "python": ".venv/bin/python", "relaxation": {"optimizer": "FIRE", "fmax_eV_per_A": 0.05, "max_steps": 400, "cell": "variable"}}`.
  Optional `protocols.ml_calibration` and `protocols.screening` set the calibration and decision policy.
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

Not implemented: PORMAKE/ToBaCCo builders, linker orientation search, non-diagonal supercells, DFT+U and hybrid functionals in the PySCF engine, WF-in-DFT embedding and the published Fe–MOF-74 embedding reference, hardware SQD and real-QPU execution (fake IBM devices only), HPC CalcJobs and concurrent execution under AiiDA, multicomponent decorations beyond the encoding layer, quartic quadratization, triplet cluster orbits, CrystalNets/MOFid, phonons, PXRD/NMR, paper reproductions (Kang, Cai, Fe–MOF-74). MOFChecker and Zeo++ run only when installed. Anything not run is reported as `SKIPPED_*`, `DISABLED_BY_CONFIG`, `NOT_RUN` or `INCOMPLETE`, never as passed. The GPU path (`device: gpu`) follows the gpu4pyscf 1.8 API but has not been exercised: no CUDA device was available.
