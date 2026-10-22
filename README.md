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
| topology screen (identity mode) | `structures.py` | 02 |
| label ingest, dataset snapshots, frozen split | `datasets.py` | integrator glue |
| parent-lattice cluster orbits (point/pair, periodic, per formula unit) | `clusters.py` | 03 |
| features, exact fit, model selection (fixture split; grouped CV + frozen holdout for physical projects) | `configuration_model.py` | 03 |
| supercell replication and transfer checks | `transfer.py` | 01, 03 |
| QUBO / HUBO compilation and 2ⁿ identity check | `encoding.py` | 03 |
| exact enumeration, simulated annealing | `optimization.py` | 04 |
| symmetry dedup, top-k pool; relaxed-geometry equivalence | `candidates.py`, `geometry_match.py` | 05 |
| acquisition policy (energy / uncertainty / diversity / audit) | `acquisition.py` | 05 |
| evaluator boundary (fixture mock backend as subprocess) | `electronic.py` | 06/07 boundary |
| periodic DFT: engine adapters, jobs, cutoff convergence of energy differences | `dft.py`, `pyscf_worker.py` | 06 |
| refinement: fixture active learning; physical rounds with declared stopping rules | `refinement.py`, `orchestrator.py` | 08 |
| ranking, reference-phase competition, structural validation, report | `ranking.py`, `references.py`, `validation.py`, `report.py` | 09 |
| proposal validation, agents disabled | `gateway.py` | 10 |

See [IMPLEMENTATION.md](IMPLEMENTATION.md) for what each stage does and what every test checks, and [TODO.md](TODO.md) for every unimplemented part (`#TODO` markers in code) and what it requires.

Not implemented: PORMAKE/ToBaCCo builders, linker orientation search, non-diagonal supercells, ML potentials and screening, DFT+U and hybrid functionals in the PySCF engine, embedding, FCI/VQE/SQD, QAOA, quadratization, triplet cluster orbits, CrystalNets/MOFid, phonons, PXRD/NMR, live literature review, runtime agents. MOFChecker and Zeo++ run only when installed. Anything not run is reported as `SKIPPED_*`, `DISABLED_BY_CONFIG`, `NOT_RUN` or `INCOMPLETE`, never as passed. The GPU path (`device: gpu`) follows the gpu4pyscf 1.8 API but has not been exercised: no CUDA device was available.
