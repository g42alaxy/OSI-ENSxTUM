Quantum-Enhanced completed pipeline for automatic MTV-MOF structure modeling

## Deterministic pipeline (`src/mofpipe`)

Implementation of the specifications in `instructions/` with runtime LLM agents disabled. Python ≥ 3.10, standard library only (pytest for tests). It runs on the synthetic fixture in `mof_no_agent_fixture/` (abstract substitution graphs, not atomistic MOFs, Milestone A) and on physical projects: a ProjectSpec v0.1 file with a parent CIF, building blocks and a Quantum ESPRESSO protocol (Milestone B). The physical path has been tested only on a SYNTHETIC toy framework with a mock `pw.x`.

```bash
# one run (case: core_pairwise | higher_order; correction: none | constant | topology_dependent)
PYTHONPATH=src python3 -m mofpipe run    --project mof_no_agent_fixture/project.json --case higher_order --correction topology_dependent
# run + harness acceptance (reads harness/ and reference_outputs/ only after the run is finalised)
PYTHONPATH=src python3 -m mofpipe accept --project mof_no_agent_fixture/project.json
# all 6 case x correction combinations
PYTHONPATH=src python3 -m mofpipe matrix --project mof_no_agent_fixture/project.json --out runs
# physical project (Milestone B): SYNTHETIC toy CIF + mock pw.x; a real project points at its CIF, pseudopotentials and pw.x
python3 tests/toy_framework.py runs/toy_project
PYTHONPATH=src python3 -m mofpipe run    --project runs/toy_project/project.json
# the physical_case recipe and the untouched template load as physical projects and stop at intake with their missing fields
PYTHONPATH=src python3 -m mofpipe run    --project mof_no_agent_fixture/physical_case/recipe.json
python3 -m pytest
```

Each run directory holds `state.json` (authoritative workflow state), `events.jsonl`, content-addressed `artifacts/`, evaluator `jobs/`, `report.json` and `report.md`. Re-running with identical inputs reuses committed stages and never resubmits evaluator jobs.

| Stage | Module | Instruction task |
|---|---|---|
| intake, project validation | `intake.py`, `project_spec.py` | 01 |
| CIF parsing, slot lattice, framework builder | `cif.py`, `framework.py` | 01 |
| decoration space, symmetry, swap reachability | `structures.py` | 01, 04 |
| topology screen (identity mode) | `structures.py` | 02 |
| label ingest, dataset snapshots, frozen split | `datasets.py` | integrator glue |
| features, exact fit, model selection (fixture split; grouped CV + frozen holdout for physical projects) | `configuration_model.py` | 03 |
| QUBO / HUBO compilation and 2ⁿ identity check | `encoding.py` | 03 |
| exact enumeration, simulated annealing | `optimization.py` | 04 |
| symmetry dedup, top-k pool | `candidates.py` | 05 |
| evaluator boundary (mock backend as subprocess) | `electronic.py` | 06/07 boundary |
| periodic DFT (Quantum ESPRESSO adapter, jobs, convergence study) | `dft.py` | 06 |
| bounded active learning (off by default) | `refinement.py` | 08 |
| ranking, competition, structural checks | `ranking.py`, `report.py` | 09 |
| proposal validation, agents disabled | `gateway.py` | 10 |

See [IMPLEMENTATION.md](IMPLEMENTATION.md) for what each stage does and what every test checks, and [TODO.md](TODO.md) for every unimplemented part (`#TODO` markers in code) and what it requires.

Not implemented: PORMAKE/ToBaCCo builders, linker orientation search, multiple supercells, ML potentials, DFT codes other than Quantum ESPRESSO, embedding, FCI/VQE/SQD, QAOA, quadratization, MOFChecker/CrystalNets/Zeo++, PXRD/NMR, reference-phase competition, live literature review, runtime agents. These are reported as `SKIPPED_NO_ATOMIC_GEOMETRY`, `DISABLED_BY_CONFIG`, `NOT_RUN` or `INCOMPLETE`, never as passed.
