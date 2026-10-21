Task 07 — Embedded electronic corrections and solver benchmarking

Assignment
Implement src/electronic/embedding and src/electronic/solvers. Separate scientific embedding development from solver integration. This is a research milestone, not a generic API-wiring task.

Inputs
Validated candidate geometry and periodic calculation, explicit embedding scheme, active-space policy, basis/pseudopotential compatibility specification, treatment of magnetic states and all relevant inequivalent nodes, plus resource budgets.

Implement
1. Freeze a mathematically specified total-energy expression and an implementable embedding backend. Identify the environment, region(s), coupling terms, frozen/core contributions, boundary/capping terms if applicable, and the contribution replaced at high level.
2. Provide separate benchmark modes for an existing published HF-embedding workflow and a production WF-in-DFT protocol if both are implemented. HF-embedded correlation cannot simply be added to DFT. A finite-cluster subtractive model must be labelled as such and must include a boundary-convergence study before periodic ranking claims.
3. Export a reproducible active-space Hamiltonian: h1, h2, scalar constants, electron counts by spin, orbital ordering, integral convention, geometry/basis/embedding hashes and orbital character diagnostics. FCIDUMP plus a richer metadata record is suitable.
4. Use a balanced selection policy across candidates, preserving relevant metal, ligand and neighbouring-node orbital character. Do not force arbitrary identical orbital counts as a substitute for balanced treatment. Resolve all spin sectors relevant to the stated comparison.
5. Benchmark PySCF CASCI/FCI where feasible and simulated VQE on exactly the same Hamiltonian, orbitals and constants. Account for fermion-to-qubit mapping, bit order, symmetry tapering and state-preparation sector.
6. Add hardware SQD as a separate adapter: quantum sample collection, configuration recovery, subspace generation and classical diagonalization. Record hardware/circuit/shot provenance, classical subspace size, solver convergence and sample sensitivity. Larger active spaces require strong approximate classical references with stated convergence; never call them exact by default.
7. Implement the embedding scheme's own energy assembler. A schematic low(full) + high(region) - low(region) expression is only valid when the chosen formalism defines those terms consistently. Prevent overlapping-region double counting and include every inequivalent contribution required for the compared cell.
8. Test active-space and environment size, electronic solver convergence, alternative spin states and sensitivity of RELATIVE energies. Store numerical/sampling uncertainty separately from embedding/basis systematic sensitivity.
9. Emit corrected energies only when the assembler and comparison group are validated. Otherwise return embedded solver results as a benchmark artifact with correction_status=UNVALIDATED; retain a clearly labelled DFT-only ranking.

Proposed tools
prepare_active_space(candidate_id, embedding_protocol_id) -> ActiveSpaceArtifact
solve_active_space(active_space_id, solver_protocol_id, budget_id) -> Job
assemble_corrected_energy(candidate_id, embedding_protocol_id, result_ids) -> CorrectedEnergy or BlockedResult

Acceptance
Identical inputs reproduce identical integrals and constants. Tiny Hamiltonians match an independent exact diagonalization. VQE/SQD comparison uses identical electronic problems. A high=low consistency test is satisfied where required by the formalism; otherwise reproduce its documented consistency limit. An unchanged correction leaves ranking unchanged. Incompatible geometries, incomplete node coverage and missing subtraction terms block assembly. Benchmark at least one published or independently validated embedding reference.

Agent/API use
The electronic agent may propose a larger space or diagnosed recovery action from approved policies. It cannot invent the embedding expression or declare convergence from one favourable result. Missing scientific specification must return BLOCKED with the unresolved choice.

