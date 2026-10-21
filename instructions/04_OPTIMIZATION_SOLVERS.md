Task 04 — Classical and quantum arrangement solvers

Assignment
Implement src/optimization with interchangeable classical and quantum adapters, common decoding and reproducible benchmarking.

Inputs
Immutable physical Hamiltonian, feasible decoration space, compiled encoding and explicit solver budgets.

Implement
1. Implement exhaustive feasible enumeration for tiny cells and an exact mixed-integer/constraint solver where practical. Report optimality gaps/timeouts honestly. If integer scaling is required, bound coefficient-rounding error.
2. Implement multi-start simulated annealing using composition-preserving swaps and a tuned schedule. Include a strong additional baseline when justified by problem size; preserve feasibility throughout or account for rejection cost.
3. Implement QAOA simulation first. Binary fixed-composition instances use a connected XY exchange graph and feasible initial states. Verify connectivity of the feasible transition graph on small examples.
4. For multicomponent one-hot states, implement exchanges that preserve both one species per site and global species counts. Do not apply independent site/species XY mixers and assume both constraints survive.
5. If domain-wall or ancilla encodings are used, verify their feasible dynamics separately. A documented penalty-based alternative is allowed, but feasibility loss and resource overhead must be reported.
6. Record mapping/endianness, circuit depth, hardware connectivity, optimizer calls, shots, initialisation, postselection/repair rate, seeds, wall time and classical overhead. Add a hardware adapter only after simulator acceptance.
7. Decode every sample and reevaluate its physical cost independently of the encoded objective. Return diverse feasible solutions, not only an expectation value.
8. Benchmark both solver-only and end-to-end cost. Report best/median quality and success probabilities across seeds. Quantum simulation is not evidence of quantum speedup.

Proposed tool
solve_arrangements(hamiltonian_id, encoding_id, solver_protocol_id, budget_id) -> Job or SolverResult

Acceptance
All methods agree with exact minima on selected small fixtures within their stated success criteria. Mixer evolution preserves each declared constraint. Decode/encode round trips pass. Ancilla or bit-order errors are caught. A timeout returns the best feasible candidate and an uncertified status.

Agent/API use
An optimization agent may allocate approved solver configurations to instance families and diagnose failures. The scheduler controls budgets and hardware submission; no LLM call is needed per optimizer iteration.

