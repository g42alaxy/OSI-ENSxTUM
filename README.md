# mofpipe: quantum-enhanced modelling of multivariate MOFs

QuMOF predicts how linkers arrange in multivariate metal–organic frameworks (MTV-MOFs), starting from a plain-text design brief. LLM agents choose topologies and Hamiltonian forms, a DFT-fitted QUBO is solved by simulated annealing and composition-preserving QAOA, and the best structures are ranked by periodic DFT with quantum-embedded corrections at the metal nodes.

## Approach

MTV-MOFs carry two or more linkers on one framework, and their properties depend on how the linkers are arranged, which experiments rarely resolve. Kang et al. (ACS Cent. Sci. 2025) showed this arrangement can be written as a QUBO, but with Hamiltonian weights chosen by hand for each material. QuMOF automates the full loop and grounds it in computed energies:

1. **Topology from text.** The user provides a design brief and a project file (metal nodes, linkers, target ratio). Missing fields are reported, never invented. Agent A1 picks candidate nets, frameworks are built from CIFs, and an ML interatomic potential screens out structurally incompatible topologies.
2. **Molecule encoding.** Each variable linker site $i$ becomes a binary variable, $x_i = 1$ if it holds linker B, so an arrangement is a bit string $\mathbf{x} \in \lbrace 0,1 \rbrace^N$ at fixed composition $\sum_i x_i = N_B$.
3. **QUBO Hamiltonian.** Agent A2 chooses the model form (point and pair cluster features, encoding, QUBO or HUBO). Code fits the coefficients to estimated DFT energies with grouped cross-validation and compiles
    
    $$
    H(\mathbf{x}) = \sum_i h_i x_i + \sum_{i<j} J_{ij}\, x_i x_j + \lambda \Big( \sum_i x_i - N_B \Big)^2 .
    $$
    
4. **Solve.** Simulated annealing and composition-preserving QAOA are benchmarked against minima certified by exact enumeration, as a statevector simulation and as noisy sampling on Qiskit's FakeBrisbane device.
5. **Analyze.** The best arrangements are relaxed and ranked with periodic DFT (which can be replaced with just VQE/SQD). Each metal node for the ranking gets a high-level correction through subtractive embedding, from CCSD (benchmark) or an active-space solver (CASCI, VQE, SQD). Agent A3 checks the top candidates against structural tests, the CoRE MOF 2024 and QMOF databases and the literature. If the energy model fails validation, agent A4 sends the run back to step 3 with a new Hamiltonian.

Every run is reproducible: immutable artifacts, an event log, and a deterministic mode with the agents switched off. The whole pipeline was built during the hackathon.

## Results

The QuMOF agentic pipeline successfully automates the discovery and optimization of MTV-MOF arrangements, demonstrated here on MUF-7 structures. As detailed in the accompanying charts:

1. Hamiltonian Generation (Left Chart): The pipeline accurately approximates the QUBO Hamiltonian for the given MUF-7 configurations, successfully replacing manual weight assignment with an automated, data-driven approach.

1. QAOA Optimization (Second & Third Charts): Solving the QUBO formulation via QAOA identifies highly optimal linker arrangements. Notably, QuMOF achieved lower pre-relaxation energies than the manually-derived Hamiltonians from the baseline literature.

1. Energy Minimization & Reconstruction (Fourth Chart): Following VQE-driven energy minimization, the pipeline exactly reconstructs the true minimal energy states. This exact reconstruction highlights the efficacy of the agentic loop (driven here by Qwen 3.8 models) in guiding the quantum solvers to the global minimum.

![](https://moment-blaiszik.s3.us-east-1.amazonaws.com/cf902999-8583-4930-ba95-15ef6f7f343f.png)


## Next steps

- Run quantum parts on actual quantum hardware for open-shell metal nodes, and scale to larger frameworks and three or more linker types
- Further fine-tuning of implemented AI agents and speedup of optimization loop
- New implementations, such as predictions of finite-temperature arrangement ensembles for comparison with NMR, and addition of a property-driven loop
- Release the fitted Hamiltonians as benchmark instances. Because the QUBO and node Hamiltonians are solver-agnostic, the pipeline can move to stronger quantum hardware without rebuilding