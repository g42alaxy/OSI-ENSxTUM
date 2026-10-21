Task 06 — Comparable periodic DFT evaluation

Assignment
Implement src/electronic/periodic and asynchronous calculation adapters. AiiDA plus aiida-quantumespresso is a reasonable open-source starting point; retain an adapter interface for existing licensed software.

Inputs
AcquisitionBatch, original geometry/mapping, approved DFT protocol, magnetic-state policy and resource budget.

Implement
1. Submit consistent cell/atomic relaxation and final single-point calculations. Record XC functional, dispersion treatment, pseudopotential versions, basis/cutoff, k-point density, smearing and zero-smearing energy convention, spin initialisation and any Hubbard parameters.
2. Converge settings for relevant ENERGY DIFFERENCES rather than adopting a universal numerical tolerance. Define the target tolerance in the common normalized units before screening claims.
3. Test the declared magnetic/structural initial states. Store each metastable outcome separately. Choosing the lowest found state does not certify a global minimum.
4. Check electronic and ionic convergence, maximum residual forces/stress, inventory, topology and reference-slot mapping after relaxation. Do not constrain a calculation to retain a net without declaring that modelling choice.
5. Preserve topology changes as transformation events; do not train the original topology model on them. Candidate reassignment requires a valid new mapping and compatible comparison group.
6. Normalize to a verified formula-unit count and define comparison groups by full composition and protocol. Distinguish a single-point energy on a DFT geometry from a structure relaxed with a higher-level method.
7. Persist all raw input/output and recovery history. Automatic retries may change numerical convergence settings inside a declared protocol policy. Changes to functional, U, charge, pseudopotential or physical state create a distinct scientific protocol.
8. Return uncertainty/convergence diagnostics and explicit blocked/failed states.

Proposed tools
submit_periodic_dft(candidate_ids, dft_protocol_id, budget_id) -> JobBatch
collect_periodic_dft(job_id) -> CalculationResult

Acceptance
Equivalent representations give matching normalized energies. Primitive and replicated cell comparisons converge appropriately with consistent sampling. Deliberately unconverged and composition-mismatched cases cannot enter rankings. Job restart does not duplicate successful work.

Agent/API use
The electronic specialist can interpret SCF failures and choose approved recovery options. No agent should silently alter a scientific approximation to obtain convergence.

