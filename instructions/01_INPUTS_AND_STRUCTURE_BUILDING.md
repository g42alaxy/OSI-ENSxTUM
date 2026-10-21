Task 01 — Inputs and framework construction

Assignment
Implement src/intake and src/structures. Deliver a strict project validator, building-block registry and reproducible builder adapter. Read the shared contract first.

Inputs
User-supplied CIFs or periodic nets, atomic building blocks with connection points, exact target composition, specified oxidation/protonation/solvent conditions and permitted supercell matrices.

Implement
1. Parse structures while retaining original files and provenance. Identify unresolved partial occupancies and disorder; require a declared resolution policy rather than filling sites silently.
2. Create a stable slot representation for linker and metal/node substitutions. A linker may occupy a net edge or a node depending on the chosen net representation. Preserve the correspondence explicitly.
3. Check connector count/geometry, allowed slot species, atom inventory and charge balance. Use explicitly supplied oxidation/protonation assumptions; uncertain chemical inference must be flagged.
4. Derive integer species counts for each permitted supercell. Reject nonintegral or infeasible counts. Store counts separately by sublattice when chemistry requires this.
5. Wrap PORMAKE first; add ToBaCCo behind the same project adapter if needed. Build alternative orientations from a finite declared catalogue. Do not assume functionalised linkers are interchangeable without orientation states.
6. Export candidate CIF plus the atom/block/slot mapping and periodic adjacency. Canonicalize input identity separately from decorated-structure symmetry.
7. Run early sanity checks and return construction failures with reasons, without treating them as calculated energetic instability.

Proposed tools
validate_project(project_spec) -> ValidationReport
build_framework(topology_id, decoration, builder_protocol_id) -> Geometry or BuildFailure

Artifacts
Validated ProjectSpec, BuildingBlock and TopologySpec records; built geometry files; reference site mappings; construction log.

Acceptance
A small known parent can be rebuilt with correct inventory/connectivity. Every legal decoration preserves required counts. An invalid connector or impossible ratio is caught before calculation. The same inputs reproduce equivalent output and a stable mapping under atom reordering.

Agent/API use
The LLM may suggest alternative orientations or builder protocols from the registered catalogue after a failure. It cannot assert successful construction, change target composition, or invent missing building-block chemistry.

