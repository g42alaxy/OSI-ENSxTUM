"""Task 01 (structures): occupations, symmetry canonicalisation, decoration space.

In graph mode a Decoration is a binary occupation of substitution slots. Atomic
framework construction is reported as SKIPPED_NO_ATOMIC_GEOMETRY.
"""
from __future__ import annotations

from collections import Counter, deque
from itertools import combinations

from .contract import PipelineError, StageStatus, bit_string, make_record


def n_ones_of(spec, topology) -> int:
    """Sites carrying encoding 1 in this topology's cell (fixture: the project-wide count)."""
    return topology.n_ones if topology.n_ones is not None else spec.n_B


def composition_of(spec, topology) -> dict:
    return dict(topology.composition) if topology.composition else dict(spec.composition_counts)


def states_of(spec, topology):
    states = feasible_states(topology.n_sites, n_ones_of(spec, topology))
    if not topology.sublattice_counts:
        return states
    return (b for b in states if all(sum(b[i] for i in sites) == k for sites, k in topology.sublattice_counts))


def require_single_count(topology, solver: str) -> None:
    """Solvers whose moves or bounds assume one global count refuse per-sublattice constraints."""
    if topology.sublattice_counts:
        raise PipelineError("UNSUPPORTED_SUBLATTICE_CONSTRAINTS",
                            f"{solver} assumes one global count; {topology.topology_id} declares {len(topology.sublattice_counts)}")


def validate_occupation(bits, n_sites: int, n_ones: int) -> tuple:
    """Validate length, strict integer binary values (bool rejected) and composition."""
    if not isinstance(bits, (list, tuple)) or len(bits) != n_sites:
        raise PipelineError("INVALID_SITE_COUNT", f"expected {n_sites}")
    if any(type(x) is not int or x not in (0, 1) for x in bits):
        raise PipelineError("INVALID_BINARY_OCCUPATION", str(bits))
    if sum(bits) != n_ones:
        raise PipelineError("INVALID_COMPOSITION", f"{sum(bits)} ones, expected {n_ones}")
    return tuple(bits)


def symmetry_image(perm, bits) -> tuple:
    # Pullback convention: image[i] = bits[perm[i]].
    return tuple(bits[j] for j in perm)


def canonical_bits(topology, bits) -> tuple:
    return min(symmetry_image(p, bits) for p in topology.permutations)


def orbit_id(topology, bits) -> str:
    return f"{topology.topology_id}:{bit_string(canonical_bits(topology, bits))}"


def configuration_id(topology, bits) -> str:
    return f"{topology.topology_id}:{bit_string(bits)}"


def feasible_states(n_sites: int, n_ones: int):
    for chosen in combinations(range(n_sites), n_ones):
        s = set(chosen)
        yield tuple(int(i in s) for i in range(n_sites))


def decode_species(bits, species_encoding: dict) -> list:
    inverse = {v: k for k, v in species_encoding.items()}
    return [inverse[b] for b in bits]


def make_decoration(spec, topology, bits) -> dict:
    #TODO: Task 01 — orientation states for asymmetric linkers (finite declared catalogue) as extra decoration variables.
    bits = validate_occupation(list(bits), topology.n_sites, n_ones_of(spec, topology))
    species = decode_species(bits, spec.species_encoding)
    return {
        "record_type": "Decoration", "topology_id": topology.topology_id,
        "configuration_id": configuration_id(topology, bits), "orbit_id": orbit_id(topology, bits),
        "bits": list(bits), "species_by_slot": species, "inventory": dict(sorted(Counter(species).items())),
        "provenance": spec.provenance,
    }


def define_decoration_space(spec, topology) -> dict:
    """Integrator glue tool: the feasible fixed-composition decoration space."""
    # Transfer supercells are separate TopologySpecs (project_spec._load_source, transfer.py).
    #TODO: Tasks 01/05 — non-diagonal supercell matrices.
    orbits = Counter(orbit_id(topology, b) for b in states_of(spec, topology))
    return make_record(
        "DecorationSpace", spec.provenance, topology_id=topology.topology_id, n_sites=topology.n_sites,
        composition=composition_of(spec, topology), encoding=spec.species_encoding,
        bit_order="bits[i] is site i; string leftmost character is site 0",
        n_feasible=sum(orbits.values()), n_symmetry_orbits=len(orbits),
        orbit_sizes=dict(sorted(orbits.items())),
        symmetry_policy={"operations": "explicit site permutations", "n_operations": len(topology.permutations),
                         "species_label_exchange": False},
    )


def check_candidate_file(spec, spaces: dict) -> dict:
    """Cross-check data/candidates_unlabelled.jsonl against the computed decoration spaces."""
    seen = set()
    per_topology = Counter()
    for row in spec.candidate_rows:
        topo = spec.topologies.get(row.get("topology_id"))
        if topo is None:
            raise PipelineError("UNKNOWN_TOPOLOGY", str(row.get("topology_id")))
        bits = validate_occupation(row["bits"], topo.n_sites, n_ones_of(spec, topo))
        if row["configuration_id"] != configuration_id(topo, bits):
            raise PipelineError("CONFIGURATION_ID_MISMATCH", row["configuration_id"])
        if row["orbit_id"] != orbit_id(topo, bits):
            raise PipelineError("ORBIT_ID_MISMATCH", row["configuration_id"])
        if row["configuration_id"] in seen:
            raise PipelineError("DUPLICATE_CANDIDATE", row["configuration_id"])
        seen.add(row["configuration_id"])
        per_topology[topo.topology_id] += 1
    for tid, space in spaces.items():
        if per_topology[tid] != space["n_feasible"]:
            raise PipelineError("CANDIDATE_FILE_INCOMPLETE", f"{tid}: {per_topology[tid]} != {space['n_feasible']}")
    return {"n_rows": len(spec.candidate_rows), "per_topology": dict(per_topology), "consistent": True}


def swap_graph_connected(topology, n_sites: int, n_ones: int) -> dict:
    """Is every fixed-count state reachable by exchanges along graph edges? (XY-mixer prerequisite)."""
    states = list(feasible_states(n_sites, n_ones))
    seen = {states[0]}
    queue = deque(seen)
    while queue:
        b = queue.popleft()
        for i, j in topology.pairs_d1:
            if b[i] != b[j]:
                moved = list(b)
                moved[i], moved[j] = moved[j], moved[i]
                moved = tuple(moved)
                if moved not in seen:
                    seen.add(moved)
                    queue.append(moved)
    return {"n_feasible": len(states), "n_reached": len(seen), "connected": len(seen) == len(states),
            "exchange_graph": "pairs_distance_1"}


def build_framework(spec, topology, decoration: dict) -> dict:
    """build_framework for graph mode: no atomic building blocks exist, so no geometry is fabricated.

    Physical projects use framework.build_framework (parent-substitution builder).
    """
    return {"record_type": "BuildResult", "topology_id": topology.topology_id,
            "decoration": decoration["configuration_id"], "geometry": None,
            "status": StageStatus.SKIPPED_NO_ATOMIC_GEOMETRY.value,
            "reason": "finite_substitution_interaction_graph mode; no CIF or building-block geometry supplied"}


def screen_topologies(spec) -> dict:
    """Task 02 in identity mode: no physical screening evidence, so every graph is retained.

    Physical projects that declare ``protocols.ml_potential_checkpoint`` use the ML-potential screen (screening.py).
    """
    #TODO: Task 01 — orientation diversity in the screening seeds needs linker orientation states (T01.4).
    screen = spec.config["topology_screen"]
    decisions = [{"topology_id": tid, "state": "retain", "evidence": "NO_PHYSICAL_SCREENING_EVIDENCE"}
                 for tid in spec.topologies if tid in screen["retain"]]
    skipped = StageStatus.SKIPPED_NO_ATOMIC_GEOMETRY.value
    return make_record("ScreeningReport", spec.provenance, mode=screen["mode"], decisions=decisions,
                       geometric_screen=screen.get("geometric_screen", skipped),
                       ml_potential=skipped if spec.physical is None else StageStatus.DISABLED_BY_CONFIG.value,
                       exclusion_ledger=[], retained=[d["topology_id"] for d in decisions])
