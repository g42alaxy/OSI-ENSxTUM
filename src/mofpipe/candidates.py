"""Task 05 (candidates): symmetry deduplication and candidate-pool selection.

Orbit key: ``topology_id:lexicographically_smallest_symmetry_image``; no A/B
label exchange. Deduplication runs separately within each graph.
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction

from .contract import PipelineError, bit_string, make_record
from .optimization import table_from_result
from .structures import canonical_bits


def deduplicate_candidates(spec, topo, solver_result: dict) -> dict:
    """deduplicate_candidates(candidate_ids, symmetry_policy_id) -> CandidateSet."""
    #TODO: Task 05 — relaxed-geometry equivalence (separate from decoration dedup) without merging distinct nets,
    #TODO: conformers or magnetic states; periodic-translation canonicalisation for supercells.
    orbits = defaultdict(list)
    for bits, energy in table_from_result(solver_result):
        orbits[bit_string(canonical_bits(topo, bits))].append((bits, energy))
    rows = []
    for image, members in orbits.items():
        energies = {e for _, e in members}
        if len(energies) != 1:
            raise PipelineError("SYMMETRY_ENERGY_MISMATCH", f"{topo.topology_id}:{image}")
        representative = min(bit_string(b) for b, _ in members)
        if representative != image:
            raise PipelineError("NONCANONICAL_REPRESENTATIVE", image)
        energy = energies.pop()
        rows.append({"candidate_id": f"{topo.topology_id}:{image}", "orbit_id": f"{topo.topology_id}:{image}",
                     "topology_id": topo.topology_id, "bits": [int(c) for c in image], "multiplicity": len(members),
                     "member_bitstrings": sorted(bit_string(b) for b, _ in members),
                     "predicted_base_energy_eV": float(energy), "predicted_base_energy_exact": str(energy)})
    rows.sort(key=lambda r: (Fraction(r["predicted_base_energy_exact"]), r["orbit_id"]))
    return make_record("CandidateSet", spec.provenance, topology_id=topo.topology_id,
                       solver_result_id=solver_result["artifact_id"], hamiltonian_id=solver_result["hamiltonian_id"],
                       symmetry_policy=spec.config["deduplication"], n_raw_states=solver_result["n_feasible"],
                       n_unique_orbits=len(rows), candidates=rows)


def select_pool(spec, candidate_set: dict) -> dict:
    """Best k unique orbits per graph, keeping every tie at the cutoff."""
    #TODO: Task 05 — select_evaluation_batch: declared energy/uncertainty/diversity acquisition with topology
    #TODO: coverage allocation, audit fraction for deprioritised regions and an evaluation budget.
    k = spec.config["candidate_pool"]["top_k_unique_orbits_per_topology"]
    rows = candidate_set["candidates"]
    energies = [Fraction(r["predicted_base_energy_exact"]) for r in rows]
    cutoff = energies[min(k, len(rows)) - 1]
    groups = {e: idx for idx, e in enumerate(sorted(set(energies)))}
    pool = []
    for rank, (row, energy) in enumerate(zip(rows, energies), start=1):
        if energy > cutoff:
            break
        n_deg = energies.count(energy)
        reason = "rank_within_top_k" if rank <= k else "tie_at_cutoff"
        pool.append({**{key: row[key] for key in ("candidate_id", "orbit_id", "topology_id", "bits", "multiplicity",
                                                  "predicted_base_energy_eV", "predicted_base_energy_exact")},
                     "surrogate_rank": rank, "degeneracy_group": groups[energy], "n_degenerate_orbits": n_deg,
                     "selection_reason": reason})
    return make_record("CandidatePool", spec.provenance, topology_id=candidate_set["topology_id"],
                       candidate_set_id=candidate_set["artifact_id"], hamiltonian_id=candidate_set["hamiltonian_id"],
                       policy={"top_k": k, "retain_all_score_ties": True, "score": "surrogate base energy"},
                       cutoff_energy_exact=str(cutoff), n_selected=len(pool), pool=pool)
