"""Tasks 01/03 (T01.5): supercell transfer checks of the working-cell configurational model.

Every topology is enumerated, fitted and searched in its working cell; further declared supercells are
transfer cells (``<topology_id>@AxBxC``). Two checks use them:

  replication  A working-cell configuration replicated into a transfer cell must have identical feature values
               per formula unit (exact), map symmetry-equivalent configurations to one transfer-cell orbit, and,
               after DFT, the same normalised energy within the target tolerance (numerical consistency of the
               k-point sampling and normalisation).
  transfer     Configurations native to the transfer cell (orderings the working cell cannot represent) are
               evaluated with DFT and compared with the working-cell model's per-formula-unit predictions.

A failed check is reported (TRANSFER_VALIDATION_FAILED), never tuned away.
"""
from __future__ import annotations

import math
import random
from fractions import Fraction

from .configuration_model import feature_vector, predict
from .contract import PipelineError, StageStatus, bit_string, make_record
from .structures import canonical_bits, n_ones_of, orbit_id

KEY_SCALE = 1000


def _key(slot_type, slot: dict, diag, period) -> tuple:
    """Slot position in parent fractional coordinates (integer 1e-3 units) modulo the working-cell period."""
    return (slot_type,) + tuple(int(round(slot["centroid_frac"][k] * diag[k] * KEY_SCALE)) % (period[k] * KEY_SCALE)
                                for k in range(3))


def slot_map(working: dict, transfer: dict) -> list | None:
    """For every transfer-cell slot, the working-cell slot it replicates (None when the cell is no replica)."""
    w_diag, t_diag = working["supercell_diagonal"], transfer["supercell_diagonal"]
    if any(t % w for t, w in zip(t_diag, w_diag)):
        return None
    index = {_key(s["slot_type"], s, w_diag, w_diag): s["slot_id"] for s in working["slots"]}
    out = []
    for s in transfer["slots"]:
        match = index.get(_key(s["slot_type"], s, t_diag, w_diag))
        if match is None:
            raise PipelineError("TRANSFER_SLOT_MAPPING_FAILED", f"transfer slot {s['slot_id']}")
        out.append(match)
    return out


def replicate(bits, mapping: list) -> list:
    return [bits[j] for j in mapping]


def is_replica(t_bits, mapping: list) -> bool:
    seen = {}
    return all(seen.setdefault(j, b) == b for b, j in zip(t_bits, mapping))


def replication_report(spec, topo, transfer_topo, mapping: list, features: list, states: list) -> dict:
    """Exact (no DFT) replication test: identical features per formula unit and consistent orbit mapping."""
    mismatches, orbit_map = [], {}
    for bits in states:
        t_bits = replicate(bits, mapping)
        if sum(t_bits) != n_ones_of(spec, transfer_topo):
            raise PipelineError("TRANSFER_COMPOSITION_MISMATCH", f"{transfer_topo.topology_id}: {sum(t_bits)} ones")
        if feature_vector(topo, bits, features) != feature_vector(transfer_topo, t_bits, features):
            mismatches.append(bit_string(bits))
        orbit_map.setdefault(orbit_id(topo, bits), set()).add(bit_string(canonical_bits(transfer_topo, t_bits)))
    split = {o: sorted(t) for o, t in orbit_map.items() if len(t) > 1}
    return {"transfer_topology_id": transfer_topo.topology_id, "n_states": len(states), "features": features,
            "feature_mismatches": mismatches, "orbits_split_by_replication": split,
            "status": StageStatus.COMPLETED.value if not mismatches and not split else "REPLICATION_INCONSISTENT"}


def native_states(spec, transfer_topo, mapping: list, n: int, seed) -> list:
    """Seeded feasible transfer-cell states that are not replicas of any working-cell state, one per orbit."""
    rng = random.Random(f"{seed}:{transfer_topo.topology_id}")
    k, size = n_ones_of(spec, transfer_topo), transfer_topo.n_sites
    out, orbits = [], set()
    for _ in range(200 * max(n, 1)):
        if len(out) >= n:
            break
        ones = set(rng.sample(range(size), k))
        bits = [int(i in ones) for i in range(size)]
        oid = orbit_id(transfer_topo, bits)
        if oid in orbits or (mapping is not None and is_replica(bits, mapping)):
            continue
        orbits.add(oid)
        out.append(bits)
    return out


def transfer_report(spec, topo, transfer_topo, hamiltonian: dict, replicated: list, native: list) -> dict:
    """SupercellTransferReport from DFT results: replication energy consistency and native-state prediction error."""
    tol = Fraction(str(spec.physical["tolerances"]["target_relative_energy_tolerance"]))
    rep_rows, nat_rows = [], []
    for row in replicated:  # {"working_result", "transfer_result"}
        w, t = row["working_result"], row["transfer_result"]
        ok = w["status"] == t["status"] == "SUCCEEDED"
        diff = abs(Fraction(t["base_energy_exact"]) - Fraction(w["base_energy_exact"])) if ok else None
        rep_rows.append({"working_configuration_id": w["configuration_id"], "transfer_configuration_id": t["configuration_id"],
                         "status": t["status"] if not ok else "SUCCEEDED",
                         "abs_difference_eV_per_fu": float(diff) if diff is not None else None,
                         "within_tolerance": diff is not None and diff <= tol})
    residuals = []
    for res in native:
        if res["status"] != "SUCCEEDED":
            nat_rows.append({"configuration_id": res["configuration_id"], "status": res["status"],
                             "failure": res.get("failure")})
            continue
        pred = predict(hamiltonian, transfer_topo, res["bits"])
        resid = pred - Fraction(res["base_energy_exact"])
        residuals.append(resid)
        nat_rows.append({"configuration_id": res["configuration_id"], "status": "SUCCEEDED",
                         "dft_eV_per_fu": res["base_energy_eV"], "predicted_eV_per_fu": float(pred),
                         "residual_eV_per_fu": float(resid), "calculation_result_id": res["artifact_id"]})
    rmse = math.sqrt(float(sum(r * r for r in residuals) / len(residuals))) if residuals else None
    replication_ok = bool(rep_rows) and all(r["within_tolerance"] for r in rep_rows)
    transfer_ok = rmse is not None and rmse <= float(tol)
    missing = [r for r in rep_rows + nat_rows if r["status"] != "SUCCEEDED"]
    status = (StageStatus.COMPLETED.value if replication_ok and transfer_ok and not missing else
              StageStatus.INCOMPLETE.value if missing and (replication_ok or not rep_rows) else "TRANSFER_VALIDATION_FAILED")
    return make_record(
        "SupercellTransferReport", spec.provenance, topology_id=topo.topology_id,
        transfer_topology_id=transfer_topo.topology_id, working_supercell=list(topo.supercell),
        transfer_supercell=list(transfer_topo.supercell), hamiltonian_id=hamiltonian["artifact_id"],
        tolerance_eV=float(tol), replication=rep_rows, native=nat_rows, transfer_rmse_eV=rmse, status=status,
        note="native transfer-cell states test orderings the working cell cannot represent; a failure means the "
             "working-cell model or cell is too small for its stated validity domain")
