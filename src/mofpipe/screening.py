"""Task 02: conservative ML-potential topology screening (T02.1-T02.3).

generate_seed_decorations(topology_id, sampling_policy_id) -> DecorationSet
submit_relaxations(decoration_ids, calculator_protocol_id) -> ML relaxation results (mlip.MLPotentialBackend)
screen_topologies(result_ids, screening_policy_id) -> ScreeningReport

Seeds per topology: pure-linker endpoints (matched references for mixing energies), the most symmetric ordered
states, seeded random states and SQS-like states (unlike-pair count closest to the random expectation), all at the
project's fixed composition except the endpoints, each tagged with its sampling method. Every seed is built and
relaxed with the same ML protocol; a failed or unconverged relaxation stays a failure and a relaxation that changes
the slot/node topology is a TRANSFORMED candidate, never an energy for the original lattice.

Energies are compared only at matched inventories: fixed-composition seeds share the common formula unit, so
their energies per formula unit are comparable across topologies; mixing energies use each topology's own
endpoints. The ML ranking is exploratory until ``protocols.ml_calibration`` measures ML-vs-DFT errors of energy
differences on representative seeds within tolerance. Decisions: reject only on structural incompatibility
(no seed builds, or every relaxed seed transforms); deprioritise only with a calibrated potential and a declared
``energetic_policy``, by a margin of the threshold plus twice the calibration error (sampled minima are upper
bounds, not the topology's minimum); uncertain topologies are kept. ``reinstate`` brings any topology back.
``energetic_policy.allow_exploratory: true`` is an explicit opt-in to deprioritise on the uncalibrated ML gap (every
seed relaxed, no domain flags); it is labelled EXPLORATORY_ML_GAP and stays auditable and reinstatable.
"""
from __future__ import annotations

import random
from collections import Counter
from fractions import Fraction
from itertools import combinations

from . import framework, structures
from .configuration_model import feature_value
from .contract import StageStatus, make_record
from .mlip import MLPotentialBackend
from .validation import _lengths_angles, _structure

DEFAULT_SAMPLING = {"include_endpoints": True, "n_ordered": 2, "n_random": 3, "n_sqs": 1, "seed": 0}
DEFAULT_MAX_CELL_STRAIN_PERCENT = 10.0
KEPT_STATES = ("retain", "uncertain")


def policy_errors(screening, calibration) -> list:
    """(code, field) for protocols.screening and protocols.ml_calibration (both optional)."""
    errors = []
    if screening is not None:
        sampling = {**DEFAULT_SAMPLING, **(screening.get("sampling") or {})}
        if not all(type(sampling[k]) is int and sampling[k] >= 0 for k in ("n_ordered", "n_random", "n_sqs", "seed")):
            errors.append(("SCIENTIFIC_SPEC_MISSING", "protocols.screening.sampling (non-negative integers)"))
        energetic = screening.get("energetic_policy")
        if energetic is not None and not isinstance(energetic.get("deprioritise_above_eV_per_formula_unit"), (int, float)):
            errors.append(("SCIENTIFIC_SPEC_MISSING", "protocols.screening.energetic_policy.deprioritise_above_eV_per_formula_unit"))
    if calibration is not None:
        if type(calibration.get("n_structures_per_topology")) is not int or calibration["n_structures_per_topology"] < 2:
            errors.append(("SCIENTIFIC_SPEC_MISSING", "protocols.ml_calibration.n_structures_per_topology (>= 2)"))
        if not isinstance(calibration.get("tolerance_eV_per_formula_unit"), (int, float)):
            errors.append(("SCIENTIFIC_SPEC_MISSING", "protocols.ml_calibration.tolerance_eV_per_formula_unit"))
    return errors


def _endpoint(spec, topo, species: str) -> dict:
    """Pure-linker decoration (every slot one species): a matched reference, not a fixed-composition state."""
    code = spec.species_encoding[species]
    return {"record_type": "Decoration", "topology_id": topo.topology_id, "sampling_method": "pure_endpoint",
            "configuration_id": f"{topo.topology_id}:pure_{species}", "orbit_id": f"{topo.topology_id}:pure_{species}",
            "bits": [code] * topo.n_sites, "species_by_slot": [species] * topo.n_sites,
            "inventory": {species: topo.n_sites}, "provenance": spec.provenance}


def generate_seed_decorations(spec, topo, sampling: dict) -> dict:
    sampling = {**DEFAULT_SAMPLING, **(sampling or {})}
    reps, sizes = {}, Counter()
    for bits in structures.states_of(spec, topo):
        orbit = structures.orbit_id(topo, bits)
        sizes[orbit] += 1
        reps.setdefault(orbit, bits)
    chosen, seeds = set(), []

    def take(orbit, method):
        chosen.add(orbit)
        seeds.append({**structures.make_decoration(spec, topo, reps[orbit]), "sampling_method": method})

    for orbit in sorted(reps, key=lambda o: (sizes[o], o))[: sampling["n_ordered"]]:
        take(orbit, "ordered_high_symmetry")
    rng = random.Random(f"{sampling['seed']}:{topo.topology_id}")
    for orbit in rng.sample(sorted(set(reps) - chosen), min(sampling["n_random"], len(set(reps) - chosen))):
        take(orbit, "random")
    n, k = topo.n_sites, structures.n_ones_of(spec, topo)
    weights = topo.pair_weights_d1 or (1,) * len(topo.pairs_d1)
    expected = sum(weights) * 2 * k * (n - k) / (n * (n - 1))  # unlike nearest-neighbour pairs of a random state
    remaining = sorted(set(reps) - chosen, key=lambda o: (abs(feature_value("unlike_distance_1", topo, reps[o]) - expected), o))
    for orbit in remaining[: sampling["n_sqs"]]:
        take(orbit, "sqs_like")
    if sampling["include_endpoints"]:
        allowed = set.intersection(*(set(s) for s in topo.allowed_species))
        seeds += [_endpoint(spec, topo, sp) for sp in sorted(allowed)]
    return make_record("DecorationSet", spec.provenance, topology_id=topo.topology_id, sampling_policy=sampling,
                       decorations=seeds, orientation_sampling="NOT_REPRESENTED (T01.4)",
                       methods=dict(Counter(s["sampling_method"] for s in seeds)))


def _relaxation_record(spec, topo, deco, geometry, result, max_strain) -> dict:
    """MLRelaxationResult: status, per-formula-unit energy and the convergence/topology/cell checks."""
    common = dict(topology_id=topo.topology_id, configuration_id=deco["configuration_id"], orbit_id=deco["orbit_id"],
                  sampling_method=deco["sampling_method"], inventory=deco["inventory"], geometry_id=geometry["artifact_id"],
                  fixed_composition=deco["sampling_method"] != "pure_endpoint", n_formula_units=topo.n_formula_units)
    if geometry["status"] != StageStatus.COMPLETED.value:
        return make_record("MLRelaxationResult", spec.provenance, **common, status="BUILD_FAILED", failure=geometry["failure"])
    if result["status"] != "SUCCEEDED":
        return make_record("MLRelaxationResult", spec.provenance, **common, status=StageStatus.FAILED.value,
                           job_id=result.get("job_id"), fidelity=result.get("fidelity"),
                           failure={"code": "ML_RELAXATION_FAILED", "detail": result.get("error")})
    builder = spec.physical["protocols"]["builder"]
    mapping = framework.map_to_parent_slots(_structure(result["relaxed"]), spec.physical["builder_info"][topo.topology_id],
                                            builder["bond_tolerance_A"], builder["slot_match_tolerance_A"])
    (l0, a0), (l1, a1) = _lengths_angles(geometry["lattice_A"]), _lengths_angles(result["relaxed"]["lattice_A"])
    strain = max([abs(100 * (b - a) / a) for a, b in zip(l0, l1)] + [0.0])
    checks = {"converged": result["converged"], "topology_retained": mapping["retained"],
              "topology_change_reason": mapping["reason"], "max_cell_length_change_percent": round(strain, 3),
              "max_angle_change_deg": round(max(abs(b - a) for a, b in zip(a0, a1)), 3),
              "max_force_eV_per_A": result["max_force_eV_per_A"], "n_steps": result["n_steps"]}
    status = ("NOT_CONVERGED" if not result["converged"] else "TRANSFORMED" if not mapping["retained"]
              else "CELL_DISTORTED" if strain > max_strain else "SUCCEEDED")
    energy = Fraction(repr(result["energy_eV"])) / topo.n_formula_units
    return make_record("MLRelaxationResult", spec.provenance, **common, status=status, job_id=result["job_id"],
                       fidelity=result["fidelity"], engine_identity=result["identity"], checks=checks,
                       energy_per_formula_unit_eV=float(energy) if status == "SUCCEEDED" else None,
                       relaxed_geometry=result["relaxed"], wall_time_s=result.get("wall_time_s"),
                       note="ML fidelity: a screening estimate, never a DFT training label")


def _mixing(records: list) -> dict:
    """Mixing energy per formula unit against this topology's own relaxed pure endpoints (matched references)."""
    ends = {next(iter(r["inventory"])): r["energy_per_formula_unit_eV"] for r in records
            if not r["fixed_composition"] and r["status"] == "SUCCEEDED"}
    if len(ends) != 2:
        return {}
    out = {}
    for r in records:
        if r["fixed_composition"] and r["status"] == "SUCCEEDED":
            total = sum(r["inventory"].values())
            out[r["configuration_id"]] = r["energy_per_formula_unit_eV"] - sum(
                n / total * ends[sp] for sp, n in r["inventory"].items())
    return out


def calibrate(spec, records: dict, evaluate, calibration: dict) -> dict:
    """ML-vs-DFT errors of energy DIFFERENCES on representative fixed-composition seeds (lowest/highest ML energy)."""
    picked = []
    for tid, recs in records.items():
        ok = sorted((r for r in recs if r["fixed_composition"] and r["status"] == "SUCCEEDED"),
                    key=lambda r: (r["energy_per_formula_unit_eV"], r["configuration_id"]))
        n = calibration["n_structures_per_topology"]
        idx = sorted({round(i * (len(ok) - 1) / (n - 1)) for i in range(n)}) if len(ok) >= 2 else range(len(ok))
        picked += [(tid, ok[i]) for i in idx]
    dft_energy, dft_records = {}, {}
    for tid in records:
        decos = [structures.make_decoration(spec, spec.topologies[tid], [int(c) for c in r["configuration_id"].split(":")[1]])
                 for t, r in picked if t == tid]
        if not decos:
            continue
        recs, results = evaluate(spec.topologies[tid], decos)
        dft_records.update(recs)
        for res in results:
            if res["status"] == "SUCCEEDED":
                dft_energy[res["decoration"]["configuration_id"]] = Fraction(res["selected"]["base_energy_exact"])
    ml = {r["configuration_id"]: Fraction(repr(r["energy_per_formula_unit_eV"])) for _, r in picked}
    both = sorted(set(ml) & set(dft_energy))
    errors, cross = [], []
    for a, b in combinations(both, 2):
        err = abs(float((ml[a] - ml[b]) - (dft_energy[a] - dft_energy[b])))
        errors.append(err)
        if a.split(":")[0] != b.split(":")[0]:
            cross.append(err)
    tol = calibration["tolerance_eV_per_formula_unit"]
    mae = sum(errors) / len(errors) if errors else None
    status = ("NOT_CALIBRATED" if not errors else "CALIBRATED" if mae <= tol and (not cross or max(cross) <= 2 * tol)
              else "CALIBRATION_FAILED")
    return {"status": status, "n_structures": len(both), "n_pairs": len(errors), "mae_eV_per_formula_unit": mae,
            "max_abs_error_eV_per_formula_unit": max(errors) if errors else None,
            "cross_topology_max_abs_error_eV_per_formula_unit": max(cross) if cross else None, "tolerance_eV_per_formula_unit": tol,
            "structures": both, "dft_records": dft_records,
            "domain": {"topologies": sorted({c.split(":")[0] for c in both}), "fidelity_pair": "ML vs DFT (relaxed energies)",
                       "composition": "project fixed composition only (endpoints not calibrated)"}}


def screen(spec, store, evaluate) -> tuple[dict, dict]:
    """Seeds -> ML relaxations -> optional DFT calibration -> decisions. Returns (artifacts, detail)."""
    protocols = spec.physical["protocols"]
    policy = protocols.get("screening") or {}
    backend = MLPotentialBackend(spec, store)
    arts, records, geometries, decos = {}, {}, {}, {}
    elements = set()
    for tid, topo in spec.topologies.items():
        seeds = arts[f"seeds/{tid}"] = generate_seed_decorations(spec, topo, policy.get("sampling"))
        for d in seeds["decorations"]:
            g = framework.build_framework(spec, topo, d)
            arts[f"seed_geometry/{d['configuration_id']}"] = geometries[d["configuration_id"]] = g
            decos[d["configuration_id"]] = (topo, d)
            elements |= set(g.get("elements") or [])
    built = [g for g in geometries.values() if g["status"] == StageStatus.COMPLETED.value]
    results = backend.relax(built, "topology_screening")
    max_strain = policy.get("max_cell_strain_percent", DEFAULT_MAX_CELL_STRAIN_PERCENT)
    for cid, (topo, d) in decos.items():
        rec = _relaxation_record(spec, topo, d, geometries[cid], results.get(geometries[cid]["artifact_id"], {}), max_strain)
        arts[f"mlrelax/{cid}"] = rec
        records.setdefault(topo.topology_id, []).append(rec)
    flags = backend.domain_flags(elements)
    calibration = None
    if protocols.get("ml_calibration"):
        calibration = calibrate(spec, records, evaluate, protocols["ml_calibration"])
        arts.update(calibration.pop("dft_records"))
        arts["ml_calibration"] = make_record("MLCalibrationReport", spec.provenance, fidelity=backend.fidelity, **calibration)
    calibrated = bool(calibration and calibration["status"] == "CALIBRATED" and not flags)
    minima = {tid: min((r["energy_per_formula_unit_eV"] for r in recs if r["fixed_composition"] and r["status"] == "SUCCEEDED"),
                       default=None) for tid, recs in records.items()}
    best = min((v for v in minima.values() if v is not None), default=None)
    energetic = policy.get("energetic_policy")
    decisions = []
    for tid, recs in records.items():
        statuses = Counter(r["status"] for r in recs if r["fixed_composition"])
        evidence = {"seed_statuses": dict(statuses), "min_energy_per_formula_unit_eV": minima[tid],
                    "gap_to_best_eV_per_formula_unit": None if minima[tid] is None else minima[tid] - best,
                    "mixing_energies_eV_per_formula_unit": _mixing(recs), "seed_result_ids": [r["artifact_id"] for r in recs]}
        if minima[tid] is None and set(statuses) <= {"BUILD_FAILED", "TRANSFORMED"}:
            state, reason = "reject", "STRUCTURAL_INCOMPATIBILITY: no seed builds or every relaxed seed changes topology"
        elif minima[tid] is None:
            state, reason = "uncertain", "NO_SUCCESSFUL_RELAXATION: failures are not energies"
        elif energetic and calibrated:
            threshold = energetic["deprioritise_above_eV_per_formula_unit"]
            margin = threshold + 2 * calibration["mae_eV_per_formula_unit"]
            gap = evidence["gap_to_best_eV_per_formula_unit"]
            state, reason = (("deprioritise", f"gap {gap:.4f} > {margin:.4f} eV/fu (threshold + 2 x calibration MAE)")
                             if gap > margin else ("uncertain", f"gap {gap:.4f} within the calibration margin")
                             if gap > threshold else ("retain", "within the declared energetic threshold"))
        elif energetic and energetic.get("allow_exploratory") is True and not flags \
                and statuses.get("SUCCEEDED", 0) == sum(statuses.values()):
            # Declared opt-in: prioritise on the uncalibrated ML gap. Deprioritised, never rejected: the topology stays
            # in the audit sample and protocols.screening.reinstate brings it back; the ranking label stays EXPLORATORY.
            threshold = energetic["deprioritise_above_eV_per_formula_unit"]
            gap = evidence["gap_to_best_eV_per_formula_unit"]
            state, reason = (("deprioritise", f"EXPLORATORY_ML_GAP: {gap:.4f} > {threshold} eV/fu (uncalibrated, "
                                              "declared allow_exploratory)") if gap > threshold
                             else ("retain", f"EXPLORATORY_ML_GAP: {gap:.4f} <= {threshold} eV/fu (uncalibrated)"))
        else:
            partial = statuses.get("SUCCEEDED", 0) < sum(statuses.values()) or flags
            state = "uncertain" if partial else "retain"
            reason = ("EXPLORATORY_ML_RANKING: " + ("no calibration declared" if calibration is None else
                      calibration["status"] if not flags else "domain flags") + ("; partial seed failures" if partial else ""))
        if tid in (policy.get("reinstate") or []) and state not in KEPT_STATES:
            state, reason = "retain", f"REINSTATED by protocols.screening.reinstate (was {state}: {reason})"
        decisions.append({"topology_id": tid, "state": state, "reason": reason, "evidence": evidence})
    retained = [d["topology_id"] for d in decisions if d["state"] in KEPT_STATES]
    excluded = [d for d in decisions if d["state"] not in KEPT_STATES]
    report = make_record(
        "ScreeningReport", spec.provenance, mode="ml_potential", fidelity=backend.fidelity,
        engine_identity=backend.identity(), domain_flags=flags,
        ranking_label="CALIBRATED" if calibrated else "EXPLORATORY", calibration_id=(arts.get("ml_calibration") or {}).get("artifact_id"),
        energetic_policy=energetic, decisions=decisions, retained=retained,
        audit_sample={d["topology_id"]: d["evidence"]["seed_result_ids"] for d in excluded},
        geometric_screen="ML relaxation: convergence, slot/node topology retention, cell distortion",
        comparison="energies per common formula unit at the fixed composition; mixing energies vs own endpoints")
    arts["screening"] = report
    detail = {"mode": "ml_potential", "ranking_label": report["ranking_label"], "retained": retained,
              "decisions": {d["topology_id"]: d["state"] for d in decisions}, "domain_flags": flags,
              "calibration": None if calibration is None else {k: calibration[k] for k in (
                  "status", "n_pairs", "mae_eV_per_formula_unit", "tolerance_eV_per_formula_unit")},
              "ml_relaxations": store.state["counters"].get("ml_relaxations", 0),
              "ml_cache_hits": store.state["counters"].get("ml_cache_hits", 0)}
    return arts, detail
