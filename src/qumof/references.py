"""Task 09 (T09.2): reference-phase competition at the candidates' overall composition.

Reference phases are periodic structures of different composition evaluated with the same DFT protocol
(``protocols.validation.reference_phases``; ``pure_endpoints`` = every slot of every candidate topology holding a
single linker species). The mixture balance

    E_ref(target) = min_c  sum_i c_i E_i   subject to   sum_i c_i inventory_i = target inventory,  c_i >= 0

is solved exactly in rational arithmetic by enumerating the basic feasible solutions of this linear programme.
The mixing energy of a candidate is E(candidate) - E_ref(target), per formula unit. The result is relative to
the listed references only: coverage is reported as incomplete unless the project declares its reference set
complete, so a negative mixing energy is never presented as an absolute stability certificate.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations

from .configuration_model import solve_exact
from .contract import PipelineError, StageStatus, make_record


def mixture_balance(target: dict, references: list) -> dict | None:
    """references: [{"reference_id", "inventory": {el: n per cell}, "energy_exact": eV per cell}] -> optimum or None."""
    elements = sorted(set(target) | {el for r in references for el in r["inventory"]})
    columns = [[Fraction(r["inventory"].get(el, 0)) for el in elements] for r in references]
    rhs = [Fraction(target.get(el, 0)) for el in elements]
    best = None
    for size in range(1, min(len(references), len(elements)) + 1):
        for subset in combinations(range(len(references)), size):
            cols = [columns[i] for i in subset]
            gram = [[sum(a * b for a, b in zip(ci, cj)) for cj in cols] for ci in cols]
            try:
                coef = solve_exact(gram, [sum(a * b for a, b in zip(ci, rhs)) for ci in cols])
            except PipelineError:
                continue  # dependent columns: covered by a smaller subset
            if any(c < 0 for c in coef):
                continue
            if [sum(c * col[e] for c, col in zip(coef, cols)) for e in range(len(elements))] != rhs:
                continue  # the composition is not reachable with this subset
            energy = sum(c * Fraction(references[i]["energy_exact"]) for c, i in zip(coef, subset))
            key = (energy, tuple(references[i]["reference_id"] for i in subset))
            if best is None or key < best[0]:
                best = (key, {"energy_exact": energy, "coefficients": [
                    {"reference_id": references[i]["reference_id"], "cells_per_formula_unit": str(c)}
                    for c, i in zip(coef, subset) if c != 0]})
    return best[1] if best else None


def evaluate_reference_mixtures(spec, ranking: dict, references: list | None = None, declared_complete: bool = False,
                                missing: list | None = None) -> dict:
    """evaluate_reference_mixtures(ranking_id, reference_set_id) -> CompetitionReport."""
    if not references:
        return make_record("CompetitionReport", spec.provenance, ranking_id=ranking["artifact_id"],
                           status=StageStatus.INCOMPLETE.value, reference_set=None, missing_references=missing or [],
                           reason="no evaluated reference phases; the result is a relative ranking, not a stability certificate")
    tol = Fraction(str(spec.physical["tolerances"]["target_relative_energy_tolerance"]))
    target = spec.physical["chemistry"]["target_elemental_inventory"]
    balance = mixture_balance(target, references)
    rows = []
    if balance is not None:
        for row in ranking["ranked"]:
            e_mix = Fraction(row["ranking_energy_exact"]) - balance["energy_exact"]
            rows.append({"configuration_id": row["configuration_id"], "topology_id": row["topology_id"],
                         "mixing_energy_eV_per_fu": float(e_mix), "mixing_energy_exact": str(e_mix),
                         "versus_references": ("below_within_listed_references" if e_mix < -tol else
                                               "tie_within_tolerance" if e_mix <= tol else "above_reference_mixture")})
    complete = declared_complete and not missing and balance is not None
    return make_record(
        "CompetitionReport", spec.provenance, ranking_id=ranking["artifact_id"],
        status=StageStatus.COMPLETED.value if complete else StageStatus.INCOMPLETE.value,
        reference_set=[{k: r[k] for k in ("reference_id", "inventory", "energy_eV", "energy_exact", "calculation_result_id",
                                          "topology_id", "species")} for r in references],
        missing_references=missing or [], target_inventory_per_fu=target,
        reference_mixture=({"energy_eV_per_fu": float(balance["energy_exact"]), "energy_exact": str(balance["energy_exact"]),
                            "coefficients": balance["coefficients"]} if balance else None),
        mixing_energies=rows, tolerance_eV=float(tol),
        coverage=("declared complete by the project" if declared_complete else
                  "listed references only (pure endpoints of the candidate topologies); other phases may compete"),
        reason=None if balance else "target composition not reachable from the evaluated references")
