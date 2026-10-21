"""Task 03 (configuration_model): features, exact fit and deterministic model selection.

Selection policy (project.json model_selection): among candidates whose
development RMSE <= threshold, choose the fewest parameters, then model ID.
No LLM participates; failed validation yields MODEL_INADEQUATE.
"""
from __future__ import annotations

import math
from fractions import Fraction

from .contract import PipelineError, StageStatus, exact, make_record

#TODO: Task 03 — generate symmetry-orbit cluster features (site/pair/triplet incl. periodic images and orientation
#TODO: variables) from the slot lattice instead of this fixed library; icet only after a validated slot mapping.
FEATURE_DEGREE = {"constant": 0, "unlike_distance_1": 2, "unlike_distance_2": 2, "BBB_paths": 3}


def feature_value(name: str, topo, bits) -> int:
    if name == "constant":
        return 1
    if name == "unlike_distance_1":
        return sum(bits[i] != bits[j] for i, j in topo.pairs_d1)
    if name == "unlike_distance_2":
        return sum(bits[i] != bits[j] for i, j in topo.pairs_d2)
    if name == "BBB_paths":
        return sum(bits[i] & bits[j] & bits[k] for i, j, k in topo.triplets)
    raise PipelineError("UNSUPPORTED_FEATURE", name)


def feature_vector(topo, bits, names) -> list:
    return [feature_value(n, topo, bits) for n in names]


def feature_polynomial(name: str, topo) -> dict:
    """Binary polynomial {sorted site tuple: coefficient}; x_i^2 = x_i."""
    poly: dict = {}

    def add(key, value):
        poly[key] = poly.get(key, 0) + value

    if name == "constant":
        add((), 1)
    elif name in ("unlike_distance_1", "unlike_distance_2"):
        # [x_i != x_j] = x_i + x_j - 2 x_i x_j
        for i, j in (topo.pairs_d1 if name == "unlike_distance_1" else topo.pairs_d2):
            add((i,), 1)
            add((j,), 1)
            add(tuple(sorted((i, j))), -2)
    elif name == "BBB_paths":
        for t in topo.triplets:
            add(tuple(sorted(t)), 1)
    else:
        raise PipelineError("UNSUPPORTED_FEATURE", name)
    return {k: v for k, v in poly.items() if v}


def check_feature_symmetry(topo, names, states) -> dict:
    """Symmetry-related arrangements must have identical features (03 acceptance)."""
    for bits in states:
        ref = feature_vector(topo, bits, names)
        for p in topo.permutations:
            if feature_vector(topo, tuple(bits[j] for j in p), names) != ref:
                raise PipelineError("FEATURE_SYMMETRY_VIOLATION", f"{topo.topology_id}:{bits}")
    return {"topology_id": topo.topology_id, "features": list(names), "n_states": len(states),
            "n_operations": len(topo.permutations), "invariant": True}


def solve_exact(matrix, rhs) -> list:
    n = len(rhs)
    aug = [[Fraction(x) for x in row] + [Fraction(b)] for row, b in zip(matrix, rhs)]
    for col in range(n):
        pivot = next((r for r in range(col, n) if aug[r][col] != 0), None)
        if pivot is None:
            raise PipelineError("RANK_DEFICIENT_FEATURES", f"column {col}")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        lead = aug[col][col]
        aug[col] = [v / lead for v in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0:
                factor = aug[r][col]
                aug[r] = [a - factor * b for a, b in zip(aug[r], aug[col])]
    return [row[-1] for row in aug]


def least_squares_exact(x_rows, y) -> list:
    if not x_rows or len(x_rows) < len(x_rows[0]):
        raise PipelineError("RANK_DEFICIENT_FEATURES", f"{len(x_rows)} rows for {len(x_rows[0]) if x_rows else 0} parameters")
    p = len(x_rows[0])
    gram = [[sum(Fraction(r[i] * r[j]) for r in x_rows) for j in range(p)] for i in range(p)]
    rhs = [sum(r[i] * v for r, v in zip(x_rows, y)) for i in range(p)]
    return solve_exact(gram, rhs)


def rmse(residuals) -> float:
    return math.sqrt(float(sum(r * r for r in residuals) / len(residuals))) if residuals else float("nan")


def predict_exact(names, coefficients, topo, bits) -> Fraction:
    return sum((c * v for c, v in zip(coefficients, feature_vector(topo, bits, names))), Fraction(0))


def fit_and_select(spec, topo, train: dict, dev: dict) -> dict:
    """fit_hamiltonian + validate_hamiltonian over the fixed candidate grid for one topology."""
    #TODO: Task 03 — grouped/nested cross-validation for real data (fixed development split only fits this fixture),
    #TODO: documented gauge for rank-deficient features under fixed composition, sparse/regularised fitting for noisy
    #TODO: labels, low-energy recall and larger-supercell transfer metrics.
    #TODO: Task 03 — calibrated predictive uncertainty from held-out residuals and multifidelity (matched-difference)
    #TODO: models for ML / DFT / embedded-corrected labels.
    threshold =spec.config["model_selection"]["max_development_rmse_eV"]
    tr = [e for e in train["entries"] if e["topology_id"] == topo.topology_id]
    dv = [e for e in dev["entries"] if e["topology_id"] == topo.topology_id]
    results = []
    for model in spec.model_candidates:
        names = [spec.feature_order[i] for i in model["feature_indices"]]
        base = {"model_id": model["id"], "features": names, "n_parameters": len(names),
                "polynomial_degree": max(FEATURE_DEGREE[n] for n in names)}
        try:
            coef = least_squares_exact([feature_vector(topo, e["bits"], names) for e in tr],
                                       [exact(e["energy_exact"]) for e in tr])
        except PipelineError as exc:
            results.append({**base, "status": StageStatus.FAILED.value, "failure": exc.as_dict(), "passes": False})
            continue
        tr_res = [predict_exact(names, coef, topo, e["bits"]) - exact(e["energy_exact"]) for e in tr]
        dv_res = [predict_exact(names, coef, topo, e["bits"]) - exact(e["energy_exact"]) for e in dv]
        dev_rmse = rmse(dv_res)
        results.append({**base, "status": StageStatus.COMPLETED.value, "coefficients_exact": [str(c) for c in coef],
                        "coefficients_eV": [float(c) for c in coef], "training_rmse_eV": rmse(tr_res),
                        "development_rmse_eV": dev_rmse, "development_max_abs_error_eV": max(float(abs(r)) for r in dv_res) if dv_res else None,
                        "passes": bool(dv_res) and dev_rmse <= threshold})
    passing = [r for r in results if r["passes"]]
    common = dict(topology_id=topo.topology_id, training_snapshot_id=train["artifact_id"],
                  development_snapshot_id=dev["artifact_id"], label_fidelity=train["fidelity"],
                  n_training=len(tr), n_development=len(dv), threshold_eV=threshold,
                  policy=spec.config["model_selection"]["policy"], candidates=results)
    if not passing:
        return make_record("ModelSelectionReport", spec.provenance, status="MODEL_INADEQUATE", selected_model=None,
                           reason=f"no candidate reached development RMSE <= {threshold} eV", **common)
    chosen = min(passing, key=lambda r: (r["n_parameters"], r["model_id"]))
    reason = (f"{chosen['model_id']}: fewest parameters ({chosen['n_parameters']}) among "
              f"{[r['model_id'] for r in passing]} with development RMSE <= {threshold} eV; ties broken by model ID")
    return make_record("ModelSelectionReport", spec.provenance, status=StageStatus.COMPLETED.value,
                       selected_model=chosen["model_id"], reason=reason, **common)


def hamiltonian_from_selection(spec, topo, selection: dict) -> dict:
    """HamiltonianArtifact: physical surrogate, kept separate from any encoding."""
    chosen = next(r for r in selection["candidates"] if r["model_id"] == selection["selected_model"])
    coef = [Fraction(c) for c in chosen["coefficients_exact"]]
    names = chosen["features"]
    offset = coef[names.index("constant")]
    return make_record(
        "HamiltonianArtifact", spec.provenance, topology_id=topo.topology_id, model_id=chosen["model_id"],
        training_snapshot_id=selection["training_snapshot_id"], selection_report_id=selection["artifact_id"],
        label_fidelity=selection["label_fidelity"], features=names, coefficients_exact=[str(c) for c in coef],
        coefficients_eV=[float(c) for c in coef], reference_offset_exact=str(offset), reference_offset_eV=float(offset),
        polynomial_degree=chosen["polynomial_degree"], energy_unit="eV",
        normalization="per_abstract_12_slot_formula_unit",
        validation={"training_rmse_eV": chosen["training_rmse_eV"], "development_rmse_eV": chosen["development_rmse_eV"]},
        uncertainty={"kind": "development_residual_rmse", "value_eV": chosen["development_rmse_eV"],
                     "note": "noiseless synthetic labels; not a certified physical error bound"},
        validity_domain={"topology_id": topo.topology_id, "n_sites": spec.n_sites,
                         "composition": spec.composition_counts, "fidelity": selection["label_fidelity"]},
    )


def predict(hamiltonian: dict, topo, bits) -> Fraction:
    return predict_exact(hamiltonian["features"], [Fraction(c) for c in hamiltonian["coefficients_exact"]], topo, bits)
