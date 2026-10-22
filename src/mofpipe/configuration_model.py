"""Task 03 (configuration_model): features, exact fit and deterministic model selection.

Selection policy (project.json model_selection): among candidates whose
development RMSE <= threshold, choose the fewest parameters, then model ID.
No LLM participates; failed validation yields MODEL_INADEQUATE.
"""
from __future__ import annotations

import math
from fractions import Fraction
from functools import lru_cache

from . import clusters
from .contract import PipelineError, StageStatus, content_hash, exact, make_record
from .structures import composition_of

#TODO: Task 03 — triplet cluster orbits and orientation variables (T01.4) in the generated cluster basis.
FEATURE_DEGREE = {"constant": 0, "unlike_distance_1": 2, "unlike_distance_2": 2, "BBB_paths": 3}
SITE_ORBIT_PREFIX = "site_orbit_"  # B count on the k-th symmetry orbit of slots in this cell (not transferable)
CLUSTER_PREFIXES = {"point_orbit_": 1, "pair_orbit_": 2}  # clusters.py: parent-lattice orbits, per formula unit
TRANSFERABLE = ("constant",) + tuple(CLUSTER_PREFIXES)


@lru_cache(maxsize=None)
def site_orbits(topo) -> tuple:
    """Slot orbits under the topology's symmetry permutations, ordered by smallest slot index."""
    orbits = {}
    for i in range(topo.n_sites):
        orbits.setdefault(min(p[i] for p in topo.permutations), set()).add(i)
    return tuple(tuple(sorted(o)) for _, o in sorted(orbits.items()))


def _cluster_prefix(name: str):
    return next((p for p in CLUSTER_PREFIXES if name.startswith(p) and name[len(p):].isdigit()), None)


def is_known_feature(name: str) -> bool:
    return (name in FEATURE_DEGREE or (name.startswith(SITE_ORBIT_PREFIX) and name[len(SITE_ORBIT_PREFIX):].isdigit())
            or _cluster_prefix(name) is not None)


def is_transferable(name: str) -> bool:
    """Defined on the parent lattice per formula unit, so comparable between supercells."""
    return name == "constant" or _cluster_prefix(name) is not None


def feature_degree(name: str) -> int:
    if not is_known_feature(name):
        raise PipelineError("UNSUPPORTED_FEATURE", name)
    prefix = _cluster_prefix(name)
    if prefix:
        return CLUSTER_PREFIXES[prefix]
    return 1 if name.startswith(SITE_ORBIT_PREFIX) else FEATURE_DEGREE[name]


def expand_tokens(tokens, topo) -> list:
    """Candidate tokens -> this topology's feature names ('site_orbits', 'point_orbits', 'pairs<=R' expand)."""
    out = []
    for token in tokens:
        if token == "site_orbits":
            out += [f"{SITE_ORBIT_PREFIX}{k}" for k in range(len(site_orbits(topo)))]
        elif token == "point_orbits" or token.startswith("pairs<="):
            out += clusters.names_for_token(topo, token)
        else:
            out.append(token)
    return out


def model_features(spec, topo, model) -> list:
    """Feature names of one candidate model for one topology (fixture: global indices; physical: tokens)."""
    if "tokens" in model:
        return expand_tokens(model["tokens"], topo)
    return [spec.feature_order[i] for i in model["feature_indices"]]


def all_model_features(spec, topo) -> list:
    names = []
    for model in spec.model_candidates:
        names += [n for n in model_features(spec, topo, model) if n not in names]
    return names


def _orbit_sites(name: str, topo) -> tuple:
    k, orbits = int(name[len(SITE_ORBIT_PREFIX):]), site_orbits(topo)
    if k >= len(orbits):
        raise PipelineError("UNSUPPORTED_FEATURE", f"{name}: {topo.topology_id} has {len(orbits)} site orbits")
    return orbits[k]


def feature_value(name: str, topo, bits):
    if name == "constant":
        return 1
    if _cluster_prefix(name):
        return clusters.cluster_value(topo, name, bits)
    if name.startswith(SITE_ORBIT_PREFIX):
        return sum(bits[i] for i in _orbit_sites(name, topo))
    if name == "unlike_distance_1":
        weights = topo.pair_weights_d1 or (1,) * len(topo.pairs_d1)
        return sum(w * (bits[i] != bits[j]) for (i, j), w in zip(topo.pairs_d1, weights))
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
    elif _cluster_prefix(name):
        return clusters.cluster_polynomial(topo, name)
    elif name in ("unlike_distance_1", "unlike_distance_2"):
        # [x_i != x_j] = x_i + x_j - 2 x_i x_j, times the periodic contact multiplicity (distance 1)
        pairs = topo.pairs_d1 if name == "unlike_distance_1" else topo.pairs_d2
        weights = (topo.pair_weights_d1 if name == "unlike_distance_1" else ()) or (1,) * len(pairs)
        for (i, j), w in zip(pairs, weights):
            add((i,), w)
            add((j,), w)
            add(tuple(sorted((i, j))), -2 * w)
    elif name == "BBB_paths":
        for t in topo.triplets:
            add(tuple(sorted(t)), 1)
    elif name.startswith(SITE_ORBIT_PREFIX):
        for i in _orbit_sites(name, topo):
            add((i,), 1)
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
    """fit_hamiltonian + validate_hamiltonian over the fixed candidate grid for one topology (fixture policy).

    Physical projects use ``fit_and_select_grouped`` (grouped cross-validation + frozen final holdout).
    """
    threshold = spec.config["model_selection"]["max_development_rmse_eV"]
    tr = [e for e in train["entries"] if e["topology_id"] == topo.topology_id]
    dv = [e for e in dev["entries"] if e["topology_id"] == topo.topology_id]
    results = []
    for model in spec.model_candidates:
        names = model_features(spec, topo, model)
        base = {"model_id": model["id"], "features": names, "n_parameters": len(names),
                "polynomial_degree": max(feature_degree(n) for n in names)}
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
        validity_domain={"topology_id": topo.topology_id, "n_sites": topo.n_sites,
                         "composition": composition_of(spec, topo), "fidelity": selection["label_fidelity"]},
    )


def predict(hamiltonian: dict, topo, bits) -> Fraction:
    return predict_exact(hamiltonian["features"], [Fraction(c) for c in hamiltonian["coefficients_exact"]], topo, bits)


# -- real (noisy) data: grouped cross-validation, gauge, ridge, low-energy metrics -------------------------
def independent_columns(x_rows, names) -> tuple[list, list]:
    """Documented gauge: keep a column only if it raises the exact rank (feature order decides); drop the rest."""
    basis, kept, dropped = [], [], []
    for c, name in enumerate(names):
        vec = [Fraction(r[c]) for r in x_rows]
        for pivot, b in basis:
            if vec[pivot]:
                factor = vec[pivot] / b[pivot]
                vec = [v - factor * w for v, w in zip(vec, b)]
        pivot = next((i for i, v in enumerate(vec) if v), None)
        if pivot is None:
            dropped.append(name)
        else:
            basis.append((pivot, vec))
            kept.append(c)
    return kept, dropped


def ridge_exact(x_rows, y, alpha: Fraction, constant_index: int | None) -> list:
    """(X^T X + alpha D) beta = X^T y, D = identity without the constant column; alpha = 0 is least squares."""
    p = len(x_rows[0])
    if len(x_rows) < p:
        raise PipelineError("RANK_DEFICIENT_FEATURES", f"{len(x_rows)} rows for {p} parameters")
    gram = [[sum(Fraction(r[i] * r[j]) for r in x_rows) + (alpha if i == j and i != constant_index else 0)
             for j in range(p)] for i in range(p)]
    rhs = [sum(r[i] * v for r, v in zip(x_rows, y)) for i in range(p)]
    return solve_exact(gram, rhs)


def assign_folds(groups: list, n_folds: int, seed) -> dict:
    """Deterministic grouped folds: groups ordered by a seeded content hash, dealt round-robin."""
    order = sorted(set(groups), key=lambda g: content_hash([seed, g]))
    return {g: i % n_folds for i, g in enumerate(order)}


def low_energy_recall(y, pred, k: int, tol: Fraction) -> float:
    """Fraction of the k lowest predictions whose label is within tol of the k-th lowest label (tie-aware)."""
    k = min(k, len(y))
    if k == 0:
        return float("nan")
    kth = sorted(y)[k - 1]
    chosen = sorted(range(len(y)), key=lambda i: (pred[i], i))[:k]
    return sum(y[i] <= kth + tol for i in chosen) / k


def spearman(a, b) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for t in range(i, j + 1):
                r[order[t]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    var = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return cov / var if var else float("nan")


def fit_and_select_grouped(spec, topo, train: dict, groups: dict) -> dict:
    """fit_hamiltonian + validate_hamiltonian for real labels (Task 03, T03.2).

    Grouped K-fold cross-validation over symmetry-orbit groups (near-duplicate relaxations share a group),
    exact rank check with a documented gauge, optional ridge, CV RMSE and tie-aware low-energy recall.
    The final holdout is not an input here; it is evaluated once after selection (evaluate_holdout).
    """
    #TODO: Task 03 — calibrated predictive uncertainty from held-out residuals and multifidelity (matched-difference)
    #TODO: models for ML / DFT / embedded-corrected labels. (Supercell transfer metric: transfer.py.)
    physical = spec.physical
    policy = physical["protocols"]["model_selection"]
    tol = Fraction(str(physical["tolerances"]["target_relative_energy_tolerance"]))
    criterion = physical["tolerances"]["low_energy_validation_criterion"]
    alpha = Fraction(str(policy["regularization"].get("alpha", 0))) if policy["regularization"]["kind"] == "ridge" else None
    if alpha is None:
        raise PipelineError("UNSUPPORTED_REGULARIZATION", str(policy["regularization"]))
    entries = [e for e in train["entries"] if e["topology_id"] == topo.topology_id]
    y = [exact(e["energy_exact"]) for e in entries]
    entry_groups = [groups[e["orbit_id"]] for e in entries]
    folds = assign_folds(entry_groups, policy["n_folds"], policy["seed"])
    if len(set(folds.values())) < 2:
        raise PipelineError("INSUFFICIENT_GROUPS_FOR_CV", f"{topo.topology_id}: {len(set(entry_groups))} groups")
    results = []
    for model in spec.model_candidates:
        names = model_features(spec, topo, model)
        x_all = [feature_vector(topo, e["bits"], names) for e in entries]
        kept, dropped = independent_columns(x_all, names)
        base = {"model_id": model["id"], "features_declared": names, "n_parameters_declared": len(names)}
        if dropped and policy["rank_deficiency"] != "drop_dependent_columns":
            results.append({**base, "status": StageStatus.FAILED.value, "passes": False,
                            "failure": {"code": "RANK_DEFICIENT_FEATURES", "detail": f"dependent columns {dropped}"}})
            continue
        eff = [names[c] for c in kept]
        x = [[row[c] for c in kept] for row in x_all]
        const = eff.index("constant") if "constant" in eff else None
        base.update(features=eff, dropped_dependent_features=dropped, n_parameters=len(eff),
                    polynomial_degree=max(feature_degree(n) for n in eff))
        try:
            oof, fold_coefs = [None] * len(entries), []
            for fold in sorted(set(folds.values())):
                fit_idx = [i for i, g in enumerate(entry_groups) if folds[g] != fold]
                coef = ridge_exact([x[i] for i in fit_idx], [y[i] for i in fit_idx], alpha, const)
                fold_coefs.append([str(c) for c in coef])
                for i, g in enumerate(entry_groups):
                    if folds[g] == fold:
                        oof[i] = sum((c * v for c, v in zip(coef, x[i])), Fraction(0))
            coef = ridge_exact(x, y, alpha, const)
        except PipelineError as exc:
            results.append({**base, "status": StageStatus.FAILED.value, "passes": False, "failure": exc.as_dict()})
            continue
        residuals = [p - t for p, t in zip(oof, y)]
        train_res = [sum((c * v for c, v in zip(coef, xi)), Fraction(0)) - t for xi, t in zip(x, y)]
        cv_rmse = rmse(residuals)
        recall = low_energy_recall(y, oof, criterion["k"], tol)
        passes = cv_rmse <= float(tol) and recall >= criterion["min_recall"]
        results.append({**base, "status": StageStatus.COMPLETED.value, "coefficients_exact": [str(c) for c in coef],
                        "coefficients_eV": [float(c) for c in coef], "training_rmse_eV": rmse(train_res),
                        "cv_rmse_eV": cv_rmse, "cv_max_abs_error_eV": max(float(abs(r)) for r in residuals),
                        "low_energy_recall": recall, "cv_spearman": spearman([float(v) for v in y], [float(v) for v in oof]),
                        "fold_coefficients_exact": fold_coefs, "passes": passes})
    passing = [r for r in results if r["passes"]]
    common = dict(topology_id=topo.topology_id, training_snapshot_id=train["artifact_id"],
                  label_fidelity=train["fidelity"], normalization=train["normalization"], n_training=len(entries),
                  n_groups=len(set(entry_groups)), n_folds=policy["n_folds"], fold_seed=policy["seed"],
                  validation="grouped_kfold_by_symmetry_orbit_and_relaxed_geometry",
                  regularization=policy["regularization"], gauge=policy["rank_deficiency"],
                  tolerance_eV=float(tol), low_energy_criterion=criterion, threshold_eV=float(tol),
                  policy=spec.config["model_selection"]["policy"], candidates=results,
                  final_holdout_used=False)
    if not passing:
        return make_record("ModelSelectionReport", spec.provenance, status="MODEL_INADEQUATE", selected_model=None,
                           reason=f"no candidate reached CV RMSE <= {float(tol)} eV and low-energy recall >= "
                                  f"{criterion['min_recall']}; more labels or a richer model family are required",
                           data_requirements={"n_labels": len(entries), "n_groups": len(set(entry_groups))}, **common)
    chosen = min(passing, key=lambda r: (r["n_parameters"], r["cv_rmse_eV"], r["model_id"]))
    reason = (f"{chosen['model_id']}: fewest effective parameters ({chosen['n_parameters']}) among "
              f"{[r['model_id'] for r in passing]} passing grouped CV; ties by CV RMSE then model ID")
    return make_record("ModelSelectionReport", spec.provenance, status=StageStatus.COMPLETED.value,
                       selected_model=chosen["model_id"], reason=reason, **common)


def hamiltonian_from_grouped_selection(spec, topo, selection: dict) -> dict:
    chosen = next(r for r in selection["candidates"] if r["model_id"] == selection["selected_model"])
    coef = [Fraction(c) for c in chosen["coefficients_exact"]]
    names = chosen["features"]
    offset = coef[names.index("constant")] if "constant" in names else Fraction(0)
    return make_record(
        "HamiltonianArtifact", spec.provenance, topology_id=topo.topology_id, model_id=chosen["model_id"],
        training_snapshot_id=selection["training_snapshot_id"], selection_report_id=selection["artifact_id"],
        label_fidelity=selection["label_fidelity"], features=names, dropped_dependent_features=chosen["dropped_dependent_features"],
        coefficients_exact=[str(c) for c in coef], coefficients_eV=[float(c) for c in coef],
        reference_offset_exact=str(offset), reference_offset_eV=float(offset),
        polynomial_degree=chosen["polynomial_degree"], energy_unit="eV", normalization=selection["normalization"],
        validation={"training_rmse_eV": chosen["training_rmse_eV"], "cv_rmse_eV": chosen["cv_rmse_eV"],
                    "low_energy_recall": chosen["low_energy_recall"]},
        uncertainty={"kind": "grouped_cv_residual_rmse", "value_eV": chosen["cv_rmse_eV"],
                     "note": "cross-validation residual scale; not a calibrated or certified error bound (T03.3)"},
        validity_domain={"topology_id": topo.topology_id, "n_sites": topo.n_sites, "composition": composition_of(spec, topo),
                         "fidelity": selection["label_fidelity"], "supercell": list(topo.supercell),
                         "transferable_features": all(is_transferable(n) for n in names)},
    )


def evaluate_holdout(spec, topo, hamiltonian: dict, holdout: dict) -> dict:
    """One-time evaluation of the frozen final holdout after selection; failure is reported, not tuned away."""
    tol = Fraction(str(spec.physical["tolerances"]["target_relative_energy_tolerance"]))
    criterion = spec.physical["tolerances"]["low_energy_validation_criterion"]
    rows = [e for e in holdout["entries"] if e["topology_id"] == topo.topology_id]
    y = [exact(e["energy_exact"]) for e in rows]
    pred = [predict(hamiltonian, topo, e["bits"]) for e in rows]
    residuals = [p - t for p, t in zip(pred, y)]
    value = rmse(residuals)
    ok = bool(rows) and value <= float(tol)
    return make_record(
        "HoldoutReport", spec.provenance, topology_id=topo.topology_id, hamiltonian_id=hamiltonian["artifact_id"],
        holdout_snapshot_id=holdout["artifact_id"], n_holdout=len(rows), holdout_rmse_eV=value,
        holdout_max_abs_error_eV=max((float(abs(r)) for r in residuals), default=None),
        low_energy_recall=low_energy_recall(y, pred, criterion["k"], tol) if rows else None, tolerance_eV=float(tol),
        status=StageStatus.COMPLETED.value if ok else "HOLDOUT_VALIDATION_FAILED",
        note="evaluated once after model selection; a holdout used for tuning must be replaced by a new one")
