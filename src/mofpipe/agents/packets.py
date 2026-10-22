"""Input packets for A1, A2 and A67 (guide sections 3, 4, 5 and 7).

Each packet carries ``run_id``, ``input_hash``, ``schema_version``, ``mode`` and the allowed IDs/options plus the
concise metadata the model needs to answer in one request. Catalogues are precomputed here, deterministically;
full structures, datasets and logs stay in the run store. Final-test (holdout) labels are never included.
"""
from __future__ import annotations

from ..configuration_model import feature_degree, model_features
from ..contract import content_hash
from ..intake import FIXTURE_PROJECT_TYPE
from ..project_spec import MAX_EXACT_SITES
from ..store import RunStore
from ..structures import composition_of, define_decoration_space
from .contracts import CONTRACT_VERSION

FEATURE_DESCRIPTIONS = {
    "constant": "topology offset c_t; always fitted, kept for cross-topology ranking",
    "unlike_distance_1": "count of graph-distance-1 slot pairs with unlike species (periodic contact multiplicity)",
    "unlike_distance_2": "count of graph-distance-2 slot pairs with unlike species",
    "BBB_paths": "count of length-2 paths whose three slots all carry the encoding-1 species (cubic term)",
}
PREFIX_DESCRIPTIONS = {
    "site_orbit_": "encoding-1 count on one slot symmetry orbit of this cell (linear, not transferable)",
    "point_orbit_": "parent-lattice point-cluster orbit, per formula unit (linear, transferable)",
    "pair_orbit_": "parent-lattice pair-cluster orbit, per formula unit (quadratic, transferable)",
}
# Structural checks whose absence or failure blocks PASS_CHECKED_CRITERIA (validation.validate_structure names).
DEFAULT_REQUIRED_CHECKS = ("post_relaxation_slot_mapping", "residual_force_stress", "min_interatomic_distance",
                           "metal_coordination", "mofchecker", "crystalnets_topology")


def mode_of(spec) -> str:
    return "synthetic_fixture" if spec.project_type == FIXTURE_PROJECT_TYPE else "physical"


def _finish(packet: dict) -> dict:
    return {**packet, "input_hash": content_hash(packet)}


def _counts(counts: dict) -> list:
    return [{"species_id": s, "count": c} for s, c in sorted(counts.items())]


def constraint_policy_id(spec, topo) -> str:
    counts = composition_of(spec, topo)
    return "exact_" + "_".join(f"{s}{c}" for s, c in sorted(counts.items())) + ("_per_sublattice" if topo.sublattice_counts else "")


def encoding_id(spec) -> str:
    return "binary_" + "_".join(f"{s}{e}" for s, e in sorted(spec.species_encoding.items(), key=lambda x: x[1]))


def _topology_summary(spec, topo) -> dict:
    return {"topology_id": topo.topology_id, "representation": topo.representation, "n_substitution_sites": topo.n_sites,
            "allowed_species": sorted({s for site in topo.allowed_species for s in site}),
            "n_pairs_distance_1": len(topo.pairs_d1), "n_pairs_distance_2": len(topo.pairs_d2),
            "n_triplet_paths": len(topo.triplets), "n_symmetry_operations": len(topo.permutations),
            "n_formula_units": topo.n_formula_units, "supercell": list(topo.supercell) or None,
            "crystallographic_net_name": topo.crystallographic_net_name, "provenance": topo.provenance}


def _composition_feasible(spec, topo) -> bool:
    counts = composition_of(spec, topo)
    return all(sum(sp in site for site in topo.allowed_species) >= n for sp, n in counts.items())


def construction_options(spec) -> list:
    """Precomputed legal options: one validated working cell per candidate topology in this build.

    #TODO: Task 01 — several options per framework (alternative supercells, orientation templates).
    """
    physical = spec.physical or {}
    backend = (((physical.get("protocols") or {}).get("builder") or {}).get("builder_id")
               if physical else "synthetic_occupation_graph")
    generated = {a["net"]: a for a in ((physical.get("topology_generation") or {}).get("accepted") or [])}
    return [{"construction_plan_id": f"{tid}_plan_0", "framework_id": tid, "backend_id": backend,
             "framework_source": (f"{physical['topology_generation']['builder']}:{tid}" if tid in generated
                                  else "supplied_cif" if physical else "fixture_graph"),
             "node_fit_rmsd_A": generated[tid]["max_node_rmsd_A"] if tid in generated else None,
             "supercell": list(topo.supercell) or None, "n_substitution_sites": topo.n_sites,
             "species_counts": _counts(composition_of(spec, topo)), "orientation_template_id": "none",
             "constraint_policy_id": constraint_policy_id(spec, topo),
             "composition_feasible": _composition_feasible(spec, topo),
             "cell_within_limit": topo.n_sites <= (MAX_EXACT_SITES if spec.physical else spec.n_sites)}
            for tid, topo in spec.topologies.items()]


def build_a1_packet(spec, run_id: str, design_brief: str, allow_scope_exclusions: bool) -> dict:
    """design_brief: the user's general description. A1 selects which catalogue topologies to analyse."""
    physical = spec.physical or {}
    if physical:
        blocks = [{k: b.get(k) for k in ("id", "role", "formula", "formal_charge", "connectivity", "slot_type",
                                         "n_connector_atoms")} for b in (physical.get("blocks") or {}).values()]
        chem = physical.get("chemistry") or {}
        constraints = {"max_substitution_sites": MAX_EXACT_SITES, "net_charge": chem.get("net_charge"),
                       "formula_unit": chem.get("formula_unit"), "permitted_transformations": []}
    else:
        blocks = [{"id": s["id"], "role": s.get("role"), "encoding": s.get("encoding"),
                   "chemical_formula": s.get("chemical_formula")} for s in spec.building_blocks.get("species", [])]
        constraints = {"max_substitution_sites": spec.n_sites, "net_charge": None, "permitted_transformations": []}
    options = construction_options(spec)
    return _finish({
        "run_id": run_id, "schema_version": CONTRACT_VERSION, "mode": mode_of(spec), "role": "A1",
        "design_brief": design_brief,
        "selection_policy": {"allow_scope_exclusions": allow_scope_exclusions, "min_selected": 1,
                             "exclusion_reasons_checked_by_code": ["COMPOSITION_INFEASIBLE", "CELL_TOO_LARGE"]},
        "candidate_frameworks": [_topology_summary(spec, t) for t in spec.topologies.values()],
        "building_blocks": blocks,
        "target_composition": {"species_ids": sorted(spec.species_encoding), "encoding": spec.species_encoding,
                               "counts_by_framework": {tid: _counts(composition_of(spec, t)) for tid, t in spec.topologies.items()}},
        "construction_options": options,
        "backend_options": sorted({o["backend_id"] for o in options if o["backend_id"]}),
        "constraints": constraints,
        "topology_search": _topology_search(spec),
    })


def _topology_search(spec):
    """How the catalogue was generated (topology builder) and which nets were rejected by code, with reasons."""
    gen = (spec.physical or {}).get("topology_generation")
    if not gen:
        return None
    return {"builder": gen["builder"], "versions": gen["versions"], "selection_rule": gen["selection_rule"],
            "n_nets_considered": gen["n_nets_considered"],
            "accepted": [{k: a.get(k) for k in ("net", "max_node_rmsd_A", "n_nodes", "n_edges", "n_substitution_sites")}
                         for a in gen["accepted"]],
            "rejected_by_code": [{k: r.get(k) for k in ("net", "reason", "max_node_rmsd_A", "count")}
                                 for r in gen["rejected"]][:40],
            "note": "builder output is idealised and unrelaxed; the ML-potential screen ranks the selected nets next"}


def _feature_catalogue(spec, topologies) -> list:
    names = []
    for topo in topologies:
        for model in spec.model_candidates:
            names += [n for n in model_features(spec, topo, model) if n not in names]
    out = []
    for n in names:
        prefix = next((p for p in PREFIX_DESCRIPTIONS if n.startswith(p)), None)
        out.append({"feature_id": n, "polynomial_degree": feature_degree(n),
                    "description": FEATURE_DESCRIPTIONS.get(n) or PREFIX_DESCRIPTIONS.get(prefix, "")})
    return out


def compilation_options(spec) -> list:
    policy = spec.config["qubo"].get("higher_order")
    options = [{"compiler_id": "direct_quadratic", "max_degree": 2,
                "description": "QUBO, upper triangle once; constraint penalty added and verified by code"}]
    if policy == "quadratize":
        options.append({"compiler_id": "quadratize_substitution", "max_degree": 3,
                        "description": "cubic terms reduced with ancilla substitution; verified over ancillas"})
    else:
        options.append({"compiler_id": "hubo_exact_enumeration", "max_degree": 3,
                        "description": "QUBO target rejected for degree > 2; HUBO keeps every triplet and is solved "
                                       "by exact enumeration (no quadratization backend in this configuration)"})
    return options


def _training_summary(spec, topo) -> dict:
    if spec.project_type == FIXTURE_PROJECT_TYPE:
        tid = topo.topology_id
        return {"topology_id": tid, "label_source": "synthetic fixture files",
                "n_training_labels": sum(r["topology_id"] == tid for r in spec.training_rows),
                "n_development_labels": sum(r["topology_id"] == tid for r in spec.development_rows),
                "fidelity": (spec.training_rows or [{}])[0].get("fidelity"), "energy_unit": "eV",
                "normalization": "per abstract formula unit", "final_test_labels": "withheld"}
    selection = (spec.physical["protocols"].get("model_selection") or {})
    return {"topology_id": topo.topology_id, "label_source": "periodic DFT labels computed by the pipeline after A2",
            "label_orbits_per_topology": selection.get("label_orbits_per_topology"),
            "fidelity": (spec.physical["protocols"].get("dft") or {}).get("protocol_id"), "energy_unit": "eV",
            "normalization": "per formula unit", "final_test_labels": "withheld"}


def _validation_policy(spec) -> dict:
    ms = spec.config["model_selection"]
    policy = {k: ms.get(k) for k in ("mode", "split_unit", "max_development_rmse_eV", "policy", "no_passing_model")}
    if spec.physical:
        selection = spec.physical["protocols"].get("model_selection") or {}
        policy.update({k: selection.get(k) for k in ("n_folds", "regularization", "rank_deficiency", "holdout_fraction")},
                      low_energy_validation_criterion=spec.physical["tolerances"].get("low_energy_validation_criterion"))
    return policy


def build_a2_packet(spec, setup: dict, run_id: str, expansion: dict | None = None, screening: dict | None = None) -> dict:
    """setup: validated A1 plans (topologies kept by screening). expansion: fit diagnostics for the optional
    model-family expansion. screening: the Task 02 decisions and ML energies (context, not training labels)."""
    planned = [p["framework_id"] for p in setup["plans"]]
    topologies = [spec.topologies[t] for t in planned]
    rows = []
    for topo in topologies:
        space = define_decoration_space(spec, topo)
        rows.append({**_topology_summary(spec, topo), "n_feasible_states": space["n_feasible"],
                     "n_symmetry_orbits": space["n_symmetry_orbits"],
                     "construction_plan_id": next(p["construction_plan_id"] for p in setup["plans"]
                                                  if p["framework_id"] == topo.topology_id),
                     "constraint_policy_id": constraint_policy_id(spec, topo)})
    models = []
    for model in spec.model_candidates:
        per = {t.topology_id: model_features(spec, t, model) for t in topologies}
        models.append({"model_id": model["id"], "features": per,
                       "n_parameters": {tid: len(names) for tid, names in per.items()},
                       "polynomial_degree": {tid: max(feature_degree(n) for n in names) for tid, names in per.items()}})
    packet = {
        "run_id": run_id, "schema_version": CONTRACT_VERSION, "mode": mode_of(spec), "role": "A2",
        "topologies": rows,
        "composition_constraints": [{"topology_id": t.topology_id, "species_counts": _counts(composition_of(spec, t)),
                                     "constraint_policy_id": constraint_policy_id(spec, t),
                                     "per_sublattice": bool(t.sublattice_counts)} for t in topologies],
        "feature_catalogue": _feature_catalogue(spec, topologies),
        "model_catalogue": models,
        "encoding_catalogue": [{"encoding_id": encoding_id(spec), "kind": "binary", "species_codes": spec.species_encoding,
                                "validity_constraint": "fixed species counts per cell, enforced by a verified penalty"}],
        "training_summary": [_training_summary(spec, t) for t in topologies],
        "compilation_options": compilation_options(spec),
        "validation_policy": _validation_policy(spec),
        "limits": {"max_models_per_topology": 3},
    }
    if screening:
        packet["screening"] = {**screening, "note": "ML-potential energies are screening estimates (another fidelity), "
                                                    "never labels; EXPLORATORY rankings are uncalibrated"}
    if expansion:
        packet["expansion_request"] = expansion
    return _finish(packet)


def fit_diagnostics(report: dict, pipeline_dir) -> dict:
    """Code-measured fit results per topology for the expansion request (no holdout values)."""
    store, out = RunStore(pipeline_dir), {}
    for key, aid in report.get("artifact_index", {}).items():
        if key.startswith("selection/"):
            sel = store.get(aid)
            out[key.split("/", 1)[1]] = {"status": sel["status"], "candidates": [
                {k: c.get(k) for k in ("model_id", "status", "passes", "n_parameters", "polynomial_degree",
                                       "training_rmse_eV", "development_rmse_eV", "cv_rmse_eV")}
                | {"failure_code": (c.get("failure") or {}).get("code")} for c in sel["candidates"]]}
    return out


def candidate_rows(report: dict, top_k: int) -> list:
    ranking = report.get("corrected_ranking") or report.get("base_ranking") or []
    rows = []
    for row in ranking[:top_k]:
        cid = row.get("configuration_id") or row.get("orbit_id")
        rows.append({"candidate_id": cid, **{k: v for k, v in row.items() if not isinstance(v, (dict, list))}})
    return rows


def _structural_checks(spec, report: dict, candidates: list, pipeline_dir, required: tuple) -> list:
    """Per-candidate check records {check_id, check, status, required, ...} from the stored validation reports."""
    out = []
    if spec.project_type == FIXTURE_PROJECT_TYPE:
        for cand in candidates:
            for name, status in (report.get("structural_checks") or {}).items():
                out.append({"check_id": f"{cand['candidate_id']}:{name}", "candidate_id": cand["candidate_id"],
                            "check": name, "status": status, "required": name in required})
        return out
    store, index = RunStore(pipeline_dir), report.get("artifact_index", {})
    for cand in candidates:
        aid = index.get(f"structure_validation/{cand.get('orbit_id')}")
        checks = store.get(aid)["checks"] if aid else {}
        for name in sorted(set(checks) | set(required)):
            c = checks.get(name, {"status": "MISSING"})
            out.append({"check_id": f"{cand['candidate_id']}:{name}", "candidate_id": cand["candidate_id"], "check": name,
                        "status": c["status"], "required": name in required,
                        "evidence": {k: v for k, v in c.items() if k != "status" and not isinstance(v, (dict, list))}})
    return out


def build_a67_packet(spec, report: dict, pipeline_dir, evidence: dict, run_id: str, top_k: int,
                     required_checks=DEFAULT_REQUIRED_CHECKS, datasets: dict | None = None) -> dict:
    """datasets: agents/datasets.gather_datasets output for these candidates (records + code-computed metrics)."""
    candidates = candidate_rows(report, top_k)
    if spec.project_type == FIXTURE_PROJECT_TYPE:  # only the checks this mode reports can be required
        required_checks = [c for c in required_checks if c in (report.get("structural_checks") or {})]
    checks = _structural_checks(spec, report, candidates, pipeline_dir, tuple(required_checks))
    datasets = datasets or {"datasets": [], "records": [], "metrics": [], "gaps": []}
    physical = report.get("physical") or {}
    return _finish({
        "run_id": run_id, "schema_version": CONTRACT_VERSION, "mode": mode_of(spec), "role": "A67",
        "final_candidates": candidates,
        "structural_checks": checks,
        "property_checks": [c for c in checks if c["check"] in ("zeopp_pores", "pxrd_comparison", "nmr_comparison")],
        "energy_results": {"base_pool_minima_eV": report.get("base_pool_minima_eV"),
                           "corrected_pool_minima_eV": report.get("corrected_pool_minima_eV"),
                           "ranking_coverage": report.get("ranking_coverage"), "competition": report.get("competition"),
                           "embedding": (physical.get("embedding") or {}).get("status"),
                           "units": "eV per formula unit; compare only within one fidelity and comparison group"},
        "numerical_verdicts": {"pipeline_status": report.get("pipeline_status"), "stage_status": report.get("stage_status"),
                               "winning_topology": report.get("winning_topology"),
                               "base_winning_topology": report.get("base_winning_topology"),
                               "selected_models": {t: (s.get("model_selection") or {}).get("selected_model")
                                                   for t, s in (report.get("topologies") or {}).items()},
                               "encoding_verification": {t: ((s.get("encoding") or {}).get("verification") or {}).get("status")
                                                         for t, s in (report.get("topologies") or {}).items()},
                               "holdout": physical.get("holdout")},
        "literature_evidence": {"sources": evidence["sources"], "passages": evidence["passages"]},
        "dataset_evidence": {"datasets": datasets["datasets"], "records": datasets["records"],
                             "note": "same_family records share net/elements, not identity; dataset energies use "
                                     "other protocols and are not comparable with pipeline energies"},
        "comparison_metrics": datasets["metrics"],
        "coverage": {"not_executed": report.get("not_executed"), "retrieval_gaps": evidence["gaps"],
                     "dataset_gaps": datasets["gaps"],
                     "required_checks": list(required_checks)},
    })


# -- A4 (optimisation) ----------------------------------------------------------------------------------------
RANK_AGREEMENT_MIN = 0.5  # Spearman correlation below this between surrogate and evaluated energies = disagreement


def _spearman(a: list, b: list):
    from ..configuration_model import spearman
    return spearman(a, b) if len(a) >= 3 else None


def candidate_quality(spec, report: dict, pipeline_dir, assessment: dict | None) -> dict:
    """Code-computed facts on whether the fitted Hamiltonian ranked the evaluated candidates correctly."""
    rows = report.get("base_ranking") or []
    if spec.physical:
        tol = float(spec.physical["tolerances"]["target_relative_energy_tolerance"])
    else:
        tol = float(spec.config["model_selection"]["max_development_rmse_eV"])
    per = {}
    for tid in sorted({r["topology_id"] for r in rows}):
        mine = [r for r in rows if r["topology_id"] == tid and r.get("surrogate_base_energy_eV") is not None]
        if not mine:
            continue
        sur = [r["surrogate_base_energy_eV"] for r in mine]
        ev = [r["ranking_energy_eV"] for r in mine]
        s_best = min(range(len(mine)), key=lambda i: (sur[i], i))
        per[tid] = {"n_evaluated": len(mine), "spearman_surrogate_vs_evaluated": _spearman(sur, ev),
                    "max_abs_surrogate_error_eV": max(abs(x - y) for x, y in zip(sur, ev)),
                    "surrogate_best": mine[s_best]["configuration_id"],
                    "evaluated_best": mine[min(range(len(mine)), key=lambda i: (ev[i], i))]["configuration_id"],
                    "surrogate_best_is_evaluated_best_within_tolerance": ev[s_best] <= min(ev) + tol}
    selections = fit_diagnostics(report, pipeline_dir)
    candidates = (assessment or {}).get("candidates") or []
    facts = {
        "surrogate_ranking_disagrees_with_dft": any(p["spearman_surrogate_vs_evaluated"] is not None
                                                    and p["spearman_surrogate_vs_evaluated"] < RANK_AGREEMENT_MIN
                                                    for p in per.values()),
        "surrogate_error_above_tolerance": any(p["max_abs_surrogate_error_eV"] > tol for p in per.values()),
        "surrogate_best_not_dft_best": any(not p["surrogate_best_is_evaluated_best_within_tolerance"] for p in per.values()),
        "model_inadequate": any(s["status"] == "MODEL_INADEQUATE" for s in selections.values())
                            or ((report.get("stage_detail") or {}).get("model_selection") or {}).get("error", {}).get("code")
                            == "MODEL_INADEQUATE",
        "structural_check_failed": any(c.get("structural_verdict") == "FAIL" for c in candidates),
        "assessment_incomplete": (assessment or {}).get("status") != "COMPLETE",
        "no_evaluated_candidates": not per,
    }
    return {"facts": facts, "per_topology": per, "tolerance_eV": tol, "rank_agreement_min": RANK_AGREEMENT_MIN,
            "evaluated_level": "periodic DFT (base)" if spec.physical else "synthetic evaluator (base)",
            "note": "surrogate = the fitted occupation Hamiltonian; evaluated = expensive energies of the same candidates"}


def build_a4_packet(spec, setup: dict, report: dict, pipeline_dir, assessment: dict | None, grid: dict,
                    history: list, run_id: str, round_index: int, max_rounds: int, screening: dict | None = None) -> dict:
    """The optimisation checkpoint after an assessment: facts, fit results and the Hamiltonian options A2 had."""
    options = build_a2_packet(spec, setup, run_id, screening=screening)
    quality = candidate_quality(spec, report, pipeline_dir, assessment)
    tried = {t: sorted({m for g in history for m in g.get(t, [])}) for t in grid}
    catalogue = [m["model_id"] for m in options["model_catalogue"]]
    quality["facts"]["untried_models_available"] = any(set(catalogue) - set(ids) for ids in tried.values())
    return _finish({
        "run_id": run_id, "schema_version": CONTRACT_VERSION, "mode": mode_of(spec), "role": "A4",
        "round": round_index, "max_rounds": max_rounds, "rounds_left": max_rounds - round_index,
        "current_model_grid": grid, "grid_history": history, "tried_models": tried,
        "candidate_quality": quality, "fit_results": fit_diagnostics(report, pipeline_dir),
        "assessment": {"status": (assessment or {}).get("status"), "unresolved_codes": (assessment or {}).get("unresolved_codes"),
                       "candidates": [{k: c.get(k) for k in ("candidate_id", "structural_verdict", "issue_codes")}
                                      for c in (assessment or {}).get("candidates") or []]},
        "pipeline": {"status": report.get("pipeline_status"),
                     "stage_status": {k: v for k, v in (report.get("stage_status") or {}).items()
                                      if k in ("model_selection", "encoding", "optimization", "evaluation", "ranking")}},
        "hamiltonian_options": {k: options[k] for k in ("topologies", "composition_constraints", "feature_catalogue",
                                                         "model_catalogue", "encoding_catalogue", "compilation_options",
                                                         "limits")},
    })
