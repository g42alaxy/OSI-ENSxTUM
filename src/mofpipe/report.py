"""Task 09 (report): static provenance report; every number traces to a stored artifact."""
from __future__ import annotations

import json
from pathlib import Path

from .contract import StageStatus, make_record

NOT_EXECUTED = ["atomic construction (PORMAKE/ToBaCCo)", "ML-potential relaxation and calibration", "periodic DFT",
                "embedding correction", "active-space FCI/CASCI", "VQE", "SQD on hardware", "QAOA simulation",
                "quadratization of higher-order models", "MOFChecker/CrystalNets/Zeo++", "phonon/elastic stability",
                "PXRD/NMR comparison", "reference phase-competition energies", "live literature review",
                "finite-temperature ensembles (out of scope)"]
EXECUTED_IN_PHYSICAL_PROJECTS = ("atomic construction (PORMAKE/ToBaCCo)", "periodic DFT")
NOT_EXECUTED_PHYSICAL = ["PORMAKE/ToBaCCo builder (parent-substitution builder used)", "linker orientation search (T01.4)",
                         "calibrated model uncertainty (T03.3)", "triplet cluster orbits (T03.1)",
                         "CrystalNets/MOFid net identification"]
PHYSICAL_EXECUTED = ("MOFChecker/CrystalNets/Zeo++", "reference phase-competition energies")


def not_executed(spec) -> list:
    executed = set()
    if spec.config["optimizer"].get("quantum_enabled"):
        executed.add("QAOA simulation")
    if spec.config["qubo"].get("higher_order") == "quadratize":
        executed.add("quadratization of higher-order models")
    if spec.project_type == "synthetic_fixture":
        return [x for x in NOT_EXECUTED if x not in executed]
    physical = spec.physical or {}
    if (physical.get("protocols") or {}).get("embedding"):
        executed.add("embedding correction")  # its validation status is in the embedding section
    out = [x for x in NOT_EXECUTED if x not in EXECUTED_IN_PHYSICAL_PROJECTS + PHYSICAL_EXECUTED + tuple(executed)]
    out += NOT_EXECUTED_PHYSICAL
    if not any((physical.get("transfer_topologies") or {}).values()):
        out.append("larger-supercell transfer test (T01.5): no transfer cell declared")
    if not (physical.get("protocols") or {}).get("stopping"):
        out.append("active learning (T08): protocols.stopping is null")
    if ((physical.get("protocols") or {}).get("validation") or {}).get("reference_phases") is None:
        out.append("reference phase-competition energies (T09.2): no reference set declared")
    return out


def _physical_section(spec, store, ctx, detail) -> dict:
    labels, selection = detail.get("labels") or {}, detail.get("model_selection") or {}
    refinement = ctx.get("refinement")
    competition = ctx.get("competition") or {}
    dft_results = [v for k, v in ctx.items() if k.startswith("dftcalc/") and v.get("status") == "SUCCEEDED"]
    identity = next((r.get("engine_identity") for r in dft_results if r.get("engine_identity") is not None), None)
    return {
        "dft_convergence": labels.get("dft_convergence"), "split_counts": labels.get("split_counts"),
        "transformation_events": (labels.get("transformation_events") or []) + ((detail.get("refinement") or {}).get("transformation_events") or []),
        "failed_labels": (labels.get("failed_labels") or []) + ((detail.get("refinement") or {}).get("failed_labels") or []),
        "holdout": selection.get("holdout"), "labelled_ground_state_recovered": selection.get("labelled_ground_state_recovered"),
        "atomic_construction": (detail.get("structures") or {}).get("atomic_construction"),
        "cluster_catalogue": (detail.get("structures") or {}).get("cluster_catalogue"),
        "supercell_replication": (detail.get("structures") or {}).get("supercell_replication"),
        "refinement": ({"stop_reason": refinement["stop_reason"], "criterion": refinement["criterion"],
                        "rounds": [{k: r.get(k) for k in ("round", "n_labels", "selected", "cv_rmse_eV", "diagnosis",
                                                          "stable_rounds", "acquired")} for r in refinement["rounds"]]}
                       if refinement else None),
        "supercell_transfer": {k[len("transfer/"):]: {f: v[f] for f in ("status", "transfer_rmse_eV", "tolerance_eV",
                                                                        "replication", "native")}
                               for k, v in ctx.items() if k.startswith("transfer/")},
        "competition": {k: competition.get(k) for k in ("status", "coverage", "reference_set", "missing_references",
                                                         "reference_mixture", "mixing_energies", "reason")},
        "structure_validation": (detail.get("validation") or {}).get("finalists"),
        "relaxed_geometry_classes": (detail.get("evaluation") or {}).get("relaxed_geometry_classes"),
        "geometry_ids": sorted(v["artifact_id"] for k, v in ctx.items() if k.startswith("geometry/")),
        "evaluation_not_evaluated": (detail.get("evaluation") or {}).get("not_evaluated"),
        "search_coverage": {tid: {"n_feasible": (ctx.get(f"space/{tid}") or {}).get("n_feasible"),
                                  "n_orbits": (ctx.get(f"space/{tid}") or {}).get("n_symmetry_orbits"),
                                  "n_labelled": (ctx.get("train") or {}).get("per_topology", {}).get(tid),
                                  "n_holdout": (ctx.get("holdout") or {}).get("per_topology", {}).get(tid),
                                  "n_pool_evaluated": (ctx.get(f"pool/{tid}") or {}).get("n_selected"),
                                  "solver": (ctx.get(f"solver/{tid}") or {}).get("optimality_status")}
                            for tid in spec.topologies},
        "reproducibility_manifest": {
            "project_fingerprint": spec.fingerprint, "sources": [{k: s.get(k) for k in ("topology_id", "cif", "sha256", "source")}
                                                                  for s in spec.physical["sources"]],
            "dft_protocol_id": (spec.physical["protocols"].get("dft") or {}).get("protocol_id"),
            "dft_fidelity": next((r["base_fidelity"] for r in dft_results), None), "engine_identity": identity,
            "dft_jobs_submitted": store.state["counters"].get("dft_submissions", 0),
            "dft_cache_hits": store.state["counters"].get("dft_cache_hits", 0),
            "rerun": "python -m mofpipe run --project <project.json> --run-dir <dir>; identical inputs reuse every job"},
        "embedding": _embedding_section(ctx, detail),
    }


def _embedding_section(ctx, detail) -> dict | None:
    validation = ctx.get("embedding_validation")
    if validation is None:
        return None
    corrections = [v for k, v in sorted(ctx.items()) if k.startswith("correction/")]
    return {"scheme": validation.get("scheme"), "label": validation.get("label"), "status": validation["status"],
            "energy_expression": validation.get("energy_expression"), "unresolved": validation["unresolved"],
            "checks": {k: v["status"] for k, v in validation["checks"].items()},
            "boundary_convergence": validation["checks"].get("boundary_convergence"),
            "reference_benchmarks": (validation["checks"].get("reference_benchmarks") or {}).get("benchmarks"),
            "validation_id": validation["artifact_id"], "stage": (detail.get("embedding") or {}).get("ranking"),
            "corrections": [{"orbit_id": c.get("orbit_id"), "record_type": c["record_type"],
                             "correction_status": c.get("correction_status"),
                             "correction_eV_per_fu": c.get("correction_eV_per_fu", c.get("unvalidated_correction_eV_per_fu")),
                             "failure": c.get("failure"), "artifact_id": c["artifact_id"]} for c in corrections]}


def _topology_section(spec, ctx, tid) -> dict:
    sel, out = ctx.get(f"selection/{tid}"), {}
    if sel:
        out["model_selection"] = {"status": sel["status"], "selected_model": sel["selected_model"], "reason": sel["reason"],
                                  "candidates": [{k: c.get(k) for k in ("model_id", "status", "n_parameters", "polynomial_degree",
                                                                         "coefficients_exact", "training_rmse_eV",
                                                                         "development_rmse_eV", "cv_rmse_eV",
                                                                         "low_energy_recall", "passes", "failure")}
                                                 for c in sel["candidates"]]}
    ham = ctx.get(f"hamiltonian/{tid}")
    if ham:
        out["hamiltonian"] = {k: ham[k] for k in ("artifact_id", "model_id", "features", "coefficients_exact",
                                                  "coefficients_eV", "reference_offset_eV", "polynomial_degree")}
    enc, ver = ctx.get(f"encoding/{tid}"), ctx.get(f"encoding_verification/{tid}")
    if enc:
        out["encoding"] = {"artifact_id": enc["artifact_id"], "target": enc["target"], "max_degree": enc["max_degree"],
                           "constant_eV": enc["constant_eV"], "physical_offset_eV": enc["physical_offset_eV"],
                           "penalty_eV": enc["penalty_eV"], "n_quadratic_terms": len(enc["quadratic_eV"]),
                           "n_higher_order_terms": len(enc["higher_order_terms"]),
                           "verification": {k: ver[k] for k in ("status", "n_states_checked", "max_abs_deviation_exact",
                                                                "penalized_minimum_eV", "all_penalized_argmins_feasible")}}
    sol = ctx.get(f"solver/{tid}")
    if sol:
        topo = spec.topologies[tid]
        from .structures import orbit_id
        out["solver"] = {"artifact_id": sol["artifact_id"], "solver": sol["solver"], "optimality_status": sol["optimality_status"],
                         "n_feasible": sol["n_feasible"], "base_minimum_energy_eV": sol["minimum_energy_eV"],
                         "minimizing_bitstrings": sol["minimizing_bitstrings"],
                         "ground_state_orbits": sorted({orbit_id(topo, [int(c) for c in b]) for b in sol["minimizing_bitstrings"]})}
    cset, pool = ctx.get(f"candidate_set/{tid}"), ctx.get(f"pool/{tid}")
    if cset:
        out["n_symmetry_orbits"] = cset["n_unique_orbits"]
    if pool:
        out["selected_pool"] = [{k: r[k] for k in ("orbit_id", "bits", "predicted_base_energy_eV", "surrogate_rank",
                                                   "n_degenerate_orbits", "selection_reason")} for r in pool["pool"]]
    return out


def build_report(spec, store, ctx, detail, status) -> dict:
    corrected, base = ctx.get("ranking/corrected"), ctx.get("ranking/base")
    executed = {k: v for k, v in status.items() if v == StageStatus.COMPLETED.value}
    lit = spec.literature or {}
    final = "COMPLETED" if all(status.get(s) in (StageStatus.COMPLETED.value, StageStatus.DISABLED_BY_CONFIG.value,
                                                  StageStatus.SKIPPED_NO_ATOMIC_GEOMETRY.value)
                               for s in status if s != "report") else "FAILED_OR_INCOMPLETE"
    extra = {} if spec.project_type == "synthetic_fixture" else {
        "project_type": spec.project_type, "physical": _physical_section(spec, store, ctx, detail),
        "base_pool_minima_eV": base["pool_minima_eV"] if base else None}
    return make_record(
        "ReportArtifacts", spec.provenance, **extra, project_id=spec.project_id, case=spec.case,
        correction_mode=spec.correction_mode,
        pipeline_status=final, project_fingerprint=spec.fingerprint, stage_order=list(status) + ["report"],
        runtime_agents_enabled=False, llm_calls=store.state["counters"].get("llm_calls", 0), network_calls=0,
        stage_status=dict(status), completed_software_stages=sorted(executed),
        stage_detail={k: v for k, v in detail.items() if k in ("structures", "labels", "refinement", "model_selection",
                                                               "supercell_checks", "encoding", "optimization", "candidates",
                                                               "evaluation", "embedding", "ranking", "validation")},
        topologies={tid: _topology_section(spec, ctx, tid) for tid in spec.topologies},
        winning_topology=(corrected or base or {}).get("winning_topology"),
        base_winning_topology=base["winning_topology"] if base else None,
        corrected_pool_minima_eV=corrected["pool_minima_eV"] if corrected else None,
        base_ranking=base["ranked"] if base else None, corrected_ranking=corrected["ranked"] if corrected else None,
        ranking_coverage=(corrected or base or {}).get("coverage"),
        competition=ctx["competition"]["status"] if "competition" in ctx else StageStatus.NOT_RUN.value,
        structural_checks=detail.get("validation", {}).get("checks"),
        #TODO: Task 09 — benchmark sections (Kang arrangements, Cai topology competition, Fe–MOF-74 electronic
        #TODO: workflow) and ablations; live literature review instead of the static manifest.
        literature={"mode": lit.get("mode"), "live_review": "NOT_RUN",
                    "entries": [{k: e.get(k) for k in ("id", "title", "doi", "url", "supports", "does_not_supply")}
                                for e in lit.get("entries", [])],
                    "note": "Listed sources and their scope only; no numerical comparison with published results."},
        not_executed=not_executed(spec),
        artifact_index={k: v["artifact_id"] for k, v in sorted(ctx.items()) if isinstance(v, dict) and "artifact_id" in v},
    )


def _md_table(rows, cols) -> str:
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "".join("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n" for r in rows)


def write_report(run_dir: Path, report: dict) -> None:
    run_dir = Path(run_dir)
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [f"# Pipeline report — {report['project_id']} ({report['case']}, correction={report['correction_mode']})", "",
             (f"**Provenance: {report['provenance']}.** "
              + ("Abstract substitution graphs; not atomistic MOF predictions. " if "physical" not in report else
                 "CIF-derived slot lattice with periodic DFT labels" + ("; labelled SYNTHETIC, not physical evidence. "
                                                                        if report["provenance"] == "SYNTHETIC" else ". "))
              + f"Runtime agents: disabled. LLM calls: {report['llm_calls']}."), "",
             (f"Pipeline status: **{report['pipeline_status']}** — winning topology (corrected pool): "
              f"**{report['winning_topology']}** (base: {report['base_winning_topology']})" if "physical" not in report else
              f"Pipeline status: **{report['pipeline_status']}** — lowest DFT energy (base level, ties within tolerance "
              f"listed below): **{(report['base_ranking'] or [{}])[0].get('configuration_id')}**"
              + (f"; with the validated embedding correction: **{report['corrected_ranking'][0]['configuration_id']}**"
                 if report.get("corrected_ranking") else "")), "", "## Stages", "",
             _md_table([{"stage": k, "status": report["stage_status"].get(k, "COMPLETED")} for k in report["stage_order"]],
                       ["stage", "status"]), "",
             "## Per topology", ""]
    for tid, sec in report["topologies"].items():
        ms, sol = sec.get("model_selection", {}), sec.get("solver", {})
        lines += [f"### {tid}", "",
                  f"- selected model: `{ms.get('selected_model')}` — {ms.get('reason')}",
                  f"- encoding: {sec.get('encoding', {}).get('target')} (identity check: "
                  f"{sec.get('encoding', {}).get('verification', {}).get('status')})",
                  f"- feasible states: {sol.get('n_feasible')}, symmetry orbits: {sec.get('n_symmetry_orbits')}",
                  f"- base minimum: {sol.get('base_minimum_energy_eV')} eV at {sol.get('minimizing_bitstrings')} "
                  f"(orbits {sol.get('ground_state_orbits')})", ""]
    if report.get("physical"):
        ph = report["physical"]
        lines += ["## Physical-project checks", "", f"- DFT cutoff convergence: {ph['dft_convergence']}",
                  f"- split counts: {ph['split_counts']}", f"- final holdout: {ph['holdout']}",
                  f"- transformation events: {len(ph['transformation_events'] or [])}, failed labels: "
                  f"{len(ph['failed_labels'] or [])}", ""]
        if ph.get("refinement"):
            ref = ph["refinement"]
            lines += ["## Refinement (active learning)", "", f"Stop reason: **{ref['stop_reason']}** — {ref['criterion']}", "",
                      _md_table([{**r, "acquired": len(r.get("acquired") or [])} for r in ref["rounds"]],
                                ["round", "n_labels", "selected", "cv_rmse_eV", "stable_rounds", "acquired"]), ""]
        if ph.get("supercell_transfer"):
            lines += ["## Supercell transfer checks", "",
                      _md_table([{"transfer_cell": k, **{f: v[f] for f in ("status", "transfer_rmse_eV", "tolerance_eV")}}
                                 for k, v in ph["supercell_transfer"].items()],
                                ["transfer_cell", "status", "transfer_rmse_eV", "tolerance_eV"]), ""]
        if report.get("base_ranking"):
            lines += ["## DFT ranking (base level, no correction; pooled over topologies)", "",
                      _md_table(report["base_ranking"], ["rank", "configuration_id", "topology_id", "base_energy_eV",
                                                         "surrogate_base_energy_eV", "n_degenerate", "base_fidelity"]),
                      "", f"Coverage: {report['ranking_coverage']}", ""]
        comp = ph.get("competition") or {}
        if comp.get("mixing_energies"):
            lines += ["## Reference-phase competition", "", f"Status: **{comp['status']}** — coverage: {comp['coverage']}", "",
                      _md_table(comp["mixing_energies"], ["configuration_id", "topology_id", "mixing_energy_eV_per_fu",
                                                          "versus_references"]), ""]
        emb = ph.get("embedding")
        if emb:
            lines += ["## Embedding correction (finite-cluster subtractive)", "",
                      f"Validation: **{emb['status']}** — {emb['stage']}. {emb['label']}.", "",
                      f"`{emb['energy_expression']}`", "",
                      _md_table([{"check": k, "status": v} for k, v in emb["checks"].items()], ["check", "status"]), "",
                      _md_table(emb["corrections"], ["orbit_id", "record_type", "correction_status", "correction_eV_per_fu"]),
                      ""]
        if ph.get("structure_validation"):
            lines += ["## Structural validation of finalists", "",
                      _md_table([{**f, "external_tools": ", ".join(f"{k}: {v}" for k, v in f["external_tools"].items())}
                                 for f in ph["structure_validation"]],
                                ["configuration_id", "status", "failed_checks", "external_tools"]), ""]
    if report.get("corrected_ranking"):
        lines += ["## Corrected pooled ranking", "",
                  _md_table(report["corrected_ranking"], ["rank", "orbit_id", "topology_id", "base_energy_eV", "correction_eV",
                                                          "corrected_energy_eV", "n_degenerate", "corrected_fidelity"]),
                  "", f"Coverage: {report['ranking_coverage']}", ""]
    lines += ["## Not executed in this run", ""] + [f"- {x}" for x in report["not_executed"]] + [""]
    (run_dir / "report.md").write_text("\n".join(lines))
