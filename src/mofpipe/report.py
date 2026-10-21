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
                         "larger-supercell transfer test (T01.5)", "calibrated model uncertainty (T03.3)"]


def not_executed(spec) -> list:
    if spec.project_type == "synthetic_fixture":
        return NOT_EXECUTED
    return [x for x in NOT_EXECUTED if x not in EXECUTED_IN_PHYSICAL_PROJECTS] + NOT_EXECUTED_PHYSICAL


def _physical_section(ctx, detail) -> dict:
    labels, selection = detail.get("labels") or {}, detail.get("model_selection") or {}
    return {
        "dft_convergence": labels.get("dft_convergence"), "split_counts": labels.get("split_counts"),
        "transformation_events": labels.get("transformation_events"), "failed_labels": labels.get("failed_labels"),
        "holdout": selection.get("holdout"), "labelled_ground_state_recovered": selection.get("labelled_ground_state_recovered"),
        "atomic_construction": (detail.get("structures") or {}).get("atomic_construction"),
        "geometry_ids": sorted(v["artifact_id"] for k, v in ctx.items() if k.startswith("geometry/")),
        "evaluation_not_evaluated": (detail.get("evaluation") or {}).get("not_evaluated"),
    }


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
        "project_type": spec.project_type, "physical": _physical_section(ctx, detail),
        "base_pool_minima_eV": base["pool_minima_eV"] if base else None}
    return make_record(
        "ReportArtifacts", spec.provenance, **extra, project_id=spec.project_id, case=spec.case,
        correction_mode=spec.correction_mode,
        pipeline_status=final, project_fingerprint=spec.fingerprint, stage_order=list(status) + ["report"],
        runtime_agents_enabled=False, llm_calls=store.state["counters"].get("llm_calls", 0), network_calls=0,
        stage_status=dict(status), completed_software_stages=sorted(executed),
        stage_detail={k: v for k, v in detail.items() if k in ("structures", "labels", "refinement", "model_selection",
                                                               "encoding", "optimization", "candidates", "evaluation", "ranking")},
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
              f"listed below): **{(report['base_ranking'] or [{}])[0].get('configuration_id')}**"), "", "## Stages", "",
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
        if report.get("base_ranking"):
            lines += ["## DFT ranking (base level, no correction)", "",
                      _md_table(report["base_ranking"], ["rank", "configuration_id", "base_energy_eV",
                                                         "surrogate_base_energy_eV", "n_degenerate", "base_fidelity"]),
                      "", f"Coverage: {report['ranking_coverage']}", ""]
    if report.get("corrected_ranking"):
        lines += ["## Corrected pooled ranking", "",
                  _md_table(report["corrected_ranking"], ["rank", "orbit_id", "topology_id", "base_energy_eV", "correction_eV",
                                                          "corrected_energy_eV", "n_degenerate", "corrected_fidelity"]),
                  "", f"Coverage: {report['ranking_coverage']}", ""]
    lines += ["## Not executed in this run", ""] + [f"- {x}" for x in report["not_executed"]] + [""]
    (run_dir / "report.md").write_text("\n".join(lines))
