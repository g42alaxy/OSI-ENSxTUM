"""Deterministic orchestrator: one authority for workflow state (10_AGENT_ARCHITECTURE).

Stages commit immutable artifacts to the run store. A re-run with identical
inputs reuses committed stages and never resubmits evaluator jobs.
"""
from __future__ import annotations

from fractions import Fraction

from . import candidates, configuration_model, datasets, dft, encoding, framework, optimization, ranking, refinement, structures
from .contract import PipelineError, StageStatus, content_hash, make_record
from .electronic import SyntheticFixtureBackend, electronic_stage_status
from .intake import FIXTURE_PROJECT_TYPE, validate_project
from .report import build_report, write_report
from .store import RunStore

#TODO: Task 10 — AiiDA WorkChain engine with CalcJobs for external DFT/QPU jobs; this local engine is the
#TODO: development stand-in obeying the same interfaces. Concurrency (optimistic state commits) is not implemented.
STAGES = ("intake", "structures", "screening", "labels", "refinement", "model_selection", "encoding",
          "optimization", "candidates", "evaluation", "ranking", "validation", "report")
# Missing inputs, resources or budget block a stage; they are not scientific failures.
BLOCKING_CODES = {"BACKEND_UNAVAILABLE", "BUDGET_EXHAUSTED", "SCIENTIFIC_SPEC_MISSING", "DFT_SETTINGS_NOT_CONVERGED"}


class StageOutcome:
    def __init__(self, status: StageStatus, artifacts: dict | None = None, detail=None, halt: bool = False):
        self.status, self.artifacts, self.detail, self.halt = status, artifacts or {}, detail, halt


class Pipeline:
    def __init__(self, spec, run_dir):
        self.spec = spec
        self.physical = spec.project_type != FIXTURE_PROJECT_TYPE
        self.store = RunStore(run_dir)
        self.store.bind_project(spec.fingerprint, spec.project_type, spec.provenance)
        self.ctx: dict = {}      # name -> artifact record
        self.detail: dict = {}   # stage -> JSON detail
        self.status: dict = {}
        self._backend = None

    @property
    def backend(self):
        if self._backend is None:
            backend_cls = dft.PeriodicDFTBackend if self.physical else SyntheticFixtureBackend
            self._backend = backend_cls(self.spec, self.store)
        return self._backend

    def run(self) -> dict:
        upstream = self.spec.fingerprint
        halted = False
        for name in STAGES:
            if halted and name != "report":  # a halted run still reports what it did and why
                self.status[name] = StageStatus.NOT_RUN.value
                continue
            fingerprint = content_hash({"stage": name, "upstream": upstream})
            previous = self.store.stage(name)
            if previous and previous["input_fingerprint"] == fingerprint and all(
                    self.store.has(a) for a in previous["artifact_ids"].values()):
                for key, aid in previous["artifact_ids"].items():
                    self.ctx[key] = self.store.get(aid)
                self.detail[name], self.status[name] = previous["detail"], previous["status"]
                self.store.log("stage_reused", stage=name)
                halted = previous["detail"].get("halt", False) if isinstance(previous["detail"], dict) else False
                upstream = content_hash({"prev": fingerprint, "ids": previous["artifact_ids"]})
                continue
            try:
                prefix = "_physical_" if self.physical and hasattr(self, f"_physical_{name}") else "_stage_"
                outcome = getattr(self, f"{prefix}{name}")()
            except PipelineError as exc:
                status = StageStatus.BLOCKED if exc.code in BLOCKING_CODES else StageStatus.FAILED
                outcome = StageOutcome(status, detail={"error": exc.as_dict()}, halt=True)
            ids = {key: self.store.put(record) for key, record in outcome.artifacts.items()}
            self.ctx.update(outcome.artifacts)
            detail = dict(outcome.detail or {}, halt=outcome.halt)
            self.detail[name], self.status[name] = detail, outcome.status.value
            self.store.commit_stage(name, outcome.status.value, fingerprint, ids, detail)
            halted = outcome.halt
            upstream = content_hash({"prev": fingerprint, "ids": ids})
        return self.ctx.get("report") or {}

    # -- stages ------------------------------------------------------------
    def _stage_intake(self):
        report = validate_project(self.spec)
        ok = report["status"] == StageStatus.COMPLETED.value
        return StageOutcome(StageStatus.COMPLETED if ok else StageStatus.BLOCKED, {"project_validation": report},
                            {"errors": report["errors"]}, halt=not ok)

    def _stage_structures(self):
        spec, arts, detail = self.spec, {}, {"symmetry": {}, "swap_reachability": {}}
        spaces = {}
        for tid, topo in spec.topologies.items():
            space = structures.define_decoration_space(spec, topo)
            spaces[tid] = arts[f"space/{tid}"] = space
            states = list(structures.feasible_states(spec.n_sites, spec.n_B))
            detail["symmetry"][tid] = configuration_model.check_feature_symmetry(topo, spec.feature_order, states)
            detail["swap_reachability"][tid] = structures.swap_graph_connected(topo, spec.n_sites, spec.n_B)
        detail["candidate_file"] = structures.check_candidate_file(spec, spaces)
        first_tid = next(iter(spec.topologies))
        sample = structures.make_decoration(spec, spec.topologies[first_tid], spec.candidate_rows[0]["bits"])
        detail["atomic_construction"] = structures.build_framework(spec, spec.topologies[first_tid], sample)
        return StageOutcome(StageStatus.COMPLETED, arts, detail)

    def _stage_screening(self):
        return StageOutcome(StageStatus.COMPLETED, {"screening": structures.screen_topologies(self.spec)})

    def _stage_labels(self):
        spec = self.spec
        train = datasets.ingest_labels(spec, spec.training_rows, "training", spec.config["inputs"]["training"])
        dev = datasets.ingest_labels(spec, spec.development_rows, "development", spec.config["inputs"]["development"])
        datasets.check_energy_compatibility(spec.training_rows + spec.development_rows, spec.provenance)
        spaces = {tid: self.ctx[f"space/{tid}"] for tid in spec.topologies}
        split = datasets.freeze_split(spec, spaces)
        datasets.check_snapshot_against_split(train, split, "training")
        datasets.check_snapshot_against_split(dev, split, "development")
        return StageOutcome(StageStatus.COMPLETED, {"train": train, "dev": dev, "split": split},
                            {"split_counts": split["counts"], "n_training": train["per_topology"],
                             "n_development": dev["per_topology"]})

    def _stage_refinement(self):
        mode = self.spec.config["active_learning"]["mode"]
        if mode != "enabled":
            return StageOutcome(StageStatus.DISABLED_BY_CONFIG, detail={"mode": mode})
        report, train = refinement.run_active_learning(self.spec, self.store, self.backend, self.ctx["train"],
                                                       self.ctx["dev"], self.ctx["split"])
        status = StageStatus.COMPLETED if report["stop_reason"] == "CONVERGED_WITHIN_SCOPE" else StageStatus.INCONCLUSIVE
        return StageOutcome(status, {"refinement": report, "train": train}, {"stop_reason": report["stop_reason"]})

    def _stage_model_selection(self):
        arts, failed = {}, []
        for tid, topo in self.spec.topologies.items():
            sel = configuration_model.fit_and_select(self.spec, topo, self.ctx["train"], self.ctx["dev"])
            arts[f"selection/{tid}"] = sel
            if sel["status"] == "MODEL_INADEQUATE":
                failed.append(tid)
            else:
                arts[f"hamiltonian/{tid}"] = configuration_model.hamiltonian_from_selection(self.spec, topo, sel)
        detail = {"selected": {t: arts[f"selection/{t}"]["selected_model"] for t in self.spec.topologies}}
        if failed:
            return StageOutcome(StageStatus.FAILED, arts, {**detail, "error": {"code": "MODEL_INADEQUATE", "detail": failed}}, halt=True)
        return StageOutcome(StageStatus.COMPLETED, arts, detail)

    def _stage_encoding(self):
        arts, detail = {}, {}
        for tid, topo in self.spec.topologies.items():
            ham = self.ctx[f"hamiltonian/{tid}"]
            try:
                enc = encoding.compile_encoding(self.spec, topo, ham, "QUBO")
                qubo_status = {"status": StageStatus.COMPLETED.value}
            except PipelineError as exc:
                if exc.code != encoding.QUADRATIZATION_REQUIRED:
                    raise
                # Policy retain_HUBO_or_explicitly_reject: keep the cubic model for direct enumeration.
                qubo_status = {"status": StageStatus.UNSUPPORTED.value, "error": exc.as_dict()}
                enc = encoding.compile_encoding(self.spec, topo, ham, "HUBO")
            verification = encoding.verify_encoding(self.spec, topo, ham, enc)
            arts[f"encoding/{tid}"], arts[f"encoding_verification/{tid}"] = enc, verification
            detail[tid] = {"target": enc["target"], "qubo_compilation": qubo_status,
                           "n_higher_order_terms": len(enc["higher_order_terms"]),
                           "verification": verification["status"]}
            if verification["status"] != StageStatus.COMPLETED.value:
                return StageOutcome(StageStatus.FAILED, arts, {**detail, "error": {"code": "ENCODING_IDENTITY_FAILED", "detail": tid}}, halt=True)
        return StageOutcome(StageStatus.COMPLETED, arts, detail)

    def _stage_optimization(self):
        arts = {}
        for tid, topo in self.spec.topologies.items():
            arts[f"solver/{tid}"] = optimization.exact_enumeration(self.spec, topo, self.ctx[f"hamiltonian/{tid}"],
                                                                   self.ctx[f"encoding/{tid}"])
        return StageOutcome(StageStatus.COMPLETED, arts, {"quantum": optimization.quantum_solver_status(self.spec)})

    def _stage_candidates(self):
        arts = {}
        for tid, topo in self.spec.topologies.items():
            cset = candidates.deduplicate_candidates(self.spec, topo, self.ctx[f"solver/{tid}"])
            arts[f"candidate_set/{tid}"] = cset
            arts[f"pool/{tid}"] = candidates.select_pool(self.spec, cset)
        return StageOutcome(StageStatus.COMPLETED, arts,
                            {"n_unique_orbits": {t: arts[f"candidate_set/{t}"]["n_unique_orbits"] for t in self.spec.topologies},
                             "n_pool": {t: arts[f"pool/{t}"]["n_selected"] for t in self.spec.topologies}})

    def _stage_evaluation(self):
        pool_rows = [row for tid in self.spec.topologies for row in self.ctx[f"pool/{tid}"]["pool"]]
        requests = [{"topology_id": r["topology_id"], "bits": r["bits"]} for r in pool_rows]
        # Fixture path. Physical projects: _physical_evaluation (build_framework -> periodic DFT).
        job_id = self.backend.submit(requests, purpose="evaluate_candidate_pool")
        train = self.ctx["train"]
        results = self.backend.collect(job_id, train["fidelity"], train["comparison_group"])
        arts = {f"calc/{r['orbit_id']}": r for r in results}
        split = self.ctx["split"]["per_topology"]
        residuals = [abs(Fraction(r["base_energy_exact"]) - Fraction(p["predicted_base_energy_exact"]))
                     for r, p in zip(results, pool_rows) if r["status"] == "SUCCEEDED"]
        detail = {"job_id": job_id, "backend": self.backend.name, "n_requests": len(requests),
                  "labels_used_for_training": False,
                  "pool_orbits_in_holdout_split": sorted(p["orbit_id"] for p in pool_rows
                                                         if p["orbit_id"] in split[p["topology_id"]]["holdout"]),
                  "surrogate_vs_evaluated_base_max_abs_eV": float(max(residuals)) if residuals else None,
                  "electronic": electronic_stage_status(self.spec)}
        return StageOutcome(StageStatus.COMPLETED, arts, detail)

    def _stage_ranking(self):
        pool_rows = [row for tid in self.spec.topologies for row in self.ctx[f"pool/{tid}"]["pool"]]
        results = [v for k, v in self.ctx.items() if k.startswith("calc/")]
        base = ranking.rank_candidates(self.spec, pool_rows, results, "base")
        corrected = ranking.rank_candidates(self.spec, pool_rows, results, "corrected")
        competition = ranking.evaluate_reference_mixtures(self.spec, corrected)
        status = StageStatus.COMPLETED if corrected["status"] == StageStatus.COMPLETED.value else StageStatus.INCOMPLETE
        return StageOutcome(status, {"ranking/base": base, "ranking/corrected": corrected, "competition": competition},
                            {"winning_topology": corrected["winning_topology"],
                             "base_winning_topology": base["winning_topology"],
                             "ranking_changed_by_correction": ranking.ranking_order_changed(base, corrected)})

    def _stage_validation(self):
        return StageOutcome(StageStatus.SKIPPED_NO_ATOMIC_GEOMETRY, detail={"checks": ranking.validate_structures(self.spec)})

    def _stage_report(self):
        report = build_report(self.spec, self.store, self.ctx, self.detail, self.status)
        write_report(self.store.root, report)
        return StageOutcome(StageStatus.COMPLETED, {"report": report})


    # -- physical projects (Milestone B: CIF slot lattice, DFT labels, grouped validation) ------------------
    def _physical_structures(self):
        spec, arts = self.spec, {}
        detail = {"symmetry": {}, "swap_reachability": {}, "atomic_construction": {}}
        for tid, topo in spec.topologies.items():
            arts[f"space/{tid}"] = structures.define_decoration_space(spec, topo)
            states = list(structures.feasible_states(spec.n_sites, spec.n_B))
            detail["symmetry"][tid] = configuration_model.check_feature_symmetry(topo, spec.feature_order, states)
            detail["swap_reachability"][tid] = structures.swap_graph_connected(topo, spec.n_sites, spec.n_B)
            # Rebuild the known parent (every slot keeps its parent linker) and compare with the parsed supercell.
            lattice = spec.physical["lattices"][tid]
            parent = {"configuration_id": f"{tid}:parent", "orbit_id": f"{tid}:parent", "topology_id": tid,
                      "species_by_slot": [s["parent_species"] for s in lattice["slots"]]}
            rebuilt = framework.build_framework(spec, topo, parent)
            reference = framework.geometry_hashes(lattice["structure"]["lattice"],
                                                  [a["element"] for a in lattice["structure"]["atoms"]],
                                                  [tuple(round(x, 6) for x in a["frac"]) for a in lattice["structure"]["atoms"]])[1]
            sample = structures.make_decoration(spec, topo, next(structures.feasible_states(spec.n_sites, spec.n_B)))
            built = framework.build_framework(spec, topo, sample)
            detail["atomic_construction"][tid] = {
                "parent_rebuild": rebuilt["status"], "parent_rebuild_identical": rebuilt.get("canonical_geometry_hash") == reference,
                "sample_decoration": sample["configuration_id"], "sample_build": built["status"],
                "sample_inventory": built.get("inventory"), "sample_failure": built.get("failure")}
            if rebuilt.get("canonical_geometry_hash") != reference or built["status"] != StageStatus.COMPLETED.value:
                return StageOutcome(StageStatus.FAILED, arts, {**detail, "error": {"code": "CONSTRUCTION_CHECK_FAILED",
                                                                                  "detail": tid}}, halt=True)
        return StageOutcome(StageStatus.COMPLETED, arts, detail)

    def _evaluate_decorations(self, topo, bits_list, purpose):
        """build_framework -> DFT relaxation + final single point per declared magnetic state (T06.2)."""
        records, results = {}, []
        for bits in bits_list:
            decoration = structures.make_decoration(self.spec, topo, bits)
            geometry = framework.build_framework(self.spec, topo, decoration)
            records[f"geometry/{decoration['configuration_id']}"] = geometry
            if geometry["status"] != StageStatus.COMPLETED.value:
                # A construction failure is a missing evaluation, never an energy.
                results.append({"decoration": decoration, "status": "BUILD_FAILED", "failure": geometry["failure"],
                                "selected": None, "outcomes": []})
                continue
            evaluation = dft.evaluate_geometry(self.backend, geometry, decoration, purpose)
            for o in evaluation["outcomes"]:
                for key in ("relaxation", "result"):
                    records[f"dftcalc/{o[key]['artifact_id']}"] = o[key]
            results.append({"decoration": decoration, **evaluation})
        return records, results

    def _physical_labels(self):
        spec, arts = self.spec, {}
        spaces = {tid: self.ctx[f"space/{tid}"] for tid in spec.topologies}
        split = datasets.freeze_physical_split(spec, spaces)
        arts["split"] = split
        detail = {"split_counts": split["counts"], "transformation_events": [], "failed_labels": []}
        rows, holdout_rows, group_of = [], [], {}
        for tid, topo in spec.topologies.items():
            groups = split["per_topology"][tid]
            # Plane-wave cutoff convergence of the relevant energy difference before any label is used.
            pair = [tuple(int(c) for c in o.split(":")[1]) for o in (groups["training"][0], groups["training"][-1])]
            built = [(framework.build_framework(spec, topo, d), d) for d in (structures.make_decoration(spec, topo, b) for b in pair)]
            if any(g["status"] != StageStatus.COMPLETED.value for g, _ in built):
                raise PipelineError("DFT_CONVERGENCE_STUDY_FAILED", "reference decorations could not be built")
            study = dft.difference_convergence_study(spec, self.backend, built)
            arts[f"dft_convergence/{tid}"] = study
            detail.setdefault("dft_convergence", {})[tid] = {k: study[k] for k in ("status", "deviation_from_highest_cutoff_eV",
                                                                                   "tolerance_eV", "protocol_cutoff_Ry")}
            if study["status"] != StageStatus.COMPLETED.value:
                return StageOutcome(StageStatus.BLOCKED, arts, {**detail, "error": {
                    "code": "DFT_SETTINGS_NOT_CONVERGED", "detail": f"{tid}: {study['deviation_from_highest_cutoff_eV']:.4g} eV "
                                                                    f"> {study['tolerance_eV']} eV per formula unit"}}, halt=True)
            labelled = groups["training"] + groups["holdout"]
            records, results = self._evaluate_decorations(
                topo, [tuple(int(c) for c in o.split(":")[1]) for o in labelled], "training_labels")
            arts.update(records)
            relaxed_groups = {}
            for r in results:
                orbit = r["decoration"]["orbit_id"]
                if r["status"] != "SUCCEEDED":
                    key = "transformation_events" if r["status"] == "TRANSFORMED" else "failed_labels"
                    detail[key].append({"orbit_id": orbit, "status": r["status"],
                                        "failure": r.get("failure") or r["selected"].get("failure")})
                    continue
                res = r["selected"]
                row = {"bits": res["bits"], "comparison_group": res["comparison_group"], "configuration_id": res["configuration_id"],
                       "energy_eV": res["base_energy_exact"], "energy_unit": res["energy_unit"], "fidelity": res["base_fidelity"],
                       "n_formula_units": res["n_formula_units"], "normalization": res["normalization"],
                       "orbit_id": orbit, "provenance": res["provenance"], "status": res["status"],
                       "topology_id": tid, "calculation_result_id": res["artifact_id"]}
                (holdout_rows if orbit in groups["holdout"] else rows).append(row)
                # Near-duplicate relaxations (identical relaxed geometry) share one CV group.
                group_of[orbit] = relaxed_groups.setdefault(res["relaxed_geometry_hash"], orbit)
        train = datasets.ingest_labels(spec, rows, "training", "periodic_dft")
        holdout = datasets.ingest_labels(spec, holdout_rows, "final_holdout", "periodic_dft")
        datasets.check_snapshot_against_split(train, split, "training")
        arts.update(train=train, holdout=holdout)
        detail.update(n_training=train["per_topology"], n_holdout=holdout["per_topology"], cv_groups=group_of,
                      dft_jobs=self.store.state["counters"].get("dft_submissions", 0))
        return StageOutcome(StageStatus.COMPLETED, arts, detail)

    def _physical_model_selection(self):
        arts, failed, detail = {}, [], {"selected": {}, "holdout": {}}
        groups = self.detail["labels"]["cv_groups"]
        for tid, topo in self.spec.topologies.items():
            sel = configuration_model.fit_and_select_grouped(self.spec, topo, self.ctx["train"], groups)
            arts[f"selection/{tid}"] = sel
            detail["selected"][tid] = sel["selected_model"]
            if sel["status"] == "MODEL_INADEQUATE":
                failed.append({"topology_id": tid, "code": "MODEL_INADEQUATE"})
                continue
            ham = configuration_model.hamiltonian_from_grouped_selection(self.spec, topo, sel)
            arts[f"hamiltonian/{tid}"] = ham
            # The frozen holdout is read exactly once, after selection.
            report = configuration_model.evaluate_holdout(self.spec, topo, ham, self.ctx["holdout"])
            arts[f"holdout_report/{tid}"] = report
            detail["holdout"][tid] = {k: report[k] for k in ("status", "n_holdout", "holdout_rmse_eV", "tolerance_eV")}
            if report["status"] != StageStatus.COMPLETED.value:
                failed.append({"topology_id": tid, "code": "HOLDOUT_VALIDATION_FAILED"})
            entries = [e for e in self.ctx["train"]["entries"] if e["topology_id"] == tid]
            preds = [configuration_model.predict(ham, topo, e["bits"]) for e in entries]
            labels = [Fraction(e["energy_exact"]) for e in entries]
            tol = Fraction(str(self.spec.physical["tolerances"]["target_relative_energy_tolerance"]))
            best = min(range(len(entries)), key=lambda i: (preds[i], i))
            detail.setdefault("labelled_ground_state_recovered", {})[tid] = labels[best] <= min(labels) + tol
        if failed:
            return StageOutcome(StageStatus.FAILED, arts, {**detail, "error": {"code": failed[0]["code"], "detail": failed}},
                                halt=True)
        return StageOutcome(StageStatus.COMPLETED, arts, detail)

    def _physical_evaluation(self):
        #TODO: Task 07 — embedding correction per candidate geometry once a validated assembler exists (T07.3).
        arts, results, detail = {}, [], {"backend": self.backend.name, "labels_used_for_training": False}
        for tid, topo in self.spec.topologies.items():
            pool = self.ctx[f"pool/{tid}"]["pool"]
            records, evaluated = self._evaluate_decorations(topo, [tuple(r["bits"]) for r in pool], "evaluate_candidate_pool")
            arts.update(records)
            for row, r in zip(pool, evaluated):
                # Failed DFT and transformed structures keep their own records; a build failure gets a stub.
                calc = r["selected"] or make_record("CalculationResult", self.spec.provenance, orbit_id=row["orbit_id"],
                                                    status=r["status"], failure=r["failure"], base_energy_exact=None)
                arts[f"calc/{row['orbit_id']}"] = calc
                results.append((row, calc))
        ok = [(p, r) for p, r in results if r["status"] == "SUCCEEDED"]
        detail.update(n_requests=len(results), n_succeeded=len(ok),
                      not_evaluated=[{"orbit_id": p["orbit_id"], "status": r["status"], "failure": r.get("failure")}
                                     for p, r in results if r["status"] != "SUCCEEDED"],
                      surrogate_vs_evaluated_base_max_abs_eV=float(max(
                          (abs(Fraction(r["base_energy_exact"]) - Fraction(p["predicted_base_energy_exact"])) for p, r in ok),
                          default=0)) if ok else None,
                      dft_jobs=self.store.state["counters"].get("dft_submissions", 0),
                      dft_cache_hits=self.store.state["counters"].get("dft_cache_hits", 0),
                      electronic=electronic_stage_status(self.spec))
        return StageOutcome(StageStatus.COMPLETED, arts, detail)

    def _physical_ranking(self):
        pool_rows = [row for tid in self.spec.topologies for row in self.ctx[f"pool/{tid}"]["pool"]]
        results = [v for k, v in self.ctx.items() if k.startswith("calc/")]
        tol = Fraction(str(self.spec.physical["tolerances"]["target_relative_energy_tolerance"]))
        base = ranking.rank_candidates(self.spec, pool_rows, results, "base", tie_tolerance=tol)
        competition = ranking.evaluate_reference_mixtures(self.spec, base)
        status = StageStatus.COMPLETED if base["status"] == StageStatus.COMPLETED.value else StageStatus.INCOMPLETE
        return StageOutcome(status, {"ranking/base": base, "competition": competition},
                            {"winning_topology": base["winning_topology"], "base_winning_topology": base["winning_topology"],
                             "best_configuration": base["ranked"][0]["configuration_id"] if base["ranked"] else None,
                             "missing_evaluations": base["missing_evaluations"], "failed_evaluations": base["failed_evaluations"],
                             "corrected_ranking": StageStatus.DISABLED_BY_CONFIG.value})

    def _physical_validation(self):
        geometries = [v for k, v in self.ctx.items() if k.startswith("geometry/")]
        calcs = [v for k, v in self.ctx.items() if k.startswith("dftcalc/")]
        checks = ranking.validate_structures(self.spec)
        checks.update(
            builder_sanity={"n_built": sum(g["status"] == StageStatus.COMPLETED.value for g in geometries),
                            "n_build_failures": sum(g["status"] != StageStatus.COMPLETED.value for g in geometries)},
            post_relaxation_slot_mapping={"n_retained": sum(c.get("slot_mapping") == "RETAINED" for c in calcs),
                                          "n_transformed": sum(c["status"] == "TRANSFORMED" for c in calcs)},
            residual_force_stress={"n_within_tolerance": sum(c["status"] == "SUCCEEDED" for c in calcs),
                                   "n_rejected": sum((c.get("failure") or {}).get("code", "").startswith("RESIDUAL_") for c in calcs)})
        return StageOutcome(StageStatus.COMPLETED, detail={"checks": checks})


def run_pipeline(spec, run_dir) -> dict:
    return Pipeline(spec, run_dir).run()
