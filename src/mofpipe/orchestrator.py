"""Deterministic orchestrator: one authority for workflow state (10_AGENT_ARCHITECTURE).

Stages commit immutable artifacts to the run store. A re-run with identical
inputs reuses committed stages and never resubmits evaluator jobs.
"""
from __future__ import annotations

from fractions import Fraction

from . import candidates, configuration_model, datasets, encoding, optimization, ranking, refinement, structures
from .contract import PipelineError, StageStatus, content_hash
from .electronic import SyntheticFixtureBackend, electronic_stage_status
from .intake import validate_project
from .report import build_report, write_report
from .store import RunStore

#TODO: Task 10 — AiiDA WorkChain engine with CalcJobs for external DFT/QPU jobs; this local engine is the
#TODO: development stand-in obeying the same interfaces. Concurrency (optimistic state commits) is not implemented.
STAGES = ("intake", "structures", "screening", "labels", "refinement", "model_selection", "encoding",
          "optimization", "candidates", "evaluation", "ranking", "validation", "report")


class StageOutcome:
    def __init__(self, status: StageStatus, artifacts: dict | None = None, detail=None, halt: bool = False):
        self.status, self.artifacts, self.detail, self.halt = status, artifacts or {}, detail, halt


class Pipeline:
    def __init__(self, spec, run_dir):
        self.spec = spec
        self.store = RunStore(run_dir)
        self.store.bind_project(spec.fingerprint)
        self.ctx: dict = {}      # name -> artifact record
        self.detail: dict = {}   # stage -> JSON detail
        self.status: dict = {}
        self._backend = None

    @property
    def backend(self):
        if self._backend is None:
            self._backend = SyntheticFixtureBackend(self.spec, self.store)
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
                outcome = getattr(self, f"_stage_{name}")()
            except PipelineError as exc:
                outcome = StageOutcome(StageStatus.FAILED, detail={"error": exc.as_dict()}, halt=True)
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
        #TODO: Tasks 01/06/07 — real path: build_framework -> submit_periodic_dft -> embedding correction per
        #TODO: candidate geometry; only the synthetic fixture backend exists now.
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


def run_pipeline(spec, run_dir) -> dict:
    return Pipeline(spec, run_dir).run()
