"""Deterministic orchestrator: one authority for workflow state (10_AGENT_ARCHITECTURE).

Stages commit immutable artifacts to the run store. A re-run with identical
inputs reuses committed stages and never resubmits evaluator jobs.
"""
from __future__ import annotations

from fractions import Fraction

from . import (acquisition, candidates, configuration_model, datasets, dft, encoding, framework, geometry_match,
               optimization, ranking, references, refinement, structures, transfer, validation)
from .contract import PipelineError, StageStatus, StopReason, content_hash, make_record
from .electronic import SyntheticFixtureBackend, electronic_stage_status
from .intake import FIXTURE_PROJECT_TYPE, validate_project
from .report import build_report, write_report
from .store import RunStore

#TODO: Task 10 — AiiDA WorkChain engine with CalcJobs for external DFT/QPU jobs; this local engine is the
#TODO: development stand-in obeying the same interfaces. Concurrency (optimistic state commits) is not implemented.
STAGES = ("intake", "structures", "screening", "labels", "refinement", "model_selection", "supercell_checks", "encoding",
          "optimization", "candidates", "evaluation", "ranking", "validation", "report")
# Missing inputs, resources or budget block a stage; they are not scientific failures.
BLOCKING_CODES = {"BACKEND_UNAVAILABLE", "BUDGET_EXHAUSTED", "SCIENTIFIC_SPEC_MISSING", "DFT_SETTINGS_NOT_CONVERGED",
                  "UNSUPPORTED_DFT_PROTOCOL"}


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
            states = list(structures.states_of(spec, topo))
            detail["symmetry"][tid] = configuration_model.check_feature_symmetry(topo, spec.feature_order, states)
            detail["swap_reachability"][tid] = structures.swap_graph_connected(topo, topo.n_sites,
                                                                               structures.n_ones_of(spec, topo))
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

    def _stage_supercell_checks(self):
        return StageOutcome(StageStatus.SKIPPED_NO_ATOMIC_GEOMETRY, detail={"reason": "abstract graphs have no supercells"})

    def _stage_validation(self):
        return StageOutcome(StageStatus.SKIPPED_NO_ATOMIC_GEOMETRY, detail={"checks": ranking.validate_structures(self.spec)})

    def _stage_report(self):
        report = build_report(self.spec, self.store, self.ctx, self.detail, self.status)
        write_report(self.store.root, report)
        return StageOutcome(StageStatus.COMPLETED, {"report": report})


    # -- physical projects (Milestones B and C: CIF slot lattices, DFT labels, refinement, transfer, references) ---
    def _physical_structures(self):
        spec, arts = self.spec, {}
        detail = {"symmetry": {}, "swap_reachability": {}, "atomic_construction": {}, "supercell_replication": {},
                  "cluster_catalogue": {s["topology_id"]: s.get("cluster_catalogue") for s in spec.physical["sources"]}}
        transfers = spec.physical.get("transfer_topologies") or {}
        for tid, topo in spec.topologies.items():
            space = arts[f"space/{tid}"] = structures.define_decoration_space(spec, topo)
            states = list(structures.states_of(spec, topo))
            features = configuration_model.all_model_features(spec, topo)
            detail["symmetry"][tid] = configuration_model.check_feature_symmetry(topo, features, states)
            detail["swap_reachability"][tid] = structures.swap_graph_connected(topo, topo.n_sites,
                                                                               structures.n_ones_of(spec, topo))
            # Rebuild the known parent (every slot keeps its parent linker) and compare with the parsed supercell.
            lattice = spec.physical["lattices"][tid]
            parent = {"configuration_id": f"{tid}:parent", "orbit_id": f"{tid}:parent", "topology_id": tid,
                      "species_by_slot": [s["parent_species"] for s in lattice["slots"]]}
            rebuilt = framework.build_framework(spec, topo, parent)
            reference = framework.geometry_hashes(lattice["structure"]["lattice"],
                                                  [a["element"] for a in lattice["structure"]["atoms"]],
                                                  [tuple(round(x, 6) for x in a["frac"]) for a in lattice["structure"]["atoms"]])[1]
            sample = structures.make_decoration(spec, topo, states[0])
            built = framework.build_framework(spec, topo, sample)
            detail["atomic_construction"][tid] = {
                "parent_rebuild": rebuilt["status"], "parent_rebuild_identical": rebuilt.get("canonical_geometry_hash") == reference,
                "sample_decoration": sample["configuration_id"], "sample_build": built["status"],
                "sample_inventory": built.get("inventory"), "sample_failure": built.get("failure")}
            if rebuilt.get("canonical_geometry_hash") != reference or built["status"] != StageStatus.COMPLETED.value:
                return StageOutcome(StageStatus.FAILED, arts, {**detail, "error": {"code": "CONSTRUCTION_CHECK_FAILED",
                                                                                  "detail": tid}}, halt=True)
            # Exact replication test (no DFT): replicated configurations keep their features and orbit identity.
            reports = []
            for cell in transfers.get(tid, []):
                mapping = transfer.slot_map(lattice, spec.physical["lattices"][cell.topology_id])
                if mapping is None:
                    reports.append({"transfer_topology_id": cell.topology_id, "status": "NOT_A_REPLICA_CELL"})
                    continue
                reps = [[int(c) for c in o.split(":")[1]] for o in space["orbit_sizes"]]
                reports.append(transfer.replication_report(spec, topo, cell, mapping, features, reps))
            detail["supercell_replication"][tid] = reports
            if any(r["status"] == "REPLICATION_INCONSISTENT" for r in reports):
                return StageOutcome(StageStatus.FAILED, arts, {**detail, "error": {"code": "REPLICATION_INCONSISTENT",
                                                                                  "detail": tid}}, halt=True)
        return StageOutcome(StageStatus.COMPLETED, arts, detail)

    def _evaluate_decorations(self, topo, decorations, purpose):
        """build_framework -> DFT relaxation + final single point per declared magnetic state (T06.2)."""
        records, results = {}, []
        for decoration in decorations:
            if not isinstance(decoration, dict):
                decoration = structures.make_decoration(self.spec, topo, decoration)
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

    def _geometry_classes(self, tid, results) -> dict:
        """T05.1: orbits whose relaxed structures are equivalent under the parent symmetry share one class."""
        lattice = self.spec.physical["lattices"][tid]
        tol = (self.spec.physical["protocols"].get("validation") or {}).get("geometry_equivalence_tolerance_A")
        items = [(r["decoration"]["orbit_id"], r["selected"]["relaxed_geometry"], r["selected"]["magnetic_state"].get("label"))
                 for r in results if r["status"] == "SUCCEEDED"]
        if tol is None:  # undeclared tolerance: only identical relaxed geometries are grouped (Milestone B rule)
            first = {}
            return {o: first.setdefault(r["selected"]["relaxed_geometry_hash"], o) for o, r in
                    ((r["decoration"]["orbit_id"], r) for r in results if r["status"] == "SUCCEEDED")}
        ops = geometry_match.supercell_operations(lattice["symmetry_operations"], lattice["supercell_diagonal"])
        return geometry_match.equivalence_classes(items, ops, lattice["supercell_diagonal"], tol)

    @staticmethod
    def _label_row(tid, res):
        return {"bits": res["bits"], "comparison_group": res["comparison_group"], "configuration_id": res["configuration_id"],
                "energy_eV": res["base_energy_exact"], "energy_unit": res["energy_unit"], "fidelity": res["base_fidelity"],
                "n_formula_units": res["n_formula_units"], "normalization": res["normalization"],
                "orbit_id": res["orbit_id"], "provenance": res["provenance"], "status": res["status"],
                "topology_id": tid, "calculation_result_id": res["artifact_id"]}

    @staticmethod
    def _record_unusable(detail, r):
        key = "transformation_events" if r["status"] == "TRANSFORMED" else "failed_labels"
        detail.setdefault(key, []).append({"orbit_id": r["decoration"]["orbit_id"], "status": r["status"],
                                           "failure": r.get("failure") or (r["selected"] or {}).get("failure")})

    def _convergence_pairs(self, split) -> dict:
        """Energy differences whose cutoff convergence gates labelling: one within every topology and, with several
        topologies, the cross-topology difference that the pooled ranking relies on (common reference)."""
        spec = self.spec
        firsts, pairs = {}, {}
        for tid, topo in spec.topologies.items():
            groups = split["per_topology"][tid]
            decos = [structures.make_decoration(spec, topo, [int(c) for c in o.split(":")[1]])
                     for o in (groups["training"][0], groups["training"][-1])]
            pairs[tid] = [(topo, d) for d in decos]
            firsts[tid] = (topo, decos[0])
        tids = list(spec.topologies)
        for tid in tids[1:]:
            pairs[f"{tids[0]}|{tid}"] = [firsts[tids[0]], firsts[tid]]
        return pairs

    def _physical_labels(self):
        spec, arts = self.spec, {}
        spaces = {tid: self.ctx[f"space/{tid}"] for tid in spec.topologies}
        split = datasets.freeze_physical_split(spec, spaces)
        arts["split"] = split
        detail = {"split_counts": split["counts"], "transformation_events": [], "failed_labels": [], "dft_convergence": {}}
        # Plane-wave cutoff convergence of the relevant energy differences before any label is used.
        for key, pair in self._convergence_pairs(split).items():
            built = [(framework.build_framework(spec, topo, d), d) for topo, d in pair]
            if any(g["status"] != StageStatus.COMPLETED.value for g, _ in built):
                raise PipelineError("DFT_CONVERGENCE_STUDY_FAILED", f"{key}: reference decorations could not be built")
            study = dft.difference_convergence_study(spec, self.backend, built)
            arts[f"dft_convergence/{key}"] = study
            detail["dft_convergence"][key] = {k: study[k] for k in ("status", "deviation_from_highest_cutoff_eV",
                                                                    "tolerance_eV", "parameter", "protocol_cutoff")}
            if study["status"] != StageStatus.COMPLETED.value:
                return StageOutcome(StageStatus.BLOCKED, arts, {**detail, "error": {
                    "code": "DFT_SETTINGS_NOT_CONVERGED", "detail": f"{key}: {study['deviation_from_highest_cutoff_eV']:.4g} eV "
                                                                    f"> {study['tolerance_eV']} eV per formula unit"}}, halt=True)
        rows, holdout_rows, group_of, attempted = [], [], {}, {}
        for tid, topo in spec.topologies.items():
            groups = split["per_topology"][tid]
            labelled = groups["training"] + groups["holdout"]
            records, results = self._evaluate_decorations(
                topo, [[int(c) for c in o.split(":")[1]] for o in labelled], "training_labels")
            arts.update(records)
            attempted[tid] = sorted(labelled)
            for r in results:
                if r["status"] != "SUCCEEDED":
                    self._record_unusable(detail, r)
                    continue
                orbit = r["decoration"]["orbit_id"]
                (holdout_rows if orbit in groups["holdout"] else rows).append(self._label_row(tid, r["selected"]))
            # Near-duplicate relaxations (equivalent relaxed structures) share one CV group.
            group_of.update(self._geometry_classes(tid, [r for r in results if r["decoration"]["orbit_id"] in groups["training"]]))
        train = datasets.ingest_labels(spec, rows, "training", "periodic_dft")
        holdout = datasets.ingest_labels(spec, holdout_rows, "final_holdout", "periodic_dft")
        datasets.check_snapshot_against_split(train, split, "training")
        arts.update(train=train, holdout=holdout)
        detail.update(n_training=train["per_topology"], n_holdout=holdout["per_topology"], cv_groups=group_of,
                      attempted=attempted, dft_jobs=self.store.state["counters"].get("dft_submissions", 0))
        return StageOutcome(StageStatus.COMPLETED, arts, detail)

    def _physical_refinement(self):
        """T08: rounds of declared acquisition + DFT labels until the stopping rules decide (never past the budget)."""
        spec = self.spec
        physical = spec.physical
        stopping = physical["protocols"].get("stopping")
        if not stopping:
            return StageOutcome(StageStatus.DISABLED_BY_CONFIG, detail={"mode": "protocols.stopping is null"})
        tol = Fraction(str(physical["tolerances"]["target_relative_energy_tolerance"]))
        required = physical["tolerances"]["consecutive_stable_rounds"]
        k = physical["protocols"]["acquisition"]["top_k_unique_orbits_per_topology"]
        labels = self.detail["labels"]
        train, split = self.ctx["train"], self.ctx["split"]
        groups = dict(labels["cv_groups"])
        attempted = {tid: set(v) for tid, v in labels["attempted"].items()}
        spaces = {tid: self.ctx[f"space/{tid}"] for tid in spec.topologies}
        arts, history, previous, stable, stop = {}, [], None, 0, None
        detail = {"transformation_events": [], "failed_labels": []}
        for round_index in range(stopping["max_rounds"] + 1):
            selections = {tid: configuration_model.fit_and_select_grouped(spec, topo, train, groups)
                          for tid, topo in spec.topologies.items()}
            lead = None
            if all(s["status"] == "COMPLETED" for s in selections.values()):
                hams = {tid: configuration_model.hamiltonian_from_grouped_selection(spec, spec.topologies[tid], s)
                        for tid, s in selections.items()}
                lead = refinement.leading_sets(spec, hams, spaces, k, tol)
            acquirable = {tid: len(set(split["per_topology"][tid]["unlabelled"]) - attempted.get(tid, set()))
                          for tid in spec.topologies}
            stop, stable, diagnosis = refinement.assess_refinement(round_index, selections, lead, previous, stable,
                                                                   required, stopping["max_rounds"], acquirable)
            entry = {"round": round_index, "training_snapshot_id": train["artifact_id"], "n_labels": train["per_topology"],
                     "selected": {t: s["selected_model"] for t, s in selections.items()},
                     "cv_rmse_eV": {t: min((c["cv_rmse_eV"] for c in s["candidates"] if c["status"] == "COMPLETED"),
                                           default=None) for t, s in selections.items()},
                     "diagnosis": diagnosis, "leading_set": lead, "stable_rounds": stable, "acquirable": acquirable}
            if stop is not None:
                history.append(entry)
                break
            batch = acquisition.select_evaluation_batch(spec, split, selections, train, attempted, round_index)
            arts[f"acquisition/{round_index}"] = batch
            if not batch["selected"]:
                stop = StopReason.INCONCLUSIVE
                history.append({**entry, "acquired": []})
                break
            rows, budget_hit = [], False
            for tid, topo in spec.topologies.items():
                picks = [p for p in batch["selected"] if p["topology_id"] == tid]
                results = []
                for pick in picks:
                    attempted.setdefault(tid, set()).add(pick["orbit_id"])
                    try:
                        records, res = self._evaluate_decorations(topo, [acquisition.bits_of(pick["orbit_id"])],
                                                                  f"refinement_round_{round_index}")
                    except PipelineError as exc:
                        if exc.code != "BUDGET_EXHAUSTED":
                            raise
                        budget_hit = True
                        break
                    arts.update(records)
                    results += res
                for r in results:
                    if r["status"] != "SUCCEEDED":
                        self._record_unusable(detail, r)
                    else:
                        rows.append(self._label_row(tid, r["selected"]))
                prior = [{"decoration": {"orbit_id": e["orbit_id"]}, "status": "SUCCEEDED",
                          "selected": arts.get(f"dftcalc/{e['calculation_result_id']}")
                          or self.ctx.get(f"dftcalc/{e['calculation_result_id']}")}
                         for e in train["entries"] if e["topology_id"] == tid]
                groups.update(self._geometry_classes(tid, [p for p in prior if p["selected"]] + results))
                if budget_hit:
                    break
            if rows:
                new = datasets.ingest_labels(spec, rows, "training", f"refinement_round_{round_index}")
                datasets.check_no_holdout(new, split)
                train = datasets.merge_snapshots(spec, train, new, role="training")
                arts[f"train/round_{round_index}"] = train
            history.append({**entry, "acquired": [p["orbit_id"] for p in batch["selected"]],
                            "n_new_labels": len(rows), "acquisition_batch_id": batch["artifact_id"]})
            if budget_hit:
                stop = StopReason.BUDGET_EXHAUSTED
                break
            previous = lead
        report = make_record(
            "RefinementReport", spec.provenance, stop_reason=stop.value, rounds=history, max_rounds=stopping["max_rounds"],
            consecutive_stable_rounds_required=required, final_training_snapshot_id=train["artifact_id"],
            criterion=("every topology selects a model passing grouped CV and the tolerance-aware leading set (orbits "
                       f"within {float(tol)} eV/fu of the {k}-th lowest prediction, per topology and pooled) is unchanged "
                       f"for {required} consecutive rounds"),
            final_holdout_used=False, coverage="screened decoration space of each working cell; not a global certificate")
        arts.update(refinement=report, train=train)
        status = StageStatus.COMPLETED if stop == StopReason.CONVERGED_WITHIN_SCOPE else StageStatus.INCONCLUSIVE
        return StageOutcome(status, arts, {**detail, "stop_reason": stop.value, "n_rounds": len(history), "cv_groups": groups,
                                           "n_training": train["per_topology"]})

    def _physical_model_selection(self):
        arts, failed, detail = {}, [], {"selected": {}, "holdout": {}}
        groups = (self.detail.get("refinement") or {}).get("cv_groups") or self.detail["labels"]["cv_groups"]
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

    def _physical_supercell_checks(self):
        """T01.5: DFT replication consistency and native transfer-cell predictions of each working-cell model."""
        spec = self.spec
        transfers = spec.physical.get("transfer_topologies") or {}
        if not any(transfers.values()):
            return StageOutcome(StageStatus.DISABLED_BY_CONFIG, detail={"reason": "no transfer supercells declared"})
        policy = (spec.physical["protocols"].get("validation") or {}).get("supercell_transfer") or {}
        arts, detail, statuses = {}, {}, []
        for tid, cells in transfers.items():
            topo, ham = spec.topologies[tid], self.ctx[f"hamiltonian/{tid}"]
            entries = sorted((e for e in self.ctx["train"]["entries"] if e["topology_id"] == tid),
                             key=lambda e: (Fraction(e["energy_exact"]), e["orbit_id"]))
            for cell in cells:
                mapping = transfer.slot_map(spec.physical["lattices"][tid], spec.physical["lattices"][cell.topology_id])
                replicated = []
                for entry in entries[:policy["n_replicated"]] if mapping is not None else []:
                    t_bits = transfer.replicate(entry["bits"], mapping)
                    records, res = self._evaluate_decorations(cell, [t_bits], "supercell_replication")
                    arts.update(records)
                    working = self.ctx[f"dftcalc/{entry['calculation_result_id']}"]
                    replicated.append({"working_result": working, "transfer_result": res[0]["selected"] or {
                        "status": res[0]["status"], "configuration_id": res[0]["decoration"]["configuration_id"]}})
                native = []
                for bits in transfer.native_states(spec, cell, mapping, policy["n_native"], policy["seed"]):
                    records, res = self._evaluate_decorations(cell, [bits], "supercell_transfer")
                    arts.update(records)
                    native.append(res[0]["selected"] or {"status": res[0]["status"], "failure": res[0]["failure"],
                                                         "configuration_id": res[0]["decoration"]["configuration_id"]})
                report = transfer.transfer_report(spec, topo, cell, ham, replicated, native)
                arts[f"transfer/{cell.topology_id}"] = report
                statuses.append(report["status"])
                detail[cell.topology_id] = {k: report[k] for k in ("status", "transfer_rmse_eV", "tolerance_eV")}
                detail[cell.topology_id]["replication_max_abs_eV"] = max(
                    (r["abs_difference_eV_per_fu"] for r in report["replication"] if r["abs_difference_eV_per_fu"] is not None),
                    default=None)
        status = (StageStatus.COMPLETED if all(s == StageStatus.COMPLETED.value for s in statuses) else
                  StageStatus.INCOMPLETE if all(s in (StageStatus.COMPLETED.value, StageStatus.INCOMPLETE.value) for s in statuses)
                  else StageStatus.FAILED)
        # A failed transfer check is reported and keeps the pipeline from COMPLETED; ranking still runs, labelled.
        return StageOutcome(status, arts, detail)

    def _physical_evaluation(self):
        #TODO: Task 07 — embedding correction per candidate geometry once a validated assembler exists (T07.3).
        arts, results, detail = {}, [], {"backend": self.backend.name, "labels_used_for_training": False,
                                          "relaxed_geometry_classes": {}}
        for tid, topo in self.spec.topologies.items():
            pool = self.ctx[f"pool/{tid}"]["pool"]
            records, evaluated = self._evaluate_decorations(topo, [list(r["bits"]) for r in pool], "evaluate_candidate_pool")
            arts.update(records)
            for row, r in zip(pool, evaluated):
                # Failed DFT and transformed structures keep their own records; a build failure gets a stub.
                calc = r["selected"] or make_record("CalculationResult", self.spec.provenance, orbit_id=row["orbit_id"],
                                                    status=r["status"], failure=r["failure"], base_energy_exact=None)
                arts[f"calc/{row['orbit_id']}"] = calc
                results.append((row, calc))
            classes = self._geometry_classes(tid, evaluated)
            detail["relaxed_geometry_classes"][tid] = {o: rep for o, rep in classes.items() if o != rep}
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

    def _reference_phases(self):
        """T09.2 reference set 'pure_endpoints': every slot of every topology holding one linker species."""
        spec = self.spec
        species = list(next(iter(spec.physical["chemistry"]["composition_by_sublattice"].values())))
        arts, refs, missing = {}, [], []
        for tid, topo in spec.topologies.items():
            n = len(spec.physical["lattices"][tid]["slots"])
            for sp in species:
                rid = f"{tid}:reference:{sp}"
                decoration = {"configuration_id": rid, "orbit_id": rid, "topology_id": tid, "bits": None,
                              "species_by_slot": [sp] * n, "normalization": "per_cell", "reference": True}
                records, res = self._evaluate_decorations(topo, [decoration], "reference_phase")
                arts.update(records)
                calc = res[0]["selected"]
                if res[0]["status"] != "SUCCEEDED":
                    missing.append({"reference_id": rid, "status": res[0]["status"],
                                    "failure": res[0].get("failure") or (calc or {}).get("failure")})
                    continue
                refs.append({"reference_id": rid, "topology_id": tid, "species": sp, "inventory": calc["inventory"],
                             "energy_exact": calc["base_energy_exact"], "energy_eV": calc["base_energy_eV"],
                             "calculation_result_id": calc["artifact_id"]})
        return arts, refs, missing

    def _physical_ranking(self):
        pool_rows = [row for tid in self.spec.topologies for row in self.ctx[f"pool/{tid}"]["pool"]]
        results = [v for k, v in self.ctx.items() if k.startswith("calc/")]
        tol = Fraction(str(self.spec.physical["tolerances"]["target_relative_energy_tolerance"]))
        base = ranking.rank_candidates(self.spec, pool_rows, results, "base", tie_tolerance=tol)
        classes = {}
        for per_topology in (self.detail.get("evaluation") or {}).get("relaxed_geometry_classes", {}).values():
            classes.update(per_topology)
        validation_policy = self.spec.physical["protocols"].get("validation") or {}
        arts = {"ranking/base": base}
        if validation_policy.get("reference_phases") == "pure_endpoints":
            ref_arts, refs, missing = self._reference_phases()
            arts.update(ref_arts)
            competition = references.evaluate_reference_mixtures(
                self.spec, base, refs, declared_complete=validation_policy.get("reference_set_complete") is True,
                missing=missing)
        else:
            competition = references.evaluate_reference_mixtures(self.spec, base)
        arts["competition"] = competition
        status = StageStatus.COMPLETED if base["status"] == StageStatus.COMPLETED.value else StageStatus.INCOMPLETE
        return StageOutcome(status, arts,
                            {"winning_topology": base["winning_topology"], "base_winning_topology": base["winning_topology"],
                             "best_configuration": base["ranked"][0]["configuration_id"] if base["ranked"] else None,
                             "topologies_ranked": sorted({r["topology_id"] for r in base["ranked"]}),
                             "relaxed_equivalent_orbits": {r["orbit_id"]: classes[r["orbit_id"]] for r in base["ranked"]
                                                           if r["orbit_id"] in classes},
                             "missing_evaluations": base["missing_evaluations"], "failed_evaluations": base["failed_evaluations"],
                             "competition": competition["status"], "corrected_ranking": StageStatus.DISABLED_BY_CONFIG.value})

    def _physical_validation(self):
        geometries = {v["decoration_id"]: v for k, v in self.ctx.items() if k.startswith("geometry/")}
        calcs = [v for k, v in self.ctx.items() if k.startswith("dftcalc/")]
        checks = {name: StageStatus.NOT_RUN.value for name in ranking.validate_structures(self.spec)}
        checks.update(
            builder_sanity={"n_built": sum(g["status"] == StageStatus.COMPLETED.value for g in geometries.values()),
                            "n_build_failures": sum(g["status"] != StageStatus.COMPLETED.value for g in geometries.values())},
            post_relaxation_slot_mapping={"n_retained": sum(c.get("slot_mapping") == "RETAINED" for c in calcs),
                                          "n_transformed": sum(c["status"] == "TRANSFORMED" for c in calcs)},
            residual_force_stress={"n_within_tolerance": sum(c["status"] == "SUCCEEDED" for c in calcs),
                                   "n_rejected": sum((c.get("failure") or {}).get("code", "").startswith("RESIDUAL_") for c in calcs)})
        arts, finalists = {}, []
        base = self.ctx.get("ranking/base")
        for row in (base or {}).get("ranked", []):
            calc = self.ctx[f"calc/{row['orbit_id']}"]
            report = validation.validate_structure(self.spec, calc, geometries[calc["configuration_id"]])
            arts[f"structure_validation/{row['orbit_id']}"] = report
            finalists.append({"configuration_id": row["configuration_id"], "status": report["status"],
                              "failed_checks": report["failed_checks"],
                              "external_tools": {k: report["checks"][k]["status"] for k in ("mofchecker", "zeopp_pores")}})
        status = StageStatus.COMPLETED if all(f["status"] == StageStatus.COMPLETED.value for f in finalists) else StageStatus.FAILED
        return StageOutcome(status, arts, {"checks": checks, "finalists": finalists})


def run_pipeline(spec, run_dir) -> dict:
    return Pipeline(spec, run_dir).run()
