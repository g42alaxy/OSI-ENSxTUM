"""Agent-assisted run: A1 -> screening -> A2 -> deterministic pipeline [-> A2 expansion] -> evidence -> A67.

A1 reads the user's general description (design brief) and decides, for every catalogue topology with validated
inputs, whether it is analysed; the project is narrowed to the selected ones. The pipeline then builds and screens
them (Task 02: ML-potential relaxations, calibrated or exploratory, when the project declares a potential) and stops;
A2 sees only the topologies screening kept, with its evidence. A2 proposes the Hamiltonian form
(features, encoding, compilation path) per selected topology; code fits its coefficients, selects the model,
compiles and verifies the QUBO/HUBO. A67 checks the result against structural checks, papers and datasets.
The expansion branch (a second A2 request after MODEL_INADEQUATE) runs only when budget.allow_model_expansion.

The graph is fixed application orchestration, not a model-driven tool loop: LangGraph (when installed, or a
sequential runner with the same routes) moves a small JSON-able state between nodes. Three LLM requests in a
normal run; the single optional extra request repairs one invalid output or expands an inadequate model family.

Run directory layout::

    <run_dir>/agent/        RunStore: input packets, DecisionRecords and AgentDecision artifacts, ledger.json,
                            evidence.json, state.json (one committed stage per decision)
    <run_dir>/pipeline/     the deterministic pipeline run: stopped after screening for A2, then resumed with the
                            A2 model grid (intake, construction and screening are reused; only fitting onward reruns)
    <run_dir>/assessment.md A67 assessment with citations expanded from the evidence packet only
    <run_dir>/agent_run.json  summary: status, request counts, decisions, attempts, redacted controls
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import TypedDict

from ..contract import PipelineError, canonical_json, content_hash, make_record
from ..gateway import review_proposal
from ..intake import AGENT_ROLES, load_project, select_topologies
from ..orchestrator import run_pipeline
from ..store import RunStore
from . import packets
from .client import AgentCallError, Cache, Ledger, OpenRouterClient, call_role
from .contracts import ROLE_ACTION
from .datasets import gather_datasets
from .literature import LocalPapers, ValencyMCP, gather_evidence
from .validate import render_assessment, validate_assessment, validate_hamiltonian_proposal, validate_setup


class AgentState(TypedDict, total=False):
    setup: dict
    screening: dict
    model_grid: dict
    attempts: list
    expansion_used: bool
    evidence: dict
    assessment: dict
    stop: dict


class AgentRun:
    def __init__(self, project, controls, run_dir, *, case=None, correction=None, client=None, valency=None,
                 local=None, brief: str | None = None):
        self.controls, self.run_dir, self.project = controls, Path(run_dir), project
        self.spec = load_project(project, case=case, correction_mode=correction)  # the full topology catalogue
        raw = self.spec.config.get("project_spec") or self.spec.config
        self.brief = brief or raw.get("design_brief") or raw.get("purpose") or ""
        self._narrowed: dict = {}
        self.run_id = f"{self.spec.project_id}:{self.spec.fingerprint[:12]}"  # stable per inputs, so caches hit
        self.store = RunStore(self.run_dir / "agent")
        self.store.bind_project(self.spec.fingerprint, self.spec.project_type, self.spec.provenance)
        self.ledger = Ledger(self.run_dir / "agent" / "ledger.json", controls["budget"]["allow_extra_request"])
        cache = controls["cache"]
        self.cache = Cache(Path(cache["dir"]) if cache["enabled"] else None)
        self.client = client or OpenRouterClient(controls)
        self.valency, self.local, self.decisions = valency, local, []

    # -- one role request + application validation + decision record -------------------------------------
    def _ask(self, role: str, packet: dict, validate, purpose: str = "initial"):
        max_tokens = self.controls["openrouter"]["max_tokens"][role]
        output, meta = call_role(role, packet, client=self.client, ledger=self.ledger, cache=self.cache,
                                 max_tokens=max_tokens, purpose=purpose)
        validation = validate(output, packet)
        if validation["status"] == "REJECTED" and not meta.get("cache_hit") and self.ledger.extra_available:
            detail = "; ".join(f"{e['code']}: {e['detail']}" for e in validation["errors"])
            repair = [{"role": "assistant", "content": canonical_json(output)},
                      {"role": "user", "content": f"Application validation rejected the output: {detail}. "
                                                  "Return one corrected, complete JSON object."}]
            output, meta = call_role(role, packet, client=self.client, ledger=self.ledger, cache=self.cache,
                                     max_tokens=max_tokens, purpose="repair", repair=repair)
            validation = validate(output, packet)
        if validation["status"] != "REJECTED" and not meta.get("cache_hit"):
            self.cache.put(meta["cache_key"], output, {k: v for k, v in meta.items() if k != "cache_key"})
        self._record(role, purpose, packet, output, meta, validation)
        return output, validation

    def _record(self, role, purpose, packet, output, meta, validation) -> None:
        packet_id = self.store.put(make_record("AgentInputPacket", self.spec.provenance, role=role, packet=packet))
        envelope = {"project_id": self.spec.project_id, "expected_state_version": self.store.version, "role": role,
                    "evidence_artifact_ids": [packet_id], "proposed_action": {"tool": ROLE_ACTION[role], "arguments": output}}
        gate = review_proposal(self.store, self.spec.project_id, envelope, agents_enabled=True,
                               provenance=self.spec.provenance)
        decision = make_record("AgentDecision", self.spec.provenance, role=role, purpose=purpose, input_packet_id=packet_id,
                               gateway_decision=gate["artifact_id"], gateway_outcome=gate["outcome"], output=output,
                               provenance_meta={k: v for k, v in meta.items() if k != "cache_key"}, validation=validation)
        ids = {"gateway": self.store.put(gate), "decision": self.store.put(decision)}
        self.store.commit_stage(f"{role}_{purpose}_{len(self.decisions) + 1}", validation["status"], packet["input_hash"],
                                ids, {"errors": validation["errors"]})
        self.decisions.append(decision["artifact_id"])

    def _stop(self, code: str, detail) -> dict:
        return {"stop": {"code": code, "detail": detail}}

    # -- nodes ---------------------------------------------------------------------------------------------
    def work_spec(self, setup: dict):
        """The project narrowed to A1's selected topologies (memoized)."""
        key = tuple(p["framework_id"] for p in setup["plans"])
        if key not in self._narrowed:
            self._narrowed[key] = select_topologies(self.spec, key)
        return self._narrowed[key]

    def node_a1(self, state: AgentState) -> dict:
        allow = self.controls["selection"]["allow_scope_exclusions"]
        try:
            _, v = self._ask("A1", packets.build_a1_packet(self.spec, self.run_id, self.brief, allow), validate_setup)
        except AgentCallError as exc:
            return self._stop(exc.code, exc.detail)
        if v["status"] != "VALID":
            return self._stop(f"A1_{v['status']}", v.get("errors") or v.get("error_codes") + v.get("missing_input_ids"))
        return {"setup": {"plans": v["plans"], "excluded": v["excluded"]}}

    def node_screen(self, state: AgentState) -> dict:
        """Construction checks and topology screening, then stop: A2 answers for the topologies screening kept."""
        out = run_pipeline(self.agent_spec(state["setup"], None), self.run_dir / "pipeline", stop_after="screening")
        report = out.get("screening")
        if out["stage_status"].get("screening") != "COMPLETED" or not report:
            failed = {k: v for k, v in out["stage_status"].items() if v not in ("COMPLETED", "SKIPPED_NO_ATOMIC_GEOMETRY")}
            return self._stop("SCREENING_NOT_COMPLETED", failed)
        return {"screening": {"mode": report.get("mode"), "ranking_label": report.get("ranking_label", "NOT_APPLICABLE"),
                              "retained": report["retained"],
                              "decisions": [{"topology_id": d["topology_id"], "state": d["state"],
                                             "reason": d.get("reason") or d.get("evidence"),
                                             **({k: d["evidence"].get(k) for k in ("min_energy_per_formula_unit_eV",
                                                                                   "gap_to_best_eV_per_formula_unit")}
                                                if isinstance(d.get("evidence"), dict) else {})}
                                            for d in report["decisions"]]}}

    def a2_inputs(self, state: AgentState):
        """The spec and setup A2 sees: A1's selection narrowed to the topologies screening kept."""
        retained = state["screening"]["retained"]
        spec = select_topologies(self.work_spec(state["setup"]), retained)
        return spec, {**state["setup"], "plans": [p for p in state["setup"]["plans"] if p["framework_id"] in retained]}

    def node_a2(self, state: AgentState) -> dict:
        spec, setup = self.a2_inputs(state)
        try:
            _, v = self._ask("A2", packets.build_a2_packet(spec, setup, self.run_id, screening=state["screening"]),
                             validate_hamiltonian_proposal)
        except AgentCallError as exc:
            return self._stop(exc.code, exc.detail)
        if v["status"] != "VALID":
            return self._stop(f"A2_{v['status']}", v.get("errors") or v.get("error_codes") + v.get("missing_input_ids"))
        return {"model_grid": v["model_grid"]}

    def agent_spec(self, setup: dict, grid: dict):
        """The project as the pipeline sees it: agents enabled, validated plans and model grid attached."""
        runtime = {"enabled": True, "llm_calls_allowed": True, "roles": list(AGENT_ROLES),
                   "max_llm_requests": self.controls.max_llm_requests}
        spec = self.work_spec(setup)
        config = {**spec.config, "runtime_agents": runtime}
        plan = {"construction_plans": setup["plans"], "excluded_topologies": setup["excluded"], "model_grid": grid,
                "design_brief": self.brief, "decision_ids": list(self.decisions),
                "llm_requests_before_pipeline": self.ledger.requests}
        # The model grid enters only the fitting stages (Pipeline._stage_inputs); decision IDs and counters stay out
        # of the fingerprint, so screening before A2 and cached re-runs reuse the committed stages.
        fingerprint = content_hash({"base": spec.fingerprint, "runtime_agents": runtime, "plans": setup["plans"]})
        return dataclasses.replace(spec, config=config, agent_plan=plan, fingerprint=fingerprint)

    def node_pipeline(self, state: AgentState) -> dict:
        attempts = list(state.get("attempts") or [])
        run_dir = self.run_dir / "pipeline"
        report = run_pipeline(self.agent_spec(state["setup"], state["model_grid"]), run_dir)
        selection = (report.get("stage_detail") or {}).get("model_selection") or {}
        attempts.append({"run_dir": str(run_dir), "model_grid": state["model_grid"],
                         "pipeline_status": report.get("pipeline_status"), "stage_status": report.get("stage_status"),
                         "model_selection_error": (selection.get("error") or {}).get("code"),
                         "winning_topology": report.get("winning_topology")})
        return {"attempts": attempts}

    def node_a2_expand(self, state: AgentState) -> dict:
        last = state["attempts"][-1]
        report = json.loads((Path(last["run_dir"]) / "report.json").read_text())
        diagnostics = packets.fit_diagnostics(report, last["run_dir"])
        failed = sorted(t for t, d in diagnostics.items() if d["status"] == "MODEL_INADEQUATE")
        expansion = {"failed_topologies": failed, "tried_model_ids": state["model_grid"], "fit_diagnostics": diagnostics}
        spec, setup = self.a2_inputs(state)
        try:
            _, v = self._ask("A2", packets.build_a2_packet(spec, setup, self.run_id, expansion, state["screening"]),
                             validate_hamiltonian_proposal, purpose="expansion")
        except AgentCallError as exc:
            return {"expansion_used": True, **self._stop(exc.code, exc.detail)}
        if v["status"] != "VALID":
            return {"expansion_used": True, **self._stop(f"A2_EXPANSION_{v['status']}", v.get("errors"))}
        return {"expansion_used": True, "model_grid": v["model_grid"]}

    def node_evidence(self, state: AgentState) -> dict:
        valency, local, gaps = self.valency, self.local, []
        v_cfg, lit = self.controls["mcp"]["valency"], self.controls["literature"]
        if valency is None and v_cfg["enabled"]:
            valency = ValencyMCP(v_cfg["url"], v_cfg["api_token"], v_cfg["timeout_s"])
        if local is None and lit["local_papers_dir"] and Path(lit["local_papers_dir"]).expanduser().is_dir():
            local = LocalPapers(lit["local_papers_dir"])
        try:
            evidence = gather_evidence(self.work_spec(state["setup"]), self.controls, valency=valency, local=local)
        except Exception as exc:  # e.g. Valency unreachable or token rejected: an incomplete assessment, not a crash
            gaps.append({"code": "RETRIEVAL_FAILED", "detail": f"{type(exc).__name__}: {str(exc)[:200]}"})
            evidence = gather_evidence(self.work_spec(state["setup"]), self.controls, valency=None, local=local)
        evidence["gaps"] = gaps + evidence["gaps"]
        (self.run_dir / "agent" / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
        return {"evidence": {k: evidence[k] for k in ("sources", "passages", "gaps", "retrieval_requests")}}

    def node_a67(self, state: AgentState) -> dict:
        last = state["attempts"][-1]
        report = json.loads((Path(last["run_dir"]) / "report.json").read_text())
        assessment_cfg, spec = self.controls["assessment"], self.agent_spec(state["setup"], state["model_grid"])
        candidates = packets.candidate_rows(report, assessment_cfg["top_k_candidates"])
        datasets = gather_datasets(spec, self.controls, report, last["run_dir"], candidates)
        (self.run_dir / "agent" / "datasets.json").write_text(json.dumps(datasets, indent=2) + "\n")
        packet = packets.build_a67_packet(spec, report, last["run_dir"], state["evidence"], self.run_id,
                                          assessment_cfg["top_k_candidates"],
                                          assessment_cfg.get("required_checks") or packets.DEFAULT_REQUIRED_CHECKS, datasets)
        try:
            output, v = self._ask("A67", packet, validate_assessment)
        except AgentCallError as exc:
            return self._stop(exc.code, exc.detail)
        (self.run_dir / "assessment.json").write_text(json.dumps({"assessment": output, "validation": v}, indent=2) + "\n")
        (self.run_dir / "assessment.md").write_text(render_assessment(output, packet, v))
        return {"assessment": {"status": output["status"], "validation": v["status"], "errors": v["errors"],
                               "n_dataset_records": len(datasets["records"]), "n_metrics": len(datasets["metrics"]),
                               "dataset_gaps": datasets["gaps"]}}

    def node_finalize(self, state: AgentState) -> dict:
        attempts = state.get("attempts") or []
        assessment = state.get("assessment") or {}
        complete = (attempts and attempts[-1]["pipeline_status"] == "COMPLETED" and assessment.get("validation") == "VALID"
                    and not state.get("stop"))
        summary = {"status": "COMPLETED" if complete else ("STOPPED" if not attempts else "INCOMPLETE"),
                   "run_id": self.run_id, "project_id": self.spec.project_id, "mode": packets.mode_of(self.spec),
                   "llm_requests": self.ledger.requests, "llm_request_ceiling": self.controls.max_llm_requests,
                   "extra_request_used": self.ledger.extra_used,
                   "cache_hits": sum(1 for e in self.ledger.entries if e.get("cache_hit")),
                   "retrieval_requests": (state.get("evidence") or {}).get("retrieval_requests", 0),
                   "stop": state.get("stop"), "design_brief": self.brief,
                   "construction_plans": (state.get("setup") or {}).get("plans"),
                   "excluded_topologies": (state.get("setup") or {}).get("excluded"),
                   "screening": state.get("screening"),
                   "model_grid": state.get("model_grid"), "expansion_used": bool(state.get("expansion_used")),
                   "pipeline_attempts": attempts, "assessment": assessment or None,
                   "evidence_gaps": (state.get("evidence") or {}).get("gaps"), "decision_ids": self.decisions,
                   "controls": self.controls.public()}
        self.ledger.save()
        (self.run_dir / "agent_run.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
        return {}

    # -- routing -------------------------------------------------------------------------------------------
    def routes(self) -> dict:
        def after_pipeline(s):
            inadequate = s["attempts"][-1]["model_selection_error"] == "MODEL_INADEQUATE"
            expand = self.controls["budget"]["allow_model_expansion"] and self.ledger.extra_available
            return "a2_expand" if inadequate and expand and not s.get("expansion_used") else "evidence"

        return {
            "a1_setup": (lambda s: "finalize" if s.get("stop") else "screen", ["screen", "finalize"]),
            "screen": (lambda s: "finalize" if s.get("stop") else "a2_hamiltonian", ["a2_hamiltonian", "finalize"]),
            "a2_hamiltonian": (lambda s: "finalize" if s.get("stop") else "pipeline", ["pipeline", "finalize"]),
            "pipeline": (after_pipeline, ["a2_expand", "evidence"]),
            "a2_expand": (lambda s: "evidence" if s.get("stop") else "pipeline", ["pipeline", "evidence"]),
            "evidence": (lambda s: "a67_assessment", ["a67_assessment"]),
            "a67_assessment": (lambda s: "finalize", ["finalize"]),
        }

    def nodes(self) -> dict:
        return {"a1_setup": self.node_a1, "screen": self.node_screen, "a2_hamiltonian": self.node_a2, "pipeline": self.node_pipeline,
                "a2_expand": self.node_a2_expand, "evidence": self.node_evidence, "a67_assessment": self.node_a67,
                "finalize": self.node_finalize}

    def run(self, engine: str = "auto") -> dict:
        graph, used = build_graph(self.nodes(), self.routes(), engine)
        graph.invoke(AgentState())
        summary = json.loads((self.run_dir / "agent_run.json").read_text())
        summary["engine"] = used
        (self.run_dir / "agent_run.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
        return summary


class SequentialGraph:
    """Same nodes and routes as the LangGraph build, for environments without langgraph."""

    def __init__(self, nodes: dict, routes: dict, start: str = "a1_setup"):
        self.nodes, self.routes, self.start = nodes, routes, start

    def invoke(self, state: dict) -> dict:
        state, node = dict(state), self.start
        for _ in range(25):
            state.update(self.nodes[node](state) or {})
            if node == "finalize":
                return state
            node = self.routes[node][0](state)
        raise PipelineError("AGENT_GRAPH_DID_NOT_TERMINATE", node)


def build_graph(nodes: dict, routes: dict, engine: str = "auto"):
    if engine in ("auto", "langgraph"):
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError:
            if engine == "langgraph":
                raise PipelineError("LANGGRAPH_NOT_INSTALLED", "pip install -e .[agents]")
        else:
            graph = StateGraph(AgentState)
            for name, fn in nodes.items():
                graph.add_node(name, fn)
            graph.add_edge(START, "a1_setup")
            for source, (router, targets) in routes.items():
                graph.add_conditional_edges(source, router, targets)
            graph.add_edge("finalize", END)
            return graph.compile(), "langgraph"
    return SequentialGraph(nodes, routes), "sequential"


def check_connections(controls) -> dict:
    """agents-check: configuration, OpenRouter key/model/structured-output support and Valency reachability."""
    import requests

    out = {"config_errors": [{"code": c, "field": f} for c, f in controls.errors()]}
    o = controls["openrouter"]
    try:
        models = requests.get(o["base_url"].rstrip("/") + "/models", timeout=30).json()["data"]
        model = next((m for m in models if m["id"] == o["model"]), None)
        out["model"] = {"id": o["model"], "listed": model is not None,
                        "structured_outputs": bool(model and "structured_outputs" in model.get("supported_parameters", [])),
                        "context_length": model and model.get("context_length")}
    except Exception as exc:
        out["model"] = {"error": type(exc).__name__}
    if o["api_key"]:
        r = requests.get(o["base_url"].rstrip("/") + "/key", headers={"Authorization": f"Bearer {o['api_key']}"}, timeout=30)
        out["openrouter_key"] = {"status_code": r.status_code, "valid": r.status_code == 200}
    v = controls["mcp"]["valency"]
    if v["enabled"]:
        try:
            health = requests.get(v["url"].rstrip("/") + "/health", timeout=30).status_code
            if not v["api_token"]:
                out["valency"] = {"health": health, "list_sources": "skipped: no api_token"}
                return out | _local_check(controls)
            sources = ValencyMCP(v["url"], v["api_token"], v["timeout_s"]).run([("list_sources", {})])[0]
            out["valency"] = {"health": health, "list_sources": "ok" if not isinstance(sources, Exception)
                              else f"{type(sources).__name__}: {str(sources)[:200]}"}
        except Exception as exc:
            out["valency"] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    return out | _local_check(controls)


def _local_check(controls) -> dict:
    lit = controls["literature"]["local_papers_dir"]
    if not lit:
        return {}
    local = LocalPapers(lit) if Path(lit).expanduser().is_dir() else None
    return {"local_papers": {"dir": lit, "documents": len(local.docs) if local else 0,
                             "gaps": local.gaps if local else [{"code": "NOT_A_DIRECTORY"}]}}
