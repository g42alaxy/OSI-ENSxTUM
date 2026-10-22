"""Task 10 (T10.1): AiiDA engine for the pipeline (``runtime.engine: aiida`` / ``mofpipe run --engine aiida``).

``MofPipeWorkChain`` steps through the stages in order. Each step runs the stage with the same interfaces as the
local engine (``Pipeline.run(stop_after=stage)``: committed stages are reused, the next one is computed and
committed) and then records the commit in the AiiDA provenance graph through the ``record_stage`` calcfunction: a
``Dict`` with the stage status, input fingerprint, state version and the content hash of every artifact it
committed, linked to the WorkChain. The run store stays the authority for scientific state (single writer with
optimistic commits, store.py); AiiDA holds the provenance of the workflow that produced it and can resume at
stage boundaries.

The default profile is a local ``verdi presto`` profile (SQLite storage, no RabbitMQ) kept in the project's
``.aiida`` folder (``AIIDA_PATH``); ``aiida.engine.run`` executes in-process, so no daemon is needed.
#TODO: Task 10 — CalcJobs for DFT/ML/QPU jobs on HPC (aiida-quantumespresso for pw.x, a PySCF code plugin) and
#TODO: concurrent topologies, DFT configurations and solver seeds once a broker and daemon are configured.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from aiida.engine import WorkChain, calcfunction, while_
from aiida.orm import Dict, Int, Str

from .contract import PipelineError

DEFAULT_PROFILE = "mofpipe"


def load_aiida(profile: str | None = None):
    """Load the AiiDA profile (default ``mofpipe``). AIIDA_PATH must be set before aiida is imported (__main__)."""
    import aiida
    try:
        return aiida.load_profile(profile or DEFAULT_PROFILE, allow_switch=True)
    except Exception as exc:  # missing profile or storage
        raise PipelineError("BACKEND_UNAVAILABLE", f"AiiDA profile {profile or DEFAULT_PROFILE}: {exc} "
                                                   f"(create it with: AIIDA_PATH=.aiida verdi presto --profile-name mofpipe)") from exc


@calcfunction
def record_stage(stage: Str, commit: Dict) -> Dict:
    """Provenance node of one committed stage (the artifacts themselves stay in the run store)."""
    return Dict(dict(commit.get_dict(), stage=stage.value))


class MofPipeWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("project", valid_type=Str)
        spec.input("run_dir", valid_type=Str)
        spec.input("case", valid_type=Str, required=False)
        spec.input("correction", valid_type=Str, required=False)
        spec.outline(cls.setup, while_(cls.has_next_stage)(cls.run_stage), cls.finalize)
        spec.output_namespace("stages", valid_type=Dict, dynamic=True)
        spec.output("report", valid_type=Dict)
        spec.output("n_stages", valid_type=Int)

    def _pipeline(self):
        from .intake import load_project
        from .orchestrator import Pipeline
        spec = load_project(self.inputs.project.value,
                            case=self.inputs.case.value if "case" in self.inputs else None,
                            correction_mode=self.inputs.correction.value if "correction" in self.inputs else None)
        return Pipeline(spec, self.inputs.run_dir.value)

    def setup(self):
        self.ctx.index = 0

    def has_next_stage(self):
        from .orchestrator import STAGES
        return self.ctx.index < len(STAGES)

    def run_stage(self):
        from .orchestrator import STAGES
        name = STAGES[self.ctx.index]
        pipeline = self._pipeline()
        pipeline.run(stop_after=name)
        committed = pipeline.store.stage(name)
        if committed is None:  # not run after a halt: nothing committed, nothing recorded as computed
            commit = {"status": pipeline.status.get(name, "NOT_RUN"), "committed": False}
        else:
            hashes = {key: hashlib.sha256(pipeline.store._path(aid).read_bytes()).hexdigest()
                      for key, aid in committed["artifact_ids"].items()}
            commit = {"status": committed["status"], "committed": True, "input_fingerprint": committed["input_fingerprint"],
                      "state_version": committed["state_version"], "artifact_ids": committed["artifact_ids"],
                      "artifact_sha256": hashes}
        self.out(f"stages.{name}", record_stage(Str(name), Dict(commit)))
        self.ctx.index += 1

    def finalize(self):
        report = json.loads((Path(self.inputs.run_dir.value) / "report.json").read_text())
        keep = ("project_id", "pipeline_status", "winning_topology", "base_winning_topology", "stage_status",
                "artifact_id", "project_fingerprint")
        self.out("report", record_stage(Str("report_summary"), Dict({k: report.get(k) for k in keep})))
        self.out("n_stages", Int(self.ctx.index).store())


def run_pipeline_aiida(project, run_dir, case=None, correction=None, profile=None) -> dict:
    """Run the pipeline as an AiiDA WorkChain; returns the run store's report plus the WorkChain identifiers."""
    load_aiida(profile)
    from aiida.engine import run_get_node

    inputs = {"project": Str(str(Path(project).resolve())), "run_dir": Str(str(Path(run_dir).resolve()))}
    if case:
        inputs["case"] = Str(case)
    if correction:
        inputs["correction"] = Str(correction)
    _, node = run_get_node(MofPipeWorkChain, **inputs)
    if not node.is_finished_ok:
        raise PipelineError("AIIDA_WORKCHAIN_FAILED", f"pk={node.pk} exit_status={node.exit_status}")
    report = json.loads((Path(run_dir) / "report.json").read_text())
    return {**report, "_aiida": {"workchain_pk": node.pk, "workchain_uuid": node.uuid, "profile": profile or DEFAULT_PROFILE,
                                 "n_stage_nodes": len(node.outputs.stages)}}
