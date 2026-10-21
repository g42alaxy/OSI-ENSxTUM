"""Persistent workflow state, immutable artifact store and event log.

Workflow state on disk is authoritative (00_SHARED_CONTRACT, execution contract).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .contract import PipelineError, canonical_json


class RunStore:
    def __init__(self, run_dir: Path):
        self.root = Path(run_dir).resolve()  # backends run with their own cwd
        self.artifacts = self.root / "artifacts"
        self.jobs = self.root / "jobs"
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.jobs.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
        else:
            self.state = {"state_version": 0, "project_fingerprint": None, "stages": {},
                          "counters": {"evaluator_submissions": 0, "evaluator_cache_hits": 0, "llm_calls": 0}}

    # -- artifacts ---------------------------------------------------------
    def _path(self, artifact_id: str) -> Path:
        return self.artifacts / (artifact_id.replace(":", "__") + ".json")

    def put(self, record: dict) -> str:
        artifact_id = record["artifact_id"]
        path = self._path(artifact_id)
        text = canonical_json(record)
        if path.exists():
            if path.read_text() != text:
                raise PipelineError("ARTIFACT_IMMUTABILITY_VIOLATION", artifact_id)
        else:
            self._atomic_write(path, text)
        return artifact_id

    def get(self, artifact_id: str) -> dict:
        path = self._path(artifact_id)
        if not path.exists():
            raise PipelineError("UNKNOWN_ARTIFACT", artifact_id)
        return json.loads(path.read_text())

    def has(self, artifact_id: str) -> bool:
        return self._path(artifact_id).exists()

    # -- state -------------------------------------------------------------
    def bind_project(self, fingerprint: str) -> None:
        current = self.state.get("project_fingerprint")
        if current is not None and current != fingerprint:
            # A changed scientific input is a new project run, never a silent resume.
            raise PipelineError("RUN_DIR_BELONGS_TO_DIFFERENT_INPUTS", str(self.root))
        self.state["project_fingerprint"] = fingerprint
        self._save_state()

    def stage(self, name: str) -> dict | None:
        return self.state["stages"].get(name)

    def commit_stage(self, name: str, status: str, input_fingerprint: str, artifact_ids: list[str], detail=None):
        self.state["state_version"] += 1
        self.state["stages"][name] = {"status": status, "input_fingerprint": input_fingerprint,
                                      "artifact_ids": artifact_ids, "detail": detail,
                                      "state_version": self.state["state_version"]}
        self._save_state()
        self.log("stage_committed", stage=name, status=status, artifact_ids=artifact_ids)

    def bump(self, counter: str, by: int = 1) -> None:
        self.state["counters"][counter] = self.state["counters"].get(counter, 0) + by
        self._save_state()

    @property
    def version(self) -> int:
        return self.state["state_version"]

    def log(self, event: str, **data) -> None:
        line = canonical_json({"event": event, "state_version": self.state["state_version"], **data})
        with open(self.root / "events.jsonl", "a") as handle:
            handle.write(line + "\n")

    def _save_state(self) -> None:
        self._atomic_write(self.state_path, json.dumps(self.state, indent=2, sort_keys=True))

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text)
        os.replace(tmp, path)
