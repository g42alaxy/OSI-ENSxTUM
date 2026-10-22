"""Persistent workflow state, immutable artifact store and event log.

Workflow state on disk is authoritative (00_SHARED_CONTRACT, execution contract). Commits are optimistic: every
state write takes a file lock and checks that the on-disk revision is the one this store last read or wrote, so a
second writer (another process, an AiiDA worker) raises STATE_VERSION_CONFLICT instead of overwriting (T10.1).
"""
from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
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
        self._revision = self.state.get("revision", 0)  # the on-disk revision this store last read or wrote

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
    def bind_project(self, fingerprint: str, project_type: str = "synthetic_fixture", provenance: str = "SYNTHETIC") -> None:
        # Synthetic and physical projects never share a run directory, whatever their fingerprints.
        for key, value in (("project_type", project_type), ("provenance", provenance)):
            bound = self.state.get(key)
            if bound is not None and bound != value:
                raise PipelineError("RUN_DIR_PROJECT_TYPE_MISMATCH", f"{key}: run dir holds {bound}, project is {value}")
            self.state[key] = value
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

    @contextmanager
    def _lock(self):
        with open(self.root / "state.lock", "a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _save_state(self) -> None:
        with self._lock():
            on_disk = json.loads(self.state_path.read_text()).get("revision", 0) if self.state_path.exists() else 0
            if on_disk != self._revision:
                raise PipelineError("STATE_VERSION_CONFLICT", f"{self.root}: revision {on_disk} on disk, {self._revision} expected")
            self.state["revision"] = self._revision = on_disk + 1
            self._atomic_write(self.state_path, json.dumps(self.state, indent=2, sort_keys=True))

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text)
        os.replace(tmp, path)
