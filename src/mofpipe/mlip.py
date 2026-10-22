"""Task 02 (T02.2): registered ML-potential calculator adapter for screening relaxations.

``protocols.ml_potential_checkpoint`` declares the potential: engine ``orb`` (Orb v3 conservative, OMat24 or MPtrj
training data; chosen for MOFs from MOFSimBench, where orb-v3-omat and eSEN-OAM gave the most successful and
accurate structure optimizations and orb has open weights) plus DFT-D3(BJ), one documented relaxation policy for
every structure, and the interpreter that runs ``mlip_worker.py`` (it needs orb-models; the pipeline core does not).

Jobs are idempotent: the key is the protocol hash plus the geometry hash, so a re-run or another stage reuses a
finished relaxation. The worker reports the engine identity (package versions, checkpoint sha256); a declared
``checkpoint_sha256`` that does not match blocks the stage. Domain flags mark use the potential does not represent
(net charge, open-shell initial states, undeclared elements); flagged screening is never treated as calibrated.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .contract import PipelineError, content_hash

ENGINES = {"orb": {"models": ("orb-v3-conservative-inf-omat", "orb-v3-conservative-20-omat", "orb-v3-conservative-inf-mpa"),
                   "dispersion": ("D3BJ", "none"), "charge_conditioning": False, "spin_input": False,
                   "training_data": {"omat": "OMat24 (PBE/PBE+U, no dispersion)", "mpa": "MPtrj + Alexandria (PBE/PBE+U)"}}}
RELAXATION_REQUIRED = ("optimizer", "fmax_eV_per_A", "max_steps", "cell")
WORKER = Path(__file__).with_name("mlip_worker.py")


def protocol_errors(protocol) -> list:
    """(code, field) pairs for protocols.ml_potential_checkpoint."""
    if not isinstance(protocol, dict):
        return [("SCIENTIFIC_SPEC_MISSING", "protocols.ml_potential_checkpoint")]
    errors, engine = [], ENGINES.get(protocol.get("engine"))
    if engine is None:
        return [("UNSUPPORTED_ML_POTENTIAL", f"protocols.ml_potential_checkpoint.engine: {protocol.get('engine')}")]
    if protocol.get("model") not in engine["models"]:
        errors.append(("UNSUPPORTED_ML_POTENTIAL", f"protocols.ml_potential_checkpoint.model: {protocol.get('model')}"))
    if protocol.get("dispersion") not in engine["dispersion"]:
        errors.append(("SCIENTIFIC_SPEC_MISSING", "protocols.ml_potential_checkpoint.dispersion (D3BJ | none)"))
    relax = protocol.get("relaxation") or {}
    errors += [("SCIENTIFIC_SPEC_MISSING", f"protocols.ml_potential_checkpoint.relaxation.{k}")
               for k in RELAXATION_REQUIRED if relax.get(k) is None]
    if relax.get("optimizer") not in (None, "FIRE", "BFGS") or relax.get("cell") not in (None, "variable", "fixed"):
        errors.append(("UNSUPPORTED_ML_POTENTIAL", "protocols.ml_potential_checkpoint.relaxation (FIRE|BFGS, variable|fixed)"))
    return errors


def _resolve(root: Path, path: str) -> str | None:
    # abspath, not resolve(): a venv interpreter is a symlink whose target lacks the venv's packages.
    return shutil.which(path) or (os.path.abspath(root / path) if (root / path).exists() else None)


class MLPotentialBackend:
    name = "ml_potential"

    def __init__(self, spec, store):
        physical = spec.physical
        self.spec, self.store = spec, store
        self.protocol = physical["protocols"]["ml_potential_checkpoint"]
        self.python = _resolve(spec.root, self.protocol.get("python") or sys.executable)
        self.worker = _resolve(spec.root, self.protocol["worker"]) if self.protocol.get("worker") else str(WORKER)
        if self.python is None or self.worker is None:
            raise PipelineError("BACKEND_UNAVAILABLE", f"ML-potential interpreter/worker not found: "
                                                       f"{self.protocol.get('python')} / {self.protocol.get('worker')}")
        self.worker_sha256 = hashlib.sha256(Path(self.worker).read_bytes()).hexdigest()
        scientific = {k: v for k, v in self.protocol.items() if k not in ("python", "worker", "timeout_s", "device")}
        self.protocol_hash = content_hash({"protocol": scientific, "worker_sha256": self.worker_sha256})
        self.fidelity = f"ML:{self.protocol['model']}+{self.protocol['dispersion']}:{self.protocol_hash[:12]}"
        self._identity = None

    def _run(self, batch_dir: Path, args: list, timeout: float):
        start = time.perf_counter()
        try:
            proc = subprocess.run([self.python, self.worker, *args], cwd=batch_dir, capture_output=True, text=True,
                                  timeout=timeout)
            out = {"returncode": proc.returncode, "stderr": proc.stderr[-2000:]}
        except subprocess.TimeoutExpired:
            out = {"returncode": None, "stderr": "TIMEOUT"}
        wall = time.perf_counter() - start
        self.store.bump("ml_core_seconds", int(math.ceil(wall)))
        return out, wall

    def identity(self) -> dict:
        """Engine identity from a worker probe (cached per protocol); verifies a declared checkpoint sha256."""
        if self._identity is not None:
            return self._identity
        probe_dir = self.store.jobs / f"mlprobe__{self.protocol_hash[:20]}"
        path = probe_dir / "probe.json"
        if not path.exists() or not json.loads(path.read_text()).get("identity"):
            probe_dir.mkdir(parents=True, exist_ok=True)
            (probe_dir / "batch.json").write_text(json.dumps({"protocol": self.protocol}))
            outcome, _ = self._run(probe_dir, ["--probe", "batch.json"], self.protocol.get("timeout_s", 3600))
            if not path.exists():
                raise PipelineError("BACKEND_UNAVAILABLE", f"ML-potential probe failed: {outcome['stderr'][-300:]}")
        probe = json.loads(path.read_text())
        if not probe.get("identity"):
            raise PipelineError("BACKEND_UNAVAILABLE", f"ML potential cannot load: {probe.get('error')}")
        identity = probe["identity"]
        declared = self.protocol.get("checkpoint_sha256")
        if declared and declared != identity.get("checkpoint_sha256"):
            raise PipelineError("ML_CHECKPOINT_MISMATCH", f"declared {declared[:12]}, found {identity['checkpoint_sha256'][:12]}")
        self._identity = identity
        return identity

    def domain_flags(self, elements) -> list:
        chem = self.spec.physical["chemistry"]
        engine, flags = ENGINES[self.protocol["engine"]], []
        if chem.get("net_charge") not in (0, None) and not engine["charge_conditioning"]:
            flags.append({"code": "CHARGED_SYSTEM_NOT_REPRESENTED", "detail": f"net_charge={chem['net_charge']}"})
        states = (chem.get("magnetic_state_policy") or {}).get("initial_states") or []
        if any(s.get("nspin", 1) != 1 or s.get("magmoms") for s in states) and not engine["spin_input"]:
            flags.append({"code": "MAGNETIC_STATE_NOT_REPRESENTED", "detail": "the potential has no spin input"})
        declared = self.protocol.get("supported_elements")
        if declared and set(elements) - set(declared):
            flags.append({"code": "ELEMENT_OUTSIDE_DECLARED_DOMAIN", "detail": str(sorted(set(elements) - set(declared)))})
        return flags

    def relax(self, geometries: list, purpose: str) -> dict:
        """geometry artifact_id -> worker result (+ job_id); finished jobs are reused, the rest run in one batch."""
        identity = self.identity()
        out, todo = {}, []
        for g in geometries:
            job_id = "mljob:" + content_hash({"protocol_hash": self.protocol_hash, "geometry_hash": g["geometry_hash"]})[:20]
            path = self.store.jobs / job_id.replace(":", "__") / "result.json"
            if path.exists():
                self.store.bump("ml_cache_hits")
                out[g["artifact_id"]] = {**json.loads(path.read_text()), "job_id": job_id}
            else:
                todo.append((g, job_id))
        if todo:
            batch_dir = self.store.jobs / f"mlbatch__{content_hash([j for _, j in todo])[:20]}"
            batch_dir.mkdir(parents=True, exist_ok=True)
            (batch_dir / "batch.json").write_text(json.dumps({
                "protocol": self.protocol, "relaxation": self.protocol["relaxation"], "purpose": purpose,
                "structures": [{"key": j, "topology_id": g["topology_id"], "configuration_id": g["decoration_id"],
                                "lattice_A": g["lattice_A"], "elements": g["elements"], "frac_coords": g["frac_coords"]}
                               for g, j in todo]}))
            self.store.bump("ml_relaxations", len(todo))
            self.store.log("ml_batch_submitted", purpose=purpose, n=len(todo))
            outcome, wall = self._run(batch_dir, ["batch.json"], self.protocol.get("timeout_s", 3600))
            results = json.loads((batch_dir / "results.json").read_text()).get("results", {}) \
                if (batch_dir / "results.json").exists() else {}
            for g, job_id in todo:
                res = results.get(job_id) or {"status": "FAILED", "error": f"worker: {outcome['stderr'][-300:]}"}
                res = {**res, "identity": identity, "fidelity": self.fidelity}
                if res["status"] == "SUCCEEDED":  # only finished relaxations are reusable
                    job_dir = self.store.jobs / job_id.replace(":", "__")
                    job_dir.mkdir(exist_ok=True)
                    (job_dir / "result.json").write_text(json.dumps(res))
                out[g["artifact_id"]] = {**res, "job_id": job_id}
            self.store.log("ml_batch_finished", n=len(todo), wall_time_s=round(wall, 1), returncode=outcome["returncode"])
        return out
