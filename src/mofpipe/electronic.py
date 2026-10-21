"""Tasks 06/07 boundary: asynchronous energy evaluation through an explicit backend.

The only backend in this build is the fixture's synthetic mock evaluator, run
as a separate process so the learner never imports the oracle formula.
Periodic DFT, embedding, VQE, SQD and FCI are reported as not executed.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from fractions import Fraction

from .contract import JobState, PipelineError, StageStatus, content_hash, exact, make_record
from .structures import configuration_id, orbit_id, validate_occupation

EVALUATOR_SCRIPT = "tools/mock_evaluator.py"

#TODO: Task 06 — PeriodicDFTBackend (AiiDA + aiida-quantumespresso, adapter for licensed codes) with the same
#TODO: submit/job_status/collect interface: cell+atomic relaxation and final single point; record XC, dispersion,
#TODO: pseudopotential hashes, cutoff, k-points, smearing, spin init, Hubbard U; converge settings for energy
#TODO: DIFFERENCES; test declared magnetic/structural initial states and store each metastable outcome; check
#TODO: forces/stress, inventory, topology and slot mapping after relaxation; route topology changes as
#TODO: transformation events; bounded retries inside a declared protocol (scientific changes = new protocol).
#TODO: Task 07 — embedding + electronic solvers: prepare_active_space (h1/h2, constants, spin counts, FCIDUMP +
#TODO: metadata), PySCF CASCI/FCI reference, simulated VQE on the identical Hamiltonian, hardware SQD adapter,
#TODO: and the scheme's own assemble_corrected_energy (double counting, all inequivalent nodes, UNVALIDATED status).


class SyntheticFixtureBackend:
    name = "synthetic_fixture_mock_evaluator"

    def __init__(self, spec, store, timeout_s: int = 300):
        self.spec, self.store, self.timeout_s = spec, store, timeout_s
        self.script = spec.root / EVALUATOR_SCRIPT
        if not self.script.exists():
            raise PipelineError("BACKEND_UNAVAILABLE", str(self.script))
        self.script_sha256 = hashlib.sha256(self.script.read_bytes()).hexdigest()

    # submit_* -> job_id; job_status(job_id); collect_result(job_id)
    def submit(self, requests: list, purpose: str) -> str:
        for r in requests:
            topo = self.spec.topologies.get(r["topology_id"])
            if topo is None:
                raise PipelineError("UNKNOWN_TOPOLOGY", r["topology_id"])
            validate_occupation(r["bits"], self.spec.n_sites, self.spec.n_B)
        payload = {"backend": self.name, "script_sha256": self.script_sha256, "case": self.spec.case,
                   "correction_mode": self.spec.correction_mode,
                   "requests": [{"topology_id": r["topology_id"], "bits": list(r["bits"])} for r in requests]}
        job_id = "job:" + content_hash(payload)[:20]  # idempotency key = scientific inputs
        job_dir = self.store.jobs / job_id.replace(":", "__")
        meta_path = job_dir / "job.json"
        if meta_path.exists() and json.loads(meta_path.read_text())["state"] == JobState.SUCCEEDED.value:
            self.store.bump("evaluator_cache_hits")
            self.store.log("job_reused", job_id=job_id, purpose=purpose)
            return job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "requests.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in payload["requests"]))
        meta = {"job_id": job_id, "purpose": purpose, "state": JobState.RUNNING.value, **{k: v for k, v in payload.items() if k != "requests"},
                "n_requests": len(requests)}
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
        self.store.bump("evaluator_submissions")
        self.store.log("job_submitted", job_id=job_id, purpose=purpose, n_requests=len(requests))
        cmd = [sys.executable, str(self.script), "--input", str(job_dir / "requests.jsonl"),
               "--out", str(job_dir / "responses.jsonl"), "--case", self.spec.case,
               "--correction", self.spec.correction_mode]
        try:
            proc = subprocess.run(cmd, cwd=self.spec.root, capture_output=True, text=True, timeout=self.timeout_s)
            ok = proc.returncode == 0
            meta["stderr"] = proc.stderr[-4000:]
        except subprocess.TimeoutExpired:
            ok, meta["stderr"] = False, "TIMEOUT"
        meta["state"] = JobState.SUCCEEDED.value if ok else JobState.FAILED.value
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
        self.store.log("job_finished", job_id=job_id, state=meta["state"])
        return job_id

    def job_status(self, job_id: str) -> dict:
        path = self.store.jobs / job_id.replace(":", "__") / "job.json"
        if not path.exists():
            raise PipelineError("UNKNOWN_JOB", job_id)
        return json.loads(path.read_text())

    def collect(self, job_id: str, expected_fidelity: str, comparison_group: str) -> list:
        """collect_result(job_id) -> [CalculationResult]; failures never become numbers."""
        meta = self.job_status(job_id)
        if meta["state"] != JobState.SUCCEEDED.value:
            raise PipelineError("JOB_FAILED", f"{job_id}: {meta.get('stderr', '')[:200]}")
        job_dir = self.store.jobs / job_id.replace(":", "__")
        requests = [json.loads(l) for l in (job_dir / "requests.jsonl").read_text().splitlines() if l.strip()]
        responses = [json.loads(l) for l in (job_dir / "responses.jsonl").read_text().splitlines() if l.strip()]
        if len(responses) != len(requests):
            raise PipelineError("RESPONSE_COUNT_MISMATCH", f"{len(responses)} != {len(requests)}")
        mode = self.spec.correction_mode
        corrected_fidelity = f"SYNTHETIC_CORRECTED_{self.spec.case.upper()}_{mode.upper()}"
        results = []
        for line, (req, resp) in enumerate(zip(requests, responses)):
            topo = self.spec.topologies[req["topology_id"]]
            if resp.get("topology_id") != req["topology_id"] or resp.get("bits") != req["bits"]:
                raise PipelineError("RESPONSE_IDENTITY_MISMATCH", f"line {line}")
            if resp.get("configuration_id") != configuration_id(topo, req["bits"]) or resp.get("orbit_id") != orbit_id(topo, req["bits"]):
                raise PipelineError("RESPONSE_IDENTITY_MISMATCH", f"line {line}")
            common = {"input_hash": content_hash(req), "job_id": job_id, "backend": self.name,
                      "configuration_id": resp["configuration_id"], "orbit_id": resp["orbit_id"],
                      "topology_id": req["topology_id"], "bits": req["bits"],
                      "raw_output_ref": f"{job_dir.name}/responses.jsonl#{line}"}
            if resp.get("status") != "SUCCEEDED":
                results.append(make_record("CalculationResult", self.spec.provenance, **common, status=resp.get("status"),
                                           base_energy_exact=None, correction_exact=None, corrected_energy_exact=None))
                continue
            for field, want in (("energy_unit", "eV"), ("normalization", "per_abstract_12_slot_formula_unit"),
                                ("n_formula_units", topo.n_formula_units), ("provenance", self.spec.provenance),
                                ("comparison_group", comparison_group), ("fidelity", expected_fidelity),
                                ("correction_mode", mode), ("corrected_fidelity", corrected_fidelity)):
                if resp.get(field) != want:
                    code = "PROVENANCE_MISMATCH" if field == "provenance" else "INCOMPATIBLE_" + field.upper()
                    raise PipelineError(code, f"{resp['configuration_id']}: {resp.get(field)!r} != {want!r}")
            base = exact(resp["energy_eV"])
            corr = exact(resp["synthetic_correction_eV"])
            if abs(base + corr - exact(resp["corrected_energy_eV"])) > Fraction(1, 10**12):
                raise PipelineError("CORRECTION_ARITHMETIC_MISMATCH", resp["configuration_id"])
            results.append(make_record(
                "CalculationResult", self.spec.provenance, **common, status="SUCCEEDED", method="synthetic_fixture_oracle",
                energy_unit="eV", normalization="per_abstract_12_slot_formula_unit", n_formula_units=topo.n_formula_units,
                comparison_group=comparison_group, base_fidelity=expected_fidelity,
                base_energy_eV=float(base), base_energy_exact=str(base),
                correction_mode=mode, correction_kind="ARTIFICIAL_BOOKKEEPING_TEST (not embedding, not DFT)",
                correction_eV=float(corr), correction_exact=str(corr),
                corrected_fidelity=corrected_fidelity, corrected_energy_eV=float(base + corr),
                corrected_energy_exact=str(base + corr), convergence="NOT_APPLICABLE_SYNTHETIC",
                uncertainty={"numerical": "exact synthetic label", "model_systematic": "not assessed"}))
        return results


def electronic_stage_status(spec) -> dict:
    ev = spec.config["evaluation"]
    off = StageStatus.DISABLED_BY_CONFIG.value
    return {
        "periodic_dft": off if not ev["dft_enabled"] else StageStatus.UNSUPPORTED.value,  #TODO: Task 06 DFT backend
        "embedding_correction": off if not ev["embedding_enabled"] else StageStatus.UNSUPPORTED.value,  #TODO: Task 07
        "active_space_fci": StageStatus.NOT_RUN.value, "vqe": StageStatus.NOT_RUN.value,  #TODO: Task 07 PySCF / VQE
        "sqd_hardware": off if not ev["hardware_enabled"] else StageStatus.UNSUPPORTED.value,  #TODO: Task 07 SQD
        "note": "Corrections in this run are artificial fixture bookkeeping tests, not electronic-structure results.",
    }
