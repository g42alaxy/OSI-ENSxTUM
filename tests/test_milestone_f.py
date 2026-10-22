"""Milestone F: ML-potential screening (Task 02), QPU fake-device adapter (T04.3), AiiDA engine and optimistic
state commits (T10.1).

ML screening runs on the SYNTHETIC two-topology toy with the mock worker (tests/mock_mlip.py) and mock pw.x. The
QPU and AiiDA tests need the project's .venv with the qpu/aiida extras and are skipped otherwise.
"""
import json
import os
import shutil
import subprocess
import sys

import pytest
from conftest import PROJECT, REPO
from toy_framework import write_toy_milestone_c_project

from qumof.contract import PipelineError
from qumof.intake import load_project
from qumof.orchestrator import run_pipeline
from qumof.store import RunStore

VENV_PYTHON = REPO / ".venv/bin/python"
ML = {"engine": "orb", "model": "orb-v3-conservative-inf-omat", "dispersion": "D3BJ", "python": sys.executable,
      "worker": "tools/mock_mlip.py", "relaxation": {"optimizer": "FIRE", "fmax_eV_per_A": 0.05, "max_steps": 300,
                                                     "cell": "variable"}}


def _toy(tmp_path, calibration=True, energetic=True):
    protocols = {"ml_potential_checkpoint": ML,
                 "ml_calibration": {"n_structures_per_topology": 2, "tolerance_eV_per_formula_unit": 0.01} if calibration else None,
                 "screening": {"energetic_policy": {"deprioritise_above_eV_per_formula_unit": 0.05} if energetic else None}}
    project = write_toy_milestone_c_project(tmp_path / "toy", {"protocols": protocols})
    shutil.copy(REPO / "tests/mock_mlip.py", tmp_path / "toy/tools/mock_mlip.py")
    return project


def test_ml_screening_deprioritises_only_calibrated_gaps_and_narrows_the_run(tmp_path, monkeypatch):
    monkeypatch.setenv("MOCK_MLIP_MIRROR_PW", "1")  # ML agrees with the (mock) DFT: calibration error ~0
    spec = load_project(_toy(tmp_path))
    report = run_pipeline(spec, tmp_path / "run")
    screening = report["physical"]["screening"]
    states = {d["topology_id"]: d["state"] for d in screening["decisions"]}
    assert screening["ranking_label"] == "CALIBRATED" and states == {"toy_pcu": "retain", "toy_pcu_staggered": "deprioritise"}
    gap = next(d for d in screening["decisions"] if d["topology_id"] == "toy_pcu_staggered")["gap_to_best_eV_per_formula_unit"]
    assert gap > 0.05  # the staggered polymorph is higher in both ML and DFT (0.056 eV per formula unit)
    assert report["pipeline_status"] == "COMPLETED" and list(report["topologies"]) == ["toy_pcu"]  # later stages narrowed
    assert set(report["stage_detail"]["labels"]["n_training"]) == {"toy_pcu"}
    seeds = json.loads(next((tmp_path / "run/artifacts").glob("DecorationSet__*.json")).read_text())
    assert seeds["methods"] == {"ordered_high_symmetry": 2, "random": 3, "sqs_like": 1, "pure_endpoint": 2}
    fixed = [d for d in seeds["decorations"] if d["sampling_method"] != "pure_endpoint"]
    assert all(sorted(d["inventory"].values()) == [6, 6] for d in fixed)  # composition survives every sampling method
    store = RunStore(tmp_path / "run")
    relaxations = store.state["counters"]["ml_relaxations"]
    run_pipeline(load_project(_toy(tmp_path)), tmp_path / "run")  # a re-run reuses every stage and every ML job
    assert RunStore(tmp_path / "run").state["counters"]["ml_relaxations"] == relaxations == 16


@pytest.mark.parametrize("calibration,fail,expected", [
    (False, False, {"toy_pcu": "retain", "toy_pcu_staggered": "retain"}),        # uncalibrated: exploratory only
    (True, False, {"toy_pcu": "retain", "toy_pcu_staggered": "retain"}),         # calibration fails: exploratory
    (True, True, {"toy_pcu": "uncertain", "toy_pcu_staggered": "uncertain"}),     # failures are never energies
])
def test_uncalibrated_or_failed_screening_keeps_every_topology(tmp_path, monkeypatch, calibration, fail, expected):
    monkeypatch.setenv("MOCK_MLIP_SHIFT_TOPOLOGY", "toy_pcu_staggered")
    monkeypatch.setenv("MOCK_MLIP_SHIFT", "0.01")
    if fail:
        monkeypatch.setenv("MOCK_MLIP_FAIL", "1")
    out = run_pipeline(load_project(_toy(tmp_path, calibration=calibration)), tmp_path / "run", stop_after="screening")
    assert {d["topology_id"]: d["state"] for d in out["screening"]["decisions"]} == expected
    assert out["topologies"] == ["toy_pcu", "toy_pcu_staggered"]
    assert out["screening"]["ranking_label"] == "EXPLORATORY"
    if calibration and not fail:  # the shifted ML energies disagree with DFT beyond tolerance
        assert out["detail"]["calibration"]["status"] == "CALIBRATION_FAILED"


def test_store_rejects_a_concurrent_writer(tmp_path):
    first, second = RunStore(tmp_path / "run"), RunStore(tmp_path / "run")
    first.commit_stage("intake", "COMPLETED", "fp", {}, {})
    with pytest.raises(PipelineError, match="STATE_VERSION_CONFLICT"):
        second.commit_stage("intake", "FAILED", "fp", {}, {})
    assert RunStore(tmp_path / "run").stage("intake")["status"] == "COMPLETED"


def _venv_has(*modules):
    if not VENV_PYTHON.exists():
        return False
    return subprocess.run([str(VENV_PYTHON), "-c", "import " + ", ".join(modules)], capture_output=True).returncode == 0


@pytest.mark.skipif(not _venv_has("qiskit", "qiskit_aer", "qiskit_ibm_runtime"), reason="needs .venv with the qpu extra")
def test_qaoa_on_fake_brisbane_matches_the_simulator_and_reports_noise(tmp_path):
    q = {"kind": "qpu_fake_device", "device": "fake_brisbane", "python": str(VENV_PYTHON), "layers": 1, "shots": 400,
         "seeds": [0], "restarts": 1, "max_evaluations": 40, "optimization_level": 1}
    spec = load_project(PROJECT, case="core_pairwise", overrides={"optimizer": {"quantum_enabled": True, "quantum_protocol": q}})
    report = run_pipeline(spec, tmp_path / "run")
    assert report["pipeline_status"] == "COMPLETED"
    runs = [json.loads(p.read_text()) for p in (tmp_path / "run/artifacts").glob("QuantumHardwareRun__*.json")]
    assert len(runs) == 2 and all(r["circuit_matches_simulator"] for r in runs)
    for r in runs:
        run = r["runs"][0]
        assert run["circuit_vs_simulator_max_abs_probability_difference"] < 1e-6 and run["shots"] == 400
        assert 0 < run["postselection_rate"] < 1  # device noise breaks the composition constraint for some shots
        assert r["device"]["device"] == "fake_brisbane" and run["transpiled"]["two_qubit_gates"] > 0


@pytest.mark.skipif(not _venv_has("aiida") or not (REPO / ".aiida").exists(), reason="needs .venv with aiida and the .aiida profile")
def test_aiida_engine_reproduces_the_local_engine(tmp_path):
    out = subprocess.run([str(VENV_PYTHON), "-m", "qumof", "run", "--project", str(PROJECT), "--case", "higher_order",
                          "--correction", "topology_dependent", "--engine", "aiida", "--run-dir", str(tmp_path / "aiida")],
                         capture_output=True, text=True, cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO / "src")})
    summary = json.loads(out.stdout)
    local = run_pipeline(load_project(PROJECT, case="higher_order", correction_mode="topology_dependent"), tmp_path / "local")
    assert summary["engine"] == "aiida" and summary["aiida"]["n_stage_nodes"] == 15
    assert summary["winning_topology"] == local["winning_topology"] and summary["stage_status"] == local["stage_status"]
    aiida_report = json.loads((tmp_path / "aiida/report.json").read_text())
    assert aiida_report["corrected_ranking"] == local["corrected_ranking"]
