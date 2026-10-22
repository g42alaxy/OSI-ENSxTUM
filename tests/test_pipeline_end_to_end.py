"""Full deterministic pipeline on the fixture: every case x correction mode against harness and reference."""
import json

import pytest
from conftest import CASES, MODES

from qumof.acceptance import check_run


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("mode", MODES)
def test_acceptance_against_harness_and_reference(runs, case, mode):
    spec, run_dir, report = runs[(case, mode)]
    result = check_run(spec, {**report, "_run_dir": str(run_dir)})
    failed = [c for c in result["checks"] if not c["passed"]]
    assert result["status"] == "PASS", failed


@pytest.mark.parametrize("case", CASES)
def test_constant_correction_preserves_full_order(runs, case):
    none = runs[(case, "none")][2]["corrected_ranking"]
    const = runs[(case, "constant")][2]["corrected_ranking"]
    assert [r["orbit_id"] for r in none] == [r["orbit_id"] for r in const]
    assert all(c["corrected_energy_eV"] - n["corrected_energy_eV"] == pytest.approx(0.05, abs=1e-12) for n, c in zip(none, const))


@pytest.mark.parametrize("case", CASES)
def test_topology_dependent_correction_flips_winner(runs, case):
    report = runs[(case, "topology_dependent")][2]
    assert report["base_winning_topology"] == "prism6" and report["winning_topology"] == "ring12"
    assert report["corrected_pool_minima_eV"] == pytest.approx({"ring12": -0.12, "prism6": -0.10}, abs=1e-12)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("mode", MODES)
def test_bookkeeping_and_provenance(runs, case, mode):
    spec, run_dir, report = runs[(case, mode)]
    artifacts = [json.loads(p.read_text()) for p in (run_dir / "artifacts").glob("*.json")]
    assert artifacts and all(a["provenance"] == "SYNTHETIC" for a in artifacts)
    train = [a for a in artifacts if a["record_type"] == "DatasetSnapshot" and a["role"] == "training"]
    assert all(t["fidelity"] == f"SYNTHETIC_BASE_{case.upper()}" for t in train)  # corrections never enter training
    for row in report["corrected_ranking"]:
        assert row["base_fidelity"] != row["corrected_fidelity"]
        assert row["corrected_energy_eV"] == pytest.approx(row["base_energy_eV"] + row["correction_eV"], abs=1e-12)
        assert row["surrogate_base_energy_eV"] == pytest.approx(row["base_energy_eV"], abs=1e-12)
    assert report["llm_calls"] == 0 and report["runtime_agents_enabled"] is False
    assert report["stage_status"]["refinement"] == "DISABLED_BY_CONFIG"
    assert report["stage_status"]["validation"] == "SKIPPED_NO_ATOMIC_GEOMETRY"
    assert set(report["structural_checks"].values()) == {"SKIPPED_NO_ATOMIC_GEOMETRY"}
    assert report["competition"] == "INCOMPLETE"
    assert report["literature"]["live_review"] == "NOT_RUN"
    electronic = report["stage_detail"]["evaluation"]["electronic"]
    assert electronic["periodic_dft"] == electronic["embedding_correction"] == "DISABLED_BY_CONFIG"
    assert (run_dir / "report.md").read_text().startswith("# Pipeline report")


def test_degeneracies_are_recorded_not_hidden(runs):
    ranked = runs[("core_pairwise", "none")][2]["corrected_ranking"]
    ties = [r for r in ranked if r["n_degenerate"] > 1]
    assert ties and any(r["tied_with_previous"] for r in ties)
