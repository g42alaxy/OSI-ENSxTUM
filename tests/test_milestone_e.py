"""Milestone E: finite-cluster subtractive embedding, corrected-energy assembly (T07.3) and its feedback into the
structural ranking.

The periodic DFT is tests/mock_pw.py and the cluster engine tests/mock_cluster.py, so every toy number is
SYNTHETIC. The reference benchmark (propanal/propenol, HF:CCSD against full CCSD) runs real PySCF.
"""
import json
from fractions import Fraction

import pytest
from conftest import REPO, load_artifacts
from toy_framework import write_toy_embedding_project

from mofpipe import embedding
from mofpipe.contract import PipelineError
from mofpipe.intake import load_project, validate_project
from mofpipe.orchestrator import run_pipeline
from mofpipe.store import RunStore

BENCHMARK = REPO / "benchmarks/embedding_keto_enol_hf_ccsd.json"


@pytest.fixture(scope="module")
def unvalidated(tmp_path_factory):
    """bond_hops 1 with a [1, 2] boundary study: the 2-hop region of this one-cell-thick toy wraps around c, and no
    reference benchmark is declared, so the scheme stays UNVALIDATED."""
    spec = load_project(write_toy_embedding_project(tmp_path_factory.mktemp("emb")))
    run_dir = tmp_path_factory.mktemp("emb_run")
    return spec, run_dir, run_pipeline(spec, run_dir)


def test_unvalidated_corrections_stay_benchmark_artifacts_and_ranking_stays_dft_only(unvalidated):
    spec, run_dir, report = unvalidated
    detail = report["stage_detail"]["embedding"]
    assert report["stage_status"]["embedding"] == "INCONCLUSIVE" and detail["validation"] == "UNVALIDATED"
    assert detail["checks"] == {"high_equals_low": "PASSED", "balanced_regions": "PASSED",
                                "boundary_convergence": "NOT_ASSESSABLE", "reference_benchmarks": "NONE_DECLARED"}
    section = report["physical"]["embedding"]
    assert section["boundary_convergence"]["failure"]["code"] == "REGION_EXCEEDS_CELL"
    assert {(c["record_type"], c["correction_status"]) for c in section["corrections"]} == \
        {("EmbeddingBenchmarkResult", "UNVALIDATED")}
    assert report["corrected_ranking"] is None and report["stage_detail"]["ranking"]["corrected_ranking"] == "UNVALIDATED"
    assert report["base_ranking"] and report["pipeline_status"] == "FAILED_OR_INCOMPLETE"
    # Every node of the cell is covered once; the four Zn clusters are one translation-equivalent cluster.
    plan = load_artifacts(run_dir, "EmbeddingPlan")[0]
    assert [c["multiplicity"] for c in plan["clusters"].values()] == [4] == [plan["n_nodes_expected"]]
    assert sorted(r["node"] for r in plan["regions"]) == [0, 1, 2, 3]
    assert len({a for r in plan["regions"] for a in r["atoms"]}) == sum(len(r["atoms"]) for r in plan["regions"])


def test_assembly_blocks_missing_terms_incomplete_coverage_foreign_geometry_and_overlap(unvalidated, monkeypatch):
    spec, run_dir, _ = unvalidated
    store = RunStore(run_dir)
    plan = load_artifacts(run_dir, "EmbeddingPlan")[0]
    calc = next(c for c in load_artifacts(run_dir, "CalculationResult") if c["artifact_id"] == plan["calculation_result_id"])
    backend = embedding.ClusterBackend(spec, store)
    before = store.state["counters"]["cluster_submissions"]
    results = backend.evaluate(plan)
    assert store.state["counters"]["cluster_submissions"] == before  # identical inputs reuse the cluster jobs
    (h, pair), = results.items()
    out = embedding.assemble_corrected_energy(spec, calc, plan, results, backend)
    delta = Fraction(repr(pair["high"]["energy_Ha"])) - Fraction(repr(pair["low"]["energy_Ha"]))
    assert Fraction(out["unvalidated_correction_exact"]) == 4 * delta * embedding.HARTREE_TO_EV / plan["n_formula_units"]
    assert out["record_type"] == "EmbeddingBenchmarkResult"  # no validation report -> never a corrected energy

    def code(*args, **kwargs):
        return embedding.assemble_corrected_energy(spec, *args, backend, **kwargs)["failure"]["code"]
    assert code(calc, plan, {h: {"high": pair["high"]}}) == "MISSING_SUBTRACTION_TERM"
    assert code(calc, plan, {h: {"low": pair["low"]}}) == "MISSING_HIGH_LEVEL_TERM"
    three = {**plan, "clusters": {h: {**plan["clusters"][h], "multiplicity": 3, "nodes": plan["clusters"][h]["nodes"][:3]}}}
    assert code(calc, three, results) == "INCOMPLETE_NODE_COVERAGE"
    assert code({**calc, "relaxed_geometry_hash": "other"}, plan, results) == "INCOMPATIBLE_GEOMETRY"
    protocol = backend.protocol
    with pytest.raises(PipelineError, match="CAPPING_RULE_MISSING"):
        embedding.plan_regions(spec, calc, 1, {**protocol, "capping": {"element": "H", "distance_A": {"Zn": 1.6}}})
    with pytest.raises(PipelineError, match="REGION_EXCEEDS_CELL"):
        embedding.plan_regions(spec, calc, 2, protocol)
    real = embedding.cut_region
    monkeypatch.setattr(embedding, "cut_region", lambda *a: {**real(*a), "atoms": [0, 1, 2]})  # every region claims atom 0
    with pytest.raises(PipelineError, match="OVERLAPPING_REGIONS"):
        embedding.plan_regions(spec, calc, 1, protocol)


def test_validated_correction_feeds_the_ranking_and_a_constant_correction_changes_nothing(tmp_path):
    pytest.importorskip("pyscf")
    spec = load_project(write_toy_embedding_project(tmp_path / "emb", bond_hops=0, series=[0, 1], benchmarks=[str(BENCHMARK)]))
    report = run_pipeline(spec, tmp_path / "run")
    detail = report["stage_detail"]["embedding"]
    assert detail["validation"] == "VALIDATED" and report["stage_status"]["embedding"] == "COMPLETED", detail
    assert report["pipeline_status"] == "COMPLETED" and report["stage_detail"]["ranking"]["corrected_ranking"] == "COMPLETED"
    base, corrected = report["base_ranking"], report["corrected_ranking"]
    assert len(corrected) == len(base) == detail["n_corrected"]
    # Every node cluster is the same in every candidate: one constant correction per formula unit, so the order,
    # the ties and the winner are unchanged.
    by_orbit = {r["orbit_id"]: Fraction(r["ranking_energy_exact"]) for r in corrected}
    shift = {by_orbit[b["orbit_id"]] - Fraction(b["ranking_energy_exact"]) for b in base}
    assert len(shift) == 1 and shift != {0}
    assert [r["orbit_id"] for r in corrected] == [r["orbit_id"] for r in base]
    assert [r["tied_with_previous"] for r in corrected] == [r["tied_with_previous"] for r in base]
    assert report["stage_detail"]["ranking"]["ranking_changed_by_correction"] is False
    assert {r["corrected_fidelity"] for r in corrected} == {f"{base[0]['base_fidelity']}+{detail['embedding_fidelity']}"}
    for row in corrected:  # every corrected number traces to its CorrectedEnergy record
        record = next(c for c in load_artifacts(tmp_path / "run", "CorrectedEnergy") if c["orbit_id"] == row["orbit_id"])
        assert record["corrected_energy_exact"] == row["ranking_energy_exact"] and record["correction_status"] == "VALIDATED"
    assert "with the validated embedding correction" in (tmp_path / "run/report.md").read_text()
    # The low level must be the periodic functional, or the subtraction removes what the cell never contained.
    bad = load_project(write_toy_embedding_project(tmp_path / "bad", low_xc="B3LYP"))
    assert "EMBEDDING_LOW_LEVEL_MISMATCH" in {e["code"] for e in validate_project(bad)["errors"]}


def test_embedding_benchmark_reproduces_the_independent_full_ccsd(tmp_path):
    pytest.importorskip("pyscf")
    from mofpipe.__main__ import main
    assert main(["embedding-benchmark", "--spec", str(BENCHMARK), "--run-dir", str(tmp_path / "run")]) == 0
    report = load_artifacts(tmp_path / "run", "EmbeddingBenchmarkReport")[0]
    checks, tol = report["checks"], report["tolerance_Ha"]
    assert report["status"] == "REPRODUCED" and all(c["passed"] for c in checks.values())
    assert abs(checks["exact_limit"]["residual_Ha"]) < 1e-8  # region = whole molecule gives the full CCSD energy
    # The embedded relative energy is within tolerance while the low level alone is not: the test discriminates.
    assert abs(checks["relative_energy"]["embedded_error_Ha"]) < tol < abs(checks["discriminating"]["low_level_error_Ha"])
    assert {json.dumps(c["cluster_inventory"]) for c in report["configurations"].values()} == {'{"C": 2, "H": 4, "O": 1}'}
