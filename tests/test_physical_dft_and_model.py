"""Milestone B: periodic DFT backend (T06.1), real evaluation path (T06.2) and grouped model validation (T03.2).

The DFT engine is tests/mock_pw.py speaking Quantum ESPRESSO formats; every number is SYNTHETIC.
"""
import json
import shutil
from fractions import Fraction

import pytest
from conftest import load_artifacts
from toy_framework import write_toy_project

from mofpipe import configuration_model as cm
from mofpipe import datasets, dft, framework, structures
from mofpipe.contract import PipelineError, make_record
from mofpipe.intake import load_project
from mofpipe.orchestrator import STAGES, run_pipeline
from mofpipe.store import RunStore


def _state(run_dir):
    return json.loads((run_dir / "state.json").read_text())


@pytest.fixture
def toy_backend(tmp_path):
    spec = load_project(write_toy_project(tmp_path / "toy"))
    store = RunStore(tmp_path / "run")
    topo = spec.topologies["toy_pcu"]
    decoration = structures.make_decoration(spec, topo, next(structures.feasible_states(12, 6)))
    geometry = framework.build_framework(spec, topo, decoration)
    return spec, store, dft.PeriodicDFTBackend(spec, store), geometry, decoration


# -- T06.1 periodic DFT backend --------------------------------------------------
def test_qe_input_records_every_scientific_setting(toy_backend):
    spec, _, backend, geometry, _ = toy_backend
    text = backend.adapter.write_input(geometry, backend.protocol, backend.tolerances, "vc-relax",
                                       {"label": "closed_shell", "nspin": 1}, {"mixing_beta": 0.7})
    for needle in ("calculation = 'vc-relax'", "input_dft = 'PBE'", "vdw_corr = 'grimme-d3'", "ecutwfc = 80",
                   "ecutrho = 640", "nspin = 1", "conv_thr = 1e-09", "mixing_beta = 0.7", "press_conv_thr = 0.5",
                   "Zn 65.38 Zn.toy.UPF", "K_POINTS automatic", "CELL_PARAMETERS angstrom", f"nat = {len(geometry['elements'])}"):
        assert needle in text, needle
    assert backend.fidelity == f"DFT:toy_qe_pbe_d3:{backend.protocol_hash[:12]}"


def test_relaxation_and_single_point_give_normalized_energy(toy_backend):
    spec, store, backend, geometry, decoration = toy_backend
    evaluation = dft.evaluate_geometry(backend, geometry, decoration, "test")
    result = evaluation["selected"]
    assert evaluation["status"] == "SUCCEEDED" and result["calculation"] == "scf"  # final single point
    assert result["n_formula_units"] == 2 and result["normalization"] == "per_formula_unit"
    total = Fraction(result["raw_energy"]["value"]) * dft.RY_TO_EV
    assert Fraction(result["base_energy_exact"]) == total / 2
    assert result["slot_mapping"] == "RETAINED" and result["convergence"]["software_version"] == "v.7.2"
    assert result["pseudopotential_sha256"]["Zn"] == spec.physical["protocols"]["dft"]["pseudopotentials"]["Zn"]["sha256"]
    assert store.state["counters"]["dft_submissions"] == 2  # relaxation + final single point


def test_scf_retry_changes_only_declared_numerics(toy_backend, monkeypatch):
    spec, store, backend, geometry, decoration = toy_backend
    monkeypatch.setenv("MOCK_PW_MODE", "scf_fail_high_beta")
    result = dft.evaluate_geometry(backend, geometry, decoration, "retry")["selected"]
    assert result["status"] == "SUCCEEDED"
    history = result["recovery_history"]
    assert [a["numerics"] for a in history] == [{"mixing_beta": 0.7}, {"mixing_beta": 0.3}]
    assert [a["scf_converged"] for a in history] == [False, True]


@pytest.mark.parametrize("mode,status,code", [
    ("unconverged_relax", "FAILED", "IONIC_NOT_CONVERGED"),
    ("large_force", "FAILED", "RESIDUAL_FORCE_ABOVE_TOLERANCE"),
    ("lose_atom", "FAILED", "COMPOSITION_CHANGED"),
    ("topology_change", "TRANSFORMED", "TOPOLOGY_CHANGED_DURING_RELAXATION"),
    ("crash", "FAILED", "JOB_FAILED"),
])
def test_unusable_calculations_never_become_energies(toy_backend, monkeypatch, mode, status, code):
    spec, store, backend, geometry, decoration = toy_backend
    monkeypatch.setenv("MOCK_PW_MODE", mode)
    evaluation = dft.evaluate_geometry(backend, geometry, decoration, mode)
    result = evaluation["selected"]
    assert evaluation["status"] == status and result["status"] == status
    assert result["failure"]["code"] == code and result["base_energy_exact"] is None
    with pytest.raises(PipelineError):
        datasets.check_energy_compatibility([{**result, "fidelity": result["base_fidelity"], "energy_unit": "eV",
                                              "normalization": "per_formula_unit"}], spec.provenance)


def test_equivalent_representations_give_matching_normalized_energies(toy_backend):
    spec, store, backend, geometry, decoration = toy_backend
    reference = dft.evaluate_geometry(backend, geometry, decoration, "ref")["selected"]
    # Same structure, atoms listed in reverse order: a distinct input, identical normalized energy.
    reordered = {**geometry, "elements": geometry["elements"][::-1], "frac_coords": geometry["frac_coords"][::-1],
                 "atom_map": geometry["atom_map"][::-1], "geometry_hash": "reordered"}
    evaluation = dft.evaluate_geometry(backend, reordered, decoration, "reordered")
    again = evaluation["selected"]
    # A new relaxation job; the final single point is keyed by the order-independent relaxed geometry.
    assert evaluation["outcomes"][0]["relaxation"]["job_id"] != reference["relaxation_result_id"]
    assert abs(Fraction(again["base_energy_exact"]) - Fraction(reference["base_energy_exact"])) <= Fraction(1, 10**6)
    # A symmetry-equivalent decoration agrees within the target tolerance (the substituent orientation
    # convention is not symmetry-adapted yet, T01.4, so agreement is not exact).
    topo = spec.topologies["toy_pcu"]
    image = next(structures.symmetry_image(p, decoration["bits"]) for p in topo.permutations
                 if structures.symmetry_image(p, decoration["bits"]) != tuple(decoration["bits"]))
    assert structures.orbit_id(topo, image) == decoration["orbit_id"]
    image_decoration = structures.make_decoration(spec, topo, image)
    image_result = dft.evaluate_geometry(backend, framework.build_framework(spec, topo, image_decoration),
                                         image_decoration, "image")["selected"]
    tol = Fraction(str(spec.physical["tolerances"]["target_relative_energy_tolerance"]))
    assert abs(Fraction(image_result["base_energy_exact"]) - Fraction(reference["base_energy_exact"])) <= tol


def test_replicated_cell_gives_the_same_energy_per_formula_unit(toy_backend):
    spec, store, backend, geometry, decoration = toy_backend
    lattice = geometry["lattice_A"]
    doubled = {**geometry, "lattice_A": [lattice[0], lattice[1], [2 * x for x in lattice[2]]],
               "elements": geometry["elements"] * 2,
               "frac_coords": [[f[0], f[1], f[2] / 2] for f in geometry["frac_coords"]]
               + [[f[0], f[1], f[2] / 2 + 0.5] for f in geometry["frac_coords"]],
               "geometry_hash": "replicated_z2",
               "inventory": {el: 2 * n for el, n in geometry["inventory"].items()}}
    energies = []
    for g in (geometry, doubled):
        job = backend.submit(g, "scf", backend.magnetic_states[0], "replication")
        meta = backend.job_status(job)
        text = (store.jobs / job.replace(":", "__") / "attempt_0/pw.out").read_text()
        parsed = backend.adapter.parse_output(text, "scf", g)
        n_fu = dft.formula_units(g["inventory"], spec.physical["chemistry"]["target_elemental_inventory"])
        energies.append(Fraction(parsed["total_energy"]) * dft.RY_TO_EV / n_fu)
        assert meta["state"] == "SUCCEEDED"
    assert abs(energies[0] - energies[1]) <= Fraction(str(spec.physical["tolerances"]["target_relative_energy_tolerance"]))


def test_job_restart_reuses_successful_work(toy_backend):
    spec, store, backend, geometry, decoration = toy_backend
    first = dft.evaluate_geometry(backend, geometry, decoration, "first")["selected"]
    fresh = dft.PeriodicDFTBackend(spec, RunStore(store.root))  # a new process resuming the same run dir
    second = dft.evaluate_geometry(fresh, geometry, decoration, "resume")["selected"]
    counters = fresh.store.state["counters"]
    assert counters["dft_submissions"] == 2 and counters["dft_cache_hits"] == 2
    assert second["base_energy_exact"] == first["base_energy_exact"]


def test_budget_is_enforced_before_submission(tmp_path):
    spec = load_project(write_toy_project(tmp_path / "toy", overrides={"budgets": {"max_dft_jobs": 10}}))
    report = run_pipeline(spec, tmp_path / "run")
    state = _state(tmp_path / "run")
    assert report["stage_status"]["labels"] == "BLOCKED"
    assert state["stages"]["labels"]["detail"]["error"]["code"] == "BUDGET_EXHAUSTED"
    assert state["counters"]["dft_submissions"] == 10 and report["base_ranking"] is None


def test_missing_executable_blocks_instead_of_failing(tmp_path):
    spec = load_project(write_toy_project(tmp_path / "toy", overrides={"protocols": {"dft": {"command": ["no-such-pw.x"]}}}))
    report = run_pipeline(spec, tmp_path / "run")
    assert report["stage_status"]["labels"] == "BLOCKED"
    assert _state(tmp_path / "run")["stages"]["labels"]["detail"]["error"]["code"] == "BACKEND_UNAVAILABLE"


def test_unconverged_cutoff_blocks_before_any_label(tmp_path):
    spec = load_project(write_toy_project(tmp_path / "toy", overrides={"protocols": {"dft": {
        "ecutwfc_Ry": 40, "ecutrho_Ry": 320, "difference_convergence": {"ecutwfc_Ry_series": [40, 60, 80]}}}}))
    report = run_pipeline(spec, tmp_path / "run")
    state = _state(tmp_path / "run")
    detail = state["stages"]["labels"]["detail"]
    assert report["stage_status"]["labels"] == "BLOCKED" and detail["error"]["code"] == "DFT_SETTINGS_NOT_CONVERGED"
    assert detail["dft_convergence"]["toy_pcu"]["deviation_from_highest_cutoff_eV"] > 0.005
    assert state["counters"]["dft_submissions"] == 6  # 2 decorations x 3 cutoffs, no relaxations
    assert not load_artifacts(tmp_path / "run", "DatasetSnapshot")


# -- T06.2 real evaluation path ---------------------------------------------------
def test_physical_pipeline_completes_with_dft_ranking(toy_run):
    spec, run_dir, report = toy_run
    assert report["pipeline_status"] == "COMPLETED", report["stage_status"]
    assert all(report["stage_status"][s] in ("COMPLETED", "DISABLED_BY_CONFIG") for s in STAGES[:-1])
    assert report["provenance"] == "SYNTHETIC" and report["project_type"] == "physical_structure"
    ranked = report["base_ranking"]
    assert ranked and report["corrected_ranking"] is None
    assert {r["base_fidelity"] for r in ranked} == {f"DFT:toy_qe_pbe_d3:{dft.PeriodicDFTBackend(spec, RunStore(run_dir)).protocol_hash[:12]}"}
    assert [r["rank"] for r in ranked] == list(range(1, len(ranked) + 1))
    # Every ranked number traces to a stored CalculationResult and its Geometry.
    for row in ranked:
        calc = json.loads((run_dir / "artifacts" / (row["calculation_result_id"].replace(":", "__") + ".json")).read_text())
        assert calc["base_energy_exact"] == row["ranking_energy_exact"] and calc["status"] == "SUCCEEDED"
        assert (run_dir / "artifacts" / (calc["geometry_id"].replace(":", "__") + ".json")).exists()
    evaluation = report["stage_detail"]["evaluation"]
    assert evaluation["electronic"]["periodic_dft"] == "COMPLETED" and evaluation["labels_used_for_training"] is False
    assert evaluation["dft_cache_hits"] >= 2 * len(ranked)  # pool members already labelled are not recomputed
    assert "periodic DFT" not in report["not_executed"]
    assert "labelled SYNTHETIC" in (run_dir / "report.md").read_text()


def test_ranking_ties_use_the_declared_tolerance(toy_run):
    spec, _, report = toy_run
    ranked = report["base_ranking"]
    tol = spec.physical["tolerances"]["target_relative_energy_tolerance"]
    for prev, row in zip(ranked, ranked[1:]):
        assert row["tied_with_previous"] == (row["ranking_energy_eV"] - prev["ranking_energy_eV"] <= tol + 1e-12)
    assert any(r["tied_with_previous"] for r in ranked)  # the toy has a near-degenerate pair


def test_rerun_reuses_every_stage_without_new_dft_jobs(toy_run, tmp_path):
    spec, run_dir, report = toy_run
    copy = tmp_path / "copy"
    shutil.copytree(run_dir, copy)
    before = _state(copy)["counters"]["dft_submissions"]
    again = run_pipeline(spec, copy)
    assert again["artifact_id"] == report["artifact_id"]
    assert _state(copy)["counters"]["dft_submissions"] == before


def test_interrupted_labelling_resumes_from_job_cache(toy_run, tmp_path):
    spec, run_dir, report = toy_run
    copy = tmp_path / "copy"
    shutil.copytree(run_dir, copy)
    state = _state(copy)
    for stage in STAGES[STAGES.index("labels"):]:
        state["stages"].pop(stage)
    (copy / "state.json").write_text(json.dumps(state))
    before = state["counters"]["dft_submissions"]
    again = run_pipeline(spec, copy)
    assert _state(copy)["counters"]["dft_submissions"] == before
    assert again["base_ranking"] == report["base_ranking"]


def test_build_failure_in_pool_makes_ranking_incomplete(toy_run, tmp_path, monkeypatch):
    spec, run_dir, report = toy_run
    copy = tmp_path / "copy"
    shutil.copytree(run_dir, copy)
    state = _state(copy)
    for stage in ("evaluation", "ranking", "validation", "report"):
        state["stages"].pop(stage)
    (copy / "state.json").write_text(json.dumps(state))
    victim = report["base_ranking"][0]["orbit_id"]
    real = framework.build_framework

    def failing(spec_, topology, decoration):
        if decoration["orbit_id"] == victim:
            return make_record("BuildFailure", spec_.provenance, topology_id=topology.topology_id,
                               decoration_id=decoration["configuration_id"], orbit_id=victim, status="FAILED",
                               failure={"code": "ATOM_OVERLAP", "detail": "injected"})
        return real(spec_, topology, decoration)

    monkeypatch.setattr(framework, "build_framework", failing)
    again = run_pipeline(spec, copy)
    assert again["stage_status"]["ranking"] == "INCOMPLETE"
    assert victim not in [r["orbit_id"] for r in again["base_ranking"]]
    detail = again["stage_detail"]["ranking"]
    assert detail["failed_evaluations"] == [{"orbit_id": victim, "status": "BUILD_FAILED"}]


# -- T03.2 grouped validation and frozen holdout ------------------------------------
def test_grouped_cv_selects_fewest_parameters_with_documented_gauge(toy_run):
    spec, run_dir, report = toy_run
    sel = load_artifacts(run_dir, "ModelSelectionReport")[0]
    assert sel["status"] == "COMPLETED" and sel["selected_model"] == "M0_nearest_pairs"
    chosen = {c["model_id"]: c for c in sel["candidates"]}
    # site_orbit_0 + site_orbit_1 = number of substituted slots = constant at fixed composition: one is dropped.
    assert chosen["M1_points_and_pairs"]["dropped_dependent_features"] == ["site_orbit_1"]
    assert chosen["M1_points_and_pairs"]["n_parameters"] == 3 and chosen["M1_points_and_pairs"]["passes"] is True
    assert chosen["M0_nearest_pairs"]["cv_rmse_eV"] <= sel["tolerance_eV"]
    assert chosen["M0_nearest_pairs"]["cv_rmse_eV"] > 0  # the orientation-dependent mock term is real model error
    assert sel["validation"].startswith("grouped_kfold") and sel["final_holdout_used"] is False
    assert report["physical"]["holdout"]["toy_pcu"]["status"] == "COMPLETED"
    assert report["physical"]["labelled_ground_state_recovered"]["toy_pcu"] is True


def test_no_symmetry_orbit_crosses_the_holdout_boundary(toy_run):
    spec, run_dir, _ = toy_run
    split = load_artifacts(run_dir, "ValidationSplit")[0]["per_topology"]["toy_pcu"]
    snaps = {s["role"]: s for s in load_artifacts(run_dir, "DatasetSnapshot")}
    train = {e["orbit_id"] for e in snaps["training"]["entries"]}
    holdout = {e["orbit_id"] for e in snaps["final_holdout"]["entries"]}
    assert train == set(split["training"]) and holdout == set(split["holdout"]) and not train & holdout
    folds = cm.assign_folds(sorted(train), 4, 11)
    assert folds == cm.assign_folds(sorted(train, reverse=True), 4, 11)  # deterministic, order independent


def test_rank_deficiency_fail_policy_and_inadequate_models_are_explicit(toy_run):
    spec, run_dir, _ = toy_run
    train = next(s for s in load_artifacts(run_dir, "DatasetSnapshot") if s["role"] == "training")
    groups = {e["orbit_id"]: e["orbit_id"] for e in train["entries"]}
    topo = spec.topologies["toy_pcu"]
    physical = spec.physical
    try:
        strict = {**physical["protocols"]["model_selection"], "rank_deficiency": "fail"}
        object.__setattr__(spec, "physical", {**physical, "protocols": {**physical["protocols"], "model_selection": strict}})
        sel = cm.fit_and_select_grouped(spec, topo, train, groups)
        failures = {c["model_id"]: (c.get("failure") or {}).get("code") for c in sel["candidates"]}
        assert failures == {"M0_nearest_pairs": None, "M1_points_and_pairs": "RANK_DEFICIENT_FEATURES",
                            "M2_points_two_shell_pairs": "RANK_DEFICIENT_FEATURES"}
        tight = {**physical["tolerances"], "target_relative_energy_tolerance": 1e-9}
        object.__setattr__(spec, "physical", {**physical, "tolerances": tight})
        sel = cm.fit_and_select_grouped(spec, topo, train, groups)
        assert sel["status"] == "MODEL_INADEQUATE" and sel["selected_model"] is None
        assert sel["data_requirements"]["n_labels"] == len(train["entries"])
    finally:
        object.__setattr__(spec, "physical", physical)


def test_holdout_failure_is_reported_not_tuned_away(toy_run):
    spec, run_dir, _ = toy_run
    ham = load_artifacts(run_dir, "HamiltonianArtifact")[0]
    holdout = next(s for s in load_artifacts(run_dir, "DatasetSnapshot") if s["role"] == "final_holdout")
    wrong = {**ham, "coefficients_exact": [str(Fraction(c) * Fraction(11, 10)) for c in ham["coefficients_exact"]]}
    report = cm.evaluate_holdout(spec, spec.topologies["toy_pcu"], wrong, holdout)
    assert report["status"] == "HOLDOUT_VALIDATION_FAILED" and report["holdout_rmse_eV"] > report["tolerance_eV"]


def test_gauge_and_ridge_helpers():
    x = [[1, 2, 1, 1], [1, 0, 1, 3], [1, 1, 1, 2]]  # column 2 equals column 0
    kept, dropped = cm.independent_columns(x, ["constant", "a", "b", "c"])
    assert kept == [0, 1] and dropped == ["b", "c"]  # c = 3*constant - a on these rows
    y = [Fraction(3), Fraction(1), Fraction(2)]
    exact = cm.ridge_exact([[r[0], r[1]] for r in x], y, Fraction(0), 0)
    assert exact == [Fraction(1), Fraction(1)]
    shrunk = cm.ridge_exact([[r[0], r[1]] for r in x], y, Fraction(10), 0)
    assert abs(shrunk[1]) < abs(exact[1])
    assert cm.low_energy_recall([0, 1, 2, 3], [3, 2, 1, 0], 2, Fraction(0)) == 0.0
    assert cm.low_energy_recall([0, 1, 2, 3], [0, 1, 3, 2], 2, Fraction(0)) == 1.0


def test_incompatible_fidelities_never_pool(toy_run):
    from conftest import PROJECT
    spec, run_dir, _ = toy_run
    physical_train = next(s for s in load_artifacts(run_dir, "DatasetSnapshot") if s["role"] == "training")
    fixture = load_project(PROJECT)
    synthetic_train = datasets.ingest_labels(fixture, fixture.training_rows, "training", "t")
    with pytest.raises(PipelineError, match="INCOMPATIBLE_FIDELITY"):
        datasets.merge_snapshots(spec, physical_train, synthetic_train, "training")
    assert physical_train["comparison_group"] != synthetic_train["comparison_group"]
