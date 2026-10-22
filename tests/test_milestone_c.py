"""Milestone C: second topology, transferable cluster features, supercell transfer checks, acquisition and
refinement, relaxed-geometry deduplication, reference-phase competition, structural validation and the report.

Runs on the SYNTHETIC toy (tests/toy_framework.py, write_toy_milestone_c_project) with the mock pw.x: two
candidate structures with the same formula unit (toy_pcu and its staggered polymorph) and a 2x2x2 transfer cell.
"""
import json
import shutil
from fractions import Fraction

import pytest
from conftest import load_artifacts
from toy_framework import write_toy_milestone_c_project

from qumof import acquisition, clusters, configuration_model as cm, geometry_match, references, refinement, transfer
from qumof import structures, validation
from qumof.contract import PipelineError, StopReason
from qumof.intake import load_project, validate_project
from qumof.orchestrator import run_pipeline


def _state(run_dir):
    return json.loads((run_dir / "state.json").read_text())


@pytest.fixture(scope="session")
def c_run(tmp_path_factory):
    spec = load_project(write_toy_milestone_c_project(tmp_path_factory.mktemp("toy_c")))
    run_dir = tmp_path_factory.mktemp("toy_c_run")
    return spec, run_dir, run_pipeline(spec, run_dir)


# -- end to end ---------------------------------------------------------------------------------------------
def test_two_topologies_rank_on_one_common_reference(c_run):
    spec, run_dir, report = c_run
    assert report["pipeline_status"] == "COMPLETED", report["stage_status"]
    ranked = report["base_ranking"]
    assert {r["topology_id"] for r in ranked} == {"toy_pcu", "toy_pcu_staggered"}
    assert len({r["base_fidelity"] for r in ranked}) == 1  # one DFT protocol, one formula unit
    for row in ranked:
        calc = json.loads((run_dir / "artifacts" / (row["calculation_result_id"].replace(":", "__") + ".json")).read_text())
        assert calc["base_energy_exact"] == row["ranking_energy_exact"] and calc["normalization"] == "per_formula_unit"
        assert calc["comparison_group"].startswith("Zn2(L)3(LN)3|")
    convergence = report["stage_detail"]["labels"]["dft_convergence"]
    assert set(convergence) == {"toy_pcu", "toy_pcu_staggered", "toy_pcu|toy_pcu_staggered"}  # cross-topology difference
    assert all(c["status"] == "COMPLETED" for c in convergence.values())


def test_report_carries_every_milestone_c_section(c_run):
    spec, run_dir, report = c_run
    ph = report["physical"]
    assert ph["refinement"]["stop_reason"] == "CONVERGED_WITHIN_SCOPE"
    assert ph["supercell_transfer"]["toy_pcu@2x2x2"]["status"] == "COMPLETED"
    assert ph["competition"]["status"] == "INCOMPLETE" and ph["competition"]["mixing_energies"]  # coverage not declared complete
    assert all(f["status"] == "COMPLETED" for f in ph["structure_validation"])
    assert ph["reproducibility_manifest"]["dft_fidelity"] == report["base_ranking"][0]["base_fidelity"]
    assert set(ph["search_coverage"]) == {"toy_pcu", "toy_pcu_staggered"}
    text = (run_dir / "report.md").read_text()
    for heading in ("## Refinement (active learning)", "## Supercell transfer checks", "## Reference-phase competition",
                    "## Structural validation of finalists"):
        assert heading in text


def test_rerun_and_interrupted_refinement_reuse_every_dft_job(c_run, tmp_path):
    spec, run_dir, report = c_run
    copy = tmp_path / "copy"
    shutil.copytree(run_dir, copy)
    state = _state(copy)
    before = state["counters"]["dft_submissions"]
    for stage in ("refinement", "model_selection", "supercell_checks", "encoding", "optimization", "candidates",
                  "evaluation", "ranking", "validation", "report"):
        state["stages"].pop(stage)
    (copy / "state.json").write_text(json.dumps(state))
    again = run_pipeline(spec, copy)
    assert _state(copy)["counters"]["dft_submissions"] == before
    assert again["base_ranking"] == report["base_ranking"]


# -- T03.1 cluster orbits, T01.5 supercells -------------------------------------------------------------------
def test_cluster_orbits_keep_inequivalent_pairs_separate_and_are_symmetric(c_run):
    spec, _, _ = c_run
    topo = spec.topologies["toy_pcu"]
    pairs = [(name, d) for name, kind, d, _ in topo.clusters if kind == "pair"]
    # collinear (through a node) and side-by-side parallel linkers are both 8.1 A apart but inequivalent
    assert [d for _, d in pairs] == [5.728, 8.1, 8.1]
    names = cm.all_model_features(spec, topo)
    states = list(structures.states_of(spec, topo))
    assert cm.check_feature_symmetry(topo, names, states)["invariant"]
    for name in names[1:]:  # the polynomial form used for encoding equals the direct count
        poly = cm.feature_polynomial(name, topo)
        for bits in states[:50]:
            assert sum(c for k, c in poly.items() if all(bits[i] for i in k)) == cm.feature_value(name, topo, bits)


def test_replicated_configuration_has_identical_features_and_orbit(c_run):
    spec, _, report = c_run
    topo, cell = spec.topologies["toy_pcu"], spec.physical["transfer_topologies"]["toy_pcu"][0]
    assert cell.topology_id == "toy_pcu@2x2x2" and cell.n_sites == 24 and cell.n_formula_units == 4
    replication = report["stage_detail"]["structures"]["supercell_replication"]["toy_pcu"][0]
    assert replication["status"] == "COMPLETED" and replication["n_states"] == 56
    mapping = transfer.slot_map(spec.physical["lattices"]["toy_pcu"], spec.physical["lattices"][cell.topology_id])
    names = cm.all_model_features(spec, topo)
    for bits in list(structures.states_of(spec, topo))[:40]:
        assert cm.feature_vector(topo, bits, names) == cm.feature_vector(cell, transfer.replicate(bits, mapping), names)


def test_transfer_check_reports_failure_of_a_wrong_model(c_run):
    spec, run_dir, _ = c_run
    good = load_artifacts(run_dir, "SupercellTransferReport")[0]
    assert good["status"] == "COMPLETED" and good["transfer_rmse_eV"] <= good["tolerance_eV"]
    assert all(r["within_tolerance"] for r in good["replication"]) and len(good["native"]) == 2
    ham = next(h for h in load_artifacts(run_dir, "HamiltonianArtifact") if h["topology_id"] == "toy_pcu")
    wrong = {**ham, "coefficients_exact": [ham["coefficients_exact"][0]] + [str(Fraction(c) * 50) for c in ham["coefficients_exact"][1:]]}
    native = [json.loads((run_dir / "artifacts" / (r["calculation_result_id"].replace(":", "__") + ".json")).read_text())
              for r in good["native"]]
    cell = spec.physical["transfer_topologies"]["toy_pcu"][0]
    bad = transfer.transfer_report(spec, spec.topologies["toy_pcu"], cell, wrong, [], native)
    assert bad["status"] == "TRANSFER_VALIDATION_FAILED" and bad["transfer_rmse_eV"] > bad["tolerance_eV"]


def test_native_transfer_states_are_not_replicas(c_run):
    spec, _, _ = c_run
    cell = spec.physical["transfer_topologies"]["toy_pcu"][0]
    mapping = transfer.slot_map(spec.physical["lattices"]["toy_pcu"], spec.physical["lattices"][cell.topology_id])
    states = transfer.native_states(spec, cell, mapping, 3, seed=3)
    assert len(states) == 3 and not any(transfer.is_replica(b, mapping) for b in states)
    assert states == transfer.native_states(spec, cell, mapping, 3, seed=3)  # seeded


# -- T05.2 acquisition, T08 refinement ------------------------------------------------------------------------
def test_refinement_acquires_only_unlabelled_orbits_with_reasons(c_run):
    spec, run_dir, report = c_run
    split = load_artifacts(run_dir, "ValidationSplit")[0]["per_topology"]
    batches = load_artifacts(run_dir, "AcquisitionBatch")
    assert batches and all(b["holdout_excluded"] for b in batches)
    for batch in batches:
        for pick in batch["selected"]:
            groups = split[pick["topology_id"]]
            assert pick["orbit_id"] in groups["unlabelled"] and pick["orbit_id"] not in groups["holdout"]
            assert pick["reason"] in ("policy_score", "audit_deprioritised_region") and pick["model_id"]
    reasons = [p["reason"] for b in batches for p in b["selected"]]
    assert "audit_deprioritised_region" in reasons
    rounds = report["physical"]["refinement"]["rounds"]
    assert rounds[-1]["stable_rounds"] >= 1 and all(d == "PASS" for d in rounds[-1]["diagnosis"].values())


def test_acquisition_is_seeded_and_favours_the_undersampled_uncertain_topology(c_run):
    spec, run_dir, _ = c_run
    split = load_artifacts(run_dir, "ValidationSplit")[0]
    train = next(s for s in load_artifacts(run_dir, "DatasetSnapshot") if s["role"] == "training" and s["source"] == "periodic_dft")
    groups = {e["orbit_id"]: e["orbit_id"] for e in train["entries"]}
    selections = {tid: cm.fit_and_select_grouped(spec, topo, train, groups) for tid, topo in spec.topologies.items()}
    one = acquisition.select_evaluation_batch(spec, split, selections, train, {}, 0)
    two = acquisition.select_evaluation_batch(spec, split, selections, train, {}, 0)
    assert one["selected"] == two["selected"]
    # Strip toy_pcu_staggered down to three labels and fake a failing model: it must get the larger share.
    keep = [e for e in train["entries"] if e["topology_id"] == "toy_pcu"] + \
           [e for e in train["entries"] if e["topology_id"] == "toy_pcu_staggered"][:3]
    sparse = {**train, "entries": keep}
    failing = {**selections["toy_pcu_staggered"], "status": "MODEL_INADEQUATE", "selected_model": None}
    batch = acquisition.select_evaluation_batch(spec, split, {**selections, "toy_pcu_staggered": failing}, sparse, {}, 1)
    assert batch["quota"]["toy_pcu_staggered"] > batch["quota"]["toy_pcu"] >= 1


def test_stopping_rules_never_report_budget_or_starvation_as_convergence():
    sel_pass = {"status": "COMPLETED", "candidates": [], "n_training": 10}
    sel_need = {"status": "MODEL_INADEQUATE", "n_training": 2,
                "candidates": [{"status": "FAILED", "n_parameters": 3}, {"status": "COMPLETED", "n_parameters": 2}]}
    lead = {"t": ["t:01"], "pooled": ["t:01"]}
    stop, stable, _ = refinement.assess_refinement(0, {"t": sel_pass}, lead, None, 0, 2, 3)
    assert stop is None and stable == 0
    stop, stable, _ = refinement.assess_refinement(1, {"t": sel_pass}, lead, lead, stable, 2, 3)
    assert stop is None and stable == 1
    stop, stable, _ = refinement.assess_refinement(2, {"t": sel_pass}, lead, lead, stable, 2, 3)
    assert stop == StopReason.CONVERGED_WITHIN_SCOPE
    stop, _, _ = refinement.assess_refinement(3, {"t": sel_pass}, {"t": ["t:10"], "pooled": ["t:10"]}, lead, 1, 2, 3)
    assert stop == StopReason.BUDGET_EXHAUSTED  # the leading set moved in the last allowed round
    stop, _, diag = refinement.assess_refinement(1, {"t": sel_need}, None, None, 0, 2, 3, acquirable={"t": 0})
    assert stop == StopReason.INCONCLUSIVE and diag["t"] == "NEEDS_LABELS"
    inadequate = {**sel_need, "n_training": 30, "candidates": [{"status": "COMPLETED", "n_parameters": 3}]}
    stop, _, _ = refinement.assess_refinement(1, {"t": inadequate}, None, None, 0, 2, 3, acquirable={"t": 5})
    assert stop == StopReason.MODEL_INADEQUATE


def test_round_cap_before_stability_is_budget_exhausted_not_converged(tmp_path):
    spec = load_project(write_toy_milestone_c_project(tmp_path / "toy", {"protocols": {"stopping": {"max_rounds": 1}},
                                                                        "tolerances": {"consecutive_stable_rounds": 3}}))
    report = run_pipeline(spec, tmp_path / "run")
    assert report["stage_detail"]["refinement"]["stop_reason"] == "BUDGET_EXHAUSTED"
    assert report["stage_status"]["refinement"] == "INCONCLUSIVE" and report["pipeline_status"] != "COMPLETED"


# -- T05.1 relaxed-geometry equivalence ------------------------------------------------------------------------
def test_relaxed_geometry_equivalence_uses_structure_not_energy(c_run):
    spec, run_dir, _ = c_run
    calcs = [c for c in load_artifacts(run_dir, "CalculationResult")
             if c["status"] == "SUCCEEDED" and c["topology_id"] == "toy_pcu" and c["calculation"] == "scf"]
    a, b = calcs[0]["relaxed_geometry"], next(c for c in calcs if c["orbit_id"] != calcs[0]["orbit_id"])["relaxed_geometry"]
    lattice = spec.physical["lattices"]["toy_pcu"]
    diag = lattice["supercell_diagonal"]
    ops = geometry_match.supercell_operations(lattice["symmetry_operations"], diag)
    # A symmetry image (operation + translation, atoms reordered, 0.01 A noise) of the same structure is equivalent.
    op = ops[len(ops) // 3]
    moved = [list(geometry_match._apply(op, f, diag)) for f in a["frac_coords"]]
    image = {"lattice_A": a["lattice_A"], "elements": a["elements"][::-1],
             "frac_coords": [[x + 0.001 for x in f] for f in moved[::-1]]}
    assert geometry_match.equivalent(a, image, ops, diag, 0.1)
    assert not geometry_match.equivalent(a, b, ops, diag, 0.1)  # a different decoration stays separate
    classes = geometry_match.equivalence_classes([("x", a, "closed_shell"), ("y", image, "closed_shell"),
                                                  ("z", image, "high_spin")], ops, diag, 0.1)
    assert classes == {"x": "x", "y": "x", "z": "z"}  # another magnetic state is never merged


# -- T09.2 references ---------------------------------------------------------------------------------------
def test_mixture_balance_is_exact_and_reports_unreachable_compositions():
    refs = [{"reference_id": "A", "inventory": {"Zn": 4, "C": 24, "H": 96, "O": 24}, "energy_exact": "-100"},
            {"reference_id": "B", "inventory": {"Zn": 4, "C": 24, "H": 108, "N": 12, "O": 24}, "energy_exact": "-120"},
            {"reference_id": "B2", "inventory": {"Zn": 4, "C": 24, "H": 108, "N": 12, "O": 24}, "energy_exact": "-121"}]
    target = {"C": 12, "H": 51, "N": 3, "O": 12, "Zn": 2}
    best = references.mixture_balance(target, refs)
    assert best["energy_exact"] == Fraction(-100, 4) + Fraction(-121, 4)  # the lower B polymorph is used
    assert {c["reference_id"]: c["cells_per_formula_unit"] for c in best["coefficients"]} == {"A": "1/4", "B2": "1/4"}
    assert references.mixture_balance({**target, "F": 1}, refs) is None


def test_reference_competition_never_certifies_with_missing_references(c_run):
    spec, run_dir, _ = c_run
    comp = load_artifacts(run_dir, "CompetitionReport")[0]
    assert len(comp["reference_set"]) == 4 and comp["status"] == "INCOMPLETE"
    ranking = load_artifacts(run_dir, "RankingReport")[0]
    complete = references.evaluate_reference_mixtures(spec, ranking, comp["reference_set"], declared_complete=True)
    assert complete["status"] == "COMPLETED" and complete["mixing_energies"] == comp["mixing_energies"]
    missing = references.evaluate_reference_mixtures(spec, ranking, comp["reference_set"], declared_complete=True,
                                                     missing=[{"reference_id": "x"}])
    assert missing["status"] == "INCOMPLETE"


# -- T09.1 validation ---------------------------------------------------------------------------------------
def test_structure_validation_reports_tools_and_catches_overlaps(c_run, monkeypatch):
    spec, run_dir, _ = c_run
    report = load_artifacts(run_dir, "StructureValidationReport")[0]
    assert report["checks"]["mofchecker"]["status"] in ("SKIPPED_TOOL_UNAVAILABLE", "COMPLETED", "FAILED")
    assert report["checks"]["phonon_elastic_stability"]["status"] == "NOT_RUN"
    monkeypatch.setattr(validation.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(validation.shutil, "which", lambda name: None)
    calc = json.loads((run_dir / "artifacts" / (report["calculation_result_id"].replace(":", "__") + ".json")).read_text())
    geometry = next(g for g in load_artifacts(run_dir, "Geometry") if g["decoration_id"] == calc["configuration_id"])
    clean = validation.validate_structure(spec, calc, geometry)
    assert clean["checks"]["mofchecker"]["status"] == clean["checks"]["zeopp_pores"]["status"] == "SKIPPED_TOOL_UNAVAILABLE"
    relaxed = calc["relaxed_geometry"]
    squashed = {**relaxed, "frac_coords": [relaxed["frac_coords"][0]] + relaxed["frac_coords"][:-1]}
    bad = validation.validate_structure(spec, {**calc, "relaxed_geometry": squashed}, geometry)
    assert "min_interatomic_distance" in bad["failed_checks"] and bad["status"] == "FAILED"


# -- intake ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("patch,code,path", [
    ({"protocols": {"validation": {"supercell_transfer": None}}}, "SCIENTIFIC_SPEC_MISSING",
     "protocols.validation.supercell_transfer.n_replicated"),
    ({"protocols": {"model_selection": {"candidates": [{"id": "X", "features": ["constant", "unlike_distance_1"]}]}}},
     "NONTRANSFERABLE_FEATURES", "protocols.model_selection.candidates.X"),
    ({"tolerances": {"consecutive_stable_rounds": None}}, "SCIENTIFIC_SPEC_MISSING", "tolerances.consecutive_stable_rounds"),
    ({"protocols": {"acquisition": {"weights": {"energy": -1.0}}}}, "SCIENTIFIC_SPEC_MISSING", "protocols.acquisition.weights"),
    ({"candidate_topology_ids": ["toy_pcu"]}, "TOPOLOGY_SOURCE_MISMATCH", "candidate_topology_ids"),
    ({"protocols": {"validation": {"reference_phases": "materials_project"}}}, "UNSUPPORTED_REFERENCE_SET",
     "protocols.validation.reference_phases"),
])
def test_milestone_c_spec_errors_block_at_intake(tmp_path, patch, code, path):
    spec = load_project(write_toy_milestone_c_project(tmp_path / "toy", patch))
    errors = validate_project(spec)["errors"]
    assert {"code": code, "check": path} in [{"code": e["code"], "check": e["check"]} for e in errors], errors


def test_topologies_keep_their_own_cell_counts(tmp_path):
    def smaller(spec):
        spec["source_structures"][1]["supercell_matrices"] = [[[2, 0, 0], [0, 1, 0], [0, 0, 1]]]
    spec = load_project(write_toy_milestone_c_project(tmp_path / "toy", mutate=smaller))
    a, b = spec.topologies["toy_pcu"], spec.topologies["toy_pcu_staggered"]
    assert (a.n_sites, a.n_ones, a.n_formula_units) == (12, 6, 2) and (b.n_sites, b.n_ones, b.n_formula_units) == (6, 3, 1)
    assert structures.define_decoration_space(spec, b)["n_feasible"] == 20
    with pytest.raises(PipelineError, match="INVALID_SITE_COUNT"):
        structures.make_decoration(spec, b, [0, 1] * 6)
    assert clusters.names_for_token(a, "pairs<=5.8") == clusters.names_for_token(b, "pairs<=5.8") == ["pair_orbit_0"]
