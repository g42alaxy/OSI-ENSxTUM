"""Tasks 01/03: symmetry quotient, grouped splits, feature invariance and deterministic model selection."""
import json
from fractions import Fraction

import pytest
from conftest import FIXTURE, PROJECT, copy_fixture, harness_json, read_jsonl

from qumof import configuration_model as cm
from qumof import datasets, structures
from qumof.intake import load_project
from qumof.orchestrator import run_pipeline

EXPECTED = harness_json("harness/expected_results.json")


@pytest.fixture(scope="module")
def spec():
    return load_project(PROJECT)


@pytest.mark.parametrize("tid", ["ring12", "prism6"])
def test_decoration_space_and_orbit_quotient(spec, tid):
    space = structures.define_decoration_space(spec, spec.topologies[tid])
    exp = EXPECTED["core_pairwise"]["topologies"][tid]
    assert space["n_feasible"] == exp["n_feasible"] == 924
    assert space["n_symmetry_orbits"] == exp["n_symmetry_orbits"]
    assert sum(space["orbit_sizes"].values()) == 924


@pytest.mark.parametrize("tid", ["ring12", "prism6"])
def test_split_partitions_orbits_without_leakage(spec, tid):
    spaces = {t: structures.define_decoration_space(spec, spec.topologies[t]) for t in spec.topologies}
    split = datasets.freeze_split(spec, spaces)
    assert split["counts"][tid] == EXPECTED["core_pairwise"]["topologies"][tid]["split_orbit_counts"]
    g = split["per_topology"][tid]
    assert not (set(g["training"]) & set(g["development"])) and not (set(g["holdout"]) & set(g["training"] + g["development"]))


def test_split_with_leak_rejected(spec):
    spaces = {t: structures.define_decoration_space(spec, spec.topologies[t]) for t in spec.topologies}
    leaky = json.loads(json.dumps(spec.split_manifest))
    leaky["ring12"]["development"].append(leaky["ring12"]["training"][0])
    object.__setattr__(spec, "split_manifest", leaky)
    try:
        with pytest.raises(Exception, match="ORBIT_LEAKAGE"):
            datasets.freeze_split(spec, spaces)
    finally:
        object.__setattr__(spec, "split_manifest", load_project(PROJECT).split_manifest)


@pytest.mark.parametrize("tid", ["ring12", "prism6"])
def test_feature_symmetry_and_swap_reachability(spec, tid):
    topo = spec.topologies[tid]
    states = list(structures.feasible_states(12, 6))
    assert cm.check_feature_symmetry(topo, spec.feature_order, states)["invariant"]
    assert structures.swap_graph_connected(topo, 12, 6)["connected"]


@pytest.mark.parametrize("tid", ["ring12", "prism6"])
def test_feature_polynomials_match_counts(spec, tid):
    topo = spec.topologies[tid]
    for name in spec.feature_order:
        poly = cm.feature_polynomial(name, topo)
        for bits in list(structures.feasible_states(12, 6))[::37]:
            value = sum(c for k, c in poly.items() if all(bits[i] for i in k))
            assert value == cm.feature_value(name, topo, bits)


@pytest.mark.parametrize("case", ["core_pairwise", "higher_order"])
def test_model_selection_and_untouched_holdout(case):
    spec = load_project(PROJECT, case=case)
    train = datasets.ingest_labels(spec, spec.training_rows, "training", "t")
    dev = datasets.ingest_labels(spec, spec.development_rows, "development", "d")
    holdout = read_jsonl(FIXTURE / f"harness/{case}_holdout.jsonl")  # harness side, after selection
    for tid, topo in spec.topologies.items():
        sel = cm.fit_and_select(spec, topo, train, dev)
        assert sel["selected_model"] == EXPECTED[case]["expected_selected_model"]
        ham = cm.hamiltonian_from_selection(spec, topo, sel)
        errors = [cm.predict(ham, topo, r["bits"]) - Fraction(repr(r["energy_eV"])) for r in holdout if r["topology_id"] == tid]
        assert errors and cm.rmse(errors) <= 1e-10
        # The constant (graph offset) is retained in the Hamiltonian.
        assert ham["reference_offset_eV"] == pytest.approx({"ring12": 0.12, "prism6": 0.2}[tid], abs=1e-12)


def test_rank_deficient_candidates_recorded_as_failed(spec):
    rows = [r for r in spec.training_rows if r["topology_id"] == "ring12"][:2]
    train = datasets.ingest_labels(spec, rows, "training", "t")
    dev = datasets.ingest_labels(spec, spec.development_rows, "development", "d")
    sel = cm.fit_and_select(spec, spec.topologies["ring12"], train, dev)
    failures = {c["model_id"]: c.get("failure", {}).get("code") for c in sel["candidates"]}
    assert failures["M1_two_shell_pairs"] == failures["M2_pairs_and_triplets"] == "RANK_DEFICIENT_FEATURES"
    assert sel["status"] == "MODEL_INADEQUATE"


def _with_models(tmp_path, keep):
    root = copy_fixture(tmp_path / "fx")
    path = root / "data/model_candidates.json"
    models = json.loads(path.read_text())
    models["candidates"] = [m for m in models["candidates"] if m["id"] in keep]
    path.write_text(json.dumps(models))
    return root / "project.json"


def test_selection_is_policy_driven_not_hardcoded(tmp_path):
    spec = load_project(_with_models(tmp_path, {"M0_nearest_pairs", "M2_pairs_and_triplets"}))
    report = run_pipeline(spec, tmp_path / "run")
    assert {s["model_selection"]["selected_model"] for s in report["topologies"].values()} == {"M2_pairs_and_triplets"}
    assert report["winning_topology"] == "prism6"


def test_inadequate_family_halts_with_model_inadequate(tmp_path):
    spec = load_project(_with_models(tmp_path, {"M0_nearest_pairs", "M1_two_shell_pairs"}), case="higher_order")
    report = run_pipeline(spec, tmp_path / "run")
    assert report["stage_status"]["model_selection"] == "FAILED"
    assert report["stage_status"]["encoding"] == "NOT_RUN"
    assert report["winning_topology"] is None
    for sec in report["topologies"].values():
        assert sec["model_selection"]["status"] == "MODEL_INADEQUATE"
