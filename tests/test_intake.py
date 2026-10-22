"""Task 01 / shared-contract boundary checks: strict inputs and malformed-record rejection."""
import json
from copy import deepcopy

import pytest
from conftest import PROJECT, copy_fixture, harness_json, read_jsonl

from mofpipe import datasets
from mofpipe.contract import PipelineError
from mofpipe.intake import load_project, validate_project
from mofpipe.orchestrator import run_pipeline
from mofpipe.structures import validate_occupation


def codes(report):
    return {e["code"] for e in report["errors"]}


def test_fixture_project_validates():
    report = validate_project(load_project(PROJECT))
    assert report["status"] == "COMPLETED", report["errors"]
    assert report["n_checks"] > 60


@pytest.mark.parametrize("case", harness_json("harness/negative_inputs.json")["occupation_cases"], ids=lambda c: c["id"])
def test_malformed_occupations_rejected(case):
    with pytest.raises(PipelineError) as exc:
        validate_occupation(case["bits"], 12, 6)
    assert str(exc.value) == case["expected_error"]


@pytest.mark.parametrize("mutation", harness_json("harness/negative_inputs.json")["energy_mutations"], ids=lambda m: m["field"])
def test_incompatible_energy_records_rejected(mutation):
    sample = read_jsonl(PROJECT.parent / "data/core_pairwise_training.jsonl")[0]
    changed = deepcopy(sample)
    changed[mutation["field"]] = mutation["value"]
    with pytest.raises(PipelineError) as exc:
        datasets.check_energy_compatibility([sample, changed])
    assert str(exc.value) == mutation["expected_error"]


def test_empty_energy_set_rejected():
    with pytest.raises(PipelineError, match="EMPTY_ENERGY_SET"):
        datasets.check_energy_compatibility([])


@pytest.mark.parametrize("case", harness_json("harness/negative_inputs.json")["normalization_cases"], ids=lambda c: c["id"])
def test_normalization(case):
    assert float(datasets.normalize_cell_energy(case["total_energy_eV"], case["n_formula_units"])) == pytest.approx(
        case["expected_normalized_eV"], abs=1e-12)


@pytest.mark.parametrize("units", [0, -1, 1.5, True])
def test_invalid_formula_unit_count(units):
    with pytest.raises(PipelineError, match="INVALID_FORMULA_UNIT_COUNT"):
        datasets.normalize_cell_energy(-0.1, units)


@pytest.mark.parametrize("field,value,code", [
    ("orbit_id", "ring12:000000111111", "ORBIT_ID_MISMATCH"),
    ("configuration_id", "ring12:000000000000", "CONFIGURATION_ID_MISMATCH"),
    ("topology_id", "pcu", "UNKNOWN_TOPOLOGY"),
    ("n_formula_units", 2, "FORMULA_UNIT_MISMATCH"),
])
def test_ingest_rejects_corrupted_labels(field, value, code):
    spec = load_project(PROJECT)
    rows = deepcopy(spec.training_rows)
    rows[0][field] = value
    with pytest.raises(PipelineError) as exc:
        datasets.ingest_labels(spec, rows, "training", "test")
    assert str(exc.value) == code


def test_duplicate_label_rejected():
    spec = load_project(PROJECT)
    with pytest.raises(PipelineError, match="DUPLICATE_LABEL"):
        datasets.ingest_labels(spec, spec.training_rows + spec.training_rows[:1], "training", "test")


@pytest.mark.parametrize("rel", ["harness/core_pairwise_holdout.jsonl", "reference_outputs/mock_labels.jsonl", "../README.md"])
def test_learner_cannot_read_harness_or_outside_files(rel):
    with pytest.raises(PipelineError, match="FORBIDDEN_LEARNER_INPUT"):
        load_project(PROJECT, overrides={"inputs": {"training": rel}})


def test_runtime_agents_enabled_blocks_run(tmp_path):
    spec = load_project(PROJECT, overrides={"runtime_agents": {"enabled": True}})
    assert "AGENT_LAYER_NOT_IMPLEMENTED" in codes(validate_project(spec))
    report = run_pipeline(spec, tmp_path / "run")
    assert report["stage_status"]["intake"] == "BLOCKED"
    assert report["stage_status"]["model_selection"] == "NOT_RUN"
    assert report["pipeline_status"] == "FAILED_OR_INCOMPLETE"


@pytest.mark.parametrize("overrides,code", [
    ({"qubo": {"ratio_penalty_eV": 0}}, "SCIENTIFIC_SPEC_MISSING"),
    ({"model_selection": {"holdout_access_during_selection": True}}, "HOLDOUT_ACCESS_FORBIDDEN"),
    ({"deduplication": {"allow_species_label_exchange": True}}, "UNSUPPORTED_SYMMETRY_POLICY"),
    ({"composition": {"counts": {"A": 7, "B": 6}}}, "INVALID_COMPOSITION_SPEC"),
    ({"topology_screen": {"retain": ["ring12"]}}, "SCREENING_POLICY_EXCLUDES_WITHOUT_EVIDENCE"),
    ({"evaluation": {"dft_enabled": True}}, "UNSUPPORTED_ELECTRONIC_BACKEND"),
    ({"optimizer": {"quantum_enabled": True}}, "SCIENTIFIC_SPEC_MISSING"),
    ({"active_learning": {"max_iterations_if_enabled": 10}}, "UNBOUNDED_ACTIVE_LEARNING"),
    ({"finite_temperature": {"enabled": True}}, "UNSUPPORTED_SCOPE_FINITE_TEMPERATURE"),
])
def test_policy_violations_block(overrides, code):
    assert code in codes(validate_project(load_project(PROJECT, overrides=overrides)))


def _edit_topologies(tmp_path, mutate):
    root = copy_fixture(tmp_path / "fx")
    path = root / "data/topologies.json"
    topologies = json.loads(path.read_text())
    mutate(topologies)
    path.write_text(json.dumps(topologies))
    return validate_project(load_project(root / "project.json"))


def test_symmetry_operation_that_breaks_graph_rejected(tmp_path):
    def mutate(t):
        p = t[0]["symmetry_pullback_permutations"][1]
        p[0], p[5] = p[5], p[0]
    found = codes(_edit_topologies(tmp_path, mutate))
    assert {"SYMMETRY_CHANGES_GRAPH", "SYMMETRY_GROUP_NOT_CLOSED"} & found


def test_graph_relabelled_as_physical_net_rejected(tmp_path):
    def mutate(t):
        t[0]["crystallographic_net_name"] = "pcu"
    assert "GRAPH_FIXTURE_CLAIMS_PHYSICAL_NET" in codes(_edit_topologies(tmp_path, mutate))


def test_cif_input_mode_unsupported(tmp_path):
    def mutate(t):
        t[1]["representation"] = "cif"
    assert "UNSUPPORTED_INPUT_MODE" in codes(_edit_topologies(tmp_path, mutate))


def test_holdout_label_smuggled_into_training_halts_run(tmp_path):
    root = copy_fixture(tmp_path / "fx")
    split = json.loads((root / "data/split_manifest.json").read_text())
    holdout_orbit = split["ring12"]["holdout"][0]
    rows = read_jsonl(root / "data/core_pairwise_training.jsonl")
    cand = next(r for r in read_jsonl(root / "data/candidates_unlabelled.jsonl") if r["orbit_id"] == holdout_orbit)
    rows.append({**rows[0], "bits": cand["bits"], "configuration_id": cand["configuration_id"], "orbit_id": holdout_orbit})
    (root / "data/core_pairwise_training.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = run_pipeline(load_project(root / "project.json"), tmp_path / "run")
    assert report["stage_status"]["labels"] == "FAILED"
    state = json.loads((tmp_path / "run/state.json").read_text())
    assert state["stages"]["labels"]["detail"]["error"]["code"] == "HOLDOUT_LABEL_IN_LEARNER_DATA"
    assert report["stage_status"]["model_selection"] == "NOT_RUN"
