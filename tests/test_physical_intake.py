"""Milestone B, Task 01: ProjectSpec loader (T01.1), CIF / slot-lattice input (T01.2) and framework builder (T01.3).

Structures here are the SYNTHETIC toy from tests/toy_framework.py; they test plumbing, not chemistry.
"""
import json
import random
from collections import Counter

import pytest
from conftest import FIXTURE, PROJECT, REPO
from toy_framework import toy_cif, write_toy_project

from mofpipe import structures
from mofpipe.cif import parse_cif, write_cif
from mofpipe.contract import PipelineError
from mofpipe.framework import build_framework
from mofpipe.intake import load_project, validate_project
from mofpipe.orchestrator import run_pipeline
from mofpipe.project_spec import REQUIRED_FIELDS, parse_formula

TEMPLATE = REPO / "instructions/project_spec.template.json"
RECIPE = FIXTURE / "physical_case/recipe.json"


def codes(report):
    return {e["code"] for e in report["errors"]}


def toy(tmp_path, **kwargs):
    return load_project(write_toy_project(tmp_path / "toy", **kwargs))


# -- T01.1 ProjectSpec loader ---------------------------------------------------
def test_untouched_template_blocked_with_one_error_per_missing_field():
    spec = load_project(TEMPLATE)
    report = validate_project(spec)
    assert spec.project_type == "physical_structure" and report["status"] == "BLOCKED"
    paths = [e["check"] for e in report["errors"]]
    assert sorted(paths) == sorted(REQUIRED_FIELDS) and len(paths) == len(set(paths))
    assert codes(report) == {"SCIENTIFIC_SPEC_MISSING"}


def test_template_run_blocks_at_intake_and_still_reports(tmp_path):
    report = run_pipeline(load_project(TEMPLATE), tmp_path / "run")
    assert report["stage_status"]["intake"] == "BLOCKED"
    assert all(report["stage_status"][s] == "NOT_RUN" for s in ("structures", "labels", "evaluation"))
    assert (tmp_path / "run/report.md").exists() and report["pipeline_status"] == "FAILED_OR_INCOMPLETE"


def test_recipe_loads_as_separate_physical_project_with_exact_missing_inputs():
    spec = load_project(RECIPE)
    report = validate_project(spec)
    assert spec.project_type == "physical_structure" and spec.provenance == "PHYSICAL"
    assert spec.project_id == "mof5_bdc_amino_bdc_1to1" and report["status"] == "BLOCKED"
    paths = {e["check"] for e in report["errors"]}
    # The verified parent CIF and the 3D linker geometry are the recipe's declared missing inputs.
    assert {"source_structures", "building_blocks[1].substitution", "building_blocks[2].substitution",
            "protocols.dft", "chemistry.solvent_and_counterion_policy"} <= paths
    # What the recipe does state is carried over, not reported missing.
    assert "chemistry.composition_by_sublattice" not in paths and "chemistry.target_elemental_inventory" not in paths


def test_recipe_inventory_is_parsed_from_blocks():
    blocks = {"Zn4O": parse_formula("Zn4O"), "BDC": parse_formula("C8H4O4"), "NH2_BDC": parse_formula("C8H5NO4")}
    assert parse_formula("Zn8O2(BDC)3(NH2_BDC)3", blocks) == {"C": 48, "H": 27, "N": 3, "O": 26, "Zn": 8}
    with pytest.raises(PipelineError, match="INVALID_FORMULA"):
        parse_formula("Zn8O2(XYZ)3", blocks)


def test_toy_project_validates_and_keeps_physical_and_synthetic_apart(tmp_path):
    spec = toy(tmp_path)
    assert validate_project(spec)["status"] == "COMPLETED"
    run_pipeline(load_project(PROJECT), tmp_path / "shared")
    with pytest.raises(PipelineError, match="RUN_DIR_PROJECT_TYPE_MISMATCH"):
        run_pipeline(spec, tmp_path / "shared")


def test_physical_project_refuses_fixture_case_modes(tmp_path):
    path = write_toy_project(tmp_path / "toy")
    with pytest.raises(PipelineError, match="UNSUPPORTED_FOR_PHYSICAL_PROJECT"):
        load_project(path, case="core_pairwise")


@pytest.mark.parametrize("overrides,path,code", [
    ({"protocols": {"model_selection": {"holdout_fraction": None}}}, "protocols.model_selection.holdout_fraction",
     "SCIENTIFIC_SPEC_MISSING"),
    ({"protocols": {"dft": {"xc_functional": None}}}, "protocols.dft.xc_functional", "SCIENTIFIC_SPEC_MISSING"),
    ({"protocols": {"dft": {"numerical_retries": [{"mixing_beta": 0.7}, {"ecutwfc": 30}]}}},
     "protocols.dft.numerical_retries[1]", "PROTOCOL_RETRY_CHANGES_SCIENCE"),
    ({"protocols": {"dft": {"difference_convergence": {"ecutwfc_Ry_series": [40, 80]}}}},
     "protocols.dft.difference_convergence.ecutwfc_Ry_series", "SCIENTIFIC_SPEC_MISSING"),
    ({"search": {"orientation_catalog_id": "amino_rotamers_v1"}}, "search.orientation_catalog_id",
     "UNSUPPORTED_ORIENTATION_CATALOG"),
    ({"runtime": {"llm_provider": "anthropic"}}, "runtime.llm_provider", "AGENT_LAYER_NOT_IMPLEMENTED"),
    ({"chemistry": {"disorder_resolution_policy": "average_sites"}}, "chemistry.disorder_resolution_policy",
     "UNSUPPORTED_DISORDER_POLICY"),
])
def test_physical_policy_fields_block_with_field_path(tmp_path, overrides, path, code):
    errors = validate_project(toy(tmp_path, overrides=overrides))["errors"]
    assert {"code": code, "check": path} in [{"code": e["code"], "check": e["check"]} for e in errors]


def test_pseudopotential_hash_is_verified(tmp_path):
    path = write_toy_project(tmp_path / "toy")
    (path.parent / "pseudo/Zn.toy.UPF").write_text("tampered\n")
    assert "PSEUDOPOTENTIAL_HASH_MISMATCH" in codes(validate_project(load_project(path)))


def test_protocol_may_be_a_registry_file(tmp_path):
    def mutate(spec):
        (tmp_path / "toy/protocols").mkdir(parents=True, exist_ok=True)
        (tmp_path / "toy/protocols/builder.json").write_text(json.dumps(spec["protocols"]["builder"]))
        spec["protocols"]["builder"] = "protocols/builder.json"
    spec = toy(tmp_path, mutate=mutate)
    assert validate_project(spec)["status"] == "COMPLETED"
    assert spec.physical["protocols"]["builder"]["builder_id"] == "parent_substitution_v1"


# -- T01.2 CIF input and slot lattice -------------------------------------------
def test_cif_parse_expands_symmetry_and_keeps_provenance():
    text = toy_cif()
    parsed = parse_cif(text, "toy")
    assert parsed["n_symmetry_operations"] == 48
    assert Counter(a["element"] for a in parsed["atoms"]) == {"Zn": 1, "O": 6, "C": 6, "H": 24}
    assert parsed["original_text"] == text and len(parsed["sha256"]) == 64
    assert not parsed["partial_occupancy_sites"] and not parsed["disordered_sites"]
    # Standard uncertainties in brackets are accepted.
    assert parse_cif(text.replace("_cell_length_a 8.1", "_cell_length_a 8.1000(3)"), "t")["cell_lengths_A"][0] == 8.1


def test_known_parent_gives_checked_slot_lattice(tmp_path):
    spec = toy(tmp_path)
    topo = spec.topologies["toy_pcu"]
    source = spec.physical["sources"][0]
    assert (topo.n_sites, source["n_nodes"], topo.n_formula_units) == (12, 4, 2)
    assert spec.composition_counts == {"L": 6, "LN": 6} and spec.species_encoding == {"L": 0, "LN": 1}
    assert source["target_inventory_per_cell"] == {"C": 24, "H": 102, "N": 6, "O": 24, "Zn": 4}
    assert source["charge_check"]["total"] == 0
    # 2x2x1 cell: x/y linkers have 7 slot neighbours, z linkers 4; periodic contacts carry multiplicity.
    degree = Counter(i for p in topo.pairs_d1 for i in p)
    assert sorted(set(degree.values())) == [4, 7] and max(topo.pair_weights_d1) == 2
    assert len(topo.permutations) == 32 and tuple(range(12)) in topo.permutations
    assert validate_project(spec)["status"] == "COMPLETED"  # group closure and graph invariance included


def test_slot_mapping_stable_under_atom_reordering(tmp_path):
    reference = toy(tmp_path / "a")
    shuffled = toy(tmp_path / "b", cif_text=toy_cif(shuffle=True))
    # Same structure written in P1 with a random atom order.
    lattice = reference.physical["lattices"]["toy_pcu"]["structure"]
    parent = parse_cif(toy_cif(), "t")
    order = list(range(len(parent["atoms"])))
    random.Random(3).shuffle(order)
    p1 = write_cif("p1", parent["lattice"], [parent["atoms"][i]["element"] for i in order],
                   [parent["atoms"][i]["frac"] for i in order])
    reordered = toy(tmp_path / "c", cif_text=p1)
    for other in (shuffled, reordered):
        a, b = reference.topologies["toy_pcu"], other.topologies["toy_pcu"]
        assert (a.pairs_d1, a.pair_weights_d1, a.pairs_d2, a.triplets) == (b.pairs_d1, b.pair_weights_d1, b.pairs_d2, b.triplets)
        assert [s["centroid_frac"] for s in reference.physical["lattices"]["toy_pcu"]["slots"]] == \
            [s["centroid_frac"] for s in other.physical["lattices"]["toy_pcu"]["slots"]]
    # P1 input has no point symmetry: only the supercell translations remain.
    assert len(reordered.topologies["toy_pcu"].permutations) == 4 and len(lattice["atoms"]) == 148


@pytest.mark.parametrize("kwargs,code", [
    ({"cif_text": toy_cif(occupancy_h=0.5)}, "DISORDER_UNRESOLVED"),
    ({"overrides": {"chemistry": {"composition_by_sublattice": {"linker": {"L": 0.4, "LN": 0.6}}}}}, "NONINTEGRAL_COMPOSITION"),
    ({"overrides": {"chemistry": {"net_charge": 1}}}, "CHARGE_IMBALANCE"),
    ({"overrides": {"chemistry": {"target_elemental_inventory": {"C": 12, "H": 50, "N": 3, "O": 12, "Zn": 2}}}},
     "INVENTORY_MISMATCH"),
    ({"overrides": {"source_structures": [{"topology_id": "toy_pcu", "cif": "structures/toy_pcu.cif", "sha256": "0" * 64,
                                           "source": "x"}]}}, "CHECKSUM_MISMATCH"),
    ({"cif_text": toy_cif(extra_atoms=[("Ow", "O", 0.5, 0.5, 0.5, 1.0)])}, "UNRESOLVED_GUEST_SPECIES"),
])
def test_invalid_structural_inputs_caught_before_any_calculation(tmp_path, kwargs, code):
    spec = toy(tmp_path, **kwargs)
    assert code in codes(validate_project(spec))
    report = run_pipeline(spec, tmp_path / "run")
    assert report["stage_status"]["intake"] == "BLOCKED"
    assert not any((tmp_path / "run/jobs").iterdir())  # no DFT job was submitted


def test_invalid_connector_declaration_caught(tmp_path):
    def mutate(spec):
        spec["building_blocks"][1]["n_connector_atoms"] = 4
        spec["building_blocks"][0]["connectivity"] = 4
    found = codes(validate_project(toy(tmp_path, mutate=mutate)))
    assert {"INVALID_CONNECTOR_COUNT", "INVALID_CONNECTIVITY"} <= found


def test_unbound_guest_removed_only_under_declared_policy(tmp_path):
    spec = toy(tmp_path, cif_text=toy_cif(extra_atoms=[("Ow", "O", 0.5, 0.5, 0.5, 1.0)]),
               overrides={"chemistry": {"solvent_and_counterion_policy": "remove_unbound_guests"}})
    assert validate_project(spec)["status"] == "COMPLETED"
    assert spec.physical["sources"][0]["removed_guests"] == [{"O": 1}] * 4


# -- T01.3 framework construction -----------------------------------------------
def test_builder_is_reproducible_and_every_legal_decoration_keeps_counts(toy_run):
    spec, _, _ = toy_run
    topo = spec.topologies["toy_pcu"]
    target = spec.physical["sources"][0]["target_inventory_per_cell"]
    seen = set()
    for bits in structures.feasible_states(spec.n_sites, spec.n_B):
        orbit = structures.orbit_id(topo, bits)
        if orbit in seen:
            continue
        seen.add(orbit)
        geometry = build_framework(spec, topo, structures.make_decoration(spec, topo, bits))
        assert geometry["status"] == "COMPLETED", geometry.get("failure")
        assert geometry["inventory"] == target
    assert len(seen) == structures.define_decoration_space(spec, topo)["n_symmetry_orbits"]
    decoration = structures.make_decoration(spec, topo, next(structures.feasible_states(12, 6)))
    first, second = build_framework(spec, topo, decoration), build_framework(spec, topo, decoration)
    assert first["artifact_id"] == second["artifact_id"] and first["geometry_hash"] == second["geometry_hash"]
    # Exported CIF round-trips with the same inventory, and the atom->block->slot map covers every atom.
    assert Counter(a["element"] for a in parse_cif(first["cif"], "built")["atoms"]) == Counter(first["inventory"])
    assert len(first["atom_map"]) == len(first["elements"])
    assert Counter(m["block"] for m in first["atom_map"] if m["block"] == "LN") == {"LN": 6 * 14}  # C2H9NO2


def test_parent_rebuild_is_identical(toy_run):
    _, run_dir, report = toy_run
    construction = report["physical"]["atomic_construction"]["toy_pcu"]
    assert construction["parent_rebuild"] == "COMPLETED" and construction["parent_rebuild_identical"] is True


def test_construction_failure_is_reported_never_an_energy(tmp_path):
    def mutate(spec):  # substituent placed on top of its anchor atom
        spec["building_blocks"][2]["substitution"]["group"][0]["frame_A"] = [0.2, 0.0, 0.0]
    spec = toy(tmp_path, mutate=mutate)
    topo = spec.topologies["toy_pcu"]
    failure = build_framework(spec, topo, structures.make_decoration(spec, topo, next(structures.feasible_states(12, 6))))
    assert failure["record_type"] == "BuildFailure" and failure["failure"]["code"] == "ATOM_OVERLAP"
    report = run_pipeline(spec, tmp_path / "run")
    assert report["stage_status"]["structures"] == "FAILED" and report["stage_status"]["labels"] == "NOT_RUN"
    assert not any((tmp_path / "run/jobs").iterdir()) and report["base_ranking"] is None


def test_pormake_adapter_never_fabricates_geometry(tmp_path):
    spec = toy(tmp_path, overrides={"protocols": {"builder": {"builder_id": "pormake"}}})
    topo = spec.topologies["toy_pcu"]
    result = build_framework(spec, topo, structures.make_decoration(spec, topo, next(structures.feasible_states(12, 6))))
    assert result["record_type"] == "BuildFailure"
    assert result["failure"]["code"] in ("BUILDER_UNAVAILABLE", "BUILDER_NOT_VERIFIED")
    assert "frac_coords" not in result
