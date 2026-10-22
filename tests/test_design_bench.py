"""A2 design mode on the Kang et al. (2025) inputs: scripted designs, no network."""
import pytest

pytest.importorskip("jsonschema")

from conftest import REPO  # noqa: E402

from mofpipe.agents.client import ReplayClient, check_schema  # noqa: E402
from mofpipe.agents.controls import load_controls  # noqa: E402
from mofpipe.agents.design import build_packet, load_materials, run_benchmark, validate_design  # noqa: E402

MATERIALS = {m["id"]: m for m in load_materials(REPO / "benchmarks/kang2025")}


def design(*terms):
    return {"status": "OK", "missing_input_ids": [], "error_codes": [], "terms": [
        {"term_id": t, "edge_kinds": kinds, "weighting": "distance", "alpha": 1.0, "relative_weight": 1.0,
         "preference": pref} for t, kinds, pref in terms]}


MIX = design(("unlike_pair_mixing", ["topological"], "penalize_like_pairs"))
SEGREGATE = design(("unlike_pair_mixing", ["topological"], "penalize_unlike_pairs"))


def test_model_designs_are_compiled_solved_and_scored_against_the_experimental_arrangement(tmp_path):
    hcb = [MATERIALS["cu_thq_hhtp"]]
    scripts = {"m/mix": [MIX], "m/segregate": [SEGREGATE]}
    rows = run_benchmark(hcb, lambda m: load_controls(None, {"budget": {"allow_extra_request": False}}), list(scripts), 1,
                         tmp_path, client_factory=lambda m: ReplayClient({"A2_design": scripts[m]}), workers=1)
    by_model = {r["model"]: r for r in rows}
    assert by_model["kang_control"]["assessment"]["outcome"] == "REPRODUCED"  # the paper's weighting, same compiler
    assert by_model["m/mix"]["assessment"]["outcome"] == "REPRODUCED"        # alternation from mixing on hcb bonds
    assert by_model["m/segregate"]["assessment"]["outcome"] == "MISSED"
    qaoa = by_model["m/mix"]["assessment"]["qaoa"]
    assert qaoa["median_p_reference"] > 3 * qaoa["uniform_baseline_p_reference"]
    assert "ground_state_bitstrings" not in str(build_packet(hcb[0]))  # the answer never reaches the model


def test_design_contract_rejects_invalid_or_constant_hamiltonians():
    packet = build_packet(MATERIALS["muf7"])
    codes = lambda d: {e["code"] for e in validate_design(d, packet)["errors"]}
    assert codes(design(("length_balance", ["topological"], "penalize_like_pairs"))) == {"PREFERENCE_MISMATCH"}
    assert codes(design(("length_balance", ["topological"], "not_applicable"))) == set()
    same_length = {**packet, "linkers": [{**l, "characteristic_length_A": 3.0} for l in packet["linkers"]]}
    balance = design(("length_balance", ["topological", "second_neighbor"], "not_applicable"))
    assert {e["code"] for e in validate_design(balance, same_length)["errors"]} == {"CONSTANT_HAMILTONIAN"}
    with pytest.raises(Exception, match="SCHEMA_VIOLATION"):
        check_schema("A2_design", {**MIX, "coefficients": [1.0]})  # numbers beyond the catalogue are not accepted


def test_reference_reproduces_the_published_spectrum_and_scores_the_paper_weighting_as_exact():
    from mofpipe.agents.design import compile_design, kang_spectrum, spectrum_agreement
    for material in MATERIALS.values():
        packet = build_packet(material)
        levels = sorted({round(v, 3) for v in kang_spectrum(material, packet).values()})
        assert levels == [round(x, 3) for x in material["reference"]["kang_hamiltonian"]["published_feasible_levels"]]
        control = spectrum_agreement(material, packet, compile_design(material["reference"]["kang_control"], packet))
        assert control["nmae"] < 0.002 and control["spearman"] > 0.999  # only the rounding of Table S3 weights differs
        segregate = spectrum_agreement(material, packet, compile_design(SEGREGATE, packet))
        assert segregate["scale"] == 0.0  # an anti-correlated design gets no credit from the affine map
