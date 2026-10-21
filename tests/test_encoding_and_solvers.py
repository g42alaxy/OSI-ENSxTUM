"""Tasks 03/04: QUBO/HUBO compilation identities and solver agreement."""
from fractions import Fraction
from itertools import product

import pytest
from conftest import PROJECT, fixture_core, harness_json

from mofpipe import configuration_model as cm
from mofpipe import datasets, encoding, optimization
from mofpipe.contract import PipelineError
from mofpipe.intake import load_project

EXPECTED = harness_json("harness/expected_results.json")


def _hamiltonians(case):
    spec = load_project(PROJECT, case=case)
    train = datasets.ingest_labels(spec, spec.training_rows, "training", "t")
    dev = datasets.ingest_labels(spec, spec.development_rows, "development", "d")
    out = {}
    for tid, topo in spec.topologies.items():
        out[tid] = cm.hamiltonian_from_selection(spec, topo, cm.fit_and_select(spec, topo, train, dev))
    return spec, out


@pytest.fixture(scope="module")
def pairwise():
    return _hamiltonians("core_pairwise")


@pytest.fixture(scope="module")
def cubic():
    return _hamiltonians("higher_order")


@pytest.mark.parametrize("tid", ["ring12", "prism6"])
def test_qubo_matches_fixture_reference_and_identity(pairwise, tid):
    spec, hams = pairwise
    topo, ham = spec.topologies[tid], hams[tid]
    enc = encoding.compile_encoding(spec, topo, ham, "QUBO")
    fc = fixture_core()
    topo_dict = next(t for t in fc.read_json(fc.ROOT / "data/topologies.json") if t["topology_id"] == tid)
    ref = fc.compile_pair_qubo(topo_dict, [Fraction(c) for c in ham["coefficients_exact"]])
    assert enc["constant_eV"] == pytest.approx(ref["constant_eV"], abs=1e-12)
    assert enc["linear_eV"] == pytest.approx(ref["linear_eV"], abs=1e-12)
    assert [(q["i"], q["j"]) for q in enc["quadratic_eV"]] == [(q["i"], q["j"]) for q in ref["quadratic_eV"]]
    assert [q["value"] for q in enc["quadratic_eV"]] == pytest.approx([q["value"] for q in ref["quadratic_eV"]], abs=1e-12)
    # Independent evaluator (fixture harness) on all 4096 strings, including infeasible ones.
    for bits in product((0, 1), repeat=12):
        direct = cm.predict(ham, topo, bits) + (sum(bits) - 6) ** 2
        assert fc.evaluate_qubo(enc, bits) == pytest.approx(float(direct), abs=1e-10)
    ver = encoding.verify_encoding(spec, topo, ham, enc)
    assert ver["status"] == "COMPLETED" and ver["all_penalized_argmins_feasible"]
    assert ver["penalized_minimum_eV"] == pytest.approx(EXPECTED["core_pairwise"]["topologies"][tid]["minimum_energy_eV"])


@pytest.mark.parametrize("tid", ["ring12", "prism6"])
def test_cubic_model_rejected_by_qubo_only_target_but_retained_as_hubo(cubic, tid):
    spec, hams = cubic
    topo, ham = spec.topologies[tid], hams[tid]
    with pytest.raises(PipelineError) as exc:
        encoding.compile_encoding(spec, topo, ham, "QUBO")
    assert str(exc.value) == "UNSUPPORTED_HIGHER_ORDER_WITHOUT_QUADRATIZATION"
    enc = encoding.compile_encoding(spec, topo, ham, "HUBO")
    assert enc["max_degree"] == 3
    assert len(enc["higher_order_terms"]) == EXPECTED["higher_order"]["topologies"][tid]["n_path_triplets"]
    assert encoding.verify_encoding(spec, topo, ham, enc)["status"] == "COMPLETED"


@pytest.mark.parametrize("tamper", ["double_count_quadratic", "drop_constant", "drop_triplet"])
def test_verifier_catches_tampered_encodings(pairwise, cubic, tamper):
    spec, hams = cubic if tamper == "drop_triplet" else pairwise
    topo, ham = spec.topologies["prism6"], hams["prism6"]
    enc = encoding.compile_encoding(spec, topo, ham, "HUBO" if tamper == "drop_triplet" else "QUBO")
    enc = dict(enc)
    if tamper == "double_count_quadratic":
        q = dict(enc["quadratic_eV"][0])
        q["exact"] = str(2 * Fraction(q["exact"]))
        enc["quadratic_eV"] = [q] + enc["quadratic_eV"][1:]
    elif tamper == "drop_constant":
        enc["constant_exact"] = "0"
    else:
        enc["higher_order_terms"] = enc["higher_order_terms"][1:]
    assert encoding.verify_encoding(spec, topo, ham, enc)["status"] == "FAILED"
    with pytest.raises(PipelineError, match="ENCODING_DECODE_MISMATCH"):
        optimization.exact_enumeration(spec, topo, ham, enc)


@pytest.mark.parametrize("case", ["core_pairwise", "higher_order"])
@pytest.mark.parametrize("tid", ["ring12", "prism6"])
def test_exact_enumeration_matches_known_minima(pairwise, cubic, case, tid):
    spec, hams = pairwise if case == "core_pairwise" else cubic
    result = optimization.exact_enumeration(spec, spec.topologies[tid], hams[tid])
    exp = EXPECTED[case]["topologies"][tid]
    assert result["n_feasible"] == 924
    assert result["minimum_energy_eV"] == pytest.approx(exp["minimum_energy_eV"], abs=1e-12)
    assert result["minimizing_bitstrings"] == exp["minimizing_bitstrings"]
    assert result["optimality_status"] == "OPTIMAL_CERTIFIED_EXHAUSTIVE"


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("tid", ["ring12", "prism6"])
def test_simulated_annealing_agrees_with_exact(cubic, tid, seed):
    spec, hams = cubic
    sa = optimization.simulated_annealing(spec, spec.topologies[tid], hams[tid], seed=seed)
    assert sa["optimality_status"] == "UNCERTIFIED_HEURISTIC"
    assert sa["minimum_energy_eV"] == pytest.approx(EXPECTED["higher_order"]["topologies"][tid]["minimum_energy_eV"], abs=1e-12)
    assert all(sum(int(c) for c in b) == 6 for b in sa["candidates"])
