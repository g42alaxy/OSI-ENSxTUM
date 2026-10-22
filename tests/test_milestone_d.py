"""Milestone D: quadratization (T03.4), constraints and one-hot encodings (T03.5), branch and bound (T04.1),
QAOA simulation (T04.2), active-space export and electronic solvers on a published benchmark (T07.1/T07.2)."""
import dataclasses
import json
from fractions import Fraction

import pytest
from conftest import PROJECT, REPO

from qumof import configuration_model as cm
from qumof import datasets, encoding, optimization, qaoa
from qumof.contract import PipelineError
from qumof.intake import load_project, validate_project
from qumof.orchestrator import run_pipeline


def _hamiltonians(case):
    spec = load_project(PROJECT, case=case)
    train = datasets.ingest_labels(spec, spec.training_rows, "training", "t")
    dev = datasets.ingest_labels(spec, spec.development_rows, "development", "d")
    return spec, {tid: cm.hamiltonian_from_selection(spec, topo, cm.fit_and_select(spec, topo, train, dev))
                  for tid, topo in spec.topologies.items()}


def test_quadratized_cubic_model_reproduces_feasible_cost():
    spec, hams = _hamiltonians("higher_order")
    for tid, topo in spec.topologies.items():
        with pytest.raises(PipelineError, match="UNSUPPORTED_HIGHER_ORDER_WITHOUT_QUADRATIZATION"):
            encoding.compile_encoding(spec, topo, hams[tid], "QUBO")  # the default policy still rejects
        enc = encoding.compile_encoding(spec, topo, hams[tid], "QUBO", quadratize_higher_order=True)
        assert enc["max_degree"] == 2 and enc["ancillas"] and not enc["higher_order_terms"]
        assert all(Fraction(a["penalty_M_exact"]) > 0 and len(a["product_of"]) == 2 for a in enc["ancillas"])
        ver = encoding.verify_encoding(spec, topo, hams[tid], enc)  # minimises over ancillas on all 4096 strings
        assert ver["status"] == "COMPLETED" and ver["max_abs_deviation_exact"] == "0"
        assert ver["n_ancillas_minimised"] == len(enc["ancillas"]) and ver["all_penalized_argmins_feasible"]
        exact = optimization.exact_enumeration(spec, topo, hams[tid], enc)  # decode cross-check with ancillas
        assert exact["minimum_energy_eV"] == pytest.approx({"ring12": -0.12, "prism6": -0.16}[tid])
        # A wrong ancilla decode rule is caught when decoded states are re-evaluated.
        i, j = enc["ancillas"][0]["product_of"]
        other = next(v for v in range(topo.n_sites) if v not in (i, j))
        broken = {**enc, "ancillas": [{**enc["ancillas"][0], "product_of": [i, other]}] + enc["ancillas"][1:]}
        with pytest.raises(PipelineError, match="ENCODING_DECODE_MISMATCH"):
            optimization.exact_enumeration(spec, topo, hams[tid], broken)


def test_penalty_bound_sublattice_counts_and_one_hot():
    spec, hams = _hamiltonians("core_pairwise")
    topo, ham = spec.topologies["ring12"], hams["ring12"]
    enc = encoding.compile_encoding(spec, topo, ham, "QUBO")
    assert enc["penalty_bound"]["certified_by_bound"] is True  # 1 eV > single-flip bound
    small = load_project(PROJECT, overrides={"qubo": {"ratio_penalty_eV": 0.001}})
    weak = encoding.compile_encoding(small, topo, ham, "QUBO")
    assert weak["penalty_bound"]["certified_by_bound"] is False
    # The bound is sufficient, not necessary: the report must say exhaustive enumeration was needed.
    assert "does not certify" in encoding.verify_encoding(small, topo, ham, weak)["penalty_certification"]
    # Two sublattices with their own counts: the penalised minimum must satisfy both, exact enumeration visits
    # only states that satisfy both, and solvers that assume one global count refuse.
    split_topo = dataclasses.replace(topo, sublattice_counts=((tuple(range(6)), 3), (tuple(range(6, 12)), 3)))
    split = encoding.compile_encoding(spec, split_topo, ham, "QUBO")
    assert [c["k"] for c in split["count_constraints"]] == [3, 3]
    assert encoding.verify_encoding(spec, split_topo, ham, split)["status"] == "COMPLETED"
    assert optimization.exact_enumeration(spec, split_topo, ham, split)["n_feasible"] == 20 * 20
    with pytest.raises(PipelineError, match="UNSUPPORTED_SUBLATTICE_CONSTRAINTS"):
        optimization.branch_and_bound(spec, split_topo, ham)
    # One-hot: 4 slots on a ring, 3 species with counts 2/1/1, nearest-neighbour like-species penalty.
    ring = [(0, 1), (1, 2), (2, 3), (3, 0)]
    physical = {((i, a), (j, a)): "0.1" for i, j in ring for a in range(3)}
    physical.update({((0, 2),): "-0.05"})
    counts = [(range(4), 0, 2), (range(4), 1, 1), (range(4), 2, 1)]
    one_hot = encoding.compile_one_hot(4, 3, physical, counts, 1.0)

    def energy(species):
        return sum(Fraction(1, 10) for i, j in ring if species[i] == species[j]) - (Fraction(1, 20) if species[0] == 2 else 0)
    report = encoding.verify_one_hot(one_hot, energy)
    feasible = encoding.one_hot_states(4, [2, 1, 1])
    best = min(energy(s) for s in feasible)
    assert report["status"] == "COMPLETED" and Fraction(report["encoded_minimum_exact"]) == best
    assert set(report["minimizers"]) == {s for s in feasible if energy(s) == best}


def test_branch_and_bound_agrees_and_reports_honest_gaps():
    spec, hams = _hamiltonians("higher_order")
    for tid, topo in spec.topologies.items():
        for coefs in (hams[tid]["coefficients_exact"], ["1/10", "1/50", "-3/100", "-1/20"], ["0", "1/40", "1/60", "-1/15"]):
            ham = {**hams[tid], "coefficients_exact": coefs}
            exact = Fraction(optimization.exact_enumeration(spec, topo, ham)["minimum_energy_exact"])
            full = optimization.branch_and_bound(spec, topo, ham)
            assert full["optimality_status"] == "OPTIMAL_CERTIFIED_BRANCH_AND_BOUND"
            assert Fraction(full["minimum_energy_exact"]) == exact
            limited = optimization.branch_and_bound(spec, topo, ham, node_limit=20)
            incumbent, lower = Fraction(limited["minimum_energy_exact"]), Fraction(limited["lower_bound_exact"])
            assert lower <= exact <= incumbent  # feasible incumbent, valid bound
            if limited["optimality_status"] != full["optimality_status"]:
                assert limited["optimality_status"] == "UNCERTIFIED_NODE_LIMIT" and limited["optimality_gap_eV"] >= 0


def test_qaoa_mixers_preserve_constraints_and_bit_order_errors_are_caught():
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    assert qaoa.verify_mixer_preserves_constraints(6, 2, [(i, (i + 1) % 6) for i in range(6)], one_hot=False)["preserves_constraints"]
    assert qaoa.verify_mixer_preserves_constraints(4, 3, edges, one_hot=True)["preserves_constraints"]
    assert not qaoa.verify_mixer_preserves_constraints(4, 3, edges, one_hot=True, mixer="x")["preserves_constraints"]
    # The exchange mixer never leaves the multicomponent feasible subspace, and its moves connect it.
    states = encoding.one_hot_states(4, [2, 1, 1])
    pairs = qaoa.exchange_pairs(states, edges)
    assert all(sorted(states[u]) == sorted(states[v]) for p in pairs for u, v in p)
    spec, hams = _hamiltonians("core_pairwise")
    topo = spec.topologies["ring12"]
    enc = encoding.compile_encoding(spec, topo, hams["ring12"], "QUBO")
    swap = {0: 2, 2: 0}  # a bit-order mistake: sites 0 and 2 exchanged in the compiled couplings (not a symmetry)
    mislabelled = {**enc, "quadratic_eV": [{**q, "i": min(swap.get(q["i"], q["i"]), swap.get(q["j"], q["j"])),
                                           "j": max(swap.get(q["i"], q["i"]), swap.get(q["j"], q["j"]))}
                                          for q in enc["quadratic_eV"]]}
    with pytest.raises(PipelineError, match="ENCODING_DECODE_MISMATCH"):
        qaoa.run_qaoa(spec, topo, hams["ring12"], mislabelled,
                      {"layers": 1, "shots": 10, "seeds": [1], "restarts": 1, "max_evaluations": 5}, seed=1)


def test_qaoa_benchmark_agrees_with_exact_and_records_provenance():
    spec, hams = _hamiltonians("core_pairwise")
    protocol = {"layers": 2, "shots": 200, "seeds": [1, 3], "restarts": 2, "max_evaluations": 60}
    topo = spec.topologies["ring12"]
    enc = encoding.compile_encoding(spec, topo, hams["ring12"], "QUBO")
    exact = optimization.exact_enumeration(spec, topo, hams["ring12"], enc)
    bench = qaoa.qaoa_benchmark(spec, topo, hams["ring12"], enc, exact, protocol)
    assert bench["status"] == "AGREES_WITH_EXACT" and bench["mixer_constraint_check"]["preserves_constraints"]
    uniform = len(exact["minimizing_bitstrings"]) / exact["n_feasible"]
    assert bench["summary"]["median_success_probability"] > 10 * uniform  # amplification over random sampling
    for run in bench["runs"]:
        assert run["postselection_rate"] == 1.0 and run["optimizer_calls"] > 0 and run["shots"] == 200
        assert Fraction(run["best_sample_energy_exact"]) == Fraction(exact["minimum_energy_exact"])
    assert {"bit_order", "n_qubits", "layers", "mixer"} <= set(bench["configuration"])
    assert "not evidence of quantum speedup" in bench["note"] and len(bench["simulated_annealing"]) == 2


def test_pipeline_quantum_benchmark_with_quadratization(tmp_path):
    protocol = {"kind": "qaoa_simulation", "layers": 1, "shots": 100, "seeds": [1], "restarts": 1, "max_evaluations": 30}
    overrides = {"qubo": {"higher_order": "quadratize"},
                 "optimizer": {"quantum_enabled": True, "cross_check_solvers": ["branch_and_bound"], "quantum_protocol": protocol}}
    spec = load_project(PROJECT, case="higher_order", overrides=overrides)
    assert validate_project(spec)["status"] == "COMPLETED"
    report = run_pipeline(spec, tmp_path / "run")
    assert report["pipeline_status"] == "COMPLETED" and report["winning_topology"] == "prism6"
    enc, opt = report["stage_detail"]["encoding"], report["stage_detail"]["optimization"]
    assert all(enc[t]["target"] == "QUBO" and enc[t]["n_ancillas"] > 0 and enc[t]["verification"] == "COMPLETED"
               for t in ("ring12", "prism6"))
    assert all(opt["cross_checks"][t]["agrees_with_exact"] for t in ("ring12", "prism6"))
    assert opt["quantum"]["qaoa_simulation"] == "COMPLETED" and "QAOA simulation" not in report["not_executed"]
    bad = load_project(PROJECT, overrides={"optimizer": {"quantum_enabled": True, "quantum_protocol": {"kind": "qpu"}}})
    assert {"SCIENTIFIC_SPEC_MISSING", "UNSUPPORTED_QUANTUM_BACKEND"} <= {e["code"] for e in validate_project(bad)["errors"]}
    zero_shots = load_project(PROJECT, overrides={"optimizer": {"quantum_enabled": True, "quantum_protocol": {**protocol, "shots": 0}}})
    assert "SCIENTIFIC_SPEC_MISSING" in {e["code"] for e in validate_project(zero_shots)["errors"]}


def test_electronic_solvers_agree_on_identical_tiny_hamiltonian():
    pytest.importorskip("pyscf")
    from qumof import electronic_solvers as es
    from qumof.active_space import prepare_active_space, problem_from_artifact, read_fcidump
    system = {"system_id": "LiH", "atoms": [["Li", 0, 0, 0], ["H", 0, 0, 1.5949]], "unit": "Angstrom", "charge": 0, "spin": 0}
    protocol = {"protocol_id": "lih_cas44", "basis": "sto-3g", "reference": "RHF", "n_active_electrons": 4,
                "n_active_orbitals": 4, "embedding": "none"}
    art = prepare_active_space(system, protocol)
    assert prepare_active_space(system, protocol)["artifact_id"] == art["artifact_id"]  # identical inputs, identical integrals
    h1, h2, ecore = read_fcidump(art["fcidump"])
    stored_h1, stored_h2, stored_ecore, nelec, _ = problem_from_artifact(art)
    assert abs(h1 - stored_h1).max() < 1e-12 and abs(h2 - stored_h2).max() < 1e-12 and abs(ecore - stored_ecore) < 1e-12
    ed, casci, vqe = es.exact_diagonalization(art), es.pyscf_casci(art), es.vqe_simulation(art)
    assert abs(ed["energy_Ha"] - casci["energy_Ha"]) < 1e-9  # independent determinant code vs PySCF FCI
    assert 0 <= vqe["energy_Ha"] - ed["energy_Ha"] < 1e-6 and vqe["mapping"] == "Jordan-Wigner"
    sqd = es.sqd_simulation(art, vqe["final_state"], shots=3000, seed=1, flip_probability=0.05)
    assert sqd["energy_Ha"] >= ed["energy_Ha"] - 1e-9 and sqd["recovered_samples"] > 0
    assert {r["hamiltonian_hash"] for r in (ed, casci, vqe, sqd)} == {art["hamiltonian_hash"]}
    with pytest.raises(PipelineError, match="UNSUPPORTED_EMBEDDING"):
        prepare_active_space(system, {**protocol, "embedding": "dft_in_dft"})


def test_published_h2_benchmark_is_reproduced(tmp_path):
    pytest.importorskip("pyscf")
    from qumof.__main__ import main
    spec_path = REPO / "benchmarks/h2_sto3g_szabo_ostlund.json"
    assert main(["electronic-benchmark", "--spec", str(spec_path), "--run-dir", str(tmp_path / "run")]) == 0
    report = next(json.loads(p.read_text()) for p in (tmp_path / "run/artifacts").glob("ElectronicBenchmarkReport__*"))
    assert report["status"] == "REPRODUCED" and report["identical_problem"] is True
    assert abs(report["comparisons"]["exact_vs_published_Ha"]) <= 1e-4
    assert abs(report["comparisons"]["hf_vs_published_Ha"]) <= 1e-4
    assert all(abs(d) < 1e-8 for d in report["comparisons"]["vqe_simulation_minus_exact_Ha"])
    assert report["solver_status"]["sqd_hardware"] == "BLOCKED" and report["correction_status"] == "UNVALIDATED"
    assert report["solver_consistency"]["consistent"] and len(report["solver_consistency"]["hamiltonian_hashes"]) == 1
