"""MOF-5-family scheme project: the whole agent pipeline of OSI_scheme.pdf on builder-generated topologies.

Text input (design brief) -> A1 selects among PORMAKE-generated nets (Zn4O + BDC, 1:1 BDC/NH2-BDC) -> ML-potential
screen (orb-v3 in .venv-mlip) -> A2 builds the occupation Hamiltonian -> QUBO solved classically and by QAOA
(statevector simulation + Qiskit fake IBM device) -> top-5 candidates evaluated with periodic DFT and node-cluster
VQE/SQD (finite-cluster embedding, PySCF) -> A67 (scheme A3) confirmation -> A4 optimisation loop.

Real components: PORMAKE frameworks and blocks, orb-v3 relaxations, QAOA/Qiskit, PySCF cluster HF/DFT/VQE/SQD and
the CCSD embedding benchmark. Stand-in: the periodic DFT engine is tests/mock_pw.py (a 200-atom MOF-5 cell is far
beyond laptop DFT), so the project is labelled SYNTHETIC and its rankings are not MOF predictions.

    python tests/mof5_scheme.py OUT_DIR    # writes OUT_DIR/project.json
"""
import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
MAIN_PYTHON = str(REPO / ".venv/bin/python")      # pormake, qiskit, pyscf
MLIP_PYTHON = str(REPO / ".venv-mlip/bin/python")  # orb-models (needs Python <= 3.13 for dm-tree)
BRIEF = ("Mixed-linker MOF-5 family: Zn4O(CO2)6 octahedral secondary building units joined by benzene-1,4-dicarboxylate "
         "(BDC) and 2-aminobenzene-1,4-dicarboxylate (NH2-BDC) at a 1:1 linker ratio, neutral and desolvated. Which "
         "net and which linker arrangement are most stable? Nets whose node geometry cannot hold an octahedral Zn4O "
         "unit are out of scope.")


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


BLOCKS = [
    {"id": "Zn4O", "role": "node", "formula": {"O": 1, "Zn": 4}, "formal_charge": 6, "connectivity": 6,
     "source": "PORMAKE block N16 (Zn4O(CO2)6); carboxylates belong to the linkers in this slot model", "license": "MIT (PORMAKE)"},
    {"id": "BDC", "role": "linker", "slot_type": "linker", "formula": {"C": 8, "H": 4, "O": 4}, "formal_charge": -2,
     "connectivity": 2, "n_connector_atoms": 4, "in_parent_cif": True,
     "source": "PORMAKE blocks N16 carboxylates + E14 (p-phenylene)", "license": "MIT (PORMAKE)"},
    {"id": "NH2_BDC", "role": "linker", "slot_type": "linker", "formula": {"C": 8, "H": 5, "N": 1, "O": 4}, "formal_charge": -2,
     "connectivity": 2, "n_connector_atoms": 4, "substitution": {"parent_block": "BDC"},
     "source": "amino group geometry derived from PORMAKE block E72 (2-aminophenylene)", "license": "MIT (PORMAKE)"},
]


def spec_dict(pseudos, pw_command, root: Path, nets="auto", fast=False):
    return {
        "schema_version": "0.1", "status": "READY", "scope": "static_lattice_candidate_ranking",
        "project_id": "mof5_bdc_nh2bdc_scheme", "provenance": "SYNTHETIC",
        "purpose": BRIEF, "design_brief": BRIEF,
        "source_structures": [], "candidate_topology_ids": [],
        "building_blocks": copy.deepcopy(BLOCKS),
        "chemistry": {"target_elemental_inventory": {"C": 48, "H": 27, "N": 3, "O": 26, "Zn": 8},
                      "formula_unit": "Zn8O2(BDC)3(NH2_BDC)3",
                      "composition_by_sublattice": {"linker": {"BDC": 0.5, "NH2_BDC": 0.5}},
                      "oxidation_and_protonation_policy": "Zn(II); mu4-oxide; dicarboxylates fully deprotonated",
                      "net_charge": 0, "solvent_and_counterion_policy": "require_desolvated",
                      "magnetic_state_policy": {"initial_states": [{"label": "closed_shell_Zn_d10", "nspin": 1}]},
                      "disorder_resolution_policy": "require_fully_ordered"},
        "search": {"supercell_matrices": [], "orientation_catalog_id": None,
                   "symmetry_policy_id": "cif_space_group_plus_supercell_translations"},
        "protocols": {
            "topology_generation": {
                "builder": "pormake", "python": MAIN_PYTHON,
                "node_block": {"block_id": "Zn4O", "pormake_id": "N16"},
                "edge_block": {"block_id": "BDC", "pormake_id": "E14"},
                "substituents": [{"block_id": "NH2_BDC", "pormake_id": "E72", "root_element": "N", "replace_element": "H"}],
                "nets": nets, "max_node_rmsd_A": 0.25, "max_nodes_per_cell": 4, "max_candidates": 6,
                "min_substitution_sites": 12, "max_substitution_sites": 12, "max_multiplier": 2, "timeout_s": 3600},
            # PORMAKE's E72 amino N-H bonds are 0.86 A (X-ray-like), so the overlap floor is 0.8 A.
            "builder": {"builder_id": "parent_substitution_v1", "bond_tolerance_A": 0.45,
                        "min_interatomic_distance_A": 0.8, "slot_match_tolerance_A": 0.6},
            "ml_potential_checkpoint": {
                "engine": "orb", "model": "orb-v3-conservative-20-omat", "dispersion": "D3BJ", "python": MLIP_PYTHON,
                "relaxation": {"optimizer": "FIRE", "fmax_eV_per_A": 0.1, "max_steps": 400 if not fast else 60,
                               "cell": "variable"},
                "timeout_s": 3600, "supported_elements": ["C", "H", "N", "O", "Zn"]},
            # Two fixed-composition seeds per net (orb-v3 on ~220 atoms is ~1 s per FIRE step on a laptop CPU).
            # Exploratory prioritisation is an explicit opt-in: uncalibrated, auditable, reinstatable.
            "screening": {"sampling": {"include_endpoints": False, "n_ordered": 1, "n_random": 1, "n_sqs": 0, "seed": 0},
                          "max_cell_strain_percent": 15.0,
                          "energetic_policy": {"deprioritise_above_eV_per_formula_unit": 0.25, "allow_exploratory": True}},
            "ml_calibration": None,
            "dft": {"protocol_id": "mock_qe_pbe", "engine": "quantum_espresso", "command": pw_command,
                    "xc_functional": "PBE", "dispersion": "none", "pseudo_dir": "pseudo",
                    "pseudopotentials": pseudos, "ecutwfc_Ry": 100, "ecutrho_Ry": 800, "kpoint_spacing_inv_A": 0.5,
                    "occupations": {"kind": "fixed"}, "zero_smearing_convention": "total_energy_as_printed",
                    "hubbard_u_eV": {}, "conv_thr_Ry": 1e-9,
                    "relaxation": {"calculation": "vc-relax", "cell_dofree": "all"}, "final_single_point": True,
                    "numerical_retries": [{"mixing_beta": 0.7}, {"mixing_beta": 0.3}],
                    "difference_convergence": {"ecutwfc_Ry_series": [60, 80, 100, 120]}, "n_cores": 1, "timeout_s": 120},
            "model_selection": {
                "candidates": [{"id": "M0_nearest_pairs", "features": ["constant", "unlike_distance_1"]},
                               {"id": "M1_two_shells", "features": ["constant", "unlike_distance_1", "unlike_distance_2"]},
                               {"id": "M2_with_triplets",
                                "features": ["constant", "unlike_distance_1", "unlike_distance_2", "BBB_paths"]},
                               {"id": "C1_pair_orbits", "features": ["constant", "point_orbits", "pairs<=9.5"]}],
                "label_orbits_per_topology": 48 if not fast else 32, "holdout_fraction": 0.25, "n_folds": 4, "seed": 11,
                "regularization": {"kind": "ridge", "alpha": 0}, "rank_deficiency": "drop_dependent_columns",
                "on_inadequate": "drop_topology"},
            "encoding": {"format": "upper_triangle_once", "ratio_penalty_eV": 2.0, "higher_order": "quadratize"},
            "classical_solvers": ["exact_enumeration", "branch_and_bound"],
            "quantum_solvers": [{"kind": "qpu_fake_device", "device": "fake_brisbane", "python": MAIN_PYTHON,
                                 "layers": 1, "shots": 512, "seeds": [1], "restarts": 2, "max_evaluations": 40,
                                 "optimization_level": 1}],
            "acquisition": {"top_k_unique_orbits_per_topology": 5},
            "embedding": {
                "protocol_id": "zn4o_formate_cluster_vqe_sqd", "scheme": "finite_cluster_subtractive", "engine": "pyscf",
                "command": [MAIN_PYTHON], "region": {"bond_hops": 2},
                "capping": {"element": "H", "distance_A": {"C": 1.09, "O": 0.97}},
                "cluster_charge": 0, "cluster_spin": 0, "basis": "sto-3g",
                # Second-order SCF: plain DIIS oscillates on this Zn4O cluster (declared numerics, part of the fidelity).
                "low_level": {"method": "DFT", "xc": "PBE", "dispersion": "none", "scf_solver": "newton", "max_cycle": 100},
                "high_level": {"method": "active_space", "solver": "sqd_simulation",
                               "active_space": {"protocol_id": "zn4o_cas44", "reference": "RHF", "scf_solver": "newton",
                                                "conv_tol": 1e-8, "n_active_electrons": 4, "n_active_orbitals": 4},
                               "sqd": {"shots": 2000, "seed": 1, "flip_probability": 0.02}},
                "conv_tol_Ha": 1e-8, "n_cores": 8, "timeout_s": 3600,
                "validation": {"bond_hops_series": [2, 3],
                               "reference_benchmarks": [str(REPO / "benchmarks/embedding_keto_enol_hf_ccsd.json")]}},
            "active_space": None,
            "validation": {"structural_checks": ["builder_sanity", "post_relaxation_slot_mapping", "residual_force_stress"],
                           "reference_phases": "pure_endpoints", "reference_set_complete": False},
            "stopping": None},
        "tolerances": {"energy_units": "eV_per_common_formula_unit", "target_relative_energy_tolerance": 0.01,
                       "force_tolerance_eV_per_angstrom": 0.01,
                       "stress_tolerance_and_units": {"value": 0.5, "units": "kbar"},
                       "low_energy_validation_criterion": {"k": 3, "min_recall": 0.6},
                       "consecutive_stable_rounds": None},
        "budgets": {"max_dft_jobs": 2000, "max_active_space_jobs": 40, "max_qpu_shots": 100000,
                    "max_cpu_core_hours": 50, "max_gpu_hours": None, "max_llm_calls": None,
                    "max_total_cost_currency_and_value": None},
        "runtime": {"engine": "local", "llm_provider": None, "llm_model": None,
                    "allowed_tools_by_role": {"structure": [], "model_search": [], "electronic": [], "reviewer": []},
                    "scientific_protocol_changes_require_new_version": True},
    }


def write_mof5_scheme_project(dst, nets="auto", fast=False, mutate=None) -> Path:
    dst = Path(dst)
    (dst / "pseudo").mkdir(parents=True, exist_ok=True)
    (dst / "tools").mkdir(exist_ok=True)
    pseudos = {}
    for el in ("C", "H", "N", "O", "Zn"):
        data = f"SYNTHETIC dummy pseudopotential for {el}; consumed only by the mock pw.x\n"
        (dst / f"pseudo/{el}.mock.UPF").write_text(data)
        pseudos[el] = {"file": f"{el}.mock.UPF", "sha256": sha(data)}
    shutil.copy(HERE / "mock_pw.py", dst / "tools/mock_pw.py")
    spec = spec_dict(pseudos, [sys.executable, str((dst / "tools/mock_pw.py").resolve())], dst, nets, fast)
    if mutate:
        mutate(spec)
    (dst / "project.json").write_text(json.dumps(spec, indent=2))
    return dst / "project.json"


if __name__ == "__main__":
    print(write_mof5_scheme_project(sys.argv[1] if len(sys.argv) > 1 else "runs/mof5_scheme"))
