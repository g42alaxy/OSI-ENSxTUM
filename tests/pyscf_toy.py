"""SYNTHETIC aluminium-formate toy framework for PySCF (GPW) tests.

Al(3+) nodes on a primitive-cubic net with one bent formate (O-C(H)-O, -1, O-C-O 125 deg) on every cell
edge; both O sit on the edge axis and the C-H points away from it, arranged so that the 3-fold rotation
about [111] is a symmetry. The substituted linker replaces H by F
(fluoroformate). The cell is small and closed shell (108 + 6 per F valence electrons in the 2x1x1 cell). One smeared GPW single
point takes 1-3 minutes on a laptop CPU at the shipped 25 Ha cutoff, which is NOT converged for the hard O/F GTH
pseudopotentials: the energy difference between two decorations moves by 0.6 Ha between 25 and 30 Ha, so the
pipeline stops at the convergence gate (DFT_SETTINGS_NOT_CONVERGED). A completing run needs ke_cutoff_Ha of order
100 Ha (hours on a CPU; use device "gpu"). The geometry is idealised and not a real MOF: every artifact is SYNTHETIC.

    python tests/pyscf_toy.py OUT_DIR   # writes OUT_DIR/project.json for `python -m qumof run`
"""
import copy
import json
import math
import sys
from pathlib import Path

from toy_framework import _merge, sha

AL_O, C_O, C_H, HALF_OCO = 1.90, 1.26, 1.09, math.radians(62.5)
C_X, C_Y = C_O * math.sin(HALF_OCO), C_O * math.cos(HALF_OCO)  # C offset from each O along / off the edge
A = 2 * (AL_O + C_X)  # 6.035 angstrom: Al-O ... O-Al along each edge
C3_OPERATIONS = ["x,y,z", "z,x,y", "y,z,x"]


def formate_cif():
    atoms = [("Al1", "Al", 0.0, 0.0, 0.0), ("O1", "O", AL_O / A, 0.0, 0.0), ("O2", "O", 1 - AL_O / A, 0.0, 0.0),
             ("C1", "C", 0.5, C_Y / A, 0.0), ("H1", "H", 0.5, (C_Y + C_H) / A, 0.0)]
    lines = ["data_toy_al_formate_SYNTHETIC", "_symmetry_space_group_name_H-M 'R 3 (cubic setting, toy)'",
             f"_cell_length_a {A}", f"_cell_length_b {A}", f"_cell_length_c {A}",
             "_cell_angle_alpha 90", "_cell_angle_beta 90", "_cell_angle_gamma 90",
             "loop_", "_space_group_symop_operation_xyz"] + [f"'{op}'" for op in C3_OPERATIONS]
    lines += ["loop_", "_atom_site_label", "_atom_site_type_symbol", "_atom_site_fract_x", "_atom_site_fract_y",
              "_atom_site_fract_z", "_atom_site_occupancy"]
    lines += [f"{l} {e} {x:.6f} {y:.6f} {z:.6f} 1.0" for l, e, x, y, z in atoms]
    return "\n".join(lines) + "\n"


BLOCKS = [
    {"id": "AL", "role": "node", "formula": {"Al": 1}, "formal_charge": 3, "connectivity": 6,
     "source": "toy (SYNTHETIC)", "license": "CC0"},
    {"id": "HCOO", "role": "linker", "slot_type": "linker", "formula": {"C": 1, "H": 1, "O": 2}, "formal_charge": -1,
     "connectivity": 2, "n_connector_atoms": 2, "in_parent_cif": True, "source": "toy (SYNTHETIC)", "license": "CC0"},
    {"id": "FCOO", "role": "linker", "slot_type": "linker", "formula": {"C": 1, "F": 1, "O": 2}, "formal_charge": -1,
     "connectivity": 2, "n_connector_atoms": 2, "source": "toy (SYNTHETIC)", "license": "CC0",
     "substitution": {"parent_block": "HCOO", "replace_element": "H",
                      "group": [{"element": "F", "frame_A": [1.35, 0.0, 0.0]}]}},
]


def pyscf_dft_protocol(python=None):
    return {"protocol_id": "toy_pyscf_gpw_pbe_d3bj", "engine": "pyscf", "command": [python or sys.executable],
            "xc_functional": "PBE", "dispersion": "d3bj", "basis": "gth-szv", "pseudo": "gth-pbe",
            "ke_cutoff_Ha": 25, "precision": 1e-8, "integral_scheme": "multigrid", "device": "cpu",
            "kpoint_spacing_inv_A": 1.1,
            # the idealised toy has a ~0.1 eV gap in this minimal basis: fixed occupations do not converge
            "occupations": {"kind": "smearing", "method": "fermi", "sigma_Ha": 0.005},
            "zero_smearing_convention": "free_energy", "hubbard_u_eV": {}, "conv_tol_Ha": 1e-8,
            "relaxation": {"calculation": "scf"}, "final_single_point": False,
            "numerical_retries": [{"max_cycle": 60}, {"max_cycle": 150, "damp": 0.3, "level_shift": 0.2}],
            "difference_convergence": {"ke_cutoff_Ha_series": [20, 25, 30]},  # 15 Ha: the smeared SCF does not converge
            "n_cores": 6, "max_memory_MB": 2000, "timeout_s": 600}


def base_spec(cif_sha, python=None):
    return {
        "schema_version": "0.1", "status": "READY", "scope": "static_lattice_candidate_ranking",
        "project_id": "toy_al_formate_FCOO_SYNTHETIC", "provenance": "SYNTHETIC",
        "source_structures": [{"topology_id": "toy_al_formate", "cif": "structures/toy_al_formate.cif",
                               "sha256": cif_sha, "source": "tests/pyscf_toy.py (generated, SYNTHETIC)",
                               "declared_net": "pcu"}],
        "candidate_topology_ids": ["toy_al_formate"],
        "building_blocks": copy.deepcopy(BLOCKS),
        "chemistry": {"target_elemental_inventory": {"Al": 2, "C": 6, "F": 3, "H": 3, "O": 12},
                      "formula_unit": "Al2(HCOO)3(FCOO)3", "composition_by_sublattice": {"linker": {"HCOO": 0.5, "FCOO": 0.5}},
                      "oxidation_and_protonation_policy": "formal charges supplied per building block",
                      "net_charge": 0, "solvent_and_counterion_policy": "require_desolvated",
                      "magnetic_state_policy": {"initial_states": [{"label": "closed_shell", "nspin": 1}]},
                      "disorder_resolution_policy": "require_fully_ordered"},
        "search": {"supercell_matrices": [[[2, 0, 0], [0, 1, 0], [0, 0, 1]]], "orientation_catalog_id": None,
                   "symmetry_policy_id": "cif_space_group_plus_supercell_translations"},
        "protocols": {
            "builder": {"builder_id": "parent_substitution_v1", "bond_tolerance_A": 0.45,
                        "min_interatomic_distance_A": 0.9, "slot_match_tolerance_A": 0.5},
            "ml_potential_checkpoint": None, "ml_calibration": None, "dft": pyscf_dft_protocol(python),
            "embedding": None, "active_space": None,
            "model_selection": {
                "candidates": [{"id": "M0_nearest_pairs", "features": ["constant", "unlike_distance_1"]},
                               {"id": "M1_points_and_pairs", "features": ["constant", "site_orbits", "unlike_distance_1"]}],
                "label_orbits_per_topology": "all", "holdout_fraction": 0.2, "n_folds": 3, "seed": 5,
                "regularization": {"kind": "ridge", "alpha": 0}, "rank_deficiency": "drop_dependent_columns"},
            "encoding": {"format": "upper_triangle_once", "ratio_penalty_eV": 5.0,
                         "higher_order": "retain_HUBO_or_explicitly_reject"},
            "classical_solvers": ["exact_enumeration"], "quantum_solvers": [],
            "acquisition": {"top_k_unique_orbits_per_topology": 2},
            "validation": {"structural_checks": ["builder_sanity", "post_relaxation_slot_mapping"]},
            "stopping": None},
        "tolerances": {"energy_units": "eV_per_common_formula_unit", "target_relative_energy_tolerance": 0.05,
                       "force_tolerance_eV_per_angstrom": 0.05,
                       "stress_tolerance_and_units": {"value": 5.0, "units": "kbar"},
                       "low_energy_validation_criterion": {"k": 2, "min_recall": 0.5},
                       "consecutive_stable_rounds": None},
        "budgets": {"max_dft_jobs": 100, "max_active_space_jobs": None, "max_qpu_shots": None,
                    "max_cpu_core_hours": 20, "max_gpu_hours": None, "max_llm_calls": None,
                    "max_total_cost_currency_and_value": None},
        "runtime": {"engine": "local", "llm_provider": None, "llm_model": None,
                    "allowed_tools_by_role": {"structure": [], "model_search": [], "electronic": [], "reviewer": []},
                    "scientific_protocol_changes_require_new_version": True},
        "deliverables": ["structures", "compatible_energy_table", "validation_reports", "reproducibility_manifest"],
        "excluded_scope": ["finite_temperature_ensembles", "quantum_mcmc"],
    }


def write_pyscf_toy_project(dst, overrides=None, python=None) -> Path:
    dst = Path(dst)
    (dst / "structures").mkdir(parents=True, exist_ok=True)
    cif = formate_cif()
    (dst / "structures/toy_al_formate.cif").write_text(cif)
    spec = _merge(base_spec(sha(cif), python), overrides or {})
    (dst / "project.json").write_text(json.dumps(spec, indent=2))
    return dst / "project.json"


if __name__ == "__main__":
    print(write_pyscf_toy_project(sys.argv[1] if len(sys.argv) > 1 else "runs/toy_pyscf_project"))
