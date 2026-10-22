"""SYNTHETIC toy physical project for Milestone B plumbing tests.

A primitive-cubic Zn/linker net (one Zn per cell, one O-C-C-O linker per cell
edge, four H per C) written as a Pm-3m asymmetric unit, a declared amino-type
substitution rule, dummy pseudopotential files and a project spec that drives
tests/mock_pw.py through the real Quantum ESPRESSO adapter. Nothing here is a
real MOF, pseudopotential or DFT result; every artifact is labelled SYNTHETIC.

    python tests/toy_framework.py OUT_DIR   # writes OUT_DIR/project.json for `python -m qumof run`
"""
import copy
import hashlib
import itertools
import json
import shutil
import sys
from pathlib import Path

A = 8.1  # angstrom; Zn-O 2.0, O-C 1.3, C-C 1.5
HERE = Path(__file__).resolve().parent


def pm3m_operations():
    ops = []
    for perm in itertools.permutations("xyz"):
        for signs in itertools.product("+-", repeat=3):
            ops.append(",".join(("-" if s == "-" else "") + v for s, v in zip(signs, perm)))
    return ops


def toy_cif(ops=None, extra_atoms=(), occupancy_h=1.0, shuffle=False):
    atoms = [("Zn1", "Zn", 0.0, 0.0, 0.0, 1.0), ("O1", "O", 2.0 / A, 0.0, 0.0, 1.0),
             ("C1", "C", 3.3 / A, 0.0, 0.0, 1.0), ("H1", "H", 3.3 / A, 1.08 / A, 0.0, occupancy_h)]
    atoms += list(extra_atoms)
    if shuffle:
        atoms = atoms[::-1]
    lines = ["data_toy_pcu_SYNTHETIC", "_symmetry_space_group_name_H-M 'P m -3 m'",
             f"_cell_length_a {A}", f"_cell_length_b {A}", f"_cell_length_c {A}",
             "_cell_angle_alpha 90", "_cell_angle_beta 90", "_cell_angle_gamma 90",
             "loop_", "_space_group_symop_operation_xyz"]
    lines += [f"'{op}'" for op in (ops or pm3m_operations())]
    lines += ["loop_", "_atom_site_label", "_atom_site_type_symbol", "_atom_site_fract_x", "_atom_site_fract_y",
              "_atom_site_fract_z", "_atom_site_occupancy"]
    lines += [f"{l} {e} {x:.6f} {y:.6f} {z:.6f} {o}" for l, e, x, y, z, o in atoms]
    return "\n".join(lines) + "\n"


def sha(text_or_bytes):
    data = text_or_bytes.encode() if isinstance(text_or_bytes, str) else text_or_bytes
    return hashlib.sha256(data).hexdigest()


BLOCKS = [
    {"id": "ZN", "role": "node", "formula": {"Zn": 1}, "formal_charge": 6, "connectivity": 6,
     "source": "toy (SYNTHETIC)", "license": "CC0"},
    {"id": "L", "role": "linker", "slot_type": "linker", "formula": {"C": 2, "H": 8, "O": 2}, "formal_charge": -2,
     "connectivity": 2, "n_connector_atoms": 2, "in_parent_cif": True, "source": "toy (SYNTHETIC)", "license": "CC0"},
    {"id": "LN", "role": "linker", "slot_type": "linker", "formula": {"C": 2, "H": 9, "N": 1, "O": 2},
     "formal_charge": -2, "connectivity": 2, "n_connector_atoms": 2, "source": "toy (SYNTHETIC)", "license": "CC0",
     "substitution": {"parent_block": "L", "replace_element": "H",
                      "group": [{"element": "N", "frame_A": [1.40, 0.0, 0.0]},
                                {"element": "H", "frame_A": [1.90, 0.0, 0.87]},
                                {"element": "H", "frame_A": [1.90, 0.0, -0.87]}]}},
]


def base_spec(cif_sha, pseudos, command):
    return {
        "schema_version": "0.1", "status": "READY", "scope": "static_lattice_candidate_ranking",
        "project_id": "toy_pcu_LN_SYNTHETIC", "provenance": "SYNTHETIC",
        "source_structures": [{"topology_id": "toy_pcu", "cif": "structures/toy_pcu.cif", "sha256": cif_sha,
                               "source": "tests/toy_framework.py (generated, SYNTHETIC)", "declared_net": "pcu"}],
        "candidate_topology_ids": ["toy_pcu"],
        "building_blocks": copy.deepcopy(BLOCKS),
        "chemistry": {"target_elemental_inventory": {"C": 12, "H": 51, "N": 3, "O": 12, "Zn": 2},
                      "formula_unit": "Zn2(L)3(LN)3", "composition_by_sublattice": {"linker": {"L": 0.5, "LN": 0.5}},
                      "oxidation_and_protonation_policy": "formal charges supplied per building block",
                      "net_charge": 0, "solvent_and_counterion_policy": "require_desolvated",
                      "magnetic_state_policy": {"initial_states": [{"label": "closed_shell", "nspin": 1}]},
                      "disorder_resolution_policy": "require_fully_ordered"},
        "search": {"supercell_matrices": [[[2, 0, 0], [0, 2, 0], [0, 0, 1]]], "orientation_catalog_id": None,
                   "symmetry_policy_id": "cif_space_group_plus_supercell_translations"},
        "protocols": {
            "builder": {"builder_id": "parent_substitution_v1", "bond_tolerance_A": 0.45,
                        "min_interatomic_distance_A": 0.9, "slot_match_tolerance_A": 0.5},
            "ml_potential_checkpoint": None, "ml_calibration": None,
            "dft": {"protocol_id": "toy_qe_pbe_d3", "engine": "quantum_espresso", "command": command,
                    "xc_functional": "PBE", "dispersion": "grimme-d3", "pseudo_dir": "pseudo",
                    "pseudopotentials": pseudos, "ecutwfc_Ry": 80, "ecutrho_Ry": 640, "kpoint_spacing_inv_A": 0.3,
                    "occupations": {"kind": "fixed"}, "zero_smearing_convention": "total_energy_as_printed",
                    "hubbard_u_eV": {}, "conv_thr_Ry": 1e-9,
                    "relaxation": {"calculation": "vc-relax", "cell_dofree": "all"}, "final_single_point": True,
                    "numerical_retries": [{"mixing_beta": 0.7}, {"mixing_beta": 0.3}],
                    "difference_convergence": {"ecutwfc_Ry_series": [40, 60, 80, 100]}, "n_cores": 1, "timeout_s": 60},
            "embedding": None, "active_space": None,
            "model_selection": {
                "candidates": [{"id": "M0_nearest_pairs", "features": ["constant", "unlike_distance_1"]},
                               {"id": "M1_points_and_pairs", "features": ["constant", "site_orbits", "unlike_distance_1"]},
                               {"id": "M2_points_two_shell_pairs",
                                "features": ["constant", "site_orbits", "unlike_distance_1", "unlike_distance_2"]}],
                "label_orbits_per_topology": "all", "holdout_fraction": 0.2, "n_folds": 4, "seed": 11,
                "regularization": {"kind": "ridge", "alpha": 0}, "rank_deficiency": "drop_dependent_columns"},
            "encoding": {"format": "upper_triangle_once", "ratio_penalty_eV": 1.0,
                         "higher_order": "retain_HUBO_or_explicitly_reject"},
            "classical_solvers": ["exact_enumeration"], "quantum_solvers": [],
            "acquisition": {"top_k_unique_orbits_per_topology": 3},
            "validation": {"structural_checks": ["builder_sanity", "post_relaxation_slot_mapping", "residual_force_stress"]},
            "stopping": None},
        "tolerances": {"energy_units": "eV_per_common_formula_unit", "target_relative_energy_tolerance": 0.005,
                       "force_tolerance_eV_per_angstrom": 0.01,
                       "stress_tolerance_and_units": {"value": 0.5, "units": "kbar"},
                       "low_energy_validation_criterion": {"k": 3, "min_recall": 0.6},
                       "consecutive_stable_rounds": None},
        "budgets": {"max_dft_jobs": 400, "max_active_space_jobs": None, "max_qpu_shots": None,
                    "max_cpu_core_hours": 10, "max_gpu_hours": None, "max_llm_calls": None,
                    "max_total_cost_currency_and_value": None},
        "runtime": {"engine": "local", "llm_provider": None, "llm_model": None,
                    "allowed_tools_by_role": {"structure": [], "model_search": [], "electronic": [], "reviewer": []},
                    "scientific_protocol_changes_require_new_version": True},
        "deliverables": ["structures", "compatible_energy_table", "validation_reports", "reproducibility_manifest"],
        "excluded_scope": ["finite_temperature_ensembles", "quantum_mcmc"],
    }


def _merge(base, extra):
    out = copy.deepcopy(base)
    for k, v in extra.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def write_toy_project(dst, overrides=None, cif_text=None, mutate=None) -> Path:
    dst = Path(dst)
    (dst / "structures").mkdir(parents=True, exist_ok=True)
    (dst / "pseudo").mkdir(exist_ok=True)
    (dst / "tools").mkdir(exist_ok=True)
    cif_text = cif_text or toy_cif()
    (dst / "structures/toy_pcu.cif").write_text(cif_text)
    pseudos = {}
    for el in ("C", "H", "N", "O", "Zn"):
        data = f"SYNTHETIC dummy pseudopotential for {el}; not usable by a real DFT code\n"
        (dst / f"pseudo/{el}.toy.UPF").write_text(data)
        pseudos[el] = {"file": f"{el}.toy.UPF", "sha256": sha(data)}
    shutil.copy(HERE / "mock_pw.py", dst / "tools/mock_pw.py")
    spec = base_spec(sha(cif_text), pseudos, [sys.executable, str((dst / "tools/mock_pw.py").resolve())])
    spec = _merge(spec, overrides or {})
    if mutate:
        mutate(spec)
    (dst / "project.json").write_text(json.dumps(spec, indent=2))
    return dst / "project.json"


if __name__ == "__main__":
    print(write_toy_project(sys.argv[1] if len(sys.argv) > 1 else "runs/toy_physical_project"))


# -- Milestone C: two candidate structures, a transfer cell, cluster features, refinement, references --------------
def staggered_cif():
    """Polymorph of the toy net: the four C-H per C rotated by 45 degrees about the linker axis (SYNTHETIC)."""
    atoms = [("Zn1", "Zn", 0.0, 0.0, 0.0, 1.0), ("O1", "O", 2.0 / A, 0.0, 0.0, 1.0), ("C1", "C", 3.3 / A, 0.0, 0.0, 1.0),
             ("H1", "H", 3.3 / A, 0.764 / A, 0.764 / A, 1.0)]
    lines = toy_cif().splitlines()
    head = lines[:lines.index("_atom_site_occupancy") + 1]
    return "\n".join(head + [f"{l} {e} {x:.6f} {y:.6f} {z:.6f} {o}" for l, e, x, y, z, o in atoms]) + "\n"


def milestone_c_overrides(staggered_sha):
    return {
        "candidate_topology_ids": ["toy_pcu", "toy_pcu_staggered"],
        "search": {"supercell_matrices": [[[2, 0, 0], [0, 2, 0], [0, 0, 1]], [[2, 0, 0], [0, 2, 0], [0, 0, 2]]]},
        "protocols": {
            "model_selection": {
                "candidates": [{"id": "C0_first_shell", "features": ["constant", "pairs<=5.8"]},
                               {"id": "C1_two_shells", "features": ["constant", "point_orbits", "pairs<=8.2"]}],
                "label_orbits_per_topology": 12, "holdout_fraction": 0.25, "n_folds": 3, "seed": 11},
            "acquisition": {"top_k_unique_orbits_per_topology": 3, "batch_size": 4, "audit_fraction": 0.25,
                            "weights": {"energy": 1.0, "uncertainty": 1.0, "diversity": 0.5},
                            "min_per_topology": 1, "seed": 7},
            "stopping": {"max_rounds": 3},
            "validation": {"structural_checks": ["builder_sanity", "post_relaxation_slot_mapping", "residual_force_stress"],
                           "geometry_equivalence_tolerance_A": 0.1, "reference_phases": "pure_endpoints",
                           "reference_set_complete": False,
                           "supercell_transfer": {"n_replicated": 1, "n_native": 2, "seed": 3}}},
        "tolerances": {"consecutive_stable_rounds": 1},
    }


def write_toy_milestone_c_project(dst, overrides=None, mutate=None) -> Path:
    """Two SYNTHETIC candidate structures with the same formula unit: toy_pcu (2x2x1 working cell, 2x2x2 transfer
    cell) and its staggered polymorph toy_pcu_staggered (2x2x1 working cell only)."""
    dst = Path(dst)
    (dst / "structures").mkdir(parents=True, exist_ok=True)
    staggered = staggered_cif()
    (dst / "structures/toy_pcu_staggered.cif").write_text(staggered)

    def add_second(spec):
        spec["source_structures"].append({"topology_id": "toy_pcu_staggered", "cif": "structures/toy_pcu_staggered.cif",
                                          "sha256": sha(staggered), "source": "tests/toy_framework.py (generated, SYNTHETIC)",
                                          "declared_net": "pcu", "supercell_matrices": [[[2, 0, 0], [0, 2, 0], [0, 0, 1]]]})
        if mutate:
            mutate(spec)
    return write_toy_project(dst, _merge(milestone_c_overrides(sha(staggered)), overrides or {}), mutate=add_second)


# -- Milestone E: finite-cluster subtractive embedding with a mock cluster engine ---------------------------------
def embedding_protocol(dst, bond_hops=1, series=(1, 2), benchmarks=(), low_xc="PBE"):
    """SYNTHETIC embedding protocol: Zn-node clusters capped with H, evaluated by tests/mock_cluster.py."""
    return {"protocol_id": "toy_subtractive_mock", "scheme": "finite_cluster_subtractive", "engine": "external",
            "command": [sys.executable, str((Path(dst) / "tools/mock_cluster.py").resolve())],
            "region": {"bond_hops": bond_hops}, "capping": {"element": "H", "distance_A": {"Zn": 1.60, "O": 0.97, "C": 1.09}},
            "cluster_charge": -4, "cluster_spin": 0, "basis": "def2-svp",
            "low_level": {"method": "DFT", "xc": low_xc, "dispersion": "grimme-d3"}, "high_level": {"method": "CCSD"},
            "conv_tol_Ha": 1e-9, "n_cores": 1, "timeout_s": 60,
            "validation": {"bond_hops_series": list(series), "reference_benchmarks": list(benchmarks)}}


def write_toy_embedding_project(dst, **protocol) -> Path:
    """The Milestone B toy (one 2x2x1 cell, four Zn nodes) with protocols.embedding and a cluster-job budget."""
    dst = Path(dst)
    (dst / "tools").mkdir(parents=True, exist_ok=True)
    shutil.copy(HERE / "mock_cluster.py", dst / "tools/mock_cluster.py")
    return write_toy_project(dst, {"protocols": {"embedding": embedding_protocol(dst, **protocol)},
                                   "budgets": {"max_active_space_jobs": 50}})
