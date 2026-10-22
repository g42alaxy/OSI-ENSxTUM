"""Task 01 (intake, physical projects): ProjectSpec v0.1 loader, recipe adapter and validator.

Loads ``instructions/project_spec.template.json``-shaped projects into the same
``ProjectSpec`` dataclass as the synthetic fixture. Every required scientific
field that is null, empty or absent becomes one BLOCKED error with code
SCIENTIFIC_SPEC_MISSING and its exact field path; nothing is defaulted. A
``physical_case/recipe.json`` is adapted into the same schema, so its missing
inputs are reported the same way.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from .cif import parse_cif
from .contract import PipelineError, StageStatus, content_hash, make_record
from .dft import dft_protocol_errors, tolerance_errors
from .framework import GRAPH_REPRESENTATION, build_slot_lattice, lattice_info_for_builder

if TYPE_CHECKING:
    from .intake import ProjectSpec

PHYSICAL_SCHEMA_VERSION = "0.1"
PROJECT_TYPE = "physical_structure"
PROVENANCES = ("SYNTHETIC", "PHYSICAL")
MAX_EXACT_SITES = 20  # exact enumeration and the 2^n encoding check; larger cells need T04.1 / T03.5

# Required for Milestone B (single binary topology, CIF input, DFT labels, validated fitted model).
# Optional fields belong to later milestones and must stay null/empty until implemented.
REQUIRED_FIELDS = (
    "project_id", "provenance", "source_structures", "candidate_topology_ids", "building_blocks",
    "chemistry.target_elemental_inventory", "chemistry.formula_unit", "chemistry.composition_by_sublattice",
    "chemistry.oxidation_and_protonation_policy", "chemistry.net_charge", "chemistry.solvent_and_counterion_policy",
    "chemistry.magnetic_state_policy", "chemistry.disorder_resolution_policy",
    "search.supercell_matrices", "search.symmetry_policy_id",
    "protocols.builder", "protocols.dft", "protocols.model_selection", "protocols.encoding",
    "protocols.classical_solvers", "protocols.acquisition", "protocols.validation",
    "tolerances.target_relative_energy_tolerance", "tolerances.force_tolerance_eV_per_angstrom",
    "tolerances.stress_tolerance_and_units", "tolerances.low_energy_validation_criterion",
    "budgets.max_dft_jobs", "budgets.max_cpu_core_hours",
)
NOT_YET_SUPPORTED = {  # field -> (code, task) when set to a non-empty value
    "search.orientation_catalog_id": ("UNSUPPORTED_ORIENTATION_CATALOG", "T01.4"),
    "protocols.ml_potential_checkpoint": ("UNSUPPORTED_ML_POTENTIAL", "T02.2"),
    "protocols.ml_calibration": ("UNSUPPORTED_ML_POTENTIAL", "T02.3"),
    # protocols.embedding (Milestone E) is validated by embedding.protocol_errors; an embedded cluster active space
    # is declared inside it (high_level.active_space). A stand-alone MOF active-space protocol is not implemented.
    "protocols.active_space": ("UNSUPPORTED_ELECTRONIC_BACKEND", "T07.1"),
    "runtime.llm_provider": ("AGENT_LAYER_NOT_IMPLEMENTED", "T10.2"),
    "runtime.llm_model": ("AGENT_LAYER_NOT_IMPLEMENTED", "T10.2"),
}
BLOCK_REQUIRED = {"node": ("id", "role", "formula", "formal_charge", "connectivity", "source", "license"),
                  "linker": ("id", "role", "formula", "formal_charge", "connectivity", "n_connector_atoms",
                             "slot_type", "source", "license")}
BUILDER_REQUIRED = ("builder_id", "bond_tolerance_A", "min_interatomic_distance_A", "slot_match_tolerance_A")
SELECTION_REQUIRED = ("candidates", "label_orbits_per_topology", "holdout_fraction", "n_folds", "seed",
                      "regularization", "rank_deficiency")
DISORDER_POLICIES = ("require_fully_ordered",)
SOLVENT_POLICIES = ("require_desolvated", "remove_unbound_guests")


def _get(d, path):
    for part in path.split("."):
        if not isinstance(d, dict) or part not in d:
            return None
        d = d[part]
    return d


def _missing(value) -> bool:
    return value is None or (isinstance(value, (list, dict, str)) and len(value) == 0)


def _err(code, path, detail=""):
    return {"code": code, "check": path, "detail": detail}


# -- recipe adapter -----------------------------------------------------------
_TOKEN = re.compile(r"\(([A-Za-z0-9_]+)\)(\d*)|([A-Z][a-z]?)(\d*)")


def parse_formula(text: str, blocks: dict | None = None) -> dict:
    """'C8H5NO4' or 'Zn8O2(BDC)3(NH2_BDC)3' (block IDs in brackets) -> element counts."""
    counts: dict = {}
    pos = 0
    for m in _TOKEN.finditer(text):
        if m.start() != pos:
            raise PipelineError("INVALID_FORMULA", text)
        pos = m.end()
        if m.group(1):
            if not blocks or m.group(1) not in blocks:
                raise PipelineError("INVALID_FORMULA", f"unknown block {m.group(1)} in {text}")
            mult, sub = int(m.group(2) or 1), blocks[m.group(1)]
        else:
            mult, sub = int(m.group(4) or 1), {m.group(3): 1}
        for el, n in sub.items():
            counts[el] = counts.get(el, 0) + n * mult
    if pos != len(text):
        raise PipelineError("INVALID_FORMULA", text)
    return dict(sorted(counts.items()))


def spec_from_recipe(recipe: dict) -> dict:
    """physical_case/recipe.json -> ProjectSpec v0.1 dict. Absent inputs stay null so validation names them."""
    node = recipe["metal_node"]
    blocks = [{"id": node["formula"], "role": "node", "formula": parse_formula(node["formula"]),
               "formal_charge": node["formal_charge"], "connectivity": node["connectivity"],
               "geometry_file": node.get("geometry_file"), "source": None, "license": None}]
    for i, l in enumerate(recipe["swappable_linkers"]):
        blocks.append({"id": l["id"], "role": "linker", "slot_type": "linker", "formula": parse_formula(l["formula"]),
                       "formal_charge": l["formal_charge"], "connectivity": 2, "n_connector_atoms": None,
                       "smiles": l.get("smiles"), "geometry_file": l.get("geometry_file"),
                       "substitution": None, "source": recipe.get("source_doi"), "license": None})
    formulas = {b["id"]: b["formula"] for b in blocks}
    inventory = parse_formula(recipe["minimum_stoichiometric_inventory"].split(";")[0].strip(), formulas)
    return {
        "schema_version": PHYSICAL_SCHEMA_VERSION, "status": recipe.get("status"), "project_id": recipe["case_id"],
        "provenance": "PHYSICAL", "recipe_case_id": recipe["case_id"],
        "source_structures": ([{"topology_id": recipe["candidate_net"], "cif": recipe["parent_cif"], "sha256": None,
                                "source": None}] if recipe.get("parent_cif") else []),
        "candidate_topology_ids": [recipe["candidate_net"]], "building_blocks": blocks,
        "chemistry": {"target_elemental_inventory": inventory,
                      "formula_unit": recipe["minimum_stoichiometric_inventory"].split(";")[0].strip(),
                      "composition_by_sublattice": {"linker": recipe["target_linker_fractions"]},
                      "oxidation_and_protonation_policy": "formal charges supplied per building block (recipe)",
                      "net_charge": 0, "solvent_and_counterion_policy": None, "magnetic_state_policy": None,
                      "disorder_resolution_policy": None},
        "search": {"supercell_matrices": [], "orientation_catalog_id": None, "symmetry_policy_id": None},
        "protocols": {}, "tolerances": {}, "budgets": {}, "runtime": {},
    }


# -- loader -------------------------------------------------------------------
def _resolve_protocols(config: dict, root: Path, errors: list) -> dict:
    """A protocol is an inline object or a project-relative JSON path (registry entry)."""
    protocols = dict(config.get("protocols") or {})
    for name, value in protocols.items():
        if isinstance(value, str) and value.endswith(".json"):
            path = (root / value).resolve()
            if root.resolve() not in path.parents or not path.exists():
                errors.append(_err("MISSING_INPUT_FILE", f"protocols.{name}", value))
                protocols[name] = None
            else:
                protocols[name] = json.loads(path.read_text())
    return protocols


def _block_errors(blocks: list, root: Path) -> list:
    errors, ids = [], set()
    for i, b in enumerate(blocks):
        role = b.get("role")
        if role not in BLOCK_REQUIRED:
            errors.append(_err("SCIENTIFIC_SPEC_MISSING", f"building_blocks[{i}].role", str(role)))
            continue
        for key in BLOCK_REQUIRED[role]:
            if _missing(b.get(key)) and not (key == "formal_charge" and b.get(key) == 0):
                errors.append(_err("SCIENTIFIC_SPEC_MISSING", f"building_blocks[{i}].{key}"))
        if b.get("id") in ids:
            errors.append(_err("DUPLICATE_BUILDING_BLOCK", f"building_blocks[{i}].id", str(b.get("id"))))
        ids.add(b.get("id"))
    by_id = {b.get("id"): b for b in blocks}
    for i, b in enumerate(blocks):
        sub = b.get("substitution")
        # A linker's 3D geometry comes either from the parent CIF or from an explicit substitution rule on a
        # parent linker; SMILES alone never define a building block.
        if b.get("role") == "linker" and not sub and b.get("in_parent_cif") is not True:
            errors.append(_err("SCIENTIFIC_SPEC_MISSING", f"building_blocks[{i}].substitution",
                               "declare in_parent_cif: true or an explicit substitution rule; SMILES are not enough"))
        if sub:
            parent = by_id.get(sub.get("parent_block"))
            if parent is None or not sub.get("group") or not sub.get("replace_element"):
                errors.append(_err("INVALID_SUBSTITUTION_RULE", f"building_blocks[{i}].substitution"))
    return errors


def load_physical_project(project_path: Path, config: dict) -> ProjectSpec:
    from .intake import ProjectSpec

    root = project_path.parent
    errors: list = []
    if "case_id" in config and "metal_node" in config:
        config = spec_from_recipe(config)
    protocols = _resolve_protocols(config, root, errors)
    config = {**config, "protocols": protocols}
    for path in REQUIRED_FIELDS:
        if _missing(_get(config, path)) and not (path == "chemistry.net_charge" and _get(config, path) == 0):
            errors.append(_err("SCIENTIFIC_SPEC_MISSING", path))
    for path, (code, task) in NOT_YET_SUPPORTED.items():
        if not _missing(_get(config, path)):
            errors.append(_err(code, path, f"not implemented in this build ({task})"))
    if config.get("provenance") is not None and config["provenance"] not in PROVENANCES:
        errors.append(_err("PROVENANCE_MISMATCH", "provenance", str(config["provenance"])))
    errors += _block_errors(config.get("building_blocks") or [], root)

    chem, search = config.get("chemistry") or {}, config.get("search") or {}
    if chem.get("disorder_resolution_policy") not in (None, *DISORDER_POLICIES):
        errors.append(_err("UNSUPPORTED_DISORDER_POLICY", "chemistry.disorder_resolution_policy",
                           str(chem["disorder_resolution_policy"])))
    if chem.get("solvent_and_counterion_policy") not in (None, *SOLVENT_POLICIES):
        errors.append(_err("UNSUPPORTED_SOLVENT_POLICY", "chemistry.solvent_and_counterion_policy",
                           str(chem["solvent_and_counterion_policy"])))
    states = (chem.get("magnetic_state_policy") or {}).get("initial_states") if isinstance(chem.get("magnetic_state_policy"), dict) else None
    if chem.get("magnetic_state_policy") is not None and not states:
        errors.append(_err("SCIENTIFIC_SPEC_MISSING", "chemistry.magnetic_state_policy.initial_states"))
    if search.get("symmetry_policy_id") not in (None, "cif_space_group_plus_supercell_translations"):
        errors.append(_err("UNSUPPORTED_SYMMETRY_POLICY", "search.symmetry_policy_id", str(search["symmetry_policy_id"])))
    declared = list(config.get("candidate_topology_ids") or [])
    supplied = [s.get("topology_id") for s in config.get("source_structures") or [] if isinstance(s, dict)]
    if declared and supplied and sorted(declared) != sorted(supplied):
        errors.append(_err("TOPOLOGY_SOURCE_MISMATCH", "candidate_topology_ids",
                           f"declared {sorted(declared)}, source structures for {sorted(supplied)}"))
    if len(set(declared)) != len(declared):
        errors.append(_err("DUPLICATE_TOPOLOGY_ID", "candidate_topology_ids", str(declared)))

    builder = protocols.get("builder") or {}
    if protocols.get("builder"):
        errors += [_err("SCIENTIFIC_SPEC_MISSING", f"protocols.builder.{k}") for k in BUILDER_REQUIRED if builder.get(k) is None]
    selection = protocols.get("model_selection") or {}
    if protocols.get("model_selection"):
        errors += [_err("SCIENTIFIC_SPEC_MISSING", f"protocols.model_selection.{k}") for k in SELECTION_REQUIRED
                   if selection.get(k) is None]
    encoding = protocols.get("encoding") or {}
    if protocols.get("encoding") and not (isinstance(encoding.get("ratio_penalty_eV"), (int, float))
                                          and not isinstance(encoding.get("ratio_penalty_eV"), bool)
                                          and encoding["ratio_penalty_eV"] > 0):
        errors.append(_err("SCIENTIFIC_SPEC_MISSING", "protocols.encoding.ratio_penalty_eV"))
    acquisition = protocols.get("acquisition") or {}
    if protocols.get("acquisition") and type(acquisition.get("top_k_unique_orbits_per_topology")) is not int:
        errors.append(_err("SCIENTIFIC_SPEC_MISSING", "protocols.acquisition.top_k_unique_orbits_per_topology"))
    stopping = protocols.get("stopping")
    if stopping:
        from .acquisition import policy_errors
        from .refinement import stopping_errors
        errors += [_err("SCIENTIFIC_SPEC_MISSING", path) for path in stopping_errors(stopping, config.get("tolerances") or {})]
        errors += [_err("SCIENTIFIC_SPEC_MISSING", f"protocols.acquisition.{k}") for k in policy_errors(acquisition)]
    validation_policy = protocols.get("validation") or {}
    if validation_policy.get("reference_phases") not in (None, "pure_endpoints"):
        errors.append(_err("UNSUPPORTED_REFERENCE_SET", "protocols.validation.reference_phases",
                           str(validation_policy["reference_phases"])))
    tol_geo = validation_policy.get("geometry_equivalence_tolerance_A")
    if tol_geo is not None and (isinstance(tol_geo, bool) or not isinstance(tol_geo, (int, float)) or tol_geo <= 0):
        errors.append(_err("INVALID_NUMBER", "protocols.validation.geometry_equivalence_tolerance_A", str(tol_geo)))
    solvers = protocols.get("classical_solvers") or ["exact_enumeration"]
    if solvers[0] != "exact_enumeration" or not set(solvers) <= {"exact_enumeration", "branch_and_bound"}:
        errors.append(_err("UNSUPPORTED_SOLVER", "protocols.classical_solvers", str(protocols["classical_solvers"])))
    from .qaoa import protocol_errors as quantum_errors
    quantum = protocols.get("quantum_solvers") or []
    if len(quantum) > 1:
        errors.append(_err("UNSUPPORTED_QUANTUM_BACKEND", "protocols.quantum_solvers", "one QAOA simulation protocol"))
    for i, q in enumerate(quantum[:1]):
        errors += [_err(code, f"protocols.quantum_solvers[{i}].{k}") for code, k in quantum_errors(q)]
    if (protocols.get("encoding") or {}).get("higher_order", "retain_HUBO_or_explicitly_reject") not in (
            "retain_HUBO_or_explicitly_reject", "quadratize"):
        errors.append(_err("UNSUPPORTED_ENCODING_POLICY", "protocols.encoding.higher_order",
                           str(protocols["encoding"]["higher_order"])))
    if protocols.get("embedding"):
        from .embedding import protocol_errors as embedding_errors
        errors += [_err(code, path, detail) for code, path, detail in
                   embedding_errors(protocols["embedding"], protocols.get("dft") or {}, config.get("budgets") or {}, root)]
    if (config.get("runtime") or {}).get("engine") not in (None, "local", "aiida_recommended"):
        errors.append(_err("UNSUPPORTED_WORKFLOW_ENGINE", "runtime.engine", str(config["runtime"]["engine"])))

    blocks = {b["id"]: b for b in config.get("building_blocks") or [] if b.get("id")}
    topologies, physical_topologies, transfer_topologies, sources = {}, {}, {}, []
    structural_inputs_ok = not any(e["check"].split(".")[0] in ("source_structures", "building_blocks", "chemistry", "search")
                                   or e["check"].startswith("protocols.builder") for e in errors)
    from .clusters import requested_cutoff
    cutoff = requested_cutoff(selection.get("candidates") or [])
    if structural_inputs_ok:
        for i, src in enumerate(config["source_structures"]):
            cells, err_i, record = _load_source(root, i, src, config, blocks, builder, cutoff)
            errors += err_i
            sources.append(record)
            if cells:
                (topo, lattice), transfers = cells[0], cells[1:]
                topologies[topo.topology_id], physical_topologies[topo.topology_id] = topo, lattice
                transfer_topologies[topo.topology_id] = [t for t, _ in transfers]
                physical_topologies.update({t.topology_id: lat for t, lat in transfers})
    elements = set()
    for info in physical_topologies.values():
        elements |= set(info["target_inventory"])
        elements |= {a["element"] for a in info["structure"]["atoms"]}
    if protocols.get("dft") and physical_topologies:
        lattices = [info["structure"]["lattice"] for info in physical_topologies.values()]
        errors += dft_protocol_errors(protocols["dft"], root, elements, lattices)
    errors += tolerance_errors(config.get("tolerances") or {})

    species = list(next(iter((chem.get("composition_by_sublattice") or {"_": {}}).values())).keys())
    encoding_map = {s: i for i, s in enumerate(species)} if len(species) == 2 else {}
    if species and len(species) != 2:
        errors.append(_err("UNSUPPORTED_MULTICOMPONENT_ENCODING", "chemistry.composition_by_sublattice", str(species)))
    if len(chem.get("composition_by_sublattice") or {}) > 1:
        errors.append(_err("UNSUPPORTED_MULTICOMPONENT_ENCODING", "chemistry.composition_by_sublattice",
                           "one substituted sublattice in this build (T03.5)"))
    if any(transfer_topologies.values()):
        policy = validation_policy.get("supercell_transfer") or {}
        for key in ("n_replicated", "n_native", "seed"):
            if policy.get(key) is None:
                errors.append(_err("SCIENTIFIC_SPEC_MISSING", f"protocols.validation.supercell_transfer.{key}",
                                   "transfer supercells are declared"))
    first = next(iter(topologies.values()), None)
    counts = dict(first.composition) if first else {}
    n_sites = first.n_sites if first else 0
    for tid, topo in topologies.items():
        if topo.n_sites > MAX_EXACT_SITES:
            errors.append(_err("UNSUPPORTED_CELL_TOO_LARGE_FOR_EXACT_SOLVER", f"search.supercell_matrices ({tid})",
                               f"{topo.n_sites} slots > {MAX_EXACT_SITES}; needs T04.1/T03.5"))
    candidates, feature_order = _model_candidates(selection), []
    tolerance = _get(config, "tolerances.target_relative_energy_tolerance")
    normalized = {
        "schema": "project_spec_v0.1", "project_spec": config, "project_id": config.get("project_id"),
        "provenance": config.get("provenance"), "random_seed": selection.get("seed"),
        "runtime_agents": {"enabled": False, "llm_calls_allowed": False},
        "finite_temperature": {"enabled": False},
        "topology_screen": {"mode": "identity", "retain": sorted(topologies), "geometric_screen": "NOT_RUN"},
        "model_selection": {"mode": "grouped_cross_validation", "split_unit": "symmetry_orbit",
                            "max_development_rmse_eV": tolerance, "holdout_access_during_selection": False,
                            "no_passing_model": "MODEL_INADEQUATE",
                            "policy": "Select the fewest effective parameters among models passing grouped CV; "
                                      "then lowest CV RMSE; then model ID"},
        "optimizer": {"method": "exact_enumeration", "restrict_to_composition": True,
                      "cross_check_solvers": [s for s in (protocols.get("classical_solvers") or []) if s != "exact_enumeration"],
                      "quantum_enabled": bool(protocols.get("quantum_solvers")),
                      "quantum_protocol": (protocols.get("quantum_solvers") or [None])[0]},
        "qubo": {"format": encoding.get("format", "upper_triangle_once"), "ratio_penalty_eV": encoding.get("ratio_penalty_eV"),
                 "higher_order": encoding.get("higher_order", "retain_HUBO_or_explicitly_reject")},
        "deduplication": {"key": "topology_id:lexicographically_smallest_symmetry_image",
                          "allow_species_label_exchange": False},
        "candidate_pool": {"top_k_unique_orbits_per_topology": acquisition.get("top_k_unique_orbits_per_topology"),
                           "retain_all_score_ties": True},
        "evaluation": {"backend": "periodic_dft", "correction_mode": "none", "dft_enabled": True,
                       "embedding_enabled": bool(protocols.get("embedding")), "hardware_enabled": False},
        "active_learning": {"mode": "disabled_milestone_B", "max_iterations_if_enabled": 0},
        "literature": {"mode": "static_manifest", "live_review_enabled": False},
        "inputs": {},
    }
    spec = ProjectSpec(
        project_id=config.get("project_id") or "UNSPECIFIED", provenance=config.get("provenance") or "UNSPECIFIED",
        root=root, config=normalized, case="physical", correction_mode="none", topologies=topologies,
        species_encoding=encoding_map, composition_counts=counts, n_sites=n_sites, feature_order=feature_order,
        model_candidates=candidates, training_rows=[], development_rows=[], split_manifest={}, candidate_rows=[],
        building_blocks={"provenance": config.get("provenance"), "blocks": list(blocks.values())}, literature={},
        project_type=PROJECT_TYPE,
        physical={"errors": errors, "sources": sources, "protocols": protocols, "blocks": blocks,
                  "chemistry": chem, "tolerances": config.get("tolerances") or {}, "budgets": config.get("budgets") or {},
                  "lattices": physical_topologies, "transfer_topologies": transfer_topologies,
                  "builder_info": {tid: lattice_info_for_builder(info, builder) for tid, info in physical_topologies.items()}},
    )
    pseudo_hashes = {el: p.get("sha256") for el, p in ((protocols.get("dft") or {}).get("pseudopotentials") or {}).items()}
    fingerprint = content_hash({"project_type": PROJECT_TYPE, "config": config, "sources": [s.get("sha256") for s in sources],
                                "pseudopotentials": pseudo_hashes})
    object.__setattr__(spec, "fingerprint", fingerprint)
    return spec


def _model_candidates(selection: dict) -> list:
    """Candidate models as feature tokens, expanded per topology (configuration_model.model_features).

    Tokens: 'constant', fixed features ('unlike_distance_1', 'unlike_distance_2', 'BBB_paths'), 'site_orbits'
    (one point feature per slot orbit of the working cell), and the transferable parent-lattice cluster basis
    'point_orbits' and 'pairs<=R' (every pair orbit with slot-centroid distance <= R angstrom). At fixed
    composition one point feature is redundant with the constant; the documented gauge (rank_deficiency) drops it.
    """
    return [{"id": c.get("id"), "tokens": list(c.get("features", []))} for c in selection.get("candidates") or []]


def _cell_label(matrix) -> str:
    return "x".join(str(matrix[i][i]) for i in range(3))


def _load_source(root: Path, index: int, src: dict, config: dict, blocks: dict, builder: dict, cutoff):
    """One source structure -> [(TopologySpec, lattice)] for the working cell then every transfer cell.

    The first supercell matrix (``source_structures[i].supercell_matrices`` or ``search.supercell_matrices``) is
    the working cell that is enumerated, fitted and searched; further matrices are transfer cells (T01.5) whose
    TopologySpec ids are ``<topology_id>@AxBxC``. Cluster orbits (``cutoff``) come from the parent lattice, so
    every cell of one topology shares the same feature names.
    """
    prefix = f"source_structures[{index}]"
    errors = [_err("SCIENTIFIC_SPEC_MISSING", f"{prefix}.{k}") for k in ("topology_id", "cif", "sha256", "source")
              if _missing(src.get(k))]
    if errors:
        return [], errors, {}
    path = (root / src["cif"]).resolve()
    if root.resolve() not in path.parents or not path.exists():
        return [], [_err("MISSING_INPUT_FILE", f"{prefix}.cif", src["cif"])], {}
    text = path.read_text()
    try:
        parent = parse_cif(text, src["source"])
    except PipelineError as exc:
        return [], [_err(exc.code, f"{prefix}.cif", exc.detail)], {}
    record = {"topology_id": src["topology_id"], "cif": src["cif"], "sha256": parent["sha256"], "source": src["source"],
              "n_atoms_p1": len(parent["atoms"]), "n_symmetry_operations": parent["n_symmetry_operations"],
              "declared_net": src.get("declared_net"), "net_verification": "NOT_RUN (CrystalNets/MOFid adapters, T09.1)"}
    if parent["sha256"] != src["sha256"]:
        return [], [_err("CHECKSUM_MISMATCH", f"{prefix}.sha256", f"{parent['sha256']} != {src['sha256']}")], record
    if parent["partial_occupancy_sites"] or parent["disordered_sites"]:
        # require_fully_ordered is the only supported policy: unresolved disorder is never filled silently.
        return [], [_err("DISORDER_UNRESOLVED", f"{prefix}.cif",
                         f"partial occupancy {parent['partial_occupancy_sites']}, disorder groups "
                         f"{parent['disordered_sites']} under {config['chemistry']['disorder_resolution_policy']}")], record
    tid = src["topology_id"]
    matrices = src.get("supercell_matrices") or config["search"]["supercell_matrices"]
    cells, catalogue = [], None
    for k, matrix in enumerate(matrices):
        cell_id = tid if k == 0 else f"{tid}@{_cell_label(matrix)}"
        try:
            lattice = build_slot_lattice(parent, cell_id, matrix, config["chemistry"], blocks, builder)
        except PipelineError as exc:
            return [], [_err(exc.code, f"{prefix}.supercell_matrices[{k}]", exc.detail)], record
        if lattice["errors"]:
            return [], lattice["errors"], record
        if cutoff is not None:
            from .clusters import build_catalogue, supercell_clusters
            try:
                catalogue = catalogue or build_catalogue(lattice, cutoff)
                cluster_terms = supercell_clusters(lattice, catalogue)
            except PipelineError as exc:
                return [], [_err(exc.code, f"{prefix}.supercell_matrices[{k}]", exc.detail)], record
        else:
            cluster_terms = ()
        cells.append((_topology_spec(cell_id, lattice, src, config, cluster_terms, None if k == 0 else tid), lattice))
    working = cells[0][1]
    record.update(n_slots=len(working["slots"]), n_nodes=len(working["nodes"]),
                  n_symmetry_permutations=len(working["permutations"]), removed_guests=working["removed_guests"],
                  charge_check=working["net_charge_check"], target_inventory_per_cell=working["target_inventory"],
                  n_formula_units=working["n_formula_units"], supercell_matrix=matrices[0],
                  transfer_cells=[{"topology_id": t.topology_id, "n_slots": t.n_sites, "n_formula_units": t.n_formula_units,
                                   "supercell": list(t.supercell)} for t, _ in cells[1:]],
                  cluster_catalogue=({"cutoff_A": catalogue["cutoff_A"], "n_point_orbits": len(catalogue["point_orbits"]),
                                      "pair_orbits": [{k: p[k] for k in ("name", "distance_A")} for p in catalogue["pair_orbits"]]}
                                     if catalogue else None))
    return cells, [], record


def _topology_spec(cell_id: str, lattice: dict, src: dict, config: dict, cluster_terms: tuple, parent_tid):
    from .intake import TopologySpec

    graph = lattice["graph"]
    species = list(next(iter(config["chemistry"]["composition_by_sublattice"].values())).keys())
    counts = lattice["species_counts"]
    return TopologySpec(
        topology_id=cell_id, representation=GRAPH_REPRESENTATION, n_sites=len(lattice["slots"]),
        site_ids=tuple(range(len(lattice["slots"]))), allowed_species=tuple(tuple(sorted(species)) for _ in lattice["slots"]),
        pairs_d1=tuple(graph["pairs_d1"]), pairs_d2=tuple(graph["pairs_d2"]), triplets=tuple(graph["triplets"]),
        permutations=tuple(lattice["permutations"]), n_formula_units=lattice["n_formula_units"],
        crystallographic_net_name=src.get("declared_net"), atomic_coordinates="parent_cif",
        provenance=config["provenance"], pair_weights_d1=tuple(graph["pair_weights_d1"]),
        n_ones=counts.get(species[1], 0) if len(species) == 2 else None,
        composition=tuple(sorted(counts.items())), clusters=cluster_terms,
        supercell=tuple(lattice["structure"]["supercell_diagonal"]), parent_topology_id=parent_tid)


# -- validator ----------------------------------------------------------------
def validate_physical_project(spec) -> dict:
    """validate_project for physical projects: spec completeness + structural checks, all before any calculation."""
    from .intake import topology_checks

    checks, errors = [], list(spec.physical["errors"])
    for e in errors:
        checks.append({"check": e["check"], "passed": False})
    if not errors:
        for tid, topo in spec.topologies.items():
            local_checks, local_errors = topology_checks(spec, topo)
            checks += local_checks
            errors += local_errors
        from .configuration_model import is_known_feature, is_transferable, model_features
        transfers = spec.physical.get("transfer_topologies") or {}
        for model in spec.model_candidates:
            for tid, topo in spec.topologies.items():
                reason = "known features with an explicit leading constant term are required"
                try:
                    names = model_features(spec, topo, model)
                except PipelineError as exc:
                    names, reason = [], f"{exc.code}: {exc.detail}"
                ok = bool(names) and names[0] == "constant" and all(is_known_feature(n) for n in names)
                checks.append({"check": f"{model['id']}:{tid}:features", "passed": ok})
                if not ok:
                    errors.append(_err("UNSUPPORTED_FEATURE", f"protocols.model_selection.candidates.{model['id']}",
                                       f"{tid}: {reason}"))
                elif transfers.get(tid) and not all(is_transferable(n) for n in names):
                    # A transfer cell can only test features defined on the parent lattice per formula unit.
                    errors.append(_err("NONTRANSFERABLE_FEATURES", f"protocols.model_selection.candidates.{model['id']}",
                                       f"{tid} has transfer cells; use 'point_orbits' / 'pairs<=R' instead of "
                                       f"{[n for n in names if not is_transferable(n)]}"))
        for tid, cells in transfers.items():
            for cell in cells:
                local_checks, local_errors = topology_checks(spec, cell)
                checks += local_checks
                errors += [e for e in local_errors if e["code"] != "INVALID_SITE_COUNT"]
        checks.append({"check": "topologies", "passed": len(spec.topologies) >= 1})
    status = StageStatus.COMPLETED if not errors else StageStatus.BLOCKED
    return make_record("ValidationReport", spec.provenance, scope="project", project_id=spec.project_id,
                       project_type=spec.project_type, status=status.value, n_checks=len(checks), checks=checks,
                       errors=errors, sources=spec.physical["sources"])
