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
from .dft import dft_protocol_errors
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
    "protocols.embedding": ("UNSUPPORTED_ELECTRONIC_BACKEND", "T07.1"),
    "protocols.active_space": ("UNSUPPORTED_ELECTRONIC_BACKEND", "T07.1"),
    "protocols.quantum_solvers": ("UNSUPPORTED_QUANTUM_BACKEND", "T04.2"),
    "protocols.stopping": ("UNSUPPORTED_ACTIVE_LEARNING_MODE", "T08.2"),
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
    if len(search.get("supercell_matrices") or []) > 1:
        errors.append(_err("UNSUPPORTED_SUPERCELL_MATRIX", "search.supercell_matrices",
                           "one working cell in this build; transfer cells are T01.5"))
    if search.get("symmetry_policy_id") not in (None, "cif_space_group_plus_supercell_translations"):
        errors.append(_err("UNSUPPORTED_SYMMETRY_POLICY", "search.symmetry_policy_id", str(search["symmetry_policy_id"])))
    if len(config.get("candidate_topology_ids") or []) > 1:
        errors.append(_err("UNSUPPORTED_MULTIPLE_TOPOLOGIES", "candidate_topology_ids", "second topology is Milestone C"))

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
    if (protocols.get("classical_solvers") or ["exact_enumeration"]) != ["exact_enumeration"]:
        errors.append(_err("UNSUPPORTED_SOLVER", "protocols.classical_solvers", str(protocols["classical_solvers"])))
    if (config.get("runtime") or {}).get("engine") not in (None, "local", "aiida_recommended"):
        errors.append(_err("UNSUPPORTED_WORKFLOW_ENGINE", "runtime.engine", str(config["runtime"]["engine"])))

    blocks = {b["id"]: b for b in config.get("building_blocks") or [] if b.get("id")}
    topologies, physical_topologies, sources = {}, {}, []
    structural_inputs_ok = not any(e["check"].split(".")[0] in ("source_structures", "building_blocks", "chemistry", "search")
                                   or e["check"].startswith("protocols.builder") for e in errors)
    if structural_inputs_ok:
        for i, src in enumerate(config["source_structures"]):
            topo, lattice, err_i, record = _load_source(root, i, src, config, blocks, builder)
            errors += err_i
            sources.append(record)
            if topo is not None:
                topologies[topo.topology_id], physical_topologies[topo.topology_id] = topo, lattice
    elements = set()
    for info in physical_topologies.values():
        elements |= set(info["target_inventory"])
        elements |= {a["element"] for a in info["structure"]["atoms"]}
    if protocols.get("dft") and physical_topologies:
        errors += dft_protocol_errors(protocols["dft"], root, elements)

    species = list(next(iter((chem.get("composition_by_sublattice") or {"_": {}}).values())).keys())
    encoding_map = {s: i for i, s in enumerate(species)} if len(species) == 2 else {}
    if species and len(species) != 2:
        errors.append(_err("UNSUPPORTED_MULTICOMPONENT_ENCODING", "chemistry.composition_by_sublattice", str(species)))
    if len(chem.get("composition_by_sublattice") or {}) > 1:
        errors.append(_err("UNSUPPORTED_MULTICOMPONENT_ENCODING", "chemistry.composition_by_sublattice",
                           "one substituted sublattice in this build (T03.5)"))
    first = next(iter(physical_topologies.values()), None)
    counts = first["species_counts"] if first else {}
    n_sites = len(first["slots"]) if first else 0
    if n_sites > MAX_EXACT_SITES:
        errors.append(_err("UNSUPPORTED_CELL_TOO_LARGE_FOR_EXACT_SOLVER", "search.supercell_matrices",
                           f"{n_sites} slots > {MAX_EXACT_SITES}; needs T04.1/T03.5"))
    from .configuration_model import site_orbits
    candidates, feature_order = _model_candidates(selection, len(site_orbits(next(iter(topologies.values()))))
                                                  if topologies else 0)
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
        "optimizer": {"method": "exact_enumeration", "restrict_to_composition": True, "quantum_enabled": False},
        "qubo": {"format": encoding.get("format", "upper_triangle_once"), "ratio_penalty_eV": encoding.get("ratio_penalty_eV"),
                 "higher_order": encoding.get("higher_order", "retain_HUBO_or_explicitly_reject")},
        "deduplication": {"key": "topology_id:lexicographically_smallest_symmetry_image",
                          "allow_species_label_exchange": False},
        "candidate_pool": {"top_k_unique_orbits_per_topology": acquisition.get("top_k_unique_orbits_per_topology"),
                           "retain_all_score_ties": True},
        "evaluation": {"backend": "periodic_dft", "correction_mode": "none", "dft_enabled": True,
                       "embedding_enabled": False, "hardware_enabled": False},
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
                  "lattices": physical_topologies,
                  "builder_info": {tid: lattice_info_for_builder(info, builder) for tid, info in physical_topologies.items()}},
    )
    pseudo_hashes = {el: p.get("sha256") for el, p in ((protocols.get("dft") or {}).get("pseudopotentials") or {}).items()}
    fingerprint = content_hash({"project_type": PROJECT_TYPE, "config": config, "sources": [s.get("sha256") for s in sources],
                                "pseudopotentials": pseudo_hashes})
    object.__setattr__(spec, "fingerprint", fingerprint)
    return spec


def _model_candidates(selection: dict, n_site_orbits: int) -> tuple[list, list]:
    """Candidate grid in the fixture's (feature_order, feature_indices) form. ``site_orbits`` expands to one
    point feature per symmetry orbit of slots; at fixed composition one of them is redundant with the constant,
    which the documented gauge (rank_deficiency) resolves."""
    from .configuration_model import FEATURE_DEGREE, SITE_ORBIT_PREFIX, feature_degree, is_known_feature

    expanded = []
    for c in selection.get("candidates") or []:
        feats = []
        for f in c.get("features", []):
            feats += [f"{SITE_ORBIT_PREFIX}{k}" for k in range(n_site_orbits)] if f == "site_orbits" else [f]
        expanded.append((c.get("id"), feats))
    names = {f for _, feats in expanded for f in feats}
    order = [f for f in FEATURE_DEGREE if f in names]
    order += sorted((f for f in names if f.startswith(SITE_ORBIT_PREFIX) and is_known_feature(f)),
                    key=lambda f: int(f[len(SITE_ORBIT_PREFIX):]))
    order += sorted(names - set(order))  # unknown names: rejected by validate_physical_project
    out = [{"id": cid, "feature_indices": [order.index(f) for f in feats], "n_parameters": len(feats),
            "polynomial_degree": max((feature_degree(f) for f in feats if is_known_feature(f)), default=0)}
           for cid, feats in expanded]
    return out, order


def _load_source(root: Path, index: int, src: dict, config: dict, blocks: dict, builder: dict):
    from .intake import TopologySpec

    prefix = f"source_structures[{index}]"
    errors = [_err("SCIENTIFIC_SPEC_MISSING", f"{prefix}.{k}") for k in ("topology_id", "cif", "sha256", "source")
              if _missing(src.get(k))]
    if errors:
        return None, None, errors, {}
    path = (root / src["cif"]).resolve()
    if root.resolve() not in path.parents or not path.exists():
        return None, None, [_err("MISSING_INPUT_FILE", f"{prefix}.cif", src["cif"])], {}
    text = path.read_text()
    try:
        parent = parse_cif(text, src["source"])
    except PipelineError as exc:
        return None, None, [_err(exc.code, f"{prefix}.cif", exc.detail)], {}
    record = {"topology_id": src["topology_id"], "cif": src["cif"], "sha256": parent["sha256"], "source": src["source"],
              "n_atoms_p1": len(parent["atoms"]), "n_symmetry_operations": parent["n_symmetry_operations"],
              "declared_net": src.get("declared_net"), "net_verification": "NOT_RUN (CrystalNets/MOFid is T09.1)"}
    if parent["sha256"] != src["sha256"]:
        return None, None, [_err("CHECKSUM_MISMATCH", f"{prefix}.sha256", f"{parent['sha256']} != {src['sha256']}")], record
    if parent["partial_occupancy_sites"] or parent["disordered_sites"]:
        # require_fully_ordered is the only supported policy: unresolved disorder is never filled silently.
        return None, None, [_err("DISORDER_UNRESOLVED", f"{prefix}.cif",
                                 f"partial occupancy {parent['partial_occupancy_sites']}, disorder groups "
                                 f"{parent['disordered_sites']} under {config['chemistry']['disorder_resolution_policy']}")], record
    tid = src["topology_id"]
    matrix = config["search"]["supercell_matrices"][0]
    try:
        lattice = build_slot_lattice(parent, tid, matrix, config["chemistry"], blocks, builder)
    except PipelineError as exc:
        return None, None, [_err(exc.code, prefix, exc.detail)], record
    if lattice["errors"]:
        return None, None, lattice["errors"], record
    graph = lattice["graph"]
    species = sorted({s for fr in config["chemistry"]["composition_by_sublattice"].values() for s in fr})
    topo = TopologySpec(
        topology_id=tid, representation=GRAPH_REPRESENTATION, n_sites=len(lattice["slots"]),
        site_ids=tuple(range(len(lattice["slots"]))), allowed_species=tuple(tuple(species) for _ in lattice["slots"]),
        pairs_d1=tuple(graph["pairs_d1"]), pairs_d2=tuple(graph["pairs_d2"]), triplets=tuple(graph["triplets"]),
        permutations=tuple(lattice["permutations"]), n_formula_units=lattice["n_formula_units"],
        crystallographic_net_name=src.get("declared_net"), atomic_coordinates="parent_cif",
        provenance=config["provenance"], pair_weights_d1=tuple(graph["pair_weights_d1"]))
    record.update(n_slots=len(lattice["slots"]), n_nodes=len(lattice["nodes"]), n_symmetry_permutations=len(lattice["permutations"]),
                  removed_guests=lattice["removed_guests"], charge_check=lattice["net_charge_check"],
                  target_inventory_per_cell=lattice["target_inventory"], n_formula_units=lattice["n_formula_units"],
                  supercell_matrix=matrix)
    return topo, lattice, [], record


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
        from .configuration_model import is_known_feature
        for model in spec.model_candidates:
            names = [spec.feature_order[i] for i in model["feature_indices"]]
            ok = bool(names) and names[0] == "constant" and all(is_known_feature(n) for n in names)
            checks.append({"check": f"{model['id']}:features", "passed": ok})
            if not ok:
                errors.append(_err("UNSUPPORTED_FEATURE", f"protocols.model_selection.candidates.{model['id']}",
                                   "known features with an explicit leading constant term are required"))
        checks.append({"check": "single_topology", "passed": len(spec.topologies) == 1})
    status = StageStatus.COMPLETED if not errors else StageStatus.BLOCKED
    return make_record("ValidationReport", spec.provenance, scope="project", project_id=spec.project_id,
                       project_type=spec.project_type, status=status.value, n_checks=len(checks), checks=checks,
                       errors=errors, sources=spec.physical["sources"])
