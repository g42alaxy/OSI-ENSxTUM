"""Task 01 (intake): fixture adapter, project loader dispatch and strict project validator.

Input mode ``finite_substitution_interaction_graph`` bypasses CIF parsing and
atomic assembly. Only learner-visible fixture files are read; ``harness/``,
``reference_outputs/`` and the oracle sources are refused. Physical projects
(ProjectSpec v0.1 or a physical_case recipe) are loaded by ``project_spec``
into the same dataclass, with ``project_type`` kept separate.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, replace
from pathlib import Path

from .contract import PipelineError, StageStatus, content_hash, make_record

FIXTURE_SCHEMA = "mof_no_agent_fixture_v1"
GRAPH_MODE = "finite_substitution_interaction_graph"
LEARNER_FORBIDDEN_DIRS = ("harness", "reference_outputs")
LEARNER_FORBIDDEN_FILES = ("tools/generate_fixture.py", "tools/fixture_core.py")
CORRECTION_MODES = ("none", "constant", "topology_dependent")
DEFAULT_SPLIT_MANIFEST = "data/split_manifest.json"  # fixture only: project.json is checksummed and omits it
FIXTURE_PROJECT_TYPE = "synthetic_fixture"


@dataclass(frozen=True)
class TopologySpec:
    topology_id: str
    representation: str
    n_sites: int
    site_ids: tuple
    allowed_species: tuple
    pairs_d1: tuple
    pairs_d2: tuple
    triplets: tuple
    permutations: tuple
    n_formula_units: int
    crystallographic_net_name: object
    atomic_coordinates: object
    provenance: str
    pair_weights_d1: tuple = ()  # periodic contact multiplicity per pairs_d1 entry; empty means 1 each
    n_ones: int | None = None  # sites carrying encoding 1 in this cell; None = the project-wide count (fixture)
    composition: tuple = ()  # (species, count) in this cell; empty = the project-wide composition (fixture)
    clusters: tuple = ()  # parent-lattice cluster orbits counted in this cell (clusters.supercell_clusters)
    supercell: tuple = ()  # diagonal supercell of the parent cell (CIF-derived lattices)
    parent_topology_id: str | None = None  # transfer cells: the working-cell topology they test
    # ((sites, k), ...): one count constraint per substituted sublattice; empty = one global count (n_ones).
    # No loader populates it yet; exact enumeration and the encoding honour it, branch and bound and QAOA refuse it.
    sublattice_counts: tuple = ()

    @classmethod
    def from_dict(cls, d: dict) -> "TopologySpec":
        try:
            return cls(
                topology_id=d["topology_id"], representation=d["representation"], n_sites=d["n_sites"],
                site_ids=tuple(d["site_ids"]), allowed_species=tuple(tuple(s) for s in d["allowed_species_by_site"]),
                pairs_d1=tuple(tuple(p) for p in d["pairs_distance_1"]),
                pairs_d2=tuple(tuple(p) for p in d["pairs_distance_2"]),
                triplets=tuple(tuple(t) for t in d["triplets_length_2_paths"]),
                permutations=tuple(tuple(p) for p in d["symmetry_pullback_permutations"]),
                n_formula_units=d["normalization"]["n_abstract_formula_units"],
                crystallographic_net_name=d.get("crystallographic_net_name"),
                atomic_coordinates=d.get("atomic_coordinates"), provenance=d.get("provenance"),
            )
        except (KeyError, TypeError) as exc:
            raise PipelineError("MALFORMED_TOPOLOGY", f"{d.get('topology_id')}: {exc!r}") from exc


@dataclass(frozen=True)
class ProjectSpec:
    project_id: str
    provenance: str
    root: Path
    config: dict
    case: str
    correction_mode: str
    topologies: dict
    species_encoding: dict
    composition_counts: dict
    n_sites: int
    feature_order: list
    model_candidates: list
    training_rows: list
    development_rows: list
    split_manifest: dict
    candidate_rows: list
    building_blocks: dict
    literature: dict
    fingerprint: str = field(default="")
    project_type: str = FIXTURE_PROJECT_TYPE
    physical: dict | None = None  # parsed structures, protocols and load errors of a physical project
    # Validated A1/A2 decisions (agents/runner.py): setup plans and the per-topology model grid. None = no agents.
    agent_plan: dict | None = None

    @property
    def n_B(self) -> int:
        """Number of sites carrying encoding 1 (fixed-composition constraint)."""
        one = [s for s, e in self.species_encoding.items() if e == 1]
        return self.composition_counts[one[0]] if one else 0


def learner_path(root: Path, rel: str) -> Path:
    path = (root / rel).resolve()
    root = root.resolve()
    if root not in path.parents and path != root:
        raise PipelineError("FORBIDDEN_LEARNER_INPUT", f"outside fixture root: {rel}")
    parts = path.relative_to(root).as_posix()
    if parts.split("/")[0] in LEARNER_FORBIDDEN_DIRS or parts in LEARNER_FORBIDDEN_FILES:
        raise PipelineError("FORBIDDEN_LEARNER_INPUT", rel)
    if not path.exists():
        raise PipelineError("MISSING_INPUT_FILE", rel)
    return path


def _read_json(root, rel):
    return json.loads(learner_path(root, rel).read_text())


def _read_jsonl(root, rel):
    return [json.loads(line) for line in learner_path(root, rel).read_text().splitlines() if line.strip()]


def _deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _case_from_fidelity(rows: list, label: str) -> str:
    fidelities = {r.get("fidelity") for r in rows}
    if len(fidelities) != 1:
        raise PipelineError("INCOMPATIBLE_FIDELITY", f"{label}: {sorted(map(str, fidelities))}")
    fidelity = fidelities.pop()
    prefix = "SYNTHETIC_BASE_"
    if not isinstance(fidelity, str) or not fidelity.startswith(prefix):
        raise PipelineError("UNSUPPORTED_FIDELITY", str(fidelity))
    return fidelity[len(prefix):].lower()


def load_project(project_path, case: str | None = None, correction_mode: str | None = None,
                 overrides: dict | None = None) -> ProjectSpec:
    project_path = Path(project_path).resolve()
    root = project_path.parent
    config = json.loads(project_path.read_text())
    if overrides:
        config = _deep_merge(config, overrides)
    if config.get("schema") != FIXTURE_SCHEMA and ("schema_version" in config or "case_id" in config):
        # ProjectSpec v0.1 (project_spec.template.json) or a physical_case recipe.
        if case not in (None, "physical") or correction_mode not in (None, "none"):
            raise PipelineError("UNSUPPORTED_FOR_PHYSICAL_PROJECT", "case/correction modes belong to the synthetic fixture")
        from .project_spec import load_physical_project
        return load_physical_project(project_path, config)
    if case is not None:
        config["inputs"]["training"] = f"data/{case}_training.jsonl"
        config["inputs"]["development"] = f"data/{case}_development.jsonl"
    if correction_mode is not None:
        config["evaluation"]["correction_mode"] = correction_mode
    inputs = config["inputs"]
    topologies_raw = _read_json(root, inputs["topologies"])
    models = _read_json(root, inputs["model_candidates"])
    training = _read_jsonl(root, inputs["training"])
    development = _read_jsonl(root, inputs["development"])
    inferred = _case_from_fidelity(training + development, "training+development")
    if case is not None and inferred != case:
        raise PipelineError("CASE_FIDELITY_MISMATCH", f"{case} vs {inferred}")
    topologies = {}
    for d in topologies_raw:
        spec = TopologySpec.from_dict(d)
        if spec.topology_id in topologies:
            raise PipelineError("DUPLICATE_TOPOLOGY_ID", spec.topology_id)
        topologies[spec.topology_id] = spec
    literature_rel = config.get("literature", {}).get("manifest")
    spec = ProjectSpec(
        project_id=config["project_id"], provenance=config["provenance"], root=root, config=config,
        case=inferred, correction_mode=config["evaluation"]["correction_mode"], topologies=topologies,
        species_encoding=dict(config["composition"]["encoding"]),
        composition_counts=dict(config["composition"]["counts"]), n_sites=config["composition"]["n_sites"],
        feature_order=list(models["feature_order"]), model_candidates=list(models["candidates"]),
        training_rows=training, development_rows=development,
        split_manifest=_read_json(root, inputs.get("split_manifest", DEFAULT_SPLIT_MANIFEST)),
        candidate_rows=_read_jsonl(root, inputs["candidates"]),
        building_blocks=_read_json(root, inputs["building_blocks"]),
        literature=_read_json(root, literature_rel) if literature_rel else {},
    )
    fingerprint = content_hash({k: getattr(spec, k) for k in (
        "config", "case", "species_encoding", "composition_counts", "feature_order", "model_candidates",
        "training_rows", "development_rows", "split_manifest", "candidate_rows", "building_blocks")}
        | {"topologies": topologies_raw})
    object.__setattr__(spec, "fingerprint", fingerprint)
    return spec


# --------------------------------------------------------------------------
def topology_checks(spec: ProjectSpec, topo: TopologySpec) -> tuple[list, list]:
    """Slot/graph checks shared by fixture graphs and CIF-derived slot lattices."""
    from .framework import GRAPH_REPRESENTATION

    checks, errors = [], []

    def check(name, ok, code, detail=""):
        checks.append({"check": name, "passed": bool(ok)})
        if not ok:
            errors.append({"code": code, "check": name, "detail": detail})

    from .structures import composition_of

    tid, n, enc, counts = topo.topology_id, topo.n_sites, spec.species_encoding, composition_of(spec, topo)
    if spec.project_type == FIXTURE_PROJECT_TYPE:
        check(f"{tid}:input_mode", topo.representation == GRAPH_MODE, "UNSUPPORTED_INPUT_MODE", topo.representation)
        check(f"{tid}:provenance", topo.provenance == "SYNTHETIC", "PROVENANCE_MISMATCH")
        check(f"{tid}:no_physical_net_claim", topo.crystallographic_net_name is None and topo.atomic_coordinates is None,
              "GRAPH_FIXTURE_CLAIMS_PHYSICAL_NET")
    else:
        check(f"{tid}:input_mode", topo.representation == GRAPH_REPRESENTATION, "UNSUPPORTED_INPUT_MODE", topo.representation)
        check(f"{tid}:provenance", topo.provenance == spec.provenance, "PROVENANCE_MISMATCH")
    site_count_ok = (n == spec.n_sites if spec.project_type == FIXTURE_PROJECT_TYPE else sum(counts.values()) == n)
    check(f"{tid}:site_count", site_count_ok and topo.site_ids == tuple(range(n)), "INVALID_SITE_COUNT")
    check(f"{tid}:allowed_species", len(topo.allowed_species) == n and all(set(s) <= set(enc) for s in topo.allowed_species),
          "INVALID_ALLOWED_SPECIES")
    for species, count in counts.items():
        hosts = sum(species in s for s in topo.allowed_species)
        check(f"{tid}:composition_feasible:{species}", hosts >= count, "INFEASIBLE_COMPOSITION", f"{hosts}<{count}")
    for label, terms, size in (("pairs_d1", topo.pairs_d1, 2), ("pairs_d2", topo.pairs_d2, 2), ("triplets", topo.triplets, 3)):
        ok = all(len(t) == size and len(set(t)) == size and all(type(i) is int and 0 <= i < n for i in t) for t in terms)
        ok = ok and len({tuple(sorted(t)) for t in terms}) == len(terms)
        check(f"{tid}:{label}_valid", ok, "MALFORMED_INTERACTION_LIST", label)
    check(f"{tid}:normalization", type(topo.n_formula_units) is int and topo.n_formula_units > 0,
          "INVALID_FORMULA_UNIT_COUNT")
    weights = topo.pair_weights_d1
    check(f"{tid}:pair_weights", not weights or (len(weights) == len(topo.pairs_d1)
                                                 and all(type(w) is int and w > 0 for w in weights)),
          "MALFORMED_INTERACTION_LIST", "pair_weights_d1")
    perms = [tuple(p) for p in topo.permutations]
    valid = all(sorted(p) == list(range(n)) for p in perms)
    check(f"{tid}:permutations_bijective", valid, "INVALID_SYMMETRY_OPERATION")
    if valid:
        pset = set(perms)
        check(f"{tid}:group_identity", tuple(range(n)) in pset, "SYMMETRY_GROUP_NOT_CLOSED", "identity missing")
        check(f"{tid}:group_closed", all(tuple(p[q[i]] for i in range(n)) in pset for p in perms for q in perms),
              "SYMMETRY_GROUP_NOT_CLOSED")
        for label, terms in (("pairs_d1", topo.pairs_d1), ("pairs_d2", topo.pairs_d2), ("triplets", topo.triplets)):
            ref = {tuple(sorted(t)) for t in terms}
            check(f"{tid}:symmetry_preserves_{label}",
                  all({tuple(sorted(p[i] for i in t)) for t in ref} == ref for p in perms),
                  "SYMMETRY_CHANGES_GRAPH", label)
        if weights and len(weights) == len(topo.pairs_d1):
            w = {tuple(sorted(t)): x for t, x in zip(topo.pairs_d1, weights)}
            check(f"{tid}:symmetry_preserves_pair_weights",
                  all(w.get(tuple(sorted((p[i], p[j])))) == x for (i, j), x in w.items() for p in perms),
                  "SYMMETRY_CHANGES_GRAPH", "pair_weights_d1")
    return checks, errors


def select_topologies(spec: ProjectSpec, selected, keep_fingerprint: bool = False) -> ProjectSpec:
    """The project restricted to a subset of its candidate topologies; unchanged if all are kept.

    Used for A1's selection (agents/runner.py, new fingerprint) and inside a run after ML screening
    (orchestrator, ``keep_fingerprint``: the run's inputs did not change, the screening artifacts record why).
    Topology-keyed rows, split manifests, physical sources, lattices, transfer cells and builder data are filtered.
    """
    keep = [t for t in spec.topologies if t in set(selected)]
    if keep == list(spec.topologies):
        return spec
    if not keep:
        raise PipelineError("NO_TOPOLOGY_SELECTED")
    rows = lambda rs: [r for r in rs if r.get("topology_id") in keep]
    config = copy.deepcopy(spec.config)
    config["topology_screen"]["retain"] = keep
    fields = dict(config=config, topologies={t: spec.topologies[t] for t in keep}, training_rows=rows(spec.training_rows),
                  development_rows=rows(spec.development_rows), candidate_rows=rows(spec.candidate_rows),
                  split_manifest={t: v for t, v in spec.split_manifest.items() if t in keep})
    if spec.physical is not None:
        physical = dict(spec.physical)
        transfers = {t: v for t, v in (physical.get("transfer_topologies") or {}).items() if t in keep}
        cells = set(keep) | {c.topology_id for v in transfers.values() for c in v}
        physical.update(sources=[s for s in physical["sources"] if s.get("topology_id") in keep], transfer_topologies=transfers,
                        lattices={t: v for t, v in physical["lattices"].items() if t in cells},
                        builder_info={t: v for t, v in physical["builder_info"].items() if t in cells})
        raw = config["project_spec"]
        raw["candidate_topology_ids"] = [t for t in raw.get("candidate_topology_ids") or [] if t in keep]
        raw["source_structures"] = [x for x in raw.get("source_structures") or [] if x.get("topology_id") in keep]
        fields["physical"] = physical
    narrowed = replace(spec, **fields)
    if not keep_fingerprint:
        object.__setattr__(narrowed, "fingerprint", content_hash({"base": spec.fingerprint, "selected_topologies": keep}))
    return narrowed


AGENT_ROLES = ("A1", "A2", "A67", "A4")  # A4 (optimisation loop) only when optimisation rounds are allowed


def agent_request_ceilings(rounds: int) -> tuple:
    """Normal requests (A1, A2, A67, then A4 and per revision A67 + A4), plus the single optional extra request."""
    normal = 3 if rounds == 0 else 4 + 2 * rounds
    return normal, normal + 1


def runtime_agent_errors(spec: ProjectSpec) -> list:
    """Agent policy (MOF_AI_Agent_Implementation_Guide-2 / Task 10): off, or A1/A2/A67(/A4) via ``qumof agents``."""
    agents = spec.config.get("runtime_agents", {})
    if agents.get("enabled") is False and agents.get("llm_calls_allowed") is False:
        return [] if spec.agent_plan is None else [("AGENT_PLAN_WITH_AGENTS_DISABLED", "agent_plan is set")]
    errors = []
    if agents.get("enabled") is not True or agents.get("llm_calls_allowed") is not True:
        errors.append(("AGENT_POLICY_INVALID", "enabled and llm_calls_allowed must both be true or both false"))
    rounds = agents.get("optimisation_rounds", 0)
    roles = AGENT_ROLES if rounds else AGENT_ROLES[:3]
    if type(rounds) is not int or not 0 <= rounds <= 3:
        errors.append(("AGENT_POLICY_INVALID", "optimisation_rounds must be an integer 0-3"))
    elif tuple(agents.get("roles") or ()) != roles:
        errors.append(("AGENT_POLICY_INVALID", f"roles must be {list(roles)}"))
    elif agents.get("max_llm_requests") not in agent_request_ceilings(rounds):
        errors.append(("AGENT_POLICY_INVALID", f"max_llm_requests must be one of {agent_request_ceilings(rounds)}"))
    if spec.agent_plan is None:
        errors.append(("AGENT_RUNNER_REQUIRED", "runtime agents run only through `python -m qumof agents`"))
    return errors


def validate_project(spec: ProjectSpec) -> dict:
    """validate_project(project_spec) -> ValidationReport (Task 01)."""
    from .configuration_model import feature_degree, is_known_feature

    if spec.project_type != FIXTURE_PROJECT_TYPE:
        from .project_spec import validate_physical_project
        return validate_physical_project(spec)

    checks, errors = [], []
    cfg = spec.config

    def check(name, ok, code, detail=""):
        checks.append({"check": name, "passed": bool(ok)})
        if not ok:
            errors.append({"code": code, "check": name, "detail": detail})

    check("schema", cfg.get("schema") == FIXTURE_SCHEMA, "UNSUPPORTED_PROJECT_SCHEMA", str(cfg.get("schema")))
    check("project_provenance", spec.provenance == "SYNTHETIC", "PROVENANCE_MISMATCH", spec.provenance)
    check("building_block_provenance", spec.building_blocks.get("provenance") == "SYNTHETIC", "PROVENANCE_MISMATCH")
    agent_errors = runtime_agent_errors(spec)
    for code, detail in agent_errors or [(None, "")]:
        check("runtime_agents" + (f":{code}" if code else ""), code is None, code, detail)
    check("finite_temperature_out_of_scope", cfg.get("finite_temperature", {}).get("enabled") is False,
          "UNSUPPORTED_SCOPE_FINITE_TEMPERATURE")

    # Composition and species encoding
    counts, enc = spec.composition_counts, spec.species_encoding
    bb_species = {s["id"]: s["encoding"] for s in spec.building_blocks.get("species", [])}
    check("composition_integral", all(type(v) is int and v >= 0 for v in counts.values()),
          "INVALID_COMPOSITION_SPEC", str(counts))
    check("composition_sums_to_sites", sum(counts.values()) == spec.n_sites, "INVALID_COMPOSITION_SPEC",
          f"{sum(counts.values())} != {spec.n_sites}")
    check("encoding_matches_building_blocks", enc == bb_species and set(enc) == set(counts),
          "ENCODING_MISMATCH", f"{enc} vs {bb_species}")
    #TODO: Task 03 — multicomponent decorations, features and enumeration across the pipeline; the encoding layer
    #TODO: (encoding.compile_one_hot, per-sublattice count constraints) and the QAOA exchange mixer already support it.
    check("binary_encoding", sorted(enc.values()) == [0, 1], "UNSUPPORTED_MULTICOMPONENT_ENCODING", str(enc))

    for topo in spec.topologies.values():
        local_checks, local_errors = topology_checks(spec, topo)
        checks += local_checks
        errors += local_errors

    screen = cfg.get("topology_screen", {})
    check("screen_retains_all_supplied_graphs",
          screen.get("mode") == "identity" and set(screen.get("retain", [])) == set(spec.topologies),
          "SCREENING_POLICY_EXCLUDES_WITHOUT_EVIDENCE", str(screen.get("retain")))

    # Model candidates
    ids = [m["id"] for m in spec.model_candidates]
    check("model_ids_unique", len(ids) == len(set(ids)), "DUPLICATE_MODEL_ID")
    for model in spec.model_candidates:
        inds = model["feature_indices"]
        valid = all(type(i) is int and 0 <= i < len(spec.feature_order) for i in inds) and len(set(inds)) == len(inds)
        check(f"{model['id']}:feature_indices", valid, "INVALID_FEATURE_INDEX")
        if not valid:
            continue
        names = [spec.feature_order[i] for i in inds]
        known = all(is_known_feature(n) for n in names)
        check(f"{model['id']}:features_known", known, "UNSUPPORTED_FEATURE", str(names))
        check(f"{model['id']}:explicit_constant_term", "constant" in names, "MISSING_CONSTANT_TERM")
        check(f"{model['id']}:n_parameters", model.get("n_parameters") == len(inds), "PARAMETER_COUNT_MISMATCH")
        if known:
            check(f"{model['id']}:declared_degree", model.get("polynomial_degree") == max(feature_degree(n) for n in names),
                  "POLYNOMIAL_DEGREE_MISMATCH")

    ms = cfg.get("model_selection", {})
    threshold = ms.get("max_development_rmse_eV")
    check("selection_threshold", isinstance(threshold, (int, float)) and not isinstance(threshold, bool) and threshold >= 0,
          "SCIENTIFIC_SPEC_MISSING", "model_selection.max_development_rmse_eV")
    check("selection_split_unit", ms.get("split_unit") == "symmetry_orbit", "UNSUPPORTED_SPLIT_UNIT")
    check("selection_holdout_blind", ms.get("holdout_access_during_selection") is False, "HOLDOUT_ACCESS_FORBIDDEN")
    check("selection_failure_code", ms.get("no_passing_model") == "MODEL_INADEQUATE", "SCIENTIFIC_SPEC_MISSING")

    opt = cfg.get("optimizer", {})
    check("optimizer_method", opt.get("method") == "exact_enumeration", "UNSUPPORTED_SOLVER", str(opt.get("method")))
    check("optimizer_fixed_composition", opt.get("restrict_to_composition") is True, "UNSUPPORTED_SOLVER")
    check("optimizer_cross_checks", set(opt.get("cross_check_solvers", [])) <= {"branch_and_bound"}, "UNSUPPORTED_SOLVER",
          str(opt.get("cross_check_solvers")))
    from .qaoa import protocol_errors
    quantum_errors = [] if opt.get("quantum_enabled") is False else protocol_errors(opt.get("quantum_protocol"))
    detail = "optimizer.quantum_protocol: " + ", ".join(f"{c}: {f}" for c, f in quantum_errors)
    for code in sorted({c for c, _ in quantum_errors}) or [None]:  # one check per error code; one passing check
        check("quantum_protocol" + (f":{code}" if code else ""), code is None, code, detail)
    qubo = cfg.get("qubo", {})
    penalty = qubo.get("ratio_penalty_eV")
    check("qubo_format", qubo.get("format") == "upper_triangle_once", "UNSUPPORTED_ENCODING_FORMAT")
    check("qubo_penalty", isinstance(penalty, (int, float)) and not isinstance(penalty, bool) and penalty > 0,
          "SCIENTIFIC_SPEC_MISSING", "qubo.ratio_penalty_eV")
    check("qubo_higher_order_policy", qubo.get("higher_order") in ("retain_HUBO_or_explicitly_reject", "quadratize"),
          "UNSUPPORTED_ENCODING_POLICY")
    dedup = cfg.get("deduplication", {})
    check("no_species_label_exchange", dedup.get("allow_species_label_exchange") is False,
          "UNSUPPORTED_SYMMETRY_POLICY")
    pool = cfg.get("candidate_pool", {})
    check("pool_size", type(pool.get("top_k_unique_orbits_per_topology")) is int and pool["top_k_unique_orbits_per_topology"] > 0,
          "SCIENTIFIC_SPEC_MISSING", "candidate_pool.top_k_unique_orbits_per_topology")
    ev = cfg.get("evaluation", {})
    check("evaluation_backend", ev.get("backend") == "synthetic_fixture", "UNSUPPORTED_EVALUATION_BACKEND")
    check("correction_mode", ev.get("correction_mode") in CORRECTION_MODES, "UNKNOWN_CORRECTION_MODE",
          str(ev.get("correction_mode")))
    # The fixture only uses its synthetic evaluator; periodic DFT runs in physical projects (dft.py).
    # Embedding corrections exist for physical projects only (embedding.py); fixture corrections are synthetic.
    #TODO: Task 07 — hardware (QPU) electronic backends.
    for flag in ("dft_enabled", "embedding_enabled", "hardware_enabled"):
        check(f"evaluation_{flag}_false", ev.get(flag) is False, "UNSUPPORTED_ELECTRONIC_BACKEND", flag)
    al = cfg.get("active_learning", {})
    check("active_learning_mode", al.get("mode") in ("disabled_for_initial_smoke_test", "enabled"),
          "UNSUPPORTED_ACTIVE_LEARNING_MODE")
    check("active_learning_bounded", type(al.get("max_iterations_if_enabled")) is int and 0 < al["max_iterations_if_enabled"] <= 3,
          "UNBOUNDED_ACTIVE_LEARNING")
    #TODO: live literature review mode (retrieve papers, map claims to cited passages); static manifest only now.
    lit = cfg.get("literature", {})
    check("literature_static", lit.get("mode") == "static_manifest" and lit.get("live_review_enabled") is False,
          "UNSUPPORTED_LITERATURE_MODE")

    status = StageStatus.COMPLETED if not errors else StageStatus.BLOCKED
    return make_record("ValidationReport", spec.provenance, scope="project", project_id=spec.project_id,
                       status=status.value, n_checks=len(checks), checks=checks, errors=errors)
