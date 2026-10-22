"""Task 03 (encoding): compile the physical surrogate into QUBO / HUBO objectives.

Convention: separate constant, linear terms, upper-triangle off-diagonal terms
counted once; objective = physical_energy + lambda * sum_c (sum_{i in c} bits_i - k_c)^2 over the declared
count constraints (one per substituted sublattice). A cubic surrogate is never labelled QUBO: a QUBO-only
target rejects it unless the encoding policy asks for explicit quadratization (Rosenberg substitution with
recorded ancillas and a stated penalty bound). ``compile_one_hot`` encodes slots with more than two species.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import product
from math import lcm

from .configuration_model import feature_polynomial, predict
from .contract import PipelineError, StageStatus, make_record
from .structures import n_ones_of

QUADRATIZATION_REQUIRED = "UNSUPPORTED_HIGHER_ORDER_WITHOUT_QUADRATIZATION"


def poly_add(a: dict, b: dict, scale=1) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + scale * v
    return {k: v for k, v in out.items() if v != 0}


def poly_mul(a: dict, b: dict) -> dict:
    out: dict = {}
    for ka, va in a.items():
        for kb, vb in b.items():
            key = tuple(sorted(set(ka) | set(kb)))  # binary variables: x^2 = x
            out[key] = out.get(key, 0) + va * vb
    return {k: v for k, v in out.items() if v != 0}


def physical_polynomial(hamiltonian: dict, topo) -> dict:
    poly: dict = {}
    for name, c in zip(hamiltonian["features"], hamiltonian["coefficients_exact"]):
        poly = poly_add(poly, feature_polynomial(name, topo), Fraction(c))
    return poly


def penalty_polynomial(n: int, k: int, weight) -> dict:
    s = {(i,): 1 for i in range(n)}
    s[()] = -k
    return {key: Fraction(weight) * v for key, v in poly_mul(s, s).items()}


def degree(poly: dict) -> int:
    return max((len(k) for k in poly), default=0)


def count_constraints(spec, topo) -> list:
    """[(sites, k)] per substituted sublattice; a single global count when the topology declares none."""
    if topo.sublattice_counts:
        return [(tuple(sites), k) for sites, k in topo.sublattice_counts]
    return [(tuple(range(topo.n_sites)), n_ones_of(spec, topo))]


def count_penalty(constraints: list, weight) -> dict:
    poly: dict = {}
    for sites, k in constraints:
        s = {(i,): 1 for i in sites}
        s[()] = -k
        poly = poly_add(poly, {key: Fraction(weight) * v for key, v in poly_mul(s, s).items()})
    return poly


def single_flip_bound(poly: dict, variables) -> Fraction:
    """max_i sum_{terms containing i} |c|: no single bit flip changes the polynomial by more than this."""
    return max((sum(abs(Fraction(c)) for key, c in poly.items() if i in key) for i in variables), default=Fraction(0))


def quadratize(poly: dict, n: int) -> tuple[dict, list]:
    """Rosenberg substitution for cubic terms: c x_i x_j x_k -> c y x_k + M (x_i x_j - 2 x_i y - 2 x_j y + 3 y).

    One ancilla y = x_i x_j per distinct substituted pair (the two smallest indices of each cubic term). The
    penalty P(x_i, x_j, y) is 0 iff y = x_i x_j and >= 1 otherwise; ancillas only multiply original variables,
    so with M_y = 2 * sum_{terms using y} |c| (> sum |c|) the minimum over y reproduces the cubic polynomial.
    """
    if degree(poly) > 3:
        raise PipelineError("UNSUPPORTED_ENCODING_FORMAT", "quadratization implemented for degree <= 3")
    out, ancilla_of = {}, {}
    for key, c in sorted(poly.items()):
        if len(key) <= 2:
            out = poly_add(out, {key: c})
            continue
        pair = key[:2]
        y = ancilla_of.setdefault(pair, n + len(ancilla_of))
        out = poly_add(out, {tuple(sorted((y, key[2]))): c})
    ancillas = []
    for (i, j), y in sorted(ancilla_of.items(), key=lambda kv: kv[1]):
        m = 2 * sum(abs(c) for key, c in out.items() if y in key)
        out = poly_add(out, {(i, j): m, (i, y): -2 * m, (j, y): -2 * m, (y,): 3 * m})
        ancillas.append({"index": y, "product_of": [i, j], "penalty_M_exact": str(m), "penalty_M_eV": float(m),
                         "rule": "M = 2 * sum |c| of substituted terms using this ancilla"})
    return out, ancillas


def compile_encoding(spec, topo, hamiltonian: dict, target: str, quadratize_higher_order: bool = False) -> dict:
    """compile_encoding(hamiltonian_id, decoration_space_id, encoding_policy_id) -> EncodingArtifact."""
    physical = physical_polynomial(hamiltonian, topo)
    if target == "QUBO" and degree(physical) > 2 and not quadratize_higher_order:
        raise PipelineError(QUADRATIZATION_REQUIRED,
                            f"{hamiltonian['model_id']} has {sum(len(k) > 2 for k in physical)} terms of degree > 2")
    if target not in ("QUBO", "HUBO"):
        raise PipelineError("UNSUPPORTED_ENCODING_FORMAT", target)
    weight = Fraction(repr(spec.config["qubo"]["ratio_penalty_eV"]))
    n = topo.n_sites
    constraints = count_constraints(spec, topo)
    penalty = count_penalty(constraints, weight)
    body, ancillas = (quadratize(physical, n) if target == "QUBO" and degree(physical) > 2 else (physical, []))
    objective = poly_add(body, penalty)
    n_vars = n + len(ancillas)
    linear = [objective.get((i,), Fraction(0)) for i in range(n_vars)]
    quadratic = [{"i": k[0], "j": k[1], "value": float(v), "exact": str(v)}
                 for k, v in sorted(objective.items()) if len(k) == 2]
    higher = [{"sites": list(k), "value": float(v), "exact": str(v)}
              for k, v in sorted(objective.items()) if len(k) > 2]
    const = objective.get((), Fraction(0))
    # Auditable sufficiency: if lambda > max single-flip change of the physical polynomial, every state with a
    # count violation d >= 1 lies above the feasible state reached by d flips (lambda d^2 >= lambda d > d L).
    bound = single_flip_bound(physical, range(n))
    k_text = "; ".join(f"sum over {len(sites)} sites = {k}" for sites, k in constraints)
    return make_record(
        "EncodingArtifact", spec.provenance, target=target, format="upper_triangle_once",
        topology_id=topo.topology_id, physical_hamiltonian_id=hamiltonian["artifact_id"],
        objective="physical_energy + lambda * sum_c (sum_{i in c} bits_i - k_c)^2",
        count_constraints=[{"sites": list(sites), "k": k} for sites, k in constraints],
        constant_eV=float(const), constant_exact=str(const),
        linear_eV=[float(v) for v in linear], linear_exact=[str(v) for v in linear],
        quadratic_eV=quadratic, higher_order_terms=higher, max_degree=degree(objective),
        penalty_eV=float(weight), penalty_exact=str(weight), penalty_constant_exact=str(penalty.get((), 0)),
        penalty_bound={"single_flip_bound_eV": float(bound), "single_flip_bound_exact": str(bound),
                       "certified_by_bound": weight > bound,
                       "rule": "lambda > max_i sum_{terms containing i} |c| guarantees a feasible penalised minimum"},
        physical_offset_eV=float(physical.get((), 0)), physical_offset_exact=str(physical.get((), 0)),
        bit_order="bits[i] is site i; string leftmost character is site 0; ancillas follow the sites",
        variable_to_slot={str(i): i for i in range(n)}, n_variables=n_vars, ancillas=ancillas,
        decode_rule=f"bit 1 -> species with encoding 1; drop ancillas; reject unless {k_text}",
    )


def complete_assignment(encoding: dict, bits) -> list:
    """Decision bits plus their consistent ancilla values (y = product of its pair)."""
    full = list(bits)
    for a in encoding.get("ancillas", []):
        i, j = a["product_of"]
        full.append(full[i] * full[j])
    return full


class _IntPoly:
    """Exact fast evaluator: coefficients scaled to integers, terms as bit masks."""

    def __init__(self, encoding: dict):
        terms = [((), Fraction(encoding["constant_exact"]))]
        terms += [((i,), Fraction(v)) for i, v in enumerate(encoding["linear_exact"])]
        terms += [((q["i"], q["j"]), Fraction(q["exact"])) for q in encoding["quadratic_eV"]]
        terms += [(tuple(h["sites"]), Fraction(h["exact"])) for h in encoding["higher_order_terms"]]
        self.scale = lcm(*(c.denominator for _, c in terms))
        self.terms = [(sum(1 << i for i in k), int(c * self.scale)) for k, c in terms]

    def __call__(self, bits) -> Fraction:
        mask = sum(1 << i for i, b in enumerate(bits) if b)
        return Fraction(sum(c for m, c in self.terms if mask & m == m), self.scale)


def evaluate_encoding(encoding: dict, bits) -> Fraction:
    return _IntPoly(encoding)(complete_assignment(encoding, bits))


def _ancilla_coefficients(encoding: dict) -> dict:
    """For each ancilla y: (linear coefficient, {original variable: coefficient of y * x_v})."""
    out = {a["index"]: [Fraction(encoding["linear_exact"][a["index"]]), {}] for a in encoding.get("ancillas", [])}
    for q in encoding["quadratic_eV"]:
        for y, v in ((q["j"], q["i"]), (q["i"], q["j"])):
            if y in out and v not in out:
                out[y][1][v] = Fraction(q["exact"])
    return out


def verify_encoding(spec, topo, hamiltonian: dict, encoding: dict) -> dict:
    """Energy identity on all 2^n decision strings (infeasible ones included), minimising over ancillas;
    the penalised minimum must be feasible."""
    evaluate = _IntPoly(encoding)
    weight = Fraction(encoding["penalty_exact"])
    constraints = [(c["sites"], c["k"]) for c in encoding.get("count_constraints") or
                   [{"sites": list(range(topo.n_sites)), "k": n_ones_of(spec, topo)}]]
    ancillas = _ancilla_coefficients(encoding)
    max_dev, best, argmins = Fraction(0), None, []
    for bits in product((0, 1), repeat=topo.n_sites):
        violation = sum((sum(bits[i] for i in sites) - k) ** 2 for sites, k in constraints)
        direct = predict(hamiltonian, topo, bits) + weight * violation
        # Ancillas couple only to decision variables, so the minimum over them is separable.
        value = evaluate(list(bits) + [0] * len(ancillas))
        for lin, couplings in ancillas.values():
            value += min(Fraction(0), lin + sum(c * bits[v] for v, c in couplings.items()))
        max_dev = max(max_dev, abs(value - direct))
        if best is None or value < best:
            best, argmins = value, [bits]
        elif value == best:
            argmins.append(bits)
    feasible_argmin = all(all(sum(b[i] for i in sites) == k for sites, k in constraints) for b in argmins)
    ok = max_dev == 0 and feasible_argmin
    return make_record(
        "EncodingVerification", spec.provenance, encoding_id=encoding["artifact_id"], topology_id=topo.topology_id,
        status=StageStatus.COMPLETED.value if ok else StageStatus.FAILED.value,
        n_states_checked=2 ** topo.n_sites, n_ancillas_minimised=len(ancillas), max_abs_deviation_exact=str(max_dev),
        penalized_minimum_eV=float(best), penalized_minimum_exact=str(best),
        n_penalized_argmins=len(argmins), all_penalized_argmins_feasible=feasible_argmin,
        penalty_certification=("exhaustive small-instance check" +
                               ("; also certified by the single-flip bound" if encoding.get("penalty_bound", {})
                                .get("certified_by_bound") else "; the single-flip bound alone does not certify it")),
    )


# -- multicomponent one-hot encoding (T03.5) ------------------------------------------------------------------
def compile_one_hot(n_sites: int, n_species: int, physical: dict, counts: list, weight) -> dict:
    """One-hot QUBO for slots with ``n_species`` species.

    Variable v = site * n_species + species. ``physical`` maps tuples of (site, species) pairs (length <= 2) to
    coefficients; ``counts`` is [(sites, species, k)] per sublattice and species. Constraints: exactly one species
    per site, and the declared counts. Penalty: lambda * (sum of squared violations of both families).
    """
    weight = Fraction(str(weight))
    var = lambda i, a: i * n_species + a  # noqa: E731
    poly: dict = {}
    for key, c in physical.items():
        poly = poly_add(poly, {tuple(sorted(var(i, a) for i, a in key)): Fraction(str(c))})
    if degree(poly) > 2:
        raise PipelineError("UNSUPPORTED_ENCODING_FORMAT", "one-hot physical terms must be at most pairwise")
    penalty: dict = {}
    families = [([var(i, a) for a in range(n_species)], 1) for i in range(n_sites)]
    families += [([var(i, a) for i in sites], k) for sites, a, k in counts]
    for members, k in families:
        s = {(v,): 1 for v in members}
        s[()] = -k
        penalty = poly_add(penalty, {key: weight * v for key, v in poly_mul(s, s).items()})
    objective = poly_add(poly, penalty)
    return {"target": "QUBO_ONE_HOT", "n_sites": n_sites, "n_species": n_species, "weight_exact": str(weight),
            "terms": {",".join(map(str, k)): str(v) for k, v in sorted(objective.items())},
            "constraint_families": [{"variables": m, "k": k} for m, k in families],
            "variable_to_slot": {str(var(i, a)): [i, a] for i in range(n_sites) for a in range(n_species)},
            "decode_rule": "site i holds the species a with x[i*n_species + a] = 1; reject unless one-hot and counts hold",
            "penalty_certification": "exhaustive small-instance check only (no general bound derived for one-hot)"}


def one_hot_energy(encoding: dict, x) -> Fraction:
    total = Fraction(0)
    for key, c in encoding["terms"].items():
        idx = [int(t) for t in key.split(",")] if key else []
        if all(x[i] for i in idx):
            total += Fraction(c)
    return total


def decode_one_hot(encoding: dict, x):
    """Species per site, or None when the bit string violates any constraint."""
    s, n = encoding["n_species"], encoding["n_sites"]
    if any(sum(x[v] for v in fam["variables"]) != fam["k"] for fam in encoding["constraint_families"]):
        return None
    return tuple(next(a for a in range(s) if x[i * s + a]) for i in range(n))


def verify_one_hot(encoding: dict, physical_energy) -> dict:
    """Exhaustive check: encoded energy equals the physical energy on every feasible string, and the minimum
    over all 2^(n*s) strings is feasible."""
    n_vars = encoding["n_sites"] * encoding["n_species"]
    if n_vars > 20:
        return {"status": "UNCERTIFIED", "reason": f"{n_vars} variables exceed the exhaustive limit"}
    best, argmins, max_dev = None, [], Fraction(0)
    for x in product((0, 1), repeat=n_vars):
        e = one_hot_energy(encoding, x)
        decoded = decode_one_hot(encoding, x)
        if decoded is not None:
            max_dev = max(max_dev, abs(e - Fraction(physical_energy(decoded))))
        if best is None or e < best:
            best, argmins = e, [x]
        elif e == best:
            argmins.append(x)
    feasible = all(decode_one_hot(encoding, x) is not None for x in argmins)
    return {"status": "COMPLETED" if feasible and max_dev == 0 else "FAILED", "n_states_checked": 2 ** n_vars,
            "max_abs_deviation_exact": str(max_dev), "encoded_minimum_exact": str(best),
            "minimizers": sorted({decode_one_hot(encoding, x) for x in argmins if decode_one_hot(encoding, x)}),
            "all_minimizers_feasible": feasible}


def one_hot_states(n_sites: int, counts_by_species: list) -> list:
    """All assignments with the given global species counts (multiset permutations), lexicographic."""
    out = []

    def rec(prefix, remaining):
        if len(prefix) == n_sites:
            out.append(tuple(prefix))
            return
        for a, left in enumerate(remaining):
            if left:
                remaining[a] -= 1
                rec(prefix + [a], remaining)
                remaining[a] += 1
    rec([], list(counts_by_species))
    return out
