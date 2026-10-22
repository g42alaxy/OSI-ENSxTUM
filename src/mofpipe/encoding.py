"""Task 03 (encoding): compile the physical surrogate into QUBO / HUBO objectives.

Convention: separate constant, linear terms, upper-triangle off-diagonal terms
counted once; objective = physical_cell_energy + lambda*(sum(bits) - k)^2.
A cubic surrogate is never labelled QUBO: a QUBO-only target rejects it.
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


def compile_encoding(spec, topo, hamiltonian: dict, target: str) -> dict:
    """compile_encoding(hamiltonian_id, decoration_space_id, encoding_policy_id) -> EncodingArtifact."""
    physical = physical_polynomial(hamiltonian, topo)
    if target == "QUBO" and degree(physical) > 2:
        #TODO: Task 03 — explicit quadratization (ancillas + validated constraints, energy-equivalence check)
        #TODO: so higher-order models can reach QUBO-only backends.
        raise PipelineError(QUADRATIZATION_REQUIRED,
                            f"{hamiltonian['model_id']} has {sum(len(k) > 2 for k in physical)} terms of degree > 2")
    if target not in ("QUBO", "HUBO"):
        raise PipelineError("UNSUPPORTED_ENCODING_FORMAT", target)
    weight = Fraction(repr(spec.config["qubo"]["ratio_penalty_eV"]))
    #TODO: Task 03 — one-hot / domain-wall encodings for multicomponent slots and an auditable penalty bound for
    #TODO: instances too large for exhaustive certification.
    n, k = topo.n_sites, n_ones_of(spec, topo)
    penalty = penalty_polynomial(n, k, weight)
    objective = poly_add(physical, penalty)
    linear = [objective.get((i,), Fraction(0)) for i in range(n)]
    quadratic = [{"i": k[0], "j": k[1], "value": float(v), "exact": str(v)}
                 for k, v in sorted(objective.items()) if len(k) == 2]
    higher = [{"sites": list(k), "value": float(v), "exact": str(v)}
              for k, v in sorted(objective.items()) if len(k) > 2]
    const = objective.get((), Fraction(0))
    return make_record(
        "EncodingArtifact", spec.provenance, target=target, format="upper_triangle_once",
        topology_id=topo.topology_id, physical_hamiltonian_id=hamiltonian["artifact_id"],
        objective="physical_energy + lambda*(sum(bits)-%d)^2" % k,
        constant_eV=float(const), constant_exact=str(const),
        linear_eV=[float(v) for v in linear], linear_exact=[str(v) for v in linear],
        quadratic_eV=quadratic, higher_order_terms=higher, max_degree=degree(objective),
        penalty_eV=float(weight), penalty_exact=str(weight), penalty_constant_exact=str(penalty.get((), 0)),
        physical_offset_eV=float(physical.get((), 0)), physical_offset_exact=str(physical.get((), 0)),
        bit_order="bits[i] is site i; string leftmost character is site 0",
        variable_to_slot={str(i): i for i in range(n)}, ancillas=[],
        decode_rule="bit 1 -> species with encoding 1; reject if sum(bits) != %d" % k,
    )


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
    return _IntPoly(encoding)(bits)


def verify_encoding(spec, topo, hamiltonian: dict, encoding: dict) -> dict:
    """Energy identity on all 2^n strings, including infeasible ones; penalised minimum must be feasible."""
    evaluate = _IntPoly(encoding)
    weight = Fraction(encoding["penalty_exact"])
    k = n_ones_of(spec, topo)
    max_dev, best, argmins = Fraction(0), None, []
    for bits in product((0, 1), repeat=topo.n_sites):
        direct = predict(hamiltonian, topo, bits) + weight * (sum(bits) - k) ** 2
        value = evaluate(bits)
        max_dev = max(max_dev, abs(value - direct))
        if best is None or value < best:
            best, argmins = value, [bits]
        elif value == best:
            argmins.append(bits)
    feasible_argmin = all(sum(b) == k for b in argmins)
    ok = max_dev == 0 and feasible_argmin
    return make_record(
        "EncodingVerification", spec.provenance, encoding_id=encoding["artifact_id"], topology_id=topo.topology_id,
        status=StageStatus.COMPLETED.value if ok else StageStatus.FAILED.value,
        n_states_checked=2 ** topo.n_sites, max_abs_deviation_exact=str(max_dev),
        penalized_minimum_eV=float(best), penalized_minimum_exact=str(best),
        n_penalized_argmins=len(argmins), all_penalized_argmins_feasible=feasible_argmin,
        penalty_certification="exhaustive small-instance check; not a general sufficiency bound",
    )
