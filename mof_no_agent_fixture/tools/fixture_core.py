"""Pure-Python utilities for a SYNTHETIC occupation-model test fixture."""
from fractions import Fraction
from itertools import combinations, product
from pathlib import Path
import json
import math

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["constant", "unlike_distance_1", "unlike_distance_2", "BBB_paths"]

def read_json(path):
    return json.loads(Path(path).read_text())

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))

def bit_string(bits):
    return "".join(map(str, bits))

def feasible_states(n=12, k=6):
    for selected in combinations(range(n), k):
        chosen = set(selected)
        yield tuple(int(i in chosen) for i in range(n))

def validate_bits(bits, n=12, k=6):
    if len(bits) != n:
        raise ValueError("INVALID_SITE_COUNT")
    if any(type(x) is not int or x not in (0, 1) for x in bits):
        raise ValueError("INVALID_BINARY_OCCUPATION")
    if sum(bits) != k:
        raise ValueError("INVALID_COMPOSITION")

def features(topology, bits):
    return [
        1,
        sum(bits[i] != bits[j] for i, j in topology["pairs_distance_1"]),
        sum(bits[i] != bits[j] for i, j in topology["pairs_distance_2"]),
        sum(bits[i] * bits[j] * bits[k] for i, j, k in topology["triplets_length_2_paths"]),
    ]

def canonical(topology, bits):
    # Convention: permutation[pullback_index] supplies the original site value.
    return min(bit_string([bits[j] for j in p]) for p in topology["symmetry_pullback_permutations"])

def solve_linear_system(matrix, rhs):
    n = len(rhs)
    augmented = [[Fraction(x) for x in row] + [Fraction(b)] for row, b in zip(matrix, rhs)]
    for col in range(n):
        pivot = next((r for r in range(col, n) if augmented[r][col]), None)
        if pivot is None:
            raise ValueError("RANK_DEFICIENT_FEATURES")
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        scale = augmented[col][col]
        augmented[col] = [v / scale for v in augmented[col]]
        for r in range(n):
            if r != col:
                scale = augmented[r][col]
                augmented[r] = [a - scale*b for a, b in zip(augmented[r], augmented[col])]
    return [row[-1] for row in augmented]

def fit_linear(x_rows, y_values):
    # Exact arithmetic is practical because this fixture has at most four features.
    n = len(x_rows[0])
    gram = [[sum(Fraction(x[i])*x[j] for x in x_rows) for j in range(n)] for i in range(n)]
    rhs = [sum(Fraction(x[i])*Fraction(str(y)) for x, y in zip(x_rows, y_values)) for i in range(n)]
    return solve_linear_system(gram, rhs)

def predict(topology, bits, feature_indices, coefficients):
    f = features(topology, bits)
    return sum(c * f[i] for i, c in zip(feature_indices, coefficients))

def rmse(values):
    return math.sqrt(sum(float(v)**2 for v in values) / len(values))

def compile_pair_qubo(topology, coefficients, ratio_penalty=1):
    if len(coefficients) != 3:
        raise ValueError("UNSUPPORTED_HIGHER_ORDER_WITHOUT_QUADRATIZATION")
    n, k = topology["n_sites"], 6
    c0, j1, j2 = map(Fraction, coefficients)
    penalty = Fraction(ratio_penalty)
    linear = [penalty*(1-2*k) for _ in range(n)]
    quadratic = {(i,j): 2*penalty for i in range(n) for j in range(i+1,n)}
    for pairs, coefficient in [(topology["pairs_distance_1"],j1),(topology["pairs_distance_2"],j2)]:
        for i,j in pairs:
            linear[i] += coefficient
            linear[j] += coefficient
            quadratic[(i,j)] += -2*coefficient
    return {
        "format": "upper_triangle_once",
        "objective": "physical_cell_energy + lambda*(sum(bits)-6)^2",
        "constant_eV": float(c0 + penalty*k*k),
        "linear_eV": [float(v) for v in linear],
        "quadratic_eV": [{"i":i,"j":j,"value":float(v)} for (i,j),v in sorted(quadratic.items()) if v],
        "penalty_eV": float(penalty),
        "physical_offset_eV": float(c0),
        "bit_order": "bits[i] is site i; string leftmost character is site 0",
        "provenance": "SYNTHETIC",
    }

def evaluate_qubo(qubo, bits):
    return qubo["constant_eV"] + sum(c*x for c,x in zip(qubo["linear_eV"],bits)) + sum(
        term["value"]*bits[term["i"]]*bits[term["j"]] for term in qubo["quadratic_eV"]
    )

def correction(topology_id, bits, mode, topology):
    if mode == "none":
        return Fraction(0)
    if mode == "constant":
        return Fraction(1,20)
    if mode == "topology_dependent":
        # An artificial routing/ranking test, not an electronic embedding formula.
        return (Fraction(3,50) if topology_id == "prism6" else Fraction(0)) + Fraction(features(topology,bits)[3],1000)
    raise ValueError("UNKNOWN_CORRECTION_MODE")

def check_energy_compatibility(rows):
    if not rows:
        raise ValueError("EMPTY_ENERGY_SET")
    fields = ["fidelity", "energy_unit", "normalization", "comparison_group"]
    for field in fields:
        if len({row[field] for row in rows}) > 1:
            raise ValueError("INCOMPATIBLE_" + field.upper())
    for row in rows:
        if row.get("status") != "SUCCEEDED":
            raise ValueError("NON_SUCCESSFUL_CALCULATION")
        if row.get("provenance") != "SYNTHETIC":
            raise ValueError("PROVENANCE_MISMATCH")

def normalize_cell_energy(total_energy_eV, n_formula_units):
    if type(n_formula_units) is not int or n_formula_units <= 0:
        raise ValueError("INVALID_FORMULA_UNIT_COUNT")
    return Fraction(str(total_energy_eV)) / n_formula_units

