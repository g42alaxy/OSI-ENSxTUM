"""Task 05 (T05.1): relaxed-geometry equivalence, separate from decoration-level symmetry deduplication.

Two relaxed structures are equivalent when they have the same inventory and magnetic state, lattices that
agree within a relative tolerance, and an operation of the parent space group combined with a supercell
translation maps every atom onto a same-element atom within ``tol_A`` (minimum image). Energies are never used:
distinct nets, conformers or magnetic states that happen to have equal energies stay separate.
"""
from __future__ import annotations

import itertools
import math
from collections import Counter


def _frac_to_cart(lattice, f) -> tuple:
    return tuple(sum(f[k] * lattice[k][c] for k in range(3)) for c in range(3))


def _lattice_params(lattice) -> tuple:
    lengths = [math.sqrt(sum(x * x for x in v)) for v in lattice]

    def angle(u, v):
        return math.degrees(math.acos(max(-1.0, min(1.0, sum(a * b for a, b in zip(u, v)) /
                                                    (math.sqrt(sum(a * a for a in u)) * math.sqrt(sum(b * b for b in v)))))))
    return lengths, [angle(lattice[1], lattice[2]), angle(lattice[0], lattice[2]), angle(lattice[0], lattice[1])]


def supercell_operations(symmetry_operations, diag) -> list:
    """Parent operations x supercell translations, acting on supercell fractional coordinates."""
    ops = []
    for (rot, trans) in symmetry_operations:
        for shift in itertools.product(*(range(n) for n in diag)):
            ops.append((rot, tuple(float(trans[r]) + shift[r] for r in range(3))))
    return ops


def _apply(op, f, diag) -> tuple:
    rot, trans = op
    p = [f[k] * diag[k] for k in range(3)]
    q = [sum(rot[r][c] * p[c] for c in range(3)) + trans[r] for r in range(3)]
    return tuple((q[k] / diag[k]) % 1.0 for k in range(3))


def fingerprint(geometry: dict, magnetic_label=None, bond_tolerance_A: float = 0.45) -> tuple:
    """Cheap invariant pre-filter: inventory, magnetic state and bond counts per element pair (cell checked later)."""
    from .framework import COVALENT_RADII_A, _close_pairs

    lattice, elements = geometry["lattice_A"], geometry["elements"]
    structure = {"lattice": lattice, "atoms": [{"element": e, "frac": tuple(f)} for e, f in zip(elements, geometry["frac_coords"])]}
    radii = [COVALENT_RADII_A.get(e, 1.5) for e in elements]
    bonds = Counter(tuple(sorted((elements[i], elements[j]))) for i, j, _, _ in
                    _close_pairs(structure, lambda i, j: radii[i] + radii[j] + bond_tolerance_A,
                                 2 * max(radii, default=0.0) + bond_tolerance_A))
    return (tuple(sorted(Counter(elements).items())), magnetic_label, tuple(sorted(bonds.items())))


def equivalent(g1: dict, g2: dict, operations: list, diag, tol_A: float = 0.1, lattice_rel_tol: float = 0.01) -> bool:
    """Is there an operation mapping g2 onto g1 atom by atom (same element, within tol_A)?"""
    if Counter(g1["elements"]) != Counter(g2["elements"]):
        return False
    (l1, a1), (l2, a2) = _lattice_params(g1["lattice_A"]), _lattice_params(g2["lattice_A"])
    if any(abs(x - y) > lattice_rel_tol * x for x, y in zip(l1, l2)) or any(abs(x - y) > 1.0 for x, y in zip(a1, a2)):
        return False
    lattice = g1["lattice_A"]
    bins = [max(1, int(l / max(2 * tol_A, 1.0))) for l in l1]
    grid = {}
    for idx, (el, f) in enumerate(zip(g1["elements"], g1["frac_coords"])):
        key = (el,) + tuple(int((f[k] % 1.0) * bins[k]) % bins[k] for k in range(3))
        grid.setdefault(key, []).append(idx)

    def close(el, f) -> int | None:
        cell = [int(f[k] * bins[k]) % bins[k] for k in range(3)]
        for off in itertools.product((-1, 0, 1), repeat=3):
            for idx in grid.get((el,) + tuple((cell[k] + off[k]) % bins[k] for k in range(3)), ()):
                g = g1["frac_coords"][idx]
                d = [f[k] - g[k] - round(f[k] - g[k]) for k in range(3)]
                if math.sqrt(sum(x * x for x in _frac_to_cart(lattice, d))) <= tol_A:
                    return idx
        return None

    for op in operations:
        used = set()
        for el, f in zip(g2["elements"], g2["frac_coords"]):
            idx = close(el, _apply(op, f, diag))
            if idx is None or idx in used:
                break
            used.add(idx)
        else:
            return True
    return False


def equivalence_classes(items: list, operations: list, diag, tol_A: float = 0.1) -> dict:
    """items: [(key, geometry, magnetic_label)] -> {key: representative key}; deterministic (first in key order)."""
    out, reps = {}, []
    for key, geometry, label in sorted(items, key=lambda x: x[0]):
        fp = fingerprint(geometry, label)
        for rep_key, rep_geometry, rep_fp in reps:
            if rep_fp == fp and equivalent(rep_geometry, geometry, operations, diag, tol_A):
                out[key] = rep_key
                break
        else:
            reps.append((key, geometry, fp))
            out[key] = key
    return out
