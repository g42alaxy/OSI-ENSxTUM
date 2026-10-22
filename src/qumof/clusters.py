"""Task 03 (T03.1): symmetry orbits of slot clusters on the parent lattice, transferable between supercells.

A slot of a supercell is a parent-cell slot site (sublattice position) plus a lattice translation. Two clusters
belong to the same orbit when a parent space-group operation (rotation, fractional translation and lattice
translation) maps one onto the other, so orbit identities do not depend on the supercell. Features are counts
per formula unit with every periodic image included:

  point_orbit_k = (1/n_fu) * sum over slots i in parent site orbit k of x_i
  pair_orbit_k  = (1/n_fu) * sum over periodic slot pairs (i, j + T) in pair orbit k of [x_i != x_j]

A configuration replicated into a larger supercell therefore has identical feature values (the replication
test of Task 03), and physically inequivalent slots or pairs stay separate even at equal distance.
Pairs are enumerated by slot-centroid distance up to the largest cutoff requested by a model candidate
(``pairs<=R`` tokens, R in angstrom). Triplet clusters are not generated (the fixed BBB_paths feature remains).
"""
from __future__ import annotations

import itertools
import math
from fractions import Fraction

from .contract import PipelineError

KEY_SCALE = 1000  # canonical coordinates rounded to 1e-3 of a parent lattice vector
DIST_ROUND = 3  # angstrom digits for distance shells


def _frac_to_cart(lattice, f) -> tuple:
    return tuple(sum(f[k] * lattice[k][c] for k in range(3)) for c in range(3))


def _norm(v) -> float:
    return math.sqrt(sum(x * x for x in v))


def _widths(lattice) -> list:
    a = lattice
    cross = [tuple(a[(k + 1) % 3][(c + 1) % 3] * a[(k + 2) % 3][(c + 2) % 3]
                   - a[(k + 1) % 3][(c + 2) % 3] * a[(k + 2) % 3][(c + 1) % 3] for c in range(3)) for k in range(3)]
    volume = abs(sum(a[0][c] * cross[0][c] for c in range(3)))
    return [volume / _norm(cross[k]) for k in range(3)]


def _apply(op, f) -> tuple:
    rot, trans = op
    return tuple(sum(rot[r][c] * f[c] for c in range(3)) + float(trans[r]) for r in range(3))


def _ikey(v) -> tuple:
    return tuple(int(round(x * KEY_SCALE)) for x in v)


def _wrap_ikey(v) -> tuple:
    return tuple(k % KEY_SCALE for k in _ikey(v))


def site_key(op_list, f) -> tuple:
    """Canonical image of a parent site position (fractional) under the space group."""
    return min(_wrap_ikey(_apply(op, f)) for op in op_list)


def pair_key(op_list, type_a, fa, type_b, fb) -> tuple:
    """Canonical image of an ordered-agnostic pair (positions in parent fractional coordinates, unwrapped)."""
    best = None
    for op in op_list:
        qa, qb = _apply(op, fa), _apply(op, fb)
        for (ta, a), (tb, b) in (((type_a, qa), (type_b, qb)), ((type_b, qb), (type_a, qa))):
            shift = tuple(math.floor(x + 0.5 / KEY_SCALE) for x in a)
            key = (ta, _ikey(tuple(x - s for x, s in zip(a, shift))), tb, _ikey(tuple(x - s for x, s in zip(b, shift))))
            if best is None or key < best:
                best = key
    return best


def parent_sites(slots: list, diag) -> list:
    """Parent-cell position (wrapped) and parent-cell translation of every supercell slot."""
    out = []
    for s in slots:
        p = tuple(s["centroid_frac"][k] * diag[k] for k in range(3))
        cell = tuple(math.floor(x + 0.5 / KEY_SCALE) for x in p)
        out.append({"slot_id": s["slot_id"], "slot_type": s["slot_type"], "absolute": p,
                    "site": tuple(x - c for x, c in zip(p, cell)), "cell": cell})
    return out


def build_catalogue(lattice_info: dict, cutoff_A: float) -> dict:
    """Point and pair orbits of the parent lattice (independent of the supercell)."""
    ops = lattice_info["symmetry_operations"]
    parent_lattice = lattice_info["parent_lattice"]
    sites = {}
    for s in parent_sites(lattice_info["slots"], lattice_info["supercell_diagonal"]):
        sites.setdefault(_wrap_ikey(s["site"]), (s["slot_type"], s["site"]))
    site_list = [sites[k] for k in sorted(sites)]
    point_keys = sorted({(t, site_key(ops, f)) for t, f in site_list})
    pairs = {}
    reach = [math.ceil(cutoff_A / w) + 1 for w in _widths(parent_lattice)]
    for (ta, fa), (tb, fb) in itertools.product(site_list, repeat=2):
        for shift in itertools.product(*(range(-n, n + 1) for n in reach)):
            fb_shift = tuple(fb[k] + shift[k] for k in range(3))
            d = _norm(_frac_to_cart(parent_lattice, tuple(fb_shift[k] - fa[k] for k in range(3))))
            if 1e-6 < d <= cutoff_A + 1e-9:
                key = pair_key(ops, ta, fa, tb, fb_shift)
                pairs.setdefault(key, round(d, DIST_ROUND))
    ordered = sorted(pairs.items(), key=lambda kv: (kv[1], kv[0]))
    return {"cutoff_A": cutoff_A, "point_orbits": [{"name": f"point_orbit_{k}", "slot_type": t, "key": key}
                                                    for k, (t, key) in enumerate(point_keys)],
            "pair_orbits": [{"name": f"pair_orbit_{k}", "distance_A": d, "key": key} for k, (key, d) in enumerate(ordered)]}


def supercell_clusters(lattice_info: dict, catalogue: dict) -> tuple:
    """Cluster terms of one supercell: ((name, kind, distance, terms), ...) with terms over supercell slots.

    point terms: ((i, 1), ...); pair terms: ((i, j, multiplicity), ...) for i <= j, multiplicity = number of
    periodic images (i, j + T) in the orbit (self-image pairs i == j are kept: they never contribute to [x_i != x_j]).
    """
    ops = lattice_info["symmetry_operations"]
    diag = lattice_info["supercell_diagonal"]
    cutoff = catalogue["cutoff_A"]
    parent_lattice = lattice_info["parent_lattice"]
    sites = parent_sites(lattice_info["slots"], diag)
    point_index = {(p["slot_type"], p["key"]): p["name"] for p in catalogue["point_orbits"]}
    pair_index = {p["key"]: p for p in catalogue["pair_orbits"]}
    point_terms = {p["name"]: [] for p in catalogue["point_orbits"]}
    for s in sites:
        name = point_index.get((s["slot_type"], site_key(ops, s["site"])))
        if name is None:
            raise PipelineError("CLUSTER_CATALOGUE_MISMATCH", f"slot {s['slot_id']} site not in the parent catalogue")
        point_terms[name].append((s["slot_id"], 1))
    pair_terms = {p["name"]: {} for p in catalogue["pair_orbits"]}
    super_lattice = [[x * diag[k] for x in parent_lattice[k]] for k in range(3)]
    reach = [math.ceil(cutoff / w) + 1 for w in _widths(super_lattice)]
    for a, b in itertools.combinations_with_replacement(sites, 2):
        for shift in itertools.product(*(range(-n, n + 1) for n in reach)):
            if a["slot_id"] == b["slot_id"] and shift <= (0, 0, 0):
                continue  # each self-image pair once
            fb = tuple(b["absolute"][k] + shift[k] * diag[k] for k in range(3))
            d = _norm(_frac_to_cart(parent_lattice, tuple(fb[k] - a["absolute"][k] for k in range(3))))
            if d > cutoff + 1e-9 or d < 1e-6:
                continue
            orbit = pair_index.get(pair_key(ops, a["slot_type"], a["absolute"], b["slot_type"], fb))
            if orbit is None:
                raise PipelineError("CLUSTER_CATALOGUE_MISMATCH", f"pair {a['slot_id']}-{b['slot_id']} not in the catalogue")
            key = (a["slot_id"], b["slot_id"])
            pair_terms[orbit["name"]][key] = pair_terms[orbit["name"]].get(key, 0) + 1
    out = [(p["name"], "point", None, tuple(sorted(point_terms[p["name"]]))) for p in catalogue["point_orbits"]]
    out += [(p["name"], "pair", p["distance_A"], tuple(sorted((i, j, w) for (i, j), w in pair_terms[p["name"]].items())))
            for p in catalogue["pair_orbits"]]
    return tuple(out)


def cluster_lookup(topo) -> dict:
    return {name: (kind, distance, terms) for name, kind, distance, terms in topo.clusters}


def cluster_value(topo, name: str, bits) -> Fraction:
    kind, _, terms = cluster_lookup(topo)[name]
    if kind == "point":
        count = sum(bits[i] * w for i, w in terms)
    else:
        count = sum(w * (bits[i] != bits[j]) for i, j, w in terms)
    return Fraction(count, topo.n_formula_units)


def cluster_polynomial(topo, name: str) -> dict:
    """Binary polynomial of a cluster feature (per formula unit); x_i^2 = x_i."""
    kind, _, terms = cluster_lookup(topo)[name]
    n = topo.n_formula_units
    poly: dict = {}

    def add(key, value):
        poly[key] = poly.get(key, 0) + Fraction(value, n)

    if kind == "point":
        for i, w in terms:
            add((i,), w)
    else:
        for i, j, w in terms:
            if i == j:
                continue  # [x_i != x_i] = 0
            add((i,), w)
            add((j,), w)
            add(tuple(sorted((i, j))), -2 * w)
    return {k: v for k, v in poly.items() if v}


def names_for_token(topo, token: str) -> list:
    """Expand a model-candidate token: 'point_orbits' or 'pairs<=R' (R in angstrom)."""
    if token == "point_orbits":
        return [name for name, kind, _, _ in topo.clusters if kind == "point"]
    if token.startswith("pairs<="):
        radius = float(token[len("pairs<="):])
        return [name for name, kind, d, _ in topo.clusters if kind == "pair" and d <= radius + 1e-9]
    raise PipelineError("UNSUPPORTED_FEATURE", token)


def requested_cutoff(candidates: list) -> float | None:
    radii = [float(t[len("pairs<="):]) for c in candidates for t in c.get("features", []) if t.startswith("pairs<=")]
    uses_points = any(t == "point_orbits" for c in candidates for t in c.get("features", []))
    if not radii and not uses_points:
        return None
    return max(radii, default=0.0)
