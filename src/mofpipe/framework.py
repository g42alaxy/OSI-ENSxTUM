"""Task 01 (structures, atomic mode): periodic bond graph, slot lattice and framework builder.

A parsed parent CIF is expanded to the declared supercell and deconstructed
into metal nodes and organic linkers. Every linker is one substitution slot;
slots that share a node are distance-1 neighbours. The resulting slot lattice
is emitted as the same TopologySpec record the graph pipeline consumes.

The builder decorates the parent by explicit, declared substitution rules
(``ParentSubstitutionBuilder``). Construction failures are BuildFailure records
with a reason, never energies.
"""
from __future__ import annotations

import importlib.util
import itertools
import math
from collections import Counter, deque
from fractions import Fraction

from .cif import write_cif
from .contract import PipelineError, StageStatus, content_hash, make_record

# Cordero et al., Dalton Trans. 2008, 2832 (low-spin values for Mn/Fe/Co).
COVALENT_RADII_A = {
    "H": 0.31, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "Si": 1.11, "P": 1.07, "S": 1.05,
    "Cl": 1.02, "Br": 1.20, "I": 1.39, "Li": 1.28, "Na": 1.66, "Mg": 1.41, "Al": 1.21, "K": 2.03, "Ca": 1.76,
    "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39, "Mn": 1.39, "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32,
    "Zn": 1.22, "Ga": 1.22, "Sr": 1.95, "Y": 1.90, "Zr": 1.75, "Nb": 1.64, "Mo": 1.54, "Ru": 1.46, "Rh": 1.42,
    "Pd": 1.39, "Ag": 1.45, "Cd": 1.44, "In": 1.42, "Sn": 1.39, "Ba": 2.15, "La": 2.07, "Ce": 2.04, "Hf": 1.75,
    "W": 1.62, "Pt": 1.36, "Au": 1.36, "Pb": 1.46, "Bi": 1.48,
}
METALS = {"Li", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Sr",
          "Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Ba", "La", "Ce", "Hf", "W", "Pt", "Au",
          "Pb", "Bi"}
SLOT_MATCH_TOL = 1e-3  # fractional coordinate tolerance when matching slot centroids under symmetry
GRAPH_REPRESENTATION = "periodic_slot_lattice_from_cif"
_IMAGES = tuple(itertools.product((-1, 0, 1), repeat=3))


# -- periodic geometry --------------------------------------------------------
def frac_to_cart(lattice, f) -> tuple:
    return tuple(sum(f[k] * lattice[k][c] for k in range(3)) for c in range(3))


def _norm(v) -> float:
    return math.sqrt(sum(x * x for x in v))


def _sub(a, b) -> tuple:
    return tuple(x - y for x, y in zip(a, b))


def _cart_to_frac(lattice, r) -> tuple:
    (a, b, c) = lattice
    det = (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]))
    inv = [[(b[1] * c[2] - b[2] * c[1]) / det, (a[2] * c[1] - a[1] * c[2]) / det, (a[1] * b[2] - a[2] * b[1]) / det],
           [(b[2] * c[0] - b[0] * c[2]) / det, (a[0] * c[2] - a[2] * c[0]) / det, (a[2] * b[0] - a[0] * b[2]) / det],
           [(b[0] * c[1] - b[1] * c[0]) / det, (a[1] * c[0] - a[0] * c[1]) / det, (a[0] * b[1] - a[1] * b[0]) / det]]
    return tuple(sum(r[k] * inv[k][j] for k in range(3)) for j in range(3))


def _wrap(x: float) -> float:
    x = x - math.floor(x)
    return 0.0 if abs(x - 1.0) < 1e-9 else x


def expand_supercell(structure: dict, matrix) -> dict:
    """Replicate a P1 structure by a diagonal integer supercell matrix."""
    diag = [matrix[i][i] for i in range(3)]
    if any(matrix[i][j] for i in range(3) for j in range(3) if i != j) or any(type(n) is not int or n < 1 for n in diag):
        raise PipelineError("UNSUPPORTED_SUPERCELL_MATRIX", f"{matrix}: only positive diagonal matrices are supported")
    lattice = tuple(tuple(v * diag[k] for v in structure["lattice"][k]) for k in range(3))
    atoms = []
    for shift in itertools.product(*(range(n) for n in diag)):
        for atom in structure["atoms"]:
            atoms.append({**atom, "frac": tuple((atom["frac"][k] + shift[k]) / diag[k] for k in range(3))})
    return {"lattice": lattice, "atoms": atoms, "supercell_diagonal": diag}


def _perpendicular_widths(lattice) -> list:
    a = lattice
    cross = [tuple(a[(k + 1) % 3][(c + 1) % 3] * a[(k + 2) % 3][(c + 2) % 3]
                   - a[(k + 1) % 3][(c + 2) % 3] * a[(k + 2) % 3][(c + 1) % 3] for c in range(3)) for k in range(3)]
    volume = abs(sum(a[0][c] * cross[0][c] for c in range(3)))
    return [volume / _norm(cross[k]) for k in range(3)]


def _close_pairs(structure: dict, cutoff_of, max_cutoff: float):
    """Yield (i, j, image, distance) for every periodic pair within cutoff_of(i, j) <= max_cutoff (self-images included).

    When every perpendicular cell width exceeds twice max_cutoff, a vector shorter than the cutoff has all
    fractional components below 1/2, so the minimum-image reduction is the only candidate (fast path).
    """
    atoms, lattice = structure["atoms"], structure["lattice"]
    fracs = [a["frac"] for a in atoms]
    if min(_perpendicular_widths(lattice)) > 2 * max_cutoff:
        (ax, ay, az), (bx, by, bz), (cx, cy, cz) = lattice
        limit = max_cutoff * max_cutoff
        n = len(fracs)
        for i in range(n):
            fi0, fi1, fi2 = fracs[i]
            for j in range(i + 1, n):
                fj = fracs[j]
                d0, d1, d2 = fj[0] - fi0, fj[1] - fi1, fj[2] - fi2
                i0, i1, i2 = -round(d0), -round(d1), -round(d2)
                d0, d1, d2 = d0 + i0, d1 + i1, d2 + i2
                x, y, z = d0 * ax + d1 * bx + d2 * cx, d0 * ay + d1 * by + d2 * cy, d0 * az + d1 * bz + d2 * cz
                r2 = x * x + y * y + z * z
                if r2 <= limit:
                    r = math.sqrt(r2)
                    if r <= cutoff_of(i, j):
                        yield i, j, (i0, i1, i2), r
        return
    for i, j in itertools.combinations_with_replacement(range(len(atoms)), 2):
        for img in _IMAGES:
            if i == j and img == (0, 0, 0):
                continue
            r = _norm(frac_to_cart(lattice, [fracs[j][k] + img[k] - fracs[i][k] for k in range(3)]))
            if r <= cutoff_of(i, j):
                yield i, j, img, r


def bond_graph(structure: dict, tolerance: float) -> list:
    """Adjacency list of (neighbour, image) with d <= r_i + r_j + tolerance under periodic images."""
    atoms = structure["atoms"]
    for a in atoms:
        if a["element"] not in COVALENT_RADII_A:
            raise PipelineError("UNSUPPORTED_ELEMENT", a["element"])
    radii = [COVALENT_RADII_A[a["element"]] for a in atoms]
    adjacency = [[] for _ in atoms]
    for i, j, img, _ in _close_pairs(structure, lambda i, j: radii[i] + radii[j] + tolerance,
                                     2 * max(radii, default=0.0) + tolerance):
        adjacency[i].append((j, img))
        if i != j:
            adjacency[j].append((i, tuple(-x for x in img)))
    return adjacency


def min_interatomic_distance(structure: dict, below: float) -> float:
    """Smallest periodic interatomic distance if it is below `below` (angstrom), else `below`."""
    return min((r for *_, r in _close_pairs(structure, lambda i, j: below, below)), default=below)


def _components(members: set, adjacency: list) -> list:
    """Connected components restricted to `members`, with per-atom unwrap offsets; flags periodic (infinite) ones."""
    seen, comps = {}, []
    for start in sorted(members):
        if start in seen:
            continue
        offsets, queue, infinite = {start: (0, 0, 0)}, deque([start]), False
        seen[start] = len(comps)
        while queue:
            i = queue.popleft()
            for j, img in adjacency[i]:
                if j not in members:
                    continue
                off = tuple(offsets[i][k] + img[k] for k in range(3))
                if j in offsets:
                    infinite |= offsets[j] != off
                    continue
                offsets[j], seen[j] = off, len(comps)
                queue.append(j)
        comps.append({"atoms": sorted(offsets), "offsets": offsets, "infinite": infinite})
    return comps


def _centroid(atoms, members, offsets) -> tuple:
    unwrapped = [tuple(atoms[i]["frac"][k] + offsets[i][k] for k in range(3)) for i in members]
    return tuple(_wrap(sum(u[k] for u in unwrapped) / len(unwrapped)) for k in range(3))


def formula_counts(elements) -> dict:
    return dict(sorted(Counter(elements).items()))


def deconstruct(structure: dict, bond_tolerance: float) -> dict:
    """Split a periodic framework into metal-oxo nodes, linkers and unbound guests."""
    atoms = structure["atoms"]
    adjacency = bond_graph(structure, bond_tolerance)
    metal = {i for i, a in enumerate(atoms) if a["element"] in METALS}
    # Non-metal atoms bonded only to metals (mu-oxo) belong to the node.
    oxo = {i for i in range(len(atoms)) if i not in metal and adjacency[i]
           and all(j in metal for j, _ in adjacency[i])}
    node_atoms = metal | oxo
    organic = set(range(len(atoms))) - node_atoms
    nodes = _components(node_atoms, adjacency)
    linkers = _components(organic, adjacency)
    node_of = {i: n for n, comp in enumerate(nodes) for i in comp["atoms"]}
    lattice = structure["lattice"]
    out_linkers, guests = [], []
    for comp in linkers:
        elements = [atoms[i]["element"] for i in comp["atoms"]]
        connectors, incidences = [], set()
        for i in comp["atoms"]:
            for j, img in adjacency[i]:
                if j in node_atoms:
                    connectors.append(i)
                    node = nodes[node_of[j]]
                    frame = tuple(img[k] + comp["offsets"][i][k] - node["offsets"][j][k] for k in range(3))
                    incidences.add((node_of[j], frame))
        record = {"atoms": comp["atoms"], "offsets": comp["offsets"], "formula": formula_counts(elements),
                  "connector_atoms": sorted(set(connectors)), "node_incidences": sorted(incidences),
                  "infinite": comp["infinite"]}
        record["centroid_frac"] = _centroid(atoms, comp["atoms"], comp["offsets"])
        if record["connector_atoms"]:
            record["connector_centroid_frac"] = _centroid(atoms, record["connector_atoms"], comp["offsets"])
        (out_linkers if incidences else guests).append(record)
    out_nodes = [{"atoms": n["atoms"], "offsets": n["offsets"], "infinite": n["infinite"],
                  "centroid_frac": _centroid(atoms, n["atoms"], n["offsets"]),
                  "formula": formula_counts(atoms[i]["element"] for i in n["atoms"])} for n in nodes]
    for n_id, node in enumerate(out_nodes):
        node["incidences"] = sorted((l_id, frame) for l_id, l in enumerate(out_linkers)
                                    for nid, frame in l["node_incidences"] if nid == n_id)
    return {"nodes": out_nodes, "linkers": out_linkers, "guests": guests, "adjacency": adjacency,
            "lattice": lattice}


# -- slot lattice -------------------------------------------------------------
def _slot_key(linker) -> tuple:
    return tuple(round(c, 3) % 1.0 for c in linker["centroid_frac"]) + (tuple(linker["formula"].items()),)


def _frac_close(f, g, tol=SLOT_MATCH_TOL) -> bool:
    return all(abs((f[k] - g[k]) - round(f[k] - g[k])) <= tol for k in range(3))


def slot_permutations(slots: list, symmetry_operations: list, diag: list) -> list:
    """Slot permutations induced by parent-cell symmetry operations combined with supercell translations."""
    perms = set()
    for (rot, trans) in symmetry_operations:
        for shift in itertools.product(*(range(n) for n in diag)):
            perm, ok = [], True
            for slot in slots:
                p = tuple(slot["centroid_frac"][k] * diag[k] for k in range(3))
                q = tuple(sum(rot[r][c] * p[c] for c in range(3)) + float(trans[r]) + shift[r] for r in range(3))
                image = tuple(_wrap(q[k] / diag[k]) for k in range(3))
                match = [s["slot_id"] for s in slots if s["slot_type"] == slot["slot_type"]
                         and _frac_close(s["centroid_frac"], image)]
                if len(match) != 1:
                    ok = False
                    break
                perm.append(match[0])
            if ok and sorted(perm) == list(range(len(slots))):
                perms.add(tuple(perm))
    return sorted(perms)


def slot_graph(n_slots: int, contacts: dict) -> dict:
    """Pairs at slot-graph distance 1 (with periodic contact multiplicity) and 2, and connected 3-slot paths.

    ``contacts[(i, j)]`` counts node-mediated contacts per cell: sum over nodes of ends(i) x ends(j). In small
    periodic cells two slots can meet at both of their nodes, or reach a node through two images.
    """
    adjacent = set(contacts)
    neigh = {i: set() for i in range(n_slots)}
    for i, j in adjacent:
        neigh[i].add(j)
        neigh[j].add(i)
    d2 = set()
    for i in range(n_slots):
        for j in neigh[i]:
            for k in neigh[j]:
                if k != i and k not in neigh[i]:
                    d2.add(tuple(sorted((i, k))))
    triplets = set()
    for j in range(n_slots):
        for i, k in itertools.combinations(sorted(neigh[j]), 2):
            triplets.add(tuple(sorted((i, j, k))))
    pairs = sorted(adjacent)
    return {"pairs_d1": pairs, "pair_weights_d1": [contacts[p] for p in pairs], "pairs_d2": sorted(d2),
            "triplets": sorted(triplets)}


def species_for_formula(formula: dict, blocks: dict, role: str):
    hits = [bid for bid, b in blocks.items() if b.get("role") == role and b.get("formula") == formula]
    return hits[0] if len(hits) == 1 else None


def build_slot_lattice(parent: dict, topology_id: str, supercell_matrix, chemistry: dict, blocks: dict,
                       builder_protocol: dict) -> dict:
    """Parsed parent CIF -> checked slot lattice (TopologySpec fields) or a list of typed errors."""
    errors = []

    def fail(code, detail):
        errors.append({"code": code, "check": f"{topology_id}:{code.lower()}", "detail": detail})

    struct = expand_supercell({"lattice": parent["lattice"], "atoms": parent["atoms"]}, supercell_matrix)
    parts = deconstruct(struct, builder_protocol["bond_tolerance_A"])
    if parts["guests"]:
        policy = chemistry.get("solvent_and_counterion_policy")
        if policy == "remove_unbound_guests":
            removed = {i for g in parts["guests"] for i in g["atoms"]}
            struct = {**struct, "atoms": [a for i, a in enumerate(struct["atoms"]) if i not in removed]}
            parts = {**deconstruct(struct, builder_protocol["bond_tolerance_A"]),
                     "removed_guests": [g["formula"] for g in parts["guests"]]}
        else:
            fail("UNRESOLVED_GUEST_SPECIES", f"{[g['formula'] for g in parts['guests']]} under policy {policy!r}")
    for kind in ("nodes", "linkers"):
        if any(c["infinite"] for c in parts[kind]):
            fail("PERIODIC_BUILDING_UNIT", f"a {kind[:-1]} component is connected to its own periodic image")
    node_species = [species_for_formula(n["formula"], blocks, "node") for n in parts["nodes"]]
    linker_species = [species_for_formula(l["formula"], blocks, "linker") for l in parts["linkers"]]
    for formula, sp in zip([n["formula"] for n in parts["nodes"]], node_species):
        if sp is None:
            fail("UNKNOWN_NODE", str(formula))
    for formula, sp in zip([l["formula"] for l in parts["linkers"]], linker_species):
        if sp is None:
            fail("UNKNOWN_LINKER", str(formula))
    if errors:
        return {"errors": errors}

    # Connector count/geometry: declared connector atoms and points of extension per block.
    for l, sp in zip(parts["linkers"], linker_species):
        block = blocks[sp]
        if len(l["connector_atoms"]) != block["n_connector_atoms"]:
            fail("INVALID_CONNECTOR_COUNT", f"{sp}: {len(l['connector_atoms'])} != {block['n_connector_atoms']}")
        if len(l["node_incidences"]) != block["connectivity"]:
            fail("INVALID_CONNECTIVITY", f"{sp}: binds {len(l['node_incidences'])} nodes, declared {block['connectivity']}")
    for n, sp in zip(parts["nodes"], node_species):
        if len(n["incidences"]) != blocks[sp]["connectivity"]:
            fail("INVALID_CONNECTIVITY", f"{sp}: {len(n['incidences'])} linker contacts, declared {blocks[sp]['connectivity']}")

    # Slots: one per linker, IDs stable under atom reordering (sorted by wrapped centroid, then formula).
    order = sorted(range(len(parts["linkers"])), key=lambda i: _slot_key(parts["linkers"][i]))
    slots = []
    for slot_id, li in enumerate(order):
        l = parts["linkers"][li]
        slots.append({"slot_id": slot_id, "linker_index": li, "slot_type": blocks[linker_species[li]]["slot_type"],
                      "parent_species": linker_species[li], "centroid_frac": l["centroid_frac"],
                      "connector_centroid_frac": l["connector_centroid_frac"],
                      "atoms": l["atoms"], "node_incidences": l["node_incidences"]})
    slot_of = {s["linker_index"]: s["slot_id"] for s in slots}
    contacts = Counter()
    for n in parts["nodes"]:
        ends = Counter(slot_of[l_id] for l_id, _ in n["incidences"])
        for i, j in itertools.combinations(sorted(ends), 2):
            contacts[(i, j)] += ends[i] * ends[j]
    graph = slot_graph(len(slots), dict(contacts))
    perms = slot_permutations(slots, parent["symmetry_operations"], struct["supercell_diagonal"])

    # Sublattice composition: binary substitution on one slot type, integral counts in this cell.
    comp = chemistry["composition_by_sublattice"]
    slot_types = Counter(s["slot_type"] for s in slots)
    counts = {}
    for sublattice, fractions in comp.items():
        n = slot_types.get(sublattice, 0)
        if n == 0:
            fail("UNKNOWN_SUBLATTICE", sublattice)
            continue
        for species, fraction in fractions.items():
            value = n * _fraction(fraction)
            if value.denominator != 1:
                fail("NONINTEGRAL_COMPOSITION", f"{sublattice}:{species} = {n} x {fraction} = {value}")
            counts[species] = int(value)
        if sum(_fraction(f) for f in fractions.values()) != 1:
            fail("INVALID_COMPOSITION_SPEC", f"{sublattice} fractions do not sum to 1")
    if set(slot_types) - set(comp):
        fail("UNDECLARED_SUBLATTICE", str(sorted(set(slot_types) - set(comp))))
    if errors:
        return {"errors": errors}

    # Inventory of the decorated target and charge balance under the supplied assignments.
    parent_inventory = Counter(a["element"] for a in struct["atoms"])
    target = Counter(parent_inventory)
    for species, count in counts.items():
        delta = species_delta(blocks, species)
        for el, d in delta.items():
            target[el] += d * count
    per_fu = chemistry["target_elemental_inventory"]
    ratios = {el: _ratio(target.get(el, 0), per_fu.get(el, 0)) for el in set(target) | set(per_fu)}
    values = set(ratios.values())
    n_fu = values.pop() if len(values) == 1 else None
    if n_fu is None or n_fu.denominator != 1 or n_fu <= 0:
        fail("INVENTORY_MISMATCH", f"cell inventory {dict(sorted(target.items()))} is not an integer multiple of "
                                   f"target_elemental_inventory {per_fu}")
    charge = sum(blocks[s]["formal_charge"] for s in node_species)
    charge += sum(blocks[s]["formal_charge"] * c for s, c in counts.items())
    expected_charge = chemistry["net_charge"] * (int(n_fu) if n_fu and n_fu.denominator == 1 else 0)
    if charge != expected_charge:
        fail("CHARGE_IMBALANCE", f"supplied formal charges sum to {charge}, expected {expected_charge}")
    if errors:
        return {"errors": errors}
    return {
        "errors": [], "structure": struct, "slots": slots, "nodes": parts["nodes"], "node_species": node_species,
        "parent_lattice": parent["lattice"], "symmetry_operations": parent["symmetry_operations"],
        "supercell_diagonal": struct["supercell_diagonal"],
        "graph": graph, "permutations": perms, "species_counts": counts, "n_formula_units": int(n_fu),
        "target_inventory": dict(sorted(target.items())), "parent_inventory": dict(sorted(parent_inventory.items())),
        "removed_guests": parts.get("removed_guests", []), "net_charge_check": {"total": charge, "assumption": "SUPPLIED_NOT_INFERRED"},
    }


def _fraction(value) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise PipelineError("INVALID_NUMBER", repr(value))
    return Fraction(str(value))


def _ratio(a, b):
    return Fraction(a, b) if b else None


def species_delta(blocks: dict, species: str) -> dict:
    """Element change when a slot's parent linker is replaced by `species` (empty for an unsubstituted block)."""
    sub = blocks[species].get("substitution")
    if not sub:
        return {}
    new, parent = Counter(blocks[species]["formula"]), Counter(blocks[sub["parent_block"]]["formula"])
    return {el: new[el] - parent[el] for el in sorted(set(new) | set(parent)) if new[el] != parent[el]}


# -- builder ------------------------------------------------------------------
def geometry_hashes(lattice, elements, fracs) -> tuple[str, str]:
    """(ordered hash in builder order, order-independent canonical hash)."""
    rows = [(e, tuple(round(_wrap(x), 5) % 1.0 for x in f)) for e, f in zip(elements, fracs)]
    lat = [[round(x, 5) for x in v] for v in lattice]
    return content_hash({"lattice": lat, "atoms": rows}), content_hash({"lattice": lat, "atoms": sorted(rows)})


class BuildFailure(Exception):
    def __init__(self, code: str, detail: str):
        super().__init__(code)
        self.code, self.detail = code, detail


class ParentSubstitutionBuilder:
    """Decorate the parent supercell by replacing parent linkers with declared substitution variants.

    Orientation convention (deterministic initial orientation, recipe orientation_policy): the replaced
    atom is the first ``replace_element`` atom of the linker in Cartesian order relative to the linker
    centroid (rounded to 1e-3 angstrom). Orientation states are not yet search variables (T01.4).
    """
    name = "parent_substitution_v1"

    def __init__(self, lattice_info: dict, blocks: dict, protocol: dict):
        self.lattice_info, self.blocks, self.protocol = lattice_info, blocks, protocol

    def _substitute(self, struct, slot, species):
        block = self.blocks[species]
        sub = block.get("substitution")
        if not sub or sub.get("parent_block") != slot["parent_species"]:
            raise BuildFailure("NO_CONSTRUCTION_RULE", f"{slot['parent_species']} -> {species}")
        lattice, atoms = struct["lattice"], struct["atoms"]
        linker = self.lattice_info["linkers_by_slot"][slot["slot_id"]]
        pos = {i: frac_to_cart(lattice, tuple(atoms[i]["frac"][k] + linker["offsets"][i][k] for k in range(3)))
               for i in slot["atoms"]}
        centre = tuple(sum(p[k] for p in pos.values()) / len(pos) for k in range(3))
        targets = sorted((i for i in slot["atoms"] if atoms[i]["element"] == sub["replace_element"]),
                         key=lambda i: tuple(round(x, 3) for x in _sub(pos[i], centre)))
        if not targets:
            raise BuildFailure("NO_SUBSTITUTION_SITE", f"slot {slot['slot_id']} has no {sub['replace_element']}")
        h = targets[0]
        anchors = [j for j, _ in self.lattice_info["adjacency"][h] if j in pos]
        if len(anchors) != 1:
            raise BuildFailure("AMBIGUOUS_SUBSTITUTION_ANCHOR", f"slot {slot['slot_id']} atom {h}")
        anchor = pos[anchors[0]]
        u = _sub(pos[h], anchor)
        u = tuple(x / _norm(u) for x in u)
        ends = sorted(pos[i] for i in linker["connector_atoms"])
        if len(ends) < 2:
            raise BuildFailure("DEGENERATE_SUBSTITUTION_FRAME", f"slot {slot['slot_id']} has < 2 connector atoms")
        axis = _sub(ends[-1], ends[0])
        proj = sum(a * b for a, b in zip(axis, u))
        v = tuple(a - proj * b for a, b in zip(axis, u))
        if _norm(v) < 1e-6:
            raise BuildFailure("DEGENERATE_SUBSTITUTION_FRAME", f"slot {slot['slot_id']}")
        v = tuple(x / _norm(v) for x in v)
        w = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
        new_atoms = []
        for g in sub["group"]:
            a, b, c = g["frame_A"]
            r = tuple(anchor[k] + a * u[k] + b * v[k] + c * w[k] for k in range(3))
            new_atoms.append({"element": g["element"], "frac": tuple(_wrap(x) for x in _cart_to_frac(lattice, r)),
                              "slot": slot["slot_id"], "block": species})
        return h, new_atoms

    def build(self, topology_id: str, species_by_slot: list) -> dict:
        info = self.lattice_info
        struct = info["structure"]
        atoms = [{**a, "slot": None, "block": None} for a in struct["atoms"]]
        for n_id, node in enumerate(info["nodes"]):
            for i in node["atoms"]:
                atoms[i].update(slot=f"node{n_id}", block=info["node_species"][n_id])
        for slot in info["slots"]:
            for i in slot["atoms"]:
                atoms[i].update(slot=slot["slot_id"], block=slot["parent_species"])
        removed, added = set(), []
        for slot, species in zip(info["slots"], species_by_slot):
            if species == slot["parent_species"]:
                continue
            h, new = self._substitute(struct, slot, species)
            removed.add(h)
            for i in slot["atoms"]:
                atoms[i]["block"] = species
            added.extend(new)
        built = [a for i, a in enumerate(atoms) if i not in removed] + added
        geometry = {"lattice": struct["lattice"], "atoms": built}
        return self.check(topology_id, geometry, species_by_slot)

    def check(self, topology_id: str, geometry: dict, species_by_slot: list) -> dict:
        """Early sanity checks: overlaps, inventory and slot/node connectivity against the parent."""
        dmin = min_interatomic_distance(geometry, self.protocol["min_interatomic_distance_A"])
        if dmin < self.protocol["min_interatomic_distance_A"]:
            raise BuildFailure("ATOM_OVERLAP", f"{dmin:.3f} A < {self.protocol['min_interatomic_distance_A']} A")
        expected = Counter(self.lattice_info["structure_inventory"])
        for slot, species in zip(self.lattice_info["slots"], species_by_slot):
            if species != slot["parent_species"]:
                for el, d in species_delta(self.blocks, species).items():
                    expected[el] += d
        inventory = Counter(a["element"] for a in geometry["atoms"])
        if +inventory != +expected:
            raise BuildFailure("INVENTORY_MISMATCH", f"{dict(inventory)} != {dict(expected)}")
        mapping = map_to_parent_slots(geometry, self.lattice_info, self.protocol["bond_tolerance_A"],
                                      self.protocol["slot_match_tolerance_A"])
        if not mapping["retained"]:
            raise BuildFailure("CONNECTIVITY_CHANGED", mapping["reason"])
        return geometry


def _min_image_distance(lattice, f, g) -> float:
    d = tuple((f[k] - g[k]) - round(f[k] - g[k]) for k in range(3))
    return min(_norm(frac_to_cart(lattice, tuple(d[k] + img[k] for k in range(3)))) for img in _IMAGES)


def _match(lattice, found: list, reference: list, tol: float):
    """Unique nearest-neighbour assignment of found -> reference centroids within tol (angstrom)."""
    out = []
    for f in found:
        hits = [r for r, g in enumerate(reference) if _min_image_distance(lattice, f, g) <= tol]
        if len(hits) != 1:
            return None
        out.append(hits[0])
    return out if sorted(out) == list(range(len(reference))) else None


def map_to_parent_slots(geometry: dict, info: dict, bond_tolerance: float, tolerance_A: float) -> dict:
    """Reference-slot mapping of a built or relaxed geometry; reports whether slot/node topology is retained."""
    parts = deconstruct(geometry, bond_tolerance)
    if parts["guests"] or any(c["infinite"] for c in parts["nodes"] + parts["linkers"]):
        return {"retained": False, "reason": "DISCONNECTED_OR_PERIODIC_UNIT"}
    if len(parts["linkers"]) != len(info["slots"]) or len(parts["nodes"]) != len(info["nodes"]):
        return {"retained": False, "reason": f"UNIT_COUNT_CHANGED: {len(parts['linkers'])} linkers, {len(parts['nodes'])} nodes"}
    lattice = geometry["lattice"]
    slot_of = _match(lattice, [l.get("connector_centroid_frac") for l in parts["linkers"]],
                     [s["connector_centroid_frac"] for s in info["slots"]], tolerance_A)
    node_of = _match(lattice, [n["centroid_frac"] for n in parts["nodes"]],
                     [n["centroid_frac"] for n in info["nodes"]], tolerance_A)
    if slot_of is None or node_of is None:
        return {"retained": False, "reason": "SLOT_MAPPING_FAILED"}
    found = sorted((slot_of[i], tuple(sorted(Counter(node_of[n] for n, _ in l["node_incidences"]).items())))
                   for i, l in enumerate(parts["linkers"]))
    reference = sorted((s["slot_id"], tuple(sorted(Counter(n for n, _ in s["node_incidences"]).items())))
                       for s in info["slots"])
    if found != reference:
        return {"retained": False, "reason": "SLOT_NODE_INCIDENCES_CHANGED"}
    return {"retained": True, "reason": None, "linker_to_slot": slot_of}


class PormakeBuilder:
    """Adapter slot for PORMAKE (topology + building-block assembly).

    The installed API and block database must be verified before use; until then every request is a
    BuildFailure that names the missing verification, never a fabricated geometry.
    """
    name = "pormake"

    def __init__(self, *_args, **_kwargs):
        self.available = importlib.util.find_spec("pormake") is not None

    def build(self, topology_id: str, species_by_slot: list) -> dict:
        #TODO: Task 01 — wire PORMAKE's topology/block API after verifying the installed version and its block
        #TODO: database (validated 3D blocks with attachment points); add ToBaCCo behind this same adapter.
        if not self.available:
            raise BuildFailure("BUILDER_UNAVAILABLE", "pormake is not installed")
        raise BuildFailure("BUILDER_NOT_VERIFIED", "pormake API/block database not verified for this project")


BUILDERS = {ParentSubstitutionBuilder.name: ParentSubstitutionBuilder, PormakeBuilder.name: PormakeBuilder}


def lattice_info_for_builder(lattice: dict, protocol: dict) -> dict:
    parts = deconstruct(lattice["structure"], protocol["bond_tolerance_A"])
    by_slot = {}
    for slot in lattice["slots"]:
        by_slot[slot["slot_id"]] = parts["linkers"][slot["linker_index"]]
    return {**lattice, "linkers_by_slot": by_slot, "adjacency": parts["adjacency"],
            "structure_inventory": Counter(a["element"] for a in lattice["structure"]["atoms"])}


def build_framework(spec, topology, decoration: dict) -> dict:
    """build_framework(topology_id, decoration, builder_protocol_id) -> Geometry | BuildFailure record."""
    physical = spec.physical
    protocol = physical["protocols"]["builder"]
    builder_cls = BUILDERS.get(protocol["builder_id"])
    common = {"topology_id": topology.topology_id, "decoration_id": decoration["configuration_id"],
              "orbit_id": decoration["orbit_id"], "builder": protocol["builder_id"],
              "builder_protocol_hash": content_hash(protocol)}
    if builder_cls is None:
        return make_record("BuildFailure", spec.provenance, **common, status=StageStatus.UNSUPPORTED.value,
                           failure={"code": "UNKNOWN_BUILDER", "detail": protocol["builder_id"]})
    info = physical["builder_info"][topology.topology_id]
    try:
        geometry = builder_cls(info, physical["blocks"], protocol).build(topology.topology_id, decoration["species_by_slot"])
    except BuildFailure as exc:
        return make_record("BuildFailure", spec.provenance, **common, status=StageStatus.FAILED.value,
                           failure={"code": exc.code, "detail": exc.detail})
    elements = [a["element"] for a in geometry["atoms"]]
    fracs = [tuple(round(x, 6) for x in a["frac"]) for a in geometry["atoms"]]
    ordered, canonical = geometry_hashes(geometry["lattice"], elements, fracs)
    return make_record(
        "Geometry", spec.provenance, **common, status=StageStatus.COMPLETED.value,
        lattice_A=[list(v) for v in geometry["lattice"]], elements=elements, frac_coords=[list(f) for f in fracs],
        atom_map=[{"block": a["block"], "slot": a["slot"]} for a in geometry["atoms"]],
        periodic_adjacency={"slot_pairs_distance_1": [list(p) for p in topology.pairs_d1],
                            "slot_node_incidences": {str(s["slot_id"]): len(s["node_incidences"]) for s in info["slots"]}},
        inventory=formula_counts(elements), geometry_hash=ordered, canonical_geometry_hash=canonical,
        orientation_convention="first replace_element atom in Cartesian order about the linker centroid",
        cif=write_cif(decoration["configuration_id"].replace(":", "_"), geometry["lattice"], elements, fracs),
        relaxation_protocol=None, validation_status="BUILDER_SANITY_PASSED",
    )
