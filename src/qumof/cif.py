"""Task 01 (intake): minimal standard-library CIF reader and writer.

Reads the cell, symmetry operations and the atom-site loop of a single-block
CIF, expands the asymmetric unit to P1 and reports partial occupancy and
disorder instead of resolving them silently. The original text and its SHA-256
are kept with every parse.
"""
from __future__ import annotations

import hashlib
import math
import re
import shlex
from fractions import Fraction

from .contract import PipelineError

POSITION_DEDUP_TOL = 1e-3  # fractional; symmetry copies closer than this are one site
_NUMBER = re.compile(r"^([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)(?:\(\d+\))?$")


def cif_number(token: str) -> float:
    """CIF numeric value with an optional standard uncertainty in brackets, e.g. 0.1234(5)."""
    match = _NUMBER.match(token.strip())
    if not match:
        raise PipelineError("CIF_PARSE_ERROR", f"not a number: {token!r}")
    return float(match.group(1))


def _tokens(line: str) -> list:
    try:
        return shlex.split(line, posix=True)
    except ValueError as exc:
        raise PipelineError("CIF_PARSE_ERROR", f"{line!r}: {exc}") from exc


def _parse_items(text: str) -> tuple[dict, list]:
    """Return (scalar items, loops). Loops are (column names, rows of tokens)."""
    items, loops = {}, []
    lines = [l for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#")]
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.lower().startswith("data_"):
            i += 1
            continue
        if line.lower() == "loop_":
            names, i = [], i + 1
            while i < len(lines) and lines[i].strip().startswith("_"):
                names.append(lines[i].strip().split()[0].lower())
                i += 1
            values = []
            while i < len(lines) and not lines[i].strip().startswith("_") and lines[i].strip().lower() != "loop_" \
                    and not lines[i].strip().lower().startswith("data_"):
                if lines[i].startswith(";"):
                    raise PipelineError("CIF_PARSE_ERROR", "multi-line text fields inside loops are not supported")
                values.extend(_tokens(lines[i]))
                i += 1
            if not names or len(values) % len(names):
                raise PipelineError("CIF_PARSE_ERROR", f"loop {names[:1]} has {len(values)} values for {len(names)} columns")
            loops.append((names, [values[k:k + len(names)] for k in range(0, len(values), len(names))]))
            continue
        if line.startswith("_"):
            parts = _tokens(line)
            if len(parts) >= 2:
                items[parts[0].lower()] = " ".join(parts[1:])
                i += 1
            elif i + 1 < len(lines) and lines[i + 1].startswith(";"):
                block, i = [], i + 2
                while i < len(lines) and not lines[i].startswith(";"):
                    block.append(lines[i])
                    i += 1
                items[parts[0].lower()] = "\n".join(block)
                i += 1
            elif i + 1 < len(lines):
                items[parts[0].lower()] = " ".join(_tokens(lines[i + 1]))
                i += 2
            else:
                raise PipelineError("CIF_PARSE_ERROR", f"item without value: {parts[0]}")
            continue
        i += 1  # unrecognised free text is ignored
    return items, loops


def _loop_with(loops, name):
    for names, rows in loops:
        if name in names:
            return names, rows
    return None, []


def parse_symop(text: str) -> tuple:
    """'x+1/2,-y,z' -> (3x3 integer rotation, translation as Fractions)."""
    parts = [p.strip().lower().replace(" ", "") for p in text.strip().strip("'\"").split(",")]
    if len(parts) != 3:
        raise PipelineError("CIF_PARSE_ERROR", f"symmetry operation {text!r}")
    rot, trans = [], []
    for part in parts:
        row, shift = [0, 0, 0], Fraction(0)
        for sign, term in re.findall(r"([+-]?)([^+-]+)", part):
            s = -1 if sign == "-" else 1
            if term in ("x", "y", "z"):
                row["xyz".index(term)] += s
            else:
                try:
                    shift += s * Fraction(term)
                except (ValueError, ZeroDivisionError) as exc:
                    raise PipelineError("CIF_PARSE_ERROR", f"symmetry operation {text!r}") from exc
        rot.append(tuple(row))
        trans.append(shift)
    return tuple(rot), tuple(trans)


def apply_symop(op, frac) -> tuple:
    rot, trans = op
    return tuple(sum(rot[r][c] * frac[c] for c in range(3)) + float(trans[r]) for r in range(3))


def cell_matrix(a, b, c, alpha, beta, gamma) -> tuple:
    """Lattice vectors as rows (angstrom), a along x, b in the xy plane."""
    al, be, ga = (math.radians(v) for v in (alpha, beta, gamma))
    ax = (a, 0.0, 0.0)
    bx = (b * math.cos(ga), b * math.sin(ga), 0.0)
    cx = c * math.cos(be)
    cy = c * (math.cos(al) - math.cos(be) * math.cos(ga)) / math.sin(ga)
    cz = math.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    return (ax, bx, (cx, cy, cz))


def _wrap(x: float) -> float:
    x = x - math.floor(x)
    return 0.0 if abs(x - 1.0) < 1e-9 else x


def _frac_distance(f, g) -> float:
    return max(abs(d - round(d)) for d in (f[k] - g[k] for k in range(3)))


def parse_cif(text: str, source: str) -> dict:
    """Parse one CIF block into a P1 atom list plus disorder diagnostics. No site is filled or dropped."""
    items, loops = _parse_items(text)
    try:
        lengths = [cif_number(items[f"_cell_length_{k}"]) for k in "abc"]
        angles = [cif_number(items[f"_cell_angle_{k}"]) for k in ("alpha", "beta", "gamma")]
    except KeyError as exc:
        raise PipelineError("CIF_PARSE_ERROR", f"missing cell parameter {exc}") from exc
    names, rows = _loop_with(loops, "_atom_site_fract_x")
    if not rows:
        raise PipelineError("CIF_PARSE_ERROR", "no _atom_site_fract_x loop")
    col = {n: names.index(n) for n in names}

    def get(row, key, default=None):
        return row[col[key]] if key in col else default

    sym_names, sym_rows = _loop_with(loops, "_space_group_symop_operation_xyz")
    if not sym_rows:
        sym_names, sym_rows = _loop_with(loops, "_symmetry_equiv_pos_as_xyz")
    key = "_space_group_symop_operation_xyz" if "_space_group_symop_operation_xyz" in (sym_names or []) \
        else "_symmetry_equiv_pos_as_xyz"
    ops = [parse_symop(r[sym_names.index(key)]) for r in sym_rows] if sym_rows else [parse_symop("x,y,z")]

    asym = []
    for row in rows:
        label = get(row, "_atom_site_label")
        element = get(row, "_atom_site_type_symbol") or re.match(r"[A-Za-z]+", label or "").group(0)
        element = re.match(r"[A-Z][a-z]?", element.strip().capitalize()).group(0)
        occ_raw = get(row, "_atom_site_occupancy", "1")
        occupancy = 1.0 if occ_raw in ("?", ".") else cif_number(occ_raw)
        group = get(row, "_atom_site_disorder_group", ".")
        frac = tuple(cif_number(get(row, f"_atom_site_fract_{k}")) for k in "xyz")
        asym.append({"label": label, "element": element, "frac": frac, "occupancy": occupancy,
                     "disorder_group": group if group not in ("?",) else "."})

    atoms = []
    for site in asym:
        images = []
        for op in ops:
            f = tuple(_wrap(v) for v in apply_symop(op, site["frac"]))
            if all(_frac_distance(f, g) > POSITION_DEDUP_TOL for g in images):
                images.append(f)
        for f in images:
            for other in atoms:
                if _frac_distance(f, other["frac"]) <= POSITION_DEDUP_TOL and other["label"] != site["label"]:
                    if other["occupancy"] >= 1 and site["occupancy"] >= 1:
                        raise PipelineError("CIF_OVERLAPPING_SITES", f"{site['label']} / {other['label']}")
            atoms.append({**site, "frac": f})

    partial = sorted({a["label"] for a in atoms if a["occupancy"] < 1.0 - 1e-6})
    disordered = sorted({a["label"] for a in atoms if a["disorder_group"] not in (".", "?")})
    return {
        "source": source, "sha256": hashlib.sha256(text.encode()).hexdigest(), "original_text": text,
        "cell_lengths_A": lengths, "cell_angles_deg": angles, "lattice": cell_matrix(*lengths, *angles),
        "n_symmetry_operations": len(ops), "symmetry_operations": ops, "atoms": atoms,
        "partial_occupancy_sites": partial, "disordered_sites": disordered,
    }


def write_cif(name: str, lattice, elements: list, frac_coords: list, labels: list | None = None) -> str:
    """P1 CIF for a built or relaxed geometry (lattice rows in angstrom)."""
    a, b, c = (math.sqrt(sum(x * x for x in v)) for v in lattice)

    def angle(u, v):
        return math.degrees(math.acos(sum(x * y for x, y in zip(u, v)) / (math.sqrt(sum(x * x for x in u)) * math.sqrt(sum(x * x for x in v)))))

    alpha, beta, gamma = angle(lattice[1], lattice[2]), angle(lattice[0], lattice[2]), angle(lattice[0], lattice[1])
    labels = labels or [f"{e}{i + 1}" for i, e in enumerate(elements)]
    out = [f"data_{name}", "_symmetry_space_group_name_H-M 'P 1'",
           f"_cell_length_a {a:.6f}", f"_cell_length_b {b:.6f}", f"_cell_length_c {c:.6f}",
           f"_cell_angle_alpha {alpha:.6f}", f"_cell_angle_beta {beta:.6f}", f"_cell_angle_gamma {gamma:.6f}",
           "loop_", "_space_group_symop_operation_xyz", "'x,y,z'",
           "loop_", "_atom_site_label", "_atom_site_type_symbol", "_atom_site_fract_x", "_atom_site_fract_y",
           "_atom_site_fract_z", "_atom_site_occupancy"]
    out += [f"{l} {e} {f[0]:.6f} {f[1]:.6f} {f[2]:.6f} 1.0" for l, e, f in zip(labels, elements, frac_coords)]
    return "\n".join(out) + "\n"
