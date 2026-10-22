"""Task 01 (topology search): PORMAKE net search and parent-framework construction (one job per process).

Run by ``topology_builders.generate_topologies`` as ``PYTHON pormake_worker.py job.json`` inside the job directory;
writes ``result.json`` and one ``<net>.cif`` per built parent framework. Self-contained (no mofpipe imports), so
any interpreter with ``pormake`` can run it.

1. Candidate nets: the declared list, or every RCSR net in PORMAKE's database whose NODE lines all have the node
   block's connectivity (read from the .cgd text, then loaded).
2. Node fit: PORMAKE ``Locator.calculate_rmsd`` of the node block on every unique local structure of the net (A);
   a net is accepted when the worst node RMSD is <= ``max_node_rmsd_A`` and it has <= ``max_nodes_per_cell`` nodes.
3. Construction: ``Builder.build_by_type`` with the node block on every node type and the parent edge block on
   every edge type (PORMAKE aligns, rescales the cell and bonds the blocks), written as a P1 CIF.
4. Substituent geometry: for each declared substituted edge block (e.g. an aminophenylene), the substituent atoms
   in the local frame used by framework.ParentSubstitutionBuilder: u = anchor ring atom -> substituent root,
   v = connection-point axis orthogonalised to u, w = u x v. This turns a validated PORMAKE 3D block into an
   explicit substitution rule instead of a hand-written one.
"""
from __future__ import annotations

import glob
import importlib.metadata as md
import json
import logging
import os
import re
import sys
import time
import traceback

WORKER_VERSION = "1"  # bump when a change alters the generated structures or rules


def _quiet():
    logging.disable(logging.CRITICAL)
    try:
        import pormake as pm
        pm.log.disable_print()
    except Exception:
        pass


def _candidate_nets(db, topo_dir: str, cn: int, declared):
    if declared:
        return list(declared), []
    names, skipped = [], 0
    for path in sorted(glob.glob(os.path.join(topo_dir, "*.cgd"))):
        cns = [int(m.group(1)) for m in re.finditer(r"^\s*NODE\s+\S+\s+(\d+)", open(path).read(), re.M)]
        if cns and set(cns) == {cn}:
            names.append(os.path.basename(path)[:-4])
        else:
            skipped += 1
    return names, [{"reason": "NODE_CONNECTIVITY_MISMATCH", "count": skipped, "detail": f"nets without only {cn}-c nodes"}]


def _substituent_rule(bb, element_root: str, replace_element: str):
    """Frame coordinates of the substituent group of an edge block (atoms beyond the anchor ring atom)."""
    import numpy as np

    atoms = bb.atoms
    symbols = atoms.get_chemical_symbols()
    pos = atoms.get_positions()
    bonds = {i: set() for i in range(len(atoms))}
    for i, j in bb.bonds:
        bonds[int(i)].add(int(j))
        bonds[int(j)].add(int(i))
    roots = [i for i, s in enumerate(symbols) if s == element_root]
    if len(roots) != 1:
        raise ValueError(f"expected one {element_root} root atom in {bb.name}, found {len(roots)}")
    root = roots[0]
    if any(symbols[j] == "X" for j in bonds[root]):  # e.g. an -NH- bridge: on the connection path, not a substituent
        raise ValueError(f"{element_root} of {bb.name} is bonded to a connection point; not a substituent")
    anchors = [j for j in bonds[root] if symbols[j] not in ("H", "X")]
    if len(anchors) != 1:
        raise ValueError(f"{element_root} root of {bb.name} must bond to exactly one framework atom")
    anchor = anchors[0]
    group = [root] + sorted(j for j in bonds[root] if j != anchor and symbols[j] == "H")
    xs = [i for i, s in enumerate(symbols) if s == "X"]
    u = pos[root] - pos[anchor]
    u /= np.linalg.norm(u)
    axis = pos[xs[1]] - pos[xs[0]]
    v = axis - axis.dot(u) * u
    v /= np.linalg.norm(v)
    w = np.cross(u, v)
    return {"replace_element": replace_element, "anchor_element": symbols[anchor],
            "group": [{"element": symbols[i], "frame_A": [round(float((pos[i] - pos[anchor]).dot(e)), 4) for e in (u, v, w)]}
                      for i in group],
            "derived_from_block": bb.name, "frame": "u = anchor->root, v = connection axis orthogonal to u, w = u x v"}


def run(job: dict, out_dir: str) -> dict:
    _quiet()
    import pormake as pm

    start = time.perf_counter()
    db = pm.Database()
    topo_dir = os.path.join(os.path.dirname(pm.__file__), "database", "topologies")
    versions = {"pormake": md.version("pormake"), "numpy": md.version("numpy"), "worker": WORKER_VERSION}
    for key in ("node_bb", "edge_bb"):
        if job[key] not in db.bb_list:
            return {"status": "FAILED", "error": {"code": "UNKNOWN_BUILDING_BLOCK", "detail": job[key]}, "versions": versions}
    node, edge = db.get_bb(job["node_bb"]), db.get_bb(job["edge_bb"])
    if edge.n_connection_points != 2:
        return {"status": "FAILED", "error": {"code": "UNSUPPORTED_EDGE_BLOCK", "detail": "edge blocks must be 2-c"},
                "versions": versions}
    cn = node.n_connection_points
    names, rejected = _candidate_nets(db, topo_dir, cn, job.get("nets"))
    locator, accepted = pm.Locator(), []
    for name in names:
        try:
            topo = db.get_topo(name)
        except Exception as exc:
            rejected.append({"net": name, "reason": "NET_NOT_LOADABLE", "detail": f"{type(exc).__name__}: {exc}"[:200]})
            continue
        cns = sorted({int(c) for c in topo.unique_cn})
        if cns != [cn]:
            rejected.append({"net": name, "reason": "NODE_CONNECTIVITY_MISMATCH", "detail": f"node CN {cns} != {cn}"})
            continue
        if topo.n_nodes > job["max_nodes_per_cell"]:
            rejected.append({"net": name, "reason": "TOO_MANY_NODES_PER_CELL", "detail": f"{topo.n_nodes}"})
            continue
        rmsd = max(float(locator.calculate_rmsd(ls, node)) for ls in topo.unique_local_structures)
        row = {"net": name, "max_node_rmsd_A": round(rmsd, 4), "n_nodes": int(topo.n_nodes), "n_edges": int(topo.n_edges),
               "n_node_types": int(topo.n_node_types), "n_edge_types": int(topo.n_edge_types)}
        if rmsd > job["max_node_rmsd_A"]:
            rejected.append({**row, "reason": "NODE_RMSD_ABOVE_THRESHOLD"})
            continue
        accepted.append((rmsd, name, topo, row))
    accepted.sort(key=lambda a: (a[0], a[1]))
    built = []
    for rmsd, name, topo, row in accepted[: job["max_candidates"]]:
        try:
            builder = pm.Builder()
            node_bbs = {int(t): node for t in topo.unique_node_types}
            edge_bbs = {tuple(int(x) for x in e): edge for e in topo.unique_edge_types}
            framework = builder.build_by_type(topology=topo, node_bbs=node_bbs, edge_bbs=edge_bbs)
            cif = os.path.join(out_dir, f"{name}.cif")
            framework.write_cif(cif)
            built.append({**row, "status": "BUILT", "cif": f"{name}.cif", "n_atoms": len(framework.atoms),
                          "cell": [round(float(x), 4) for x in framework.atoms.cell.cellpar()]})
        except Exception as exc:  # a construction failure is recorded, never a structure
            built.append({**row, "status": "BUILD_FAILED", "error": f"{type(exc).__name__}: {exc}"[:300]})
    for rmsd, name, topo, row in accepted[job["max_candidates"]:]:
        rejected.append({**row, "reason": "BEYOND_MAX_CANDIDATES"})
    substituents = {}
    for block_id, spec in (job.get("substituents") or {}).items():
        try:
            substituents[block_id] = _substituent_rule(db.get_bb(spec["pormake_block"]), spec["root_element"],
                                                       spec["replace_element"])
        except Exception as exc:
            substituents[block_id] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
    return {"status": "SUCCEEDED", "versions": versions, "node_bb": job["node_bb"], "edge_bb": job["edge_bb"],
            "node_connectivity": cn, "n_nets_considered": len(names), "candidates": built,
            "rejected": rejected, "substituents": substituents, "wall_time_s": round(time.perf_counter() - start, 2)}


def main(argv) -> int:
    if len(argv) != 2:
        print("usage: pormake_worker.py job.json", file=sys.stderr)
        return 2
    job = json.load(open(argv[1]))
    try:
        result = run(job, os.getcwd())
    except Exception as exc:
        result = {"status": "FAILED", "error": {"code": "WORKER_ERROR", "detail": f"{type(exc).__name__}: {exc}"},
                  "traceback": traceback.format_exc()[-2000:]}
    with open("result.json", "w") as handle:
        json.dump(result, handle, indent=1, sort_keys=True)
    return 0 if result["status"] == "SUCCEEDED" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
