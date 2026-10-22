"""MUF-7 through the OSI scheme with real structures (approximate run).

A2 (Qwen, design mode) -> QUBO solved exactly + QAOA -> top-k arrangements of each Hamiltonian -> PORMAKE build
(Zn4O N16 + BTB on ith-d; BDC E14 / BPDC E34 on the six Zn4O-Zn4O edges) -> orb-v3 + D3(BJ) relaxation = the
classical ground-state energy -> Zn4O(HCO2)6 cluster cut from the relaxed structure -> HF + CAS(4,4) VQE + SQD
(cluster_worker) -> scores against the paper's experimental arrangement and the orb energies.

Approximations, stated in the report: ML-potential energies instead of DFT; translation-only cells (no symmetry
reduction); Kang's six sites mapped onto the three pairs of parallel linkers (pairing exact, the assignment of his
second-neighbour edges between pairs is an assumption); the cluster VQE/SQD is a local electronic check, not the
arrangement energy. Run with the project .venv (pormake, orb-models, pyscf):

    PYTHONPATH=src .venv/bin/python benchmarks/kang2025/run_muf7_full.py --models qwen/qwen3.8-max-0902,qwen/qwen3.8-flash
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
# Kang site i -> PORMAKE ith-d edge slot; parallel pairs (0,1) (2,3) (4,5) bridge the same two Zn4O clusters.
SITE_TO_EDGE = [14, 16, 15, 19, 17, 18]
HARTREE_EV = 27.211386245988


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# -- structures ---------------------------------------------------------------------------------------------
class Builder:
    def __init__(self):
        import pormake as pm
        logging.disable(logging.CRITICAL)
        db = pm.Database()
        self.pm, self.topo = pm, db.get_topo("ith-d")
        self.btb, self.zn4o = pm.BuildingBlock(str(HERE / "blocks/BTB.xyz")), db.get_bb("N16")
        self.edge = {0: db.get_bb("E14"), 1: db.get_bb("E34")}  # 0 = BDC, 1 = BPDC
        t = self.topo
        cn = {int(i): int(t.unique_cn[list(t.unique_node_types).index(nt)]) for i, nt in zip(t.node_indices, t.node_types)}
        assert sorted(e for e in map(int, t.edge_indices) if all(cn[int(n.index)] == 6 for n in t.neighbor_list[e])) == sorted(SITE_TO_EDGE)
        self.cn = cn

    def build(self, bits: str):
        t = self.topo
        bbs = [self.btb if self.cn[i] == 3 else self.zn4o for i in range(t.n_nodes)] + [None] * t.n_edges
        for site, b in enumerate(bits):
            bbs[SITE_TO_EDGE[site]] = self.edge[int(b)]
        return self.pm.Builder().build(t, bbs).atoms


class Relaxer:
    def __init__(self, threads: int):
        import torch
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.inference.calculator import ORBCalculator
        from orb_models.forcefield.inference.d3_model import AlchemiDFTD3, D3SumModel
        torch.set_num_threads(threads)
        model, adapter = pretrained.orb_v3_conservative_inf_omat(device="cpu", compile=False)
        self.calc = ORBCalculator(D3SumModel(model, AlchemiDFTD3(functional="PBE", damping="BJ")), adapter, device="cpu")

    def relax(self, atoms, logfile, fmax=0.1, steps=500):
        from ase.filters import FrechetCellFilter
        from ase.optimize import FIRE
        atoms.calc = self.calc
        opt = FIRE(FrechetCellFilter(atoms), logfile=str(logfile), maxstep=0.1)
        converged = bool(opt.run(fmax=fmax, steps=steps))
        return {"energy_eV": float(atoms.get_potential_energy()), "converged": converged, "n_steps": int(opt.nsteps),
                "max_force_eV_per_A": float(np.linalg.norm(atoms.get_forces(), axis=1).max()),
                "cell_A": [round(x, 3) for x in atoms.cell.cellpar()], "n_atoms": len(atoms)}


def zn4o_cluster(atoms) -> dict:
    """Zn4O(HCO2)6 cut from one Zn4O node: mu4-O, the 6 carboxylates, ring carbons replaced by H at 1.09 A."""
    from ase.neighborlist import NeighborList, natural_cutoffs
    nl = NeighborList(natural_cutoffs(atoms, mult=1.15), self_interaction=False, bothways=True)
    nl.update(atoms)
    sym = atoms.get_chemical_symbols()
    zn = [i for i, s in enumerate(sym) if s == "Zn"]
    nb = lambda i: nl.get_neighbors(i)
    mu4 = next(i for i, s in enumerate(sym) if s == "O" and sum(sym[j] == "Zn" for j in nb(i)[0]) == 4)
    zns = [j for j in nb(mu4)[0] if sym[j] == "Zn"]
    ref = atoms.positions[mu4]
    def pos(i, off):  # unwrapped position next to the cluster
        return atoms.positions[i] + off @ atoms.cell.array
    keep, coords, elements = set(), [], []
    def add(i, p):
        if i not in keep:
            keep.add(i); elements.append(sym[i]); coords.append(p)
    add(mu4, ref)
    o_carb, c_carb = {}, {}
    for z in zns:
        zi, zoff = nb(mu4)[0].tolist().index(z), nb(mu4)[1][nb(mu4)[0].tolist().index(z)]
        zp = pos(z, zoff); add(z, zp)
        for o, ooff in zip(*nb(z)):
            if sym[o] == "O" and o != mu4:
                op = pos(o, ooff + zoff); add(o, op); o_carb[o] = op
                for c, coff in zip(*nb(o)):
                    if sym[c] == "C":
                        c_carb[c] = pos(c, coff + ooff + zoff)
    for c, cp in c_carb.items():
        add(c, cp)
        for r, roff in zip(*nb(c)):
            if sym[r] == "C":  # ring carbon -> capping H along the C-C bond
                rp = cp + (atoms.positions[r] + roff @ atoms.cell.array - atoms.positions[c])
                v = rp - cp
                elements.append("H"); coords.append(cp + 1.09 * v / np.linalg.norm(v))
    return {"elements": elements, "coords_A": [list(map(float, p)) for p in coords]}


def cluster_vqe_sqd(cluster: dict, workdir: Path) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    job = {"schema": "qumof_cluster_job_v1", "cluster_hash": "", **cluster, "charge": 0, "spin": 0, "basis": "sto-3g",
           "ecp": None, "conv_tol_Ha": 1e-8, "n_threads": 8, "provenance": "MUF-7 orb-relaxed PORMAKE build",
           "level": {"method": "active_space", "solver": "sqd_simulation",
                     "active_space": {"protocol_id": "zn4o_cas44", "reference": "RHF", "scf_solver": "newton",
                                      "conv_tol": 1e-8, "n_active_electrons": 4, "n_active_orbitals": 4},
                     "sqd": {"shots": 2000, "seed": 1, "flip_probability": 0.02}}}
    (workdir / "input.json").write_text(json.dumps(job))
    proc = subprocess.run([sys.executable, str(REPO / "src/qumof/cluster_worker.py"), "input.json"], cwd=workdir,
                          capture_output=True, text=True, timeout=3600)
    if not (workdir / "result.json").exists():
        return {"status": "FAILED", "error": proc.stderr[-300:]}
    r = json.loads((workdir / "result.json").read_text())
    d = r.get("details") or {}
    return {"status": r["status"], "sqd_energy_Ha": r.get("energy_Ha"), "vqe_energy_Ha": d.get("vqe_energy_Ha"),
            "exact_cas_energy_Ha": d.get("exact_active_space_energy_Ha"), "hf_energy_Ha": d.get("mean_field_energy_Ha"),
            "sqd_subspace": d.get("sqd_subspace_dimension"), "wall_time_s": r.get("wall_time_s"),
            "n_atoms": len(cluster["elements"]), "error": r.get("error")}


# -- run ----------------------------------------------------------------------------------------------------
def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 and np.std(ra) and np.std(rb) else None


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--controls", default="controls.yaml")
    p.add_argument("--models", default="qwen/qwen3.8-max-0902,qwen/qwen3.8-flash")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--out", default="runs/muf7_full")
    p.add_argument("--threads", type=int, default=10)
    p.add_argument("--fmax", type=float, default=0.1)
    args = p.parse_args(argv)
    sys.path.insert(0, str(REPO / "src"))
    from qumof.agents.controls import load_controls
    from qumof.agents.design import (build_packet, compile_design, evaluate_polynomial, feasible_states, kang_spectrum,
                                       load_materials, run_one)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    material = next(m for m in load_materials(HERE) if m["id"] == "muf7")
    packet = build_packet(material)
    reference = set(material["reference"]["ground_state_bitstrings"])
    states = ["".join(map(str, s)) for s in feasible_states(packet)]

    # 1. A2 designs (Qwen) + QUBO exact + QAOA, and the paper's own weighting as the control.
    models = [m for m in args.models.split(",") if m]
    jobs = [("kang_control", 0)] + [(m, r) for m in models for r in range(args.repeats)]
    log(f"A2 design + QUBO/QAOA: {len(jobs)} runs")
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(lambda j: run_one(material, None if j[0] == "kang_control" else load_controls(
            args.controls, {"openrouter": {"model": j[0], "reasoning": {"effort": "low"}}}), j[0], j[1], out / "design",
            None if j[0] == "kang_control" else __import__("qumof.agents.client", fromlist=["x"]).OpenRouterClient(
                load_controls(args.controls, {"openrouter": {"model": j[0], "reasoning": {"effort": "low"}}}))), jobs))
    kang = kang_spectrum(material, packet)
    for r in rows:
        if r.get("design") and r["validation"] == "VALID":
            poly = compile_design(r["design"], packet)
            values = {s: evaluate_polynomial(poly, tuple(int(c) for c in s)) for s in states}
            r["qubo_values"] = values
            r["top_k"] = sorted(states, key=lambda s: (values[s], s))[:args.top_k]
    candidates = sorted({s for r in rows for s in r.get("top_k", [])} | reference)
    log(f"distinct candidate arrangements to build and relax: {len(candidates)}")

    # 2. Ground state of every candidate: PORMAKE build + orb-v3 + D3 relaxation (cached per arrangement).
    builder, relaxer = Builder(), Relaxer(args.threads)
    from ase.io import write
    energies = {}
    for s in candidates:
        path = out / "structures" / f"{s}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            energies[s] = json.loads(path.read_text())
            continue
        t0 = time.time()
        atoms = builder.build(s)
        res = relaxer.relax(atoms, out / "structures" / f"{s}.log", args.fmax)
        write(str(out / "structures" / f"{s}_relaxed.cif"), atoms)
        res.update(arrangement=s, experimental=s in reference, wall_time_s=round(time.time() - t0, 1),
                   energy_per_fu_eV=res["energy_eV"] / 6)  # 6 Zn4O per cell
        path.write_text(json.dumps(res, indent=1))
        energies[s] = res
        log(f"relaxed {s} ({'experimental' if s in reference else 'other'}): {res['energy_eV']:.3f} eV, "
            f"converged {res['converged']} in {res['n_steps']} steps, {res['wall_time_s']} s")

    # 3. Quantum ground-state step: Zn4O cluster VQE/SQD on the lowest-energy relaxed candidate and the best other one.
    ordered = sorted(candidates, key=lambda s: energies[s]["energy_eV"])
    vqe = {}
    for s in ordered[:2]:
        from ase.io import read
        cluster = zn4o_cluster(read(str(out / "structures" / f"{s}_relaxed.cif")))
        log(f"cluster VQE/SQD on {s}: {len(cluster['elements'])} atoms")
        vqe[s] = cluster_vqe_sqd(cluster, out / "cluster" / s)
        log(f"  {vqe[s]}")

    # 4. Scores.
    e = {s: energies[s]["energy_eV"] for s in candidates}
    best = ordered[0]
    def score(r):
        top = r["top_k"]
        q = [r["qubo_values"][s] for s in top]
        o = [e[s] for s in top]
        return {"qubo_ground_states": sorted(s for s in states if r["qubo_values"][s] - min(r["qubo_values"].values()) < 1e-9),
                "qubo_ground_is_experimental": all(s in reference for s in states
                                                    if r["qubo_values"][s] - min(r["qubo_values"].values()) < 1e-9),
                "top1": top[0], "top1_is_orb_lowest_of_top_k": e[top[0]] == min(o),
                "orb_lowest_of_top_k": min(top, key=lambda s: e[s]),
                "orb_lowest_is_experimental": min(top, key=lambda s: e[s]) in reference,
                "experimental_in_top_k": bool(set(top) & reference),
                "spearman_qubo_vs_orb_top_k": spearman(q, o),
                "qaoa_p_reference": (r.get("assessment") or {}).get("qaoa", {}).get("median_p_reference")}
    scored = [{**{k: r[k] for k in ("model", "repeat", "validation", "cost_usd", "llm_seconds")}, "design": r.get("design"),
               **(score(r) if r.get("top_k") else {})} for r in rows]
    kang_vals = [kang[s] for s in candidates]
    summary = {"material": "MUF-7", "n_candidates_relaxed": len(candidates),
               "orb_lowest_overall": best, "orb_lowest_overall_is_experimental": best in reference,
               "energies_eV": {s: round(e[s], 4) for s in ordered},
               "experimental_minus_best_other_eV_per_cell": (min(e[s] for s in candidates if s in reference)
                                                             - min((e[s] for s in candidates if s not in reference), default=float("nan"))),
               "paper_hamiltonian_spearman_vs_orb": spearman(kang_vals, [e[s] for s in candidates]),
               "cluster_vqe_sqd": vqe, "runs": scored,
               "approximations": ["orb-v3 + D3(BJ) energies instead of DFT (fmax %.2f eV/A)" % args.fmax,
                                  "PORMAKE translation-only cell (ith-d, 540 atoms), no symmetry reduction",
                                  "Kang sites mapped to the three parallel-linker pairs; second-neighbour assignment assumed",
                                  "cluster VQE/SQD: Zn4O(HCO2)6, STO-3G, CAS(4,4); a local electronic check"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    lines = [f"# MUF-7 full-scheme run (approximate)", "",
             f"Relaxed candidates: {len(candidates)}; orb-lowest: `{best}` ({'experimental' if best in reference else 'NOT experimental'}); "
             f"experimental minus best other: {summary['experimental_minus_best_other_eV_per_cell']:.3f} eV/cell; "
             f"paper Hamiltonian vs orb Spearman: {summary['paper_hamiltonian_spearman_vs_orb']}", "",
             "| model | runs | valid | QUBO ground = experimental | experimental in top-k | orb-lowest of top-k = experimental | "
             "top-1 = orb-lowest of top-k | mean Spearman QUBO vs orb | cost $ |", "|---|---|---|---|---|---|---|---|---|"]
    for model in ["kang_control"] + models:
        rs = [x for x in scored if x["model"] == model]
        ok = [x for x in rs if "top1" in x]
        f = lambda k: f"{sum(bool(x[k]) for x in ok)}/{len(rs)}"
        sp = [x["spearman_qubo_vs_orb_top_k"] for x in ok if x["spearman_qubo_vs_orb_top_k"] is not None]
        lines.append(f"| {model} | {len(rs)} | {len(ok)} | {f('qubo_ground_is_experimental')} | {f('experimental_in_top_k')} | "
                     f"{f('orb_lowest_is_experimental')} | {f('top1_is_orb_lowest_of_top_k')} | "
                     f"{(sum(sp) / len(sp)) if sp else '–'} | {sum(x['cost_usd'] or 0 for x in rs):.4f} |")
    lines += ["", "## orb-v3 energies (eV per cell, lowest first)", ""] + \
             [f"- `{s}` {e[s]:.3f}{'  ← experimental' if s in reference else ''}" for s in ordered]
    lines += ["", "## Cluster VQE/SQD (Zn4O(HCO2)6)", ""] + [f"- `{s}`: {v}" for s, v in vqe.items()]
    lines += ["", "## Approximations", ""] + [f"- {a}" for a in summary["approximations"]]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    log("done:", out / "summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
