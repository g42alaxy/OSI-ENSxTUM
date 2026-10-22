"""Task 02: ML-potential relaxation worker (one batch per process).

Run by ``mlip.MLPotentialBackend`` as ``PYTHON mlip_worker.py batch.json`` inside the batch directory; it rewrites
``results.json`` after every structure, so finished relaxations survive a timeout. One model load serves the whole
batch; ``n_threads`` (default: all cores) sets the torch CPU threads. ``--probe probe.json`` reports the engine identity
(package versions, checkpoint path and sha256) without relaxing anything.

Engine ``orb``: Orb v3 conservative models (orb-models, Apache-2.0) through their ASE calculator; forces and stress
are energy gradients. ``dispersion: D3BJ`` adds DFT-D3(BJ) for the declared functional (the OMat24-trained models
carry none). Relaxation: ASE FIRE (or BFGS) on a FrechetCellFilter (variable cell) or fixed cell, to ``fmax``.

Energies eV per cell; forces eV/A; stress GPa (ASE sign convention).
This file is self-contained (no mofpipe imports) so any interpreter with the engine can run it.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback

WORKER_VERSION = "1"  # bump when a change alters the computed quantities (part of the ML fidelity)
EV_PER_A3_TO_GPA = 160.21766208
ORB_MODELS = {"orb-v3-conservative-inf-omat": "orb_v3_conservative_inf_omat",
              "orb-v3-conservative-20-omat": "orb_v3_conservative_20_omat",
              "orb-v3-conservative-inf-mpa": "orb_v3_conservative_inf_mpa"}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _orb(protocol: dict):
    import importlib.metadata as md

    import torch
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.inference.calculator import ORBCalculator

    loader = getattr(pretrained, ORB_MODELS[protocol["model"]])
    weights = protocol.get("checkpoint_path")
    if not weights:  # orb's default URL, cached locally by cached_path after the first download
        from cached_path import cached_path
        import inspect
        weights = str(cached_path(inspect.signature(loader).parameters["weights_path"].default))
    device = protocol.get("device", "cpu")
    torch.set_num_threads(int(protocol.get("n_threads") or os.cpu_count() or 1))
    model, adapter = loader(weights_path=weights, device=device, precision=protocol.get("precision", "float32-high"),
                            compile=False)
    if protocol.get("dispersion") == "D3BJ":
        from orb_models.forcefield.inference.d3_model import AlchemiDFTD3, D3SumModel
        model = D3SumModel(model, AlchemiDFTD3(functional=protocol.get("d3_functional", "PBE"), damping="BJ"))
    identity = {"engine": "orb", "model": protocol["model"], "dispersion": protocol.get("dispersion"),
                "orb_models_version": md.version("orb-models"), "torch_version": torch.__version__,
                "checkpoint_path": weights, "checkpoint_sha256": _sha256(weights), "worker_version": WORKER_VERSION}
    return ORBCalculator(model, adapter, device=device), identity


def _relax(atoms, calc, policy: dict) -> dict:
    import numpy as np
    from ase.filters import FrechetCellFilter
    from ase.optimize import BFGS, FIRE

    atoms.calc = calc
    initial = float(atoms.get_potential_energy())
    target = FrechetCellFilter(atoms) if policy.get("cell", "variable") == "variable" else atoms
    optimizer = {"FIRE": FIRE, "BFGS": BFGS}[policy.get("optimizer", "FIRE")](target, logfile=None)
    converged = bool(optimizer.run(fmax=policy["fmax_eV_per_A"], steps=policy["max_steps"]))
    forces = atoms.get_forces()
    out = {"initial_energy_eV": initial, "energy_eV": float(atoms.get_potential_energy()), "converged": converged,
           "n_steps": int(optimizer.nsteps), "max_force_eV_per_A": float(np.linalg.norm(forces, axis=1).max())}
    if policy.get("cell", "variable") == "variable":
        out["max_stress_GPa"] = float(np.abs(atoms.get_stress()).max() * EV_PER_A3_TO_GPA)
    out["relaxed"] = {"lattice_A": atoms.cell.array.tolist(), "elements": atoms.get_chemical_symbols(),
                      "frac_coords": atoms.get_scaled_positions(wrap=True).tolist()}
    return out


def main(argv) -> int:
    probe = argv[0] == "--probe"
    path = argv[1] if probe else argv[0]
    batch = json.loads(open(path).read())
    protocol = batch["protocol"]
    try:
        if protocol["engine"] != "orb" or protocol["model"] not in ORB_MODELS:
            raise ValueError(f"unsupported engine/model {protocol['engine']}/{protocol['model']}")
        calc, identity = _orb(protocol)
    except Exception as exc:  # the engine is missing or the checkpoint cannot load: report, do not guess
        json.dump({"identity": None, "error": f"{type(exc).__name__}: {exc}", "results": {}},
                  open("probe.json" if probe else "results.json", "w"))
        return 2
    if probe:
        json.dump({"identity": identity}, open("probe.json", "w"), indent=2)
        return 0
    from ase import Atoms

    results = {}
    for item in batch["structures"]:
        start = time.perf_counter()
        try:
            atoms = Atoms(symbols=item["elements"], scaled_positions=item["frac_coords"], cell=item["lattice_A"], pbc=True)
            results[item["key"]] = {"status": "SUCCEEDED", **_relax(atoms, calc, batch["relaxation"])}
        except Exception as exc:  # a failed relaxation is a failure, never an energy
            results[item["key"]] = {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}",
                                    "traceback": traceback.format_exc()[-1500:]}
        results[item["key"]]["wall_time_s"] = round(time.perf_counter() - start, 3)
        with open("results.json.tmp", "w") as handle:  # atomic per-structure checkpoint
            json.dump({"identity": identity, "results": results}, handle, indent=1)
        os.replace("results.json.tmp", "results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
