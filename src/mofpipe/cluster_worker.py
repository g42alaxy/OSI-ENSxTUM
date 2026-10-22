"""Task 07 (T07.3): finite-cluster energies for the subtractive embedding (one calculation per process).

Run by ``embedding.ClusterBackend`` as ``PYTHON cluster_worker.py job.json`` inside the job directory; writes
``result.json``. ``compute(job)`` is also importable (the embedding benchmark calls it directly).

Levels (``job["level"]["method"]``), all on the same Mole (geometry, basis, ECP, charge, spin):
  DFT            RKS/UKS with ``xc`` and optional ``dispersion`` (pyscf-dispersion; QE's ``grimme-d3`` is D3 zero damping)
  HF             RHF/UHF
  MP2            RMP2/UMP2 on the HF reference
  CCSD, CCSD(T)  RCCSD/UCCSD (+ perturbative triples) on the HF reference
  active_space   ``mofpipe.active_space.prepare_active_space`` + one ``mofpipe.electronic_solvers`` solver;
                 the energy is the solver's total energy (E_core included). ``sqd_simulation`` samples the
                 simulated VQE state of the same Hamiltonian and reports both energies (plus exact ED when small)
Closed shells (spin 0) use restricted references, open shells unrestricted ones. All electrons are correlated.
SCF numerics are declared per level (part of the embedding fidelity): ``scf_solver`` ``diis`` (default) or
``newton`` (second-order SCF, for transition-metal clusters where DIIS oscillates), ``max_cycle``, ``level_shift_Ha``.
One OpenMP thread unless ``n_threads`` says otherwise; energies are Hartree.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

WORKER_VERSION = "3"  # bump when a change alters the computed quantities (part of the embedding fidelity)
DISPERSION_NAMES = {"grimme-d3": "d3zero", "d3zero": "d3zero", "d3bj": "d3bj", "d4": "d4"}
METHODS = ("DFT", "HF", "MP2", "CCSD", "CCSD(T)", "active_space")


def _scf(mf, level: dict, conv_tol: float):
    """Apply the level's declared SCF numerics; returns the (possibly second-order) solver object."""
    mf.conv_tol = conv_tol
    mf.max_cycle = level.get("max_cycle", mf.max_cycle)
    if level.get("level_shift_Ha"):
        mf.level_shift = level["level_shift_Ha"]
    return mf.newton() if level.get("scf_solver") == "newton" else mf


def compute(job: dict) -> dict:
    from pyscf import __version__ as pyscf_version, cc, dft, gto, lib, mp, scf
    level, start = job["level"], time.perf_counter()
    method = level["method"]
    if method not in METHODS:
        return {"status": "FAILED", "error": {"code": "UNSUPPORTED_METHOD", "detail": method}}
    with lib.with_omp_threads(job.get("n_threads", 1)):
        mol = gto.M(atom=[[e, tuple(r)] for e, r in zip(job["elements"], job["coords_A"])], unit="Angstrom",
                    basis=job["basis"], ecp=job.get("ecp") or {}, charge=job["charge"], spin=job["spin"], verbose=0)
        open_shell = job["spin"] != 0
        details = {"n_ao": mol.nao_nr(), "n_electrons": int(mol.nelectron), "reference": "U" if open_shell else "R"}
        if method == "active_space":
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from mofpipe import electronic_solvers
            from mofpipe.active_space import prepare_active_space
            system = {"system_id": job["cluster_hash"], "atoms": [[e, *r] for e, r in zip(job["elements"], job["coords_A"])],
                      "unit": "Angstrom", "charge": job["charge"], "spin": job["spin"], "provenance": job.get("provenance")}
            from mofpipe.contract import PipelineError
            try:
                artifact = prepare_active_space(system, {**level["active_space"], "basis": job["basis"], "embedding": "none"})
            except PipelineError as exc:  # e.g. the reference SCF did not converge: a failure, never an energy
                return {"status": "FAILED", "energy_Ha": None, "error": exc.as_dict(), "details": details,
                        "versions": {"pyscf": pyscf_version, "worker": WORKER_VERSION}}
            if level["solver"] == "sqd_simulation":
                # SQD samples a prepared state: the simulated VQE state on the same Hamiltonian is the source, and
                # both energies are reported (the SQD energy is the level's energy).
                vqe = electronic_solvers.vqe_simulation(artifact)
                cfg = level.get("sqd") or {}
                solver = electronic_solvers.sqd_simulation(artifact, vqe["final_state"], cfg.get("shots", 2000),
                                                           cfg.get("seed", 1), cfg.get("flip_probability", 0.0),
                                                           vqe["artifact_id"])
                details.update(vqe_energy_Ha=vqe["energy_Ha"], vqe_parameters=vqe["n_parameters"],
                               sqd_subspace_dimension=solver["subspace_dimension"],
                               sqd_full_sector_dimension=solver["full_sector_dimension"],
                               sqd_in_sector_fraction=solver["in_sector_fraction"])
            else:
                solver = getattr(electronic_solvers, level["solver"])(artifact)
            if artifact["n_active_orbitals"] <= 8:  # exact reference for the same active space
                details["exact_active_space_energy_Ha"] = electronic_solvers.exact_diagonalization(artifact)["energy_Ha"]
            energy, converged = solver["energy_Ha"], solver["status"] == "COMPLETED"
            details.update(active_space_id=artifact["artifact_id"], hamiltonian_hash=artifact["hamiltonian_hash"],
                           solver=level["solver"], exactness=solver.get("exactness"),
                           mean_field_energy_Ha=artifact["mean_field"]["energy_Ha"])
        elif method == "DFT":
            mf = (dft.UKS if open_shell else dft.RKS)(mol)
            mf.xc = level["xc"]
            if level.get("dispersion") not in (None, "none"):
                if level["dispersion"] not in DISPERSION_NAMES:
                    return {"status": "FAILED", "error": {"code": "UNSUPPORTED_DISPERSION", "detail": level["dispersion"]}}
                mf.disp = DISPERSION_NAMES[level["dispersion"]]
            mf = _scf(mf, level, job["conv_tol_Ha"])
            energy, converged = mf.kernel(), mf.converged
        else:
            mf = _scf((scf.UHF if open_shell else scf.RHF)(mol), level, job["conv_tol_Ha"])
            energy, converged = mf.kernel(), mf.converged
            details["hf_energy_Ha"] = float(energy)
            if method == "MP2":
                energy = mp.MP2(mf).run().e_tot
            elif method in ("CCSD", "CCSD(T)"):
                mycc = cc.CCSD(mf)
                mycc.conv_tol = job["conv_tol_Ha"]
                mycc.kernel()
                converged = converged and mycc.converged
                energy = mycc.e_tot + (mycc.ccsd_t() if method == "CCSD(T)" else 0.0)
    return {"status": "SUCCEEDED" if converged else "FAILED", "energy_Ha": float(energy), "converged": bool(converged),
            "error": None if converged else {"code": "NOT_CONVERGED", "detail": method}, "details": details,
            "versions": {"pyscf": pyscf_version, "worker": WORKER_VERSION}, "wall_time_s": round(time.perf_counter() - start, 3)}


def main(argv) -> int:
    if len(argv) != 2:
        print("usage: cluster_worker.py job.json", file=sys.stderr)
        return 2
    with open(argv[1]) as handle:
        result = compute(json.load(handle))
    with open("result.json", "w") as handle:
        json.dump(result, handle, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
