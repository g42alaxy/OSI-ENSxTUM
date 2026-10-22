"""Task 06: PySCF periodic GPW worker (one calculation per process).

Run by ``dft.PySCFAdapter`` as ``PYTHON pyscf_worker.py job.json`` inside the job's attempt directory; it
writes ``result.json`` (the parsed raw output) and the PySCF log ``pyscf.log``. ``--probe probe.json``
reports versions, basis/pseudopotential data hashes and capability errors without running an SCF.

Electronic structure: Gaussian orbitals (GTH basis sets) with GTH pseudopotentials and a plane-wave
representation of the density and potentials (GPW). ``integral_scheme``:
  multigrid  Gamma point only; CP2K-style multigrid GPW (``multigrid_numint``) for SCF and nuclear gradients.
             The stress tensor is evaluated with the FFTDF integrator at the converged orbitals.
  fftdf      any Monkhorst-Pack mesh; FFT-based GPW (KRKS/KUKS) for SCF, gradients and stress.
Dispersion (DFT-D3/D4) comes from pyscf-dispersion (energy, gradient and virial), relaxations from ASE
(BFGS/FIRE/LBFGS, FrechetCellFilter for variable cell). ``device: gpu`` runs the SCF, gradients and stress
through gpu4pyscf (``to_gpu``).

Sign conventions: stress = (1/V) dE/d(strain) (ASE convention; compressed cells have negative diagonal
stress, pressure = -trace/3). Energies in the result are Hartree; forces eV/A; stress eV/A^3 and kbar.

This file is self-contained (no mofpipe imports) so any interpreter with PySCF can run it.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import traceback

WORKER_VERSION = "1"  # bump when a change alters the computed quantities (part of the DFT fidelity)
HARTREE_EV = 27.211386245988  # CODATA 2018
BOHR_A = 0.529177210903
EV_PER_A3_TO_KBAR = 1602.1766208
NUMERICAL_KEYS = ("damp", "level_shift", "diis_space", "diis_start_cycle", "max_cycle")
# multigrid + stress: the FFTDF stress is the derivative of the FFTDF energy functional. It is used for the GPW
# (multigrid) energy only if both integrators give the same energy for the converged density within this bound.
STRESS_CONSISTENCY_HA_PER_ATOM = 1e-4
D3_VERSIONS = ("d3bj", "d3zero", "d3bjm", "d3zerom", "d3op")
OPTIMIZERS = ("BFGS", "LBFGS", "FIRE")


class WorkerError(Exception):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(code)
        self.code, self.detail = code, detail


class SCFNotConverged(Exception):
    pass


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=repr).encode()).hexdigest()


def _host(x):
    """numpy array from numpy or cupy (gpu4pyscf) data."""
    import numpy
    return numpy.asarray(x.get() if hasattr(x, "get") else x)


def parse_dispersion(name: str):
    """'none' | 'd3bj' | 'd3zero' | 'd3bjm' | 'd3zerom' | 'd3op' (each optionally '-atm') | 'd4'."""
    if name == "none":
        return None
    if name == "d4":
        return ("d4", "d4", True)
    base, atm = (name[:-4], True) if name.endswith("-atm") else (name, False)
    if base not in D3_VERSIONS:
        raise WorkerError("UNSUPPORTED_DISPERSION", name)
    return ("d3", base, atm)


# -- environment ---------------------------------------------------------------
def versions(settings: dict) -> dict:
    from importlib import metadata

    import numpy
    import pyscf
    out = {"python": sys.version.split()[0], "numpy": numpy.__version__, "pyscf": pyscf.__version__,
           "worker": WORKER_VERSION}
    if parse_dispersion(settings["dispersion"]):
        out["pyscf_dispersion"] = metadata.version("pyscf-dispersion")
    if settings.get("relaxation_calculation") in ("relax", "vc-relax"):
        import ase
        out["ase"] = ase.__version__
    if settings["device"] == "gpu":
        import gpu4pyscf
        out["gpu4pyscf"] = gpu4pyscf.__version__
    return out


def _per_element(value, element):
    return value[element] if isinstance(value, dict) else value


def basis_and_pseudo_hashes(settings: dict, elements) -> dict:
    """Hashes of the actual basis/pseudopotential parameters PySCF loads for each element."""
    from pyscf.pbc import gto
    out = {"basis": {}, "pseudo": {}}
    for el in sorted(set(elements)):
        try:
            out["basis"][el] = _sha(gto.basis.load(_per_element(settings["basis"], el), el))
        except Exception as exc:  # noqa: BLE001 - reported as a typed error
            raise WorkerError("UNSUPPORTED_BASIS_FOR_ELEMENT", f"{el}: {exc}") from exc
        try:
            out["pseudo"][el] = _sha(gto.pseudo.load(_per_element(settings["pseudo"], el), el))
        except Exception as exc:  # noqa: BLE001
            raise WorkerError("UNSUPPORTED_PSEUDOPOTENTIAL_FOR_ELEMENT", f"{el}: {exc}") from exc
    return out


def probe(spec: dict) -> dict:
    """Capability check for a protocol: versions, data hashes and typed errors (no SCF)."""
    settings, errors = spec["settings"], []
    result = {"versions": None, "data_hashes": None, "errors": errors}
    try:
        result["versions"] = versions(settings)
    except ImportError as exc:
        errors.append({"code": "BACKEND_UNAVAILABLE", "detail": f"import failed: {exc}"})
        return result
    try:
        result["data_hashes"] = basis_and_pseudo_hashes(settings, spec["elements"])
    except WorkerError as exc:
        errors.append({"code": exc.code, "detail": exc.detail})
    from pyscf.dft import libxc
    xc = settings["xc"]
    try:
        xc_type = libxc.xc_type(xc)
        if libxc.is_hybrid_xc(xc) or libxc.is_nlc(xc) or xc_type not in ("LDA", "GGA"):
            errors.append({"code": "UNSUPPORTED_XC_FOR_PERIODIC_GRADIENTS",
                           "detail": f"{xc} ({xc_type}); forces and stress need a local or semilocal functional"})
    except Exception as exc:  # noqa: BLE001
        errors.append({"code": "UNSUPPORTED_XC", "detail": f"{xc}: {exc}"})
    try:
        disp = parse_dispersion(settings["dispersion"])
        if disp:
            from pyscf.pbc import gto
            cell = gto.Cell(atom="H 0 0 0; H 0 0 0.74", a=[[6, 0, 0], [0, 6, 0], [0, 0, 6]], unit="A",
                            basis="gth-szv", pseudo="gth-pbe", verbose=0)
            cell.build(parse_arg=False)
            _dispersion_model(cell, settings["dispersion_xc"], disp)
    except WorkerError as exc:
        errors.append({"code": exc.code, "detail": exc.detail})
    except ImportError as exc:
        errors.append({"code": "BACKEND_UNAVAILABLE", "detail": f"pyscf-dispersion: {exc}"})
    except Exception as exc:  # noqa: BLE001 - e.g. no damping parameters for this functional
        errors.append({"code": "UNSUPPORTED_DISPERSION", "detail": f"{settings['dispersion']}/{settings['dispersion_xc']}: {exc}"})
    if settings.get("relaxation_calculation") in ("relax", "vc-relax"):
        try:
            from ase import filters, optimize
            if not hasattr(filters, "FrechetCellFilter") or not hasattr(optimize, settings["optimizer"]):
                errors.append({"code": "BACKEND_UNAVAILABLE", "detail": "ase lacks FrechetCellFilter or the optimizer"})
        except ImportError as exc:
            errors.append({"code": "BACKEND_UNAVAILABLE", "detail": f"ase: {exc}"})
    if settings["device"] == "gpu":
        try:
            import cupy
            if cupy.cuda.runtime.getDeviceCount() < 1:
                errors.append({"code": "BACKEND_UNAVAILABLE", "detail": "no CUDA device"})
        except Exception as exc:  # noqa: BLE001
            errors.append({"code": "BACKEND_UNAVAILABLE", "detail": f"gpu4pyscf/cupy: {exc}"})
    return result


# -- one electronic-structure evaluation ----------------------------------------------
def build_cell(lattice_A, elements, cart_A, settings: dict, log):
    from pyscf.pbc import gto
    cell = gto.Cell()
    cell.unit = "A"
    cell.a = [list(map(float, v)) for v in lattice_A]
    cell.atom = [[el, [float(x) for x in r]] for el, r in zip(elements, cart_A)]
    cell.basis = settings["basis"]
    cell.pseudo = settings["pseudo"]
    cell.ke_cutoff = float(settings["ke_cutoff_Ha"])
    cell.precision = float(settings["precision"])
    cell.spin = int(settings["magnetic_state"].get("spin_2S", 0))
    cell.charge = int(settings.get("cell_charge", 0))
    cell.max_memory = int(settings["max_memory_MB"])
    cell.stdout, cell.verbose = log, 4
    cell.build(parse_arg=False, dump_input=False)
    return cell


def _scf_object(cell, settings: dict, kmesh):
    from pyscf.pbc import dft
    open_shell = settings["magnetic_state"]["nspin"] == 2
    scheme, gpu = settings["integral_scheme"], settings["device"] == "gpu"
    if scheme == "multigrid":
        if list(kmesh) != [1, 1, 1]:
            raise WorkerError("UNSUPPORTED_KPOINTS_FOR_MULTIGRID", f"kmesh {kmesh}")
        mf = (dft.UKS if open_shell else dft.RKS)(cell, xc=settings["xc"])
        mf = mf.to_gpu().multigrid_numint() if gpu else mf.multigrid_numint()
    elif scheme == "fftdf":
        mf = (dft.KUKS if open_shell else dft.KRKS)(cell, kpts=cell.make_kpts(list(kmesh)), xc=settings["xc"])
        mf = mf.to_gpu() if gpu else mf
    else:
        raise WorkerError("UNSUPPORTED_INTEGRAL_SCHEME", scheme)
    occ = settings["occupations"]
    if occ["kind"] == "smearing":
        if gpu:
            from gpu4pyscf.pbc.scf.smearing import smearing
        else:
            from pyscf.pbc.scf.smearing import smearing
        mf = smearing(mf, sigma=float(occ["sigma_Ha"]), method=occ["method"])
    elif occ["kind"] != "fixed":
        raise WorkerError("UNSUPPORTED_OCCUPATIONS", str(occ))
    mf.conv_tol = float(settings["conv_tol_Ha"])
    for key, value in settings["numerics"].items():
        if key not in NUMERICAL_KEYS:
            raise WorkerError("PROTOCOL_RETRY_CHANGES_SCIENCE", key)
        setattr(mf, key, value)
    return mf


def _initial_guess(mf, cell, settings: dict):
    """Atomic guess; for nspin=2 each element's AO block gets the declared starting magnetization (-1..1)."""
    state = settings["magnetic_state"]
    if state["nspin"] != 2 or not state.get("starting_magnetization"):
        return None
    dm = _host(mf.get_init_guess())
    total = dm[0] + dm[1]
    for ia, (_, _, p0, p1) in enumerate(cell.aoslice_by_atom()):
        m = state["starting_magnetization"].get(cell.atom_pure_symbol(ia))
        if m is None:
            continue
        block = total[..., p0:p1, p0:p1]
        dm[0][..., p0:p1, p0:p1] = 0.5 * (1 + m) * block
        dm[1][..., p0:p1, p0:p1] = 0.5 * (1 - m) * block
    return dm


def _dispersion_model(cell, xc: str, disp):
    kind, version, atm = disp
    if kind == "d3":
        from pyscf.dispersion import dftd3
        return dftd3.DFTD3Dispersion(cell, xc, version=version, atm=atm)
    from pyscf.dispersion import dftd4
    return dftd4.DFTD4Dispersion(cell, xc, atm=atm)


def _stress_cpu(mf, cell, settings: dict, kmesh):
    """Analytic stress (Hartree/Bohr^3, symmetrised) and the integrator consistency (Hartree, multigrid only).

    multigrid: the FFTDF stress at the converged GPW orbitals, plus E_FFTDF(dm) - E_multigrid(dm) for that density.
    """
    import numpy
    from pyscf.pbc import dft
    open_shell = settings["magnetic_state"]["nspin"] == 2
    delta = None
    if settings["integral_scheme"] == "multigrid":
        from pyscf.pbc.grad import rks as rks_grad, rks_stress, uks as uks_grad, uks_stress
        ref = (dft.UKS if open_shell else dft.RKS)(cell, xc=settings["xc"])
        ref.mo_coeff, ref.mo_occ, ref.mo_energy = mf.mo_coeff, mf.mo_occ, mf.mo_energy
        delta = float(ref.energy_tot(dm=ref.make_rdm1())) - float(mf.e_tot)
        sigma = (uks_stress.kernel(uks_grad.Gradients(ref)) if open_shell else
                 rks_stress.kernel(rks_grad.Gradients(ref)))
    else:
        from pyscf.pbc.grad import krks_stress, kuks_stress
        grad = mf.Gradients()
        sigma = kuks_stress.kernel(grad) if open_shell else krks_stress.kernel(grad)
    sigma = numpy.asarray(sigma)
    return 0.5 * (sigma + sigma.T), delta


class Evaluator:
    """Energy, forces and (optionally) stress for one geometry, with density-matrix warm starts."""

    def __init__(self, settings: dict, log):
        self.settings, self.log = settings, log
        self.disp = parse_dispersion(settings["dispersion"])
        self.dm_last = None
        self.n_scf = 0
        self.meta = {}

    def __call__(self, lattice_A, elements, cart_A, need_stress: bool, need_forces: bool = True) -> dict:
        import numpy
        s = self.settings
        cell = build_cell(lattice_A, elements, cart_A, s, self.log)
        kmesh = s["kmesh"]
        mf = _scf_object(cell, s, kmesh)
        dm0 = self.dm_last if self.dm_last is not None else _initial_guess(mf, cell, s)
        mf.kernel(dm0=dm0)
        self.n_scf += 1
        self.meta = {"nao": int(cell.nao_nr()), "nelectron": int(cell.nelectron), "mesh": [int(x) for x in cell.mesh],
                     "kmesh": list(kmesh), "spin_2S": int(cell.spin)}
        if not mf.converged:
            raise SCFNotConverged(f"SCF not converged after {mf.max_cycle} cycles")
        self.dm_last = _host(mf.make_rdm1())
        energies = {"e_tot": float(mf.e_tot)}
        if s["occupations"]["kind"] == "smearing":
            energies.update(e_free=float(mf.e_free), e_zero=float(mf.e_zero))
        grad = _host(mf.Gradients().kernel()) if need_forces or need_stress else None
        if need_stress and s["device"] == "gpu":
            sigma = _host(mf.Gradients().get_stress())  # gpu4pyscf: multigrid and FFTDF stress on the device
            sigma = 0.5 * (sigma + sigma.T)
        elif need_stress:
            sigma, delta = _stress_cpu(mf, cell, s, kmesh)
            if delta is not None:
                self.meta["stress_integrator_delta_Ha"] = delta
                if abs(delta) > STRESS_CONSISTENCY_HA_PER_ATOM * cell.natm:
                    raise WorkerError("STRESS_INTEGRATOR_INCONSISTENT",
                                      f"E_FFTDF - E_multigrid = {delta:.3e} Ha for {cell.natm} atoms at ke_cutoff "
                                      f"{cell.ke_cutoff} Ha; raise ke_cutoff_Ha or use integral_scheme fftdf")
        else:
            sigma = None
        if self.disp:
            res = _dispersion_model(cell, s["dispersion_xc"], self.disp).get_dispersion(grad=grad is not None)
            energies["e_disp"] = float(res["energy"])
            if grad is not None:
                grad = grad + numpy.asarray(res["gradient"])
            if sigma is not None:
                sigma = sigma + numpy.asarray(res["virial"]) / cell.vol
        # The relaxation target is the variational energy whose derivatives the forces are: E, or F with smearing.
        variational = energies.get("e_free", energies["e_tot"]) + energies.get("e_disp", 0.0)
        forces = None if grad is None else -grad * HARTREE_EV / BOHR_A
        stress = None if sigma is None else sigma * HARTREE_EV / BOHR_A ** 3
        return {"energies_Ha": energies, "variational_energy_Ha": variational, "forces_eV_per_A": forces,
                "stress_eV_per_A3": stress}


def reported_energy(energies: dict, convention: str) -> float:
    key = {"internal_energy": "e_tot", "free_energy": "e_free", "zero_temperature_extrapolated": "e_zero"}[convention]
    if key not in energies:
        key = "e_tot"  # fixed occupations: all three coincide
    return energies[key] + energies.get("e_disp", 0.0)


def _summary(ev: dict) -> dict:
    import numpy
    out = {"max_force_eV_per_A": None, "forces_eV_per_A": None, "stress_eV_per_A3": None, "stress_kbar": None,
           "max_stress_kbar": None, "pressure_kbar": None}
    if ev["forces_eV_per_A"] is not None:
        forces = numpy.asarray(ev["forces_eV_per_A"])
        out.update(max_force_eV_per_A=float(numpy.linalg.norm(forces, axis=1).max()) if len(forces) else 0.0,
                   forces_eV_per_A=forces.tolist())
    if ev["stress_eV_per_A3"] is not None:
        kbar = numpy.asarray(ev["stress_eV_per_A3"]) * EV_PER_A3_TO_KBAR
        out.update(stress_eV_per_A3=numpy.asarray(ev["stress_eV_per_A3"]).tolist(), stress_kbar=kbar.tolist(),
                   max_stress_kbar=float(abs(kbar).max()), pressure_kbar=float(-numpy.trace(kbar) / 3))
    return out


def _relax(job: dict, evaluator: Evaluator, log):
    """ASE optimisation until max |F| <= force tol (and max |stress_ij| <= stress tol for vc-relax)."""
    import numpy
    from ase import Atoms, optimize
    from ase.calculators.calculator import Calculator, all_changes
    from ase.filters import FrechetCellFilter

    s, geo = job["settings"], job["geometry"]
    variable_cell = job["calculation"] == "vc-relax"
    lattice = numpy.asarray(geo["lattice_A"], dtype=float)
    atoms = Atoms(symbols=geo["elements"], scaled_positions=geo["frac_coords"], cell=lattice, pbc=True)
    trajectory, last = [], {}

    class PySCFCalculator(Calculator):
        implemented_properties = ["energy", "free_energy", "forces", "stress"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            ev = evaluator(self.atoms.cell.array, self.atoms.get_chemical_symbols(), self.atoms.positions, variable_cell)
            last.clear()
            last.update(ev)
            energy = ev["variational_energy_Ha"] * HARTREE_EV
            stress = ev["stress_eV_per_A3"]
            voigt = numpy.zeros(6) if stress is None else numpy.array(
                [stress[0][0], stress[1][1], stress[2][2], stress[1][2], stress[0][2], stress[0][1]])
            self.results = {"energy": energy, "free_energy": energy, "forces": numpy.asarray(ev["forces_eV_per_A"]),
                            "stress": voigt}

    atoms.calc = PySCFCalculator()
    relax = s["relaxation"]
    target = FrechetCellFilter(atoms) if variable_cell else atoms
    opt = getattr(optimize, relax["optimizer"])(target, logfile=log, maxstep=float(relax["max_step_A"]))
    f_tol, s_tol = float(relax["force_tolerance_eV_per_A"]), relax.get("stress_tolerance_kbar")
    converged = False
    # fmax=0 disables ASE's own criterion; convergence is decided here on the physical quantities.
    for step, _ in enumerate(opt.irun(fmax=0.0, steps=int(relax["max_steps"]))):
        summary = _summary(last)
        trajectory.append({"step": step, "energy_eV": last["variational_energy_Ha"] * HARTREE_EV,
                           "max_force_eV_per_A": summary["max_force_eV_per_A"],
                           "max_stress_kbar": summary["max_stress_kbar"]})
        if summary["max_force_eV_per_A"] <= f_tol and (not variable_cell or summary["max_stress_kbar"] <= float(s_tol)):
            converged = True
            break
    final = {"lattice_A": atoms.cell.array.tolist(), "elements": atoms.get_chemical_symbols(),
             "frac_coords": atoms.get_scaled_positions(wrap=False).tolist()}
    return converged, dict(last), final, trajectory


def run(job: dict, log) -> dict:
    import numpy
    from pyscf import lib
    s, geo = job["settings"], job["geometry"]
    lib.num_threads(int(s["n_threads"]))
    evaluator = Evaluator(s, log)
    result = {"schema": "mofpipe_pyscf_result_v1", "calculation": job["calculation"], "job_done": False,
              "scf_converged": False, "ionic_converged": None, "error": None}
    start = time.perf_counter()
    try:
        result["versions"] = versions(s)
        result["data_hashes"] = basis_and_pseudo_hashes(s, geo["elements"])
        if job["calculation"] == "scf":
            lattice = numpy.asarray(geo["lattice_A"], dtype=float)
            cart = numpy.asarray(geo["frac_coords"], dtype=float) @ lattice
            ev = evaluator(lattice, geo["elements"], cart, bool(s["compute_stress"]), bool(s["compute_forces"]))
            final, trajectory, ionic = dict(geo), [], None
        elif job["calculation"] in ("relax", "vc-relax"):
            ionic, ev, final, trajectory = _relax(job, evaluator, log)
        else:
            raise WorkerError("UNSUPPORTED_CALCULATION", job["calculation"])
        result.update(scf_converged=True, ionic_converged=ionic, energies_Ha=ev["energies_Ha"],
                      total_energy_Ha=repr(reported_energy(ev["energies_Ha"], s["energy_convention"])),
                      energy_convention=s["energy_convention"], final_geometry=final, trajectory=trajectory,
                      n_scf=evaluator.n_scf, **evaluator.meta, **_summary(ev))
    except SCFNotConverged as exc:
        result.update(scf_converged=False, scf_failure=str(exc), n_scf=evaluator.n_scf, **evaluator.meta)
    except WorkerError as exc:
        result["error"] = {"code": exc.code, "detail": exc.detail}
    except Exception as exc:  # noqa: BLE001 - every failure leaves a typed record, never an energy
        result["error"] = {"code": "ENGINE_ERROR", "detail": f"{type(exc).__name__}: {exc}",
                           "traceback": traceback.format_exc()[-4000:]}
    result["wall_time_s"] = round(time.perf_counter() - start, 3)
    result["job_done"] = result["error"] is None
    return result


def main(argv) -> int:
    if len(argv) == 3 and argv[1] == "--probe":
        with open(argv[2]) as handle:
            print(json.dumps(probe(json.load(handle)), sort_keys=True))
        return 0
    if len(argv) != 2:
        print("usage: pyscf_worker.py job.json | --probe probe.json", file=sys.stderr)
        return 2
    with open(argv[1]) as handle:
        job = json.load(handle)
    with open("pyscf.log", "a") as log:
        result = run(job, log)
    with open("result.json", "w") as handle:
        json.dump(result, handle, indent=1, sort_keys=True)
    return 0 if result["error"] is None else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
