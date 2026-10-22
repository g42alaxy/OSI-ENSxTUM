"""Task 06 with a real electronic-structure engine: PySCF periodic GPW (pyscf_worker.py + dft.PySCFAdapter).

Physics checks run the worker in-process on tiny cells (Si2, H2) so they take seconds: energies equal a direct
PySCF + D3 calculation, analytic forces and stress (DFT + D3) equal finite differences, a k-mesh calculation
equals the replicated cell at the folded k-mesh, smearing conventions, relaxations reach the declared
tolerances, and SCF failure is reported without an energy. Backend checks (intake validation, engine probe,
fidelity, GPU switch) need no SCF. The full pipeline on the SYNTHETIC formate toy is opt-in:
MOFPIPE_SLOW_TESTS=1.
"""
import json
import os
import sys
from fractions import Fraction

import pytest

pytest.importorskip("pyscf")
pytest.importorskip("pyscf.dispersion")
pytestmark = pytest.mark.pyscf
import numpy as np  # noqa: E402
from pyscf_toy import pyscf_dft_protocol, write_pyscf_toy_project  # noqa: E402

from mofpipe import dft, pyscf_worker as worker  # noqa: E402
from mofpipe.intake import load_project, validate_project  # noqa: E402
from mofpipe.orchestrator import run_pipeline  # noqa: E402
from mofpipe.store import RunStore  # noqa: E402

SI_A = 5.43
SI_LATTICE = [[0.0, SI_A / 2, SI_A / 2], [SI_A / 2, 0.0, SI_A / 2], [SI_A / 2, SI_A / 2, 0.0]]
SI_FRAC = [[0.0, 0.0, 0.0], [0.26, 0.24, 0.25]]  # second atom slightly off the diamond site: nonzero forces


def settings(**overrides):
    base = {"xc": "PBE", "dispersion": "none", "dispersion_xc": "pbe", "basis": "gth-szv", "pseudo": "gth-pbe",
            "ke_cutoff_Ha": 25, "precision": 1e-8, "integral_scheme": "multigrid", "kmesh": [1, 1, 1],
            "occupations": {"kind": "fixed"}, "energy_convention": "internal_energy", "conv_tol_Ha": 1e-10,
            "magnetic_state": {"label": "closed_shell", "nspin": 1}, "numerics": {"max_cycle": 100},
            "device": "cpu", "n_threads": 4, "max_memory_MB": 2000, "cell_charge": 0, "compute_forces": True,
            "compute_stress": True, "relaxation_calculation": "scf", "optimizer": "BFGS",
            "relaxation": {"optimizer": "BFGS", "max_steps": 40, "max_step_A": 0.1, "force_tolerance_eV_per_A": 0.02,
                           "stress_tolerance_kbar": 2.0}}
    base.update(overrides)
    return base


def run(tmp_path, calculation, lattice, elements, frac, s):
    with open(tmp_path / "pyscf.log", "a") as log:
        return worker.run({"calculation": calculation, "settings": s,
                           "geometry": {"lattice_A": lattice, "elements": elements, "frac_coords": frac}}, log)


def evaluate(s, lattice, elements, frac):
    import io
    lattice = np.asarray(lattice, dtype=float)
    return worker.Evaluator(s, io.StringIO())(lattice, elements, np.asarray(frac) @ lattice, True)


def test_worker_energy_equals_direct_pyscf_plus_d3(tmp_path):
    from pyscf.dispersion import dftd3
    from pyscf.pbc import dft as pbcdft, gto
    s = settings(dispersion="d3bj", compute_stress=False)
    result = run(tmp_path, "scf", SI_LATTICE, ["Si", "Si"], SI_FRAC, s)
    assert result["job_done"] and result["scf_converged"] and result["error"] is None
    cell = gto.Cell(a=SI_LATTICE, atom=[["Si", np.asarray(f) @ np.asarray(SI_LATTICE)] for f in SI_FRAC], unit="A",
                    basis="gth-szv", pseudo="gth-pbe", ke_cutoff=25, verbose=0)
    cell.build()
    mf = pbcdft.RKS(cell, xc="PBE").multigrid_numint()
    mf.conv_tol = 1e-10
    e_dft = mf.kernel()
    e_d3 = float(dftd3.DFTD3Dispersion(cell, "pbe", version="d3bj").get_dispersion()["energy"])
    assert abs(float(result["total_energy_Ha"]) - (e_dft + e_d3)) < 1e-7
    assert abs(result["energies_Ha"]["e_disp"] - e_d3) < 1e-12
    assert result["versions"]["pyscf"] and set(result["data_hashes"]["basis"]) == {"Si"}
    assert result["mesh"] == [int(x) for x in cell.mesh] and result["kmesh"] == [1, 1, 1]


@pytest.mark.parametrize("scheme,kmesh", [("multigrid", [1, 1, 1]), ("fftdf", [2, 2, 2])])
def test_forces_and_stress_match_finite_differences(scheme, kmesh):
    """Analytic DFT + D3 forces (eV/A) and stress (eV/A^3, (1/V) dE/d strain) against central differences."""
    s = settings(dispersion="d3bj", integral_scheme=scheme, kmesh=kmesh, ke_cutoff_Ha=30)
    ref = evaluate(s, SI_LATTICE, ["Si", "Si"], SI_FRAC)
    lattice = np.asarray(SI_LATTICE)

    def energy_eV(lat, frac):
        return evaluate({**s, "compute_stress": False}, lat, ["Si", "Si"], frac)["variational_energy_Ha"] * worker.HARTREE_EV

    h = 2e-3  # angstrom, atom 1 along x
    cart = np.asarray(SI_FRAC) @ lattice
    shifted = [np.linalg.solve(lattice.T, (cart + d).T).T for d in (np.array([[0, 0, 0], [h, 0, 0]]),
                                                                       np.array([[0, 0, 0], [-h, 0, 0]]))]
    fd_force = -(energy_eV(lattice, shifted[0]) - energy_eV(lattice, shifted[1])) / (2 * h)
    assert abs(ref["forces_eV_per_A"][1][0] - fd_force) < 2e-4
    volume = abs(np.linalg.det(lattice))
    eps = 1e-4
    for i, j in ((0, 0), (0, 1)):
        d = np.zeros((3, 3))
        d[i, j] = d[j, i] = eps / (2 if i != j else 1)  # symmetric strain
        plus, minus = lattice @ (np.eye(3) + d).T, lattice @ (np.eye(3) - d).T
        fd = (energy_eV(plus, SI_FRAC) - energy_eV(minus, SI_FRAC)) / (2 * eps) / volume
        analytic = ref["stress_eV_per_A3"][i][j]
        assert abs(analytic - fd) < 2e-5, (i, j, analytic, fd)


def test_kpoint_mesh_equals_replicated_cell_at_folded_mesh(tmp_path):
    """Consistent sampling: primitive cell with mesh [2,2,2] == cell doubled along a3 with mesh [2,2,1]."""
    spacing = 1.5
    doubled = [SI_LATTICE[0], SI_LATTICE[1], [2 * x for x in SI_LATTICE[2]]]
    assert dft._kmesh(SI_LATTICE, spacing) == [2, 2, 2] and dft._kmesh(doubled, spacing) == [2, 2, 1]
    frac2 = [[f[0], f[1], f[2] / 2] for f in SI_FRAC] + [[f[0], f[1], f[2] / 2 + 0.5] for f in SI_FRAC]
    energies = []
    for lattice, elements, frac in ((SI_LATTICE, ["Si"] * 2, SI_FRAC), (doubled, ["Si"] * 4, frac2)):
        s = settings(integral_scheme="fftdf", ke_cutoff_Ha=30, kmesh=dft._kmesh(lattice, spacing), compute_stress=False)
        result = run(tmp_path, "scf", lattice, elements, frac, s)
        energies.append(float(result["total_energy_Ha"]) / (len(elements) // 2))
    assert abs(energies[0] - energies[1]) < 2e-5


def test_smearing_conventions_and_reported_energy(tmp_path):
    s = settings(occupations={"kind": "smearing", "method": "fermi", "sigma_Ha": 0.01}, compute_stress=False,
                 energy_convention="zero_temperature_extrapolated")
    result = run(tmp_path, "scf", SI_LATTICE, ["Si", "Si"], SI_FRAC, s)
    e = result["energies_Ha"]
    assert e["e_free"] <= e["e_tot"] and abs(e["e_zero"] - 0.5 * (e["e_tot"] + e["e_free"])) < 1e-10
    assert float(result["total_energy_Ha"]) == e["e_zero"]


def test_relax_reaches_the_declared_force_tolerance(tmp_path):
    lattice = [[6.0, 0, 0], [0, 6.0, 0], [0, 0, 6.0]]
    frac = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5 + 1.0 / 6.0]]  # H2 stretched to 1.0 A
    s = settings(relaxation_calculation="relax", compute_stress=False, ke_cutoff_Ha=30)
    result = run(tmp_path, "relax", lattice, ["H", "H"], frac, s)
    assert result["ionic_converged"] is True and result["max_force_eV_per_A"] <= 0.02
    final = np.asarray(result["final_geometry"]["frac_coords"]) @ np.asarray(lattice)
    assert 0.75 < np.linalg.norm(final[1] - final[0]) < 0.90  # minimal-basis (gth-szv) PBE minimum ~0.84 A
    trajectory = result["trajectory"]
    assert trajectory[0]["max_force_eV_per_A"] > 0.5 and trajectory[-1]["energy_eV"] < trajectory[0]["energy_eV"]


def test_vc_relax_reaches_the_declared_stress_tolerance(tmp_path):
    strained = [[x * 1.04 for x in v] for v in SI_LATTICE]
    s = settings(relaxation_calculation="vc-relax", integral_scheme="fftdf", ke_cutoff_Ha=30,
                 kmesh=dft._kmesh(strained, 1.5))
    result = run(tmp_path, "vc-relax", strained, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]], s)
    assert result["ionic_converged"] is True
    assert result["max_stress_kbar"] <= 2.0 and result["max_force_eV_per_A"] <= 0.02
    assert result["trajectory"][0]["max_stress_kbar"] > 10  # it started far from equilibrium
    assert result["final_geometry"]["lattice_A"] != strained


def test_scf_failure_is_reported_without_energy(tmp_path):
    s = settings(numerics={"max_cycle": 1}, compute_stress=False)
    result = run(tmp_path, "scf", SI_LATTICE, ["Si", "Si"], SI_FRAC, s)
    assert result["scf_converged"] is False and "total_energy_Ha" not in result and result["error"] is None
    parsed = dft.PySCFAdapter().parse_output(json.dumps(result), "scf", {"lattice_A": SI_LATTICE, "elements": ["Si", "Si"],
                                                                        "frac_coords": SI_FRAC})
    assert parsed["scf_converged"] is False and parsed["total_energy"] is None


def test_unsupported_settings_raise_typed_errors(tmp_path):
    result = run(tmp_path, "scf", SI_LATTICE, ["Si", "Si"], SI_FRAC, settings(numerics={"xc": "PBE0"}))
    assert result["error"]["code"] == "PROTOCOL_RETRY_CHANGES_SCIENCE" and result["job_done"] is False
    result = run(tmp_path, "scf", SI_LATTICE, ["Si", "Si"], SI_FRAC, settings(kmesh=[2, 2, 2]))
    assert result["error"]["code"] == "UNSUPPORTED_KPOINTS_FOR_MULTIGRID"


# -- backend: intake validation, probe, fidelity, GPU switch (no SCF) -----------------------------
@pytest.mark.parametrize("patch,code,path", [
    ({"integral_scheme": "multigrid", "kpoint_spacing_inv_A": 0.3}, "UNSUPPORTED_KPOINTS_FOR_MULTIGRID",
     "protocols.dft.kpoint_spacing_inv_A"),
    ({"hubbard_u_eV": {"Al-3p": 2.0}}, "UNSUPPORTED_HUBBARD_U", "protocols.dft.hubbard_u_eV"),
    ({"numerical_retries": [{"xc_functional": "PBE0"}]}, "PROTOCOL_RETRY_CHANGES_SCIENCE", "protocols.dft.numerical_retries[0]"),
    ({"difference_convergence": {"ke_cutoff_Ha_series": [20, 25]}}, "SCIENTIFIC_SPEC_MISSING",
     "protocols.dft.difference_convergence.ke_cutoff_Ha_series"),
    ({"dispersion": "d5"}, "UNSUPPORTED_DISPERSION", "protocols.dft.dispersion"),
    ({"zero_smearing_convention": "total_energy_as_printed"}, "UNSUPPORTED_ENERGY_CONVENTION",
     "protocols.dft.zero_smearing_convention"),
    ({"basis": {"Al": "gth-szv"}}, "SCIENTIFIC_SPEC_MISSING", "protocols.dft.basis"),
    ({"device": "tpu"}, "UNSUPPORTED_DEVICE", "protocols.dft.device"),
    ({"engine": "vasp"}, "UNSUPPORTED_DFT_ENGINE", "protocols.dft.engine"),
])
def test_pyscf_protocol_errors_block_at_intake(tmp_path, patch, code, path):
    spec = load_project(write_pyscf_toy_project(tmp_path / "toy", {"protocols": {"dft": patch}}))
    errors = validate_project(spec)["errors"]
    assert {"code": code, "check": path} in [{"code": e["code"], "check": e["check"]} for e in errors], errors


def test_engine_identity_is_part_of_the_fidelity(tmp_path):
    spec = load_project(write_pyscf_toy_project(tmp_path / "toy"))
    backend = dft.PeriodicDFTBackend(spec, RunStore(tmp_path / "run"))
    identity = backend.engine_identity
    assert identity["versions"]["pyscf"] and identity["versions"]["pyscf_dispersion"]
    assert set(identity["data_hashes"]["basis"]) == {"Al", "C", "F", "H", "O"}
    physical = spec.physical
    other = dft.protocol_fingerprint(backend.protocol, physical["chemistry"]["magnetic_state_policy"], backend.tolerances,
                                     {**identity, "versions": {**identity["versions"], "pyscf": "0.0"}})
    assert other != backend.protocol_hash and backend.fidelity.endswith(backend.protocol_hash[:12])
    assert backend.command[0] == os.path.abspath(sys.executable) or backend.command[0].endswith("python")


@pytest.mark.parametrize("patch,code", [({"device": "gpu"}, "BACKEND_UNAVAILABLE"),
                                        ({"xc_functional": "PBE0"}, "UNSUPPORTED_DFT_PROTOCOL")])
def test_probe_refusals_block_the_labels_stage(tmp_path, patch, code):
    try:
        import gpu4pyscf  # noqa: F401
        if patch.get("device") == "gpu":
            pytest.skip("gpu4pyscf is installed here; the refusal path needs it absent")
    except ImportError:
        pass
    spec = load_project(write_pyscf_toy_project(tmp_path / "toy", {"protocols": {"dft": patch}}))
    report = run_pipeline(spec, tmp_path / "run")
    state = json.loads((tmp_path / "run" / "state.json").read_text())
    assert report["stage_status"]["labels"] == "BLOCKED"
    assert state["stages"]["labels"]["detail"]["error"]["code"] == code
    assert state["counters"].get("dft_submissions", 0) == 0


def test_pyscf_job_input_carries_every_scientific_setting(tmp_path):
    spec = load_project(write_pyscf_toy_project(tmp_path / "toy"))
    backend = dft.PeriodicDFTBackend(spec, RunStore(tmp_path / "run"))
    geometry = {"lattice_A": SI_LATTICE, "elements": ["Si", "Si"], "frac_coords": SI_FRAC}
    job = json.loads(backend.adapter.write_input(geometry, backend.protocol, backend.tolerances, "relax",
                                                 {"label": "closed_shell", "nspin": 1}, {"max_cycle": 60},
                                                 {"ke_cutoff_Ha": 15}, ".", ("energy", "forces"), backend=backend))
    s = job["settings"]
    assert (s["xc"], s["dispersion"], s["basis"], s["pseudo"], s["integral_scheme"]) == ("PBE", "d3bj", "gth-szv", "gth-pbe", "multigrid")
    assert s["ke_cutoff_Ha"] == 15 and s["numerics"] == {"max_cycle": 60} and s["kmesh"] == dft._kmesh(SI_LATTICE, 1.1)
    assert s["relaxation"]["force_tolerance_eV_per_A"] == 0.05 and s["relaxation"]["stress_tolerance_kbar"] == 5.0
    assert s["compute_stress"] is False and s["energy_convention"] == "free_energy"


@pytest.mark.skipif(os.environ.get("MOFPIPE_SLOW_TESTS") != "1", reason="real DFT on the formate toy (~10 min on 10 CPU cores)")
def test_pyscf_physical_pipeline_end_to_end(tmp_path):
    """With the shipped 20/25/30 Ha series the energy difference is not converged (grid error of the hard O/F GTH
    pseudopotentials), so the run must stop at labels with DFT_SETTINGS_NOT_CONVERGED; with a converged cutoff it
    must complete and every ranked energy must trace to its Hartree raw energy."""
    spec = load_project(write_pyscf_toy_project(tmp_path / "toy"))
    report = run_pipeline(spec, tmp_path / "run")
    if report["pipeline_status"] == "COMPLETED":
        labels = report["stage_detail"]["labels"]
        assert labels["dft_convergence"]["toy_al_formate"]["parameter"] == "ke_cutoff_Ha"
        for row in report["base_ranking"]:
            assert row["base_fidelity"].startswith("DFT:toy_pyscf_gpw_pbe_d3bj:")
            calc = json.loads((tmp_path / "run" / "artifacts" / (row["calculation_result_id"].replace(":", "__") + ".json")).read_text())
            assert calc["engine"] == "pyscf_gpw" and calc["raw_energy"]["unit"] == "Ha"
            assert Fraction(calc["base_energy_exact"]) == Fraction(calc["raw_energy"]["value"]) * dft.HARTREE_TO_EV / calc["n_formula_units"]
    else:  # an honest stop (e.g. MODEL_INADEQUATE on real energies) must name its reason
        halted = [s for s, v in report["stage_status"].items() if v in ("FAILED", "BLOCKED")]
        state = json.loads((tmp_path / "run" / "state.json").read_text())
        assert halted and state["stages"][halted[0]]["detail"].get("error", {}).get("code")
