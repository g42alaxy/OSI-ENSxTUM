"""Task 07 (T07.1): reproducible active-space Hamiltonians (ActiveSpaceArtifact + FCIDUMP).

``prepare_active_space(system, protocol)`` takes one specific geometry (and the calculation it came from), runs
the declared mean-field reference with PySCF and exports the active-space problem every electronic solver must
share:

    E = E_core + sum_pq h_pq g_pq + 1/2 sum_pqrs (pq|rs) G_pqrs

h1 and E_core include the nuclear repulsion and the frozen doubly occupied core (PySCF CASCI ``get_h1eff``);
h2 is in chemists' notation over real active orbitals, ordered by ascending orbital energy. Electron counts
are stored by spin. Embedding: only ``none`` (a bare molecule or cluster) is implemented. A bare-cluster
energy is not a periodic correction (T07.3), so every artifact carries ``correction_status = UNVALIDATED``.
Identical inputs give identical integrals, constants and hash.
"""
from __future__ import annotations

import hashlib
import os
import tempfile

from .contract import PipelineError, content_hash, make_record

PROTOCOL_REQUIRED = ("protocol_id", "basis", "reference", "n_active_electrons", "n_active_orbitals", "embedding")
SYSTEM_REQUIRED = ("system_id", "atoms", "unit", "charge", "spin")


def _require(record: dict, keys, prefix: str) -> None:
    missing = [k for k in keys if record.get(k) is None]
    if missing:
        raise PipelineError("SCIENTIFIC_SPEC_MISSING", ", ".join(f"{prefix}.{k}" for k in missing))


def integral_hash(h1, h2, ecore: float, nelec) -> str:
    """Hash of the electronic problem itself (rounded to 1e-10 Eh, so platform noise below that is ignored)."""
    return content_hash({"h1": [[round(float(x), 10) + 0.0 for x in row] for row in h1],
                         "h2": [round(float(x), 10) + 0.0 for x in h2.reshape(-1)],
                         "ecore": round(float(ecore), 10), "nelec": list(nelec)})


def prepare_active_space(system: dict, protocol: dict) -> dict:
    """prepare_active_space(geometry/calculation, active_space_protocol) -> ActiveSpaceArtifact."""
    _require(system, SYSTEM_REQUIRED, "system")
    _require(protocol, PROTOCOL_REQUIRED, "protocol")
    if protocol["embedding"] != "none":
        #TODO: Task 07 — active spaces in an embedding potential (WF-in-DFT); the finite-cluster subtractive
        #TODO: scheme (embedding.py) exports bare capped clusters, which is what 'none' means here.
        raise PipelineError("UNSUPPORTED_EMBEDDING", f"{protocol['embedding']}: only 'none' is implemented")
    try:
        import numpy as np
        from pyscf import __version__ as pyscf_version, ao2mo, gto, lib, mcscf, scf
        from pyscf.tools import fcidump
    except ImportError as exc:
        raise PipelineError("BACKEND_UNAVAILABLE", f"PySCF is required for active-space export: {exc}") from exc
    mol = gto.M(atom=[[a[0], tuple(a[1:4])] for a in system["atoms"]], unit=system["unit"], basis=protocol["basis"],
                charge=system["charge"], spin=system["spin"], verbose=0)
    if protocol["reference"] not in ("RHF", "ROHF"):
        raise PipelineError("UNSUPPORTED_REFERENCE", protocol["reference"])
    # One OpenMP thread: threaded integral reductions differ at ~1e-14 Eh between runs, which would change the
    # stored integrals, the FCIDUMP hash and the artifact ID for identical inputs.
    with lib.with_omp_threads(1):
        mf = (scf.RHF if protocol["reference"] == "RHF" else scf.ROHF)(mol)
        mf.conv_tol = protocol.get("conv_tol", 1e-12)
        mf.max_cycle = protocol.get("max_cycle", mf.max_cycle)
        if protocol.get("scf_solver") == "newton":  # declared numerics: second-order SCF for transition-metal clusters
            mf = mf.newton()
        mf.kernel()
        if not mf.converged:
            raise PipelineError("SCF_NOT_CONVERGED", system["system_id"])
        ncas, nelecas = protocol["n_active_orbitals"], protocol["n_active_electrons"]
        chosen = protocol.get("active_orbitals")
        if chosen is not None and (len(set(chosen)) != ncas or not all(type(i) is int and 0 <= i < mol.nao_nr() for i in chosen)):
            raise PipelineError("INVALID_ACTIVE_SPACE", f"active_orbitals {chosen} must be {ncas} distinct MO indices")
        mc = mcscf.CASCI(mf, ncas, nelecas)
        if protocol.get("active_orbitals") is not None:  # explicit 0-based MO indices
            mo = mc.sort_mo(protocol["active_orbitals"], base=0)
            selection = sorted(protocol["active_orbitals"])
        else:
            mo = mf.mo_coeff
            selection = list(range(mc.ncore, mc.ncore + ncas))
        h1, ecore = mc.get_h1eff(mo)
        h2 = ao2mo.restore(1, mc.get_h2eff(mo), ncas)
    nalpha, nbeta = mc.nelecas
    ms2 = nalpha - nbeta
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "FCIDUMP")
        fcidump.from_integrals(path, h1, h2, ncas, (nalpha, nbeta), nuc=ecore, ms=ms2, tol=1e-15)
        text = open(path).read()
    # Orbital character: Mulliken weight of every atom in each active orbital.
    s = mol.intor("int1e_ovlp")
    active = mo[:, mc.ncore:mc.ncore + ncas]
    character = []
    for k in range(ncas):
        c = active[:, k]
        weights = c * (s @ c)
        per_atom = {}
        for ao, (atom_index, *_rest) in enumerate(mol.ao_labels(fmt=False)):
            label = f"{mol.atom_symbol(atom_index)}{atom_index}"
            per_atom[label] = per_atom.get(label, 0.0) + float(weights[ao])
        character.append({k_: round(v, 4) for k_, v in sorted(per_atom.items(), key=lambda kv: -abs(kv[1])) if abs(v) > 1e-3})
    geometry = {"atoms": system["atoms"], "unit": system["unit"], "charge": system["charge"], "spin": system["spin"]}
    problem_hash = integral_hash(h1, h2, ecore, (nalpha, nbeta))
    return make_record(
        "ActiveSpaceArtifact", system.get("provenance", "PHYSICAL"), system_id=system["system_id"],
        calculation_id=system.get("calculation_id"), geometry_source=system.get("geometry_source"),
        geometry_hash=content_hash(geometry), protocol=protocol, protocol_hash=content_hash(protocol),
        embedding={"scheme": "none", "note": "bare system; not an embedded correction"},
        correction_status="UNVALIDATED",
        software={"pyscf": pyscf_version, "numpy": np.__version__, "omp_threads": 1},
        basis={"name": protocol["basis"], "n_ao": mol.nao_nr(), "hash": content_hash([protocol["basis"], mol._basis])},
        mean_field={"method": protocol["reference"], "energy_Ha": float(mf.e_tot), "converged": bool(mf.converged),
                    "mo_energies_Ha": [round(float(e), 10) for e in mf.mo_energy]},
        n_core_orbitals=mc.ncore, active_orbital_indices=selection,
        orbital_ordering="active orbitals in ascending mean-field orbital energy",
        n_active_orbitals=ncas, electrons_by_spin={"alpha": nalpha, "beta": nbeta},
        integral_convention="chemists' (pq|rs), real orbitals; E = E_core + h.g + 1/2 (pq|rs) G; E_core includes "
                            "nuclear repulsion and frozen-core energy; units Hartree",
        h1=[[float(x) for x in row] for row in h1], h2=[float(x) for x in h2.reshape(-1)],
        scalar_constants={"E_core_Ha": float(ecore), "nuclear_repulsion_Ha": float(mol.energy_nuc())},
        hamiltonian_hash=problem_hash, fcidump=text, fcidump_sha256=hashlib.sha256(text.encode()).hexdigest(),
        orbital_character=character)


def problem_from_artifact(artifact: dict):
    """(h1, h2 as norb^4 array, E_core, (n_alpha, n_beta), norb) exactly as stored; every solver starts here."""
    import numpy as np
    n = artifact["n_active_orbitals"]
    return (np.array(artifact["h1"]), np.array(artifact["h2"]).reshape(n, n, n, n), artifact["scalar_constants"]["E_core_Ha"],
            (artifact["electrons_by_spin"]["alpha"], artifact["electrons_by_spin"]["beta"]), n)


def read_fcidump(text: str):
    """Independent FCIDUMP reader (8-fold symmetric integrals) for round-trip checks."""
    import numpy as np
    header, _, body = text.partition("&END")
    if not body:
        header, _, body = text.partition("/")
    norb = int(header.split("NORB=")[1].split(",")[0])
    h1, h2, ecore = np.zeros((norb, norb)), np.zeros((norb,) * 4), 0.0
    for line in body.strip().splitlines():
        v, p, q, r, s = line.split()
        v, p, q, r, s = float(v), int(p) - 1, int(q) - 1, int(r) - 1, int(s) - 1
        if p < 0:
            ecore = v
        elif r < 0:
            h1[p, q] = h1[q, p] = v
        else:
            for a, b, c, d in ((p, q, r, s), (q, p, r, s), (p, q, s, r), (q, p, s, r),
                               (r, s, p, q), (s, r, p, q), (r, s, q, p), (s, r, q, p)):
                h2[a, b, c, d] = v
    return h1, h2, ecore
