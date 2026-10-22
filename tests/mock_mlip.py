"""Mock ML-potential worker (same JSON protocol as src/qumof/mlip_worker.py) for tests without torch.

Energy (eV/cell): fixed per-element terms, minus 0.05 eV for every N-N pair closer than 4.5 A (a deterministic
ordering preference), plus ``MOCK_MLIP_SHIFT`` eV per atom for structures of topology ``MOCK_MLIP_SHIFT_TOPOLOGY``
(to make one topology higher). Geometries are returned unrelaxed and converged.
``MOCK_MLIP_FAIL=1`` makes every relaxation fail. ``MOCK_MLIP_MIRROR_PW=1`` returns exactly the mock pw.x energy
(tests/mock_pw.py at ecutwfc 80 Ry, in eV) so ML and DFT agree and calibration passes with zero error.
"""
import json
import math
import os
import sys

E_ATOM = {"H": -3.0, "C": -9.0, "N": -8.0, "O": -7.0, "Zn": -1.5}


RY_EV = 13.605693122994


def mirror_pw(item):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import mock_pw as pw
    lat, el = item["lattice_A"], item["elements"]
    pos = [[sum(f[j] * lat[j][i] for j in range(3)) for i in range(3)] for f in item["frac_coords"]]
    energy = sum(pw.MU_RY[x] for x in el)
    n_pos = [r for x, r in zip(el, pos) if x == "N"]
    zn_pos = [r for x, r in zip(el, pos) if x == "Zn"]
    images = [[a * lat[0][c] + b * lat[1][c] + d * lat[2][c] for c in range(3)]
              for a in (-1, 0, 1) for b in (-1, 0, 1) for d in (-1, 0, 1)]
    contacts = lambda u, v: sum(math.dist(u, [v[c] + img[c] for c in range(3)]) < pw.NODE_RADIUS_A for img in images)
    nodes = [[contacts(n, r) for r in zn_pos] for n in n_pos]
    scale = 1 + pw.KAPPA / 80.0 ** 2
    for p in range(len(n_pos)):
        for q in range(p, len(n_pos)):
            if p != q:
                energy += pw.J_NODE_RY * sum(a * b for a, b in zip(nodes[p], nodes[q])) * scale
            else:
                energy += pw.J_NODE_RY * sum(a * (a - 1) // 2 for a in nodes[p]) * scale
            for img in images:
                dist = math.dist(n_pos[p], [n_pos[q][c] + img[c] for c in range(3)])
                if dist > 1e-6:
                    energy += (0.5 if p == q else 1.0) * pw.J_SCREENED_RY * math.exp(-dist / pw.LAMBDA_A) * scale
    return energy * RY_EV


def energy(item):
    if os.environ.get("MOCK_MLIP_MIRROR_PW"):
        return mirror_pw(item)
    lat, fr, el = item["lattice_A"], item["frac_coords"], item["elements"]
    cart = [[sum(f[j] * lat[j][i] for j in range(3)) for i in range(3)] for f in fr]
    e = sum(E_ATOM.get(x, -5.0) for x in el)
    n_idx = [i for i, x in enumerate(el) if x == "N"]
    for a in range(len(n_idx)):
        for b in range(a + 1, len(n_idx)):
            d = [cart[n_idx[a]][k] - cart[n_idx[b]][k] for k in range(3)]
            if math.sqrt(sum(x * x for x in d)) < 4.5:
                e -= 0.05
    if item.get("topology_id") == os.environ.get("MOCK_MLIP_SHIFT_TOPOLOGY"):
        e += float(os.environ.get("MOCK_MLIP_SHIFT", "0")) * len(el)
    return e


def main(argv):
    probe = argv[0] == "--probe"
    batch = json.load(open(argv[-1]))
    identity = {"engine": "mock", "model": batch["protocol"]["model"], "checkpoint_sha256": "0" * 64, "worker_version": "mock"}
    if probe:
        json.dump({"identity": identity}, open("probe.json", "w"))
        return 0
    results = {}
    for item in batch["structures"]:
        if os.environ.get("MOCK_MLIP_FAIL"):
            results[item["key"]] = {"status": "FAILED", "error": "mock failure"}
            continue
        e = energy(item)
        results[item["key"]] = {"status": "SUCCEEDED", "initial_energy_eV": e, "energy_eV": e, "converged": True,
                                "n_steps": 0, "max_force_eV_per_A": 0.0, "max_stress_GPa": 0.0, "wall_time_s": 0.0,
                                "relaxed": {k: item[k] for k in ("lattice_A", "elements", "frac_coords")}}
    json.dump({"identity": identity, "results": results}, open("results.json", "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
