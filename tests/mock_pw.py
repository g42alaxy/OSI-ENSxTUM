"""Mock Quantum ESPRESSO pw.x for plumbing tests. SYNTHETIC: never physical evidence.

Reads a pw.x input (``-in FILE``), writes pw.x-formatted output to stdout. The
energy is an explicit toy formula: per-element constants plus a screened
pair interaction between N atoms (periodic images included) whose strength
depends weakly on ecutwfc, so energy differences converge with cutoff.
Failure modes for tests come from MOCK_PW_MODE (comma-separated):
scf_fail_high_beta, crash, topology_change, unconverged_relax, large_force, lose_atom.
"""
import math
import os
import re
import sys

MU_RY = {"H": -0.9, "C": -10.8, "N": -19.6, "O": -31.5, "Zn": -460.0}
J_NODE_RY, NODE_RADIUS_A = 0.004, 6.0
J_SCREENED_RY, LAMBDA_A, KAPPA = 0.0004, 3.0, 200.0


def namelist_value(text, key, default=None):
    m = re.search(rf"^\s*{key}\s*=\s*'?([^'\n]+?)'?\s*$", text, re.M)
    return m.group(1) if m else default


def main():
    path = sys.argv[sys.argv.index("-in") + 1]
    text = open(path).read()
    modes = set(filter(None, os.environ.get("MOCK_PW_MODE", "").split(",")))
    if "crash" in modes:
        sys.exit("mock pw.x: simulated crash")
    calc = namelist_value(text, "calculation")
    ecut = float(namelist_value(text, "ecutwfc"))
    beta = float(namelist_value(text, "mixing_beta", "0.7"))
    lines = text.splitlines()
    i = lines.index("CELL_PARAMETERS angstrom")
    lattice = [[float(x) for x in lines[i + k].split()] for k in (1, 2, 3)]
    j = lines.index("ATOMIC_POSITIONS angstrom")
    nat = int(namelist_value(text, "nat"))
    atoms = [(l.split()[0], [float(x) for x in l.split()[1:4]]) for l in lines[j + 1:j + 1 + nat]]

    out = ["     Program PWSCF v.7.2 starts on  1Jan2026 at  0: 0: 0", "     SYNTHETIC MOCK - not a DFT calculation", ""]
    if "scf_fail_high_beta" in modes and beta >= 0.5:
        out += ["     convergence NOT achieved after 100 iterations: stopping", "", "     JOB DONE."]
        print("\n".join(out))
        return
    out += ["     convergence has been achieved in   9 iterations", ""]

    energy = sum(MU_RY[el] for el, _ in atoms)
    n_pos = [r for el, r in atoms if el == "N"]
    zn_pos = [r for el, r in atoms if el == "Zn"]
    images = [[a * lattice[0][c] + b * lattice[1][c] + d * lattice[2][c] for c in range(3)]
              for a in (-1, 0, 1) for b in (-1, 0, 1) for d in (-1, 0, 1)]

    def contacts(u, v):
        return sum(math.dist(u, [v[c] + img[c] for c in range(3)]) < NODE_RADIUS_A for img in images)

    nodes = [[contacts(n, r) for r in zn_pos] for n in n_pos]
    scale = 1 + KAPPA / ecut ** 2
    for p in range(len(n_pos)):
        for q in range(p, len(n_pos)):
            if p != q:
                energy += J_NODE_RY * sum(a * b for a, b in zip(nodes[p], nodes[q])) * scale
            else:  # an N meeting its own periodic image at one Zn
                energy += J_NODE_RY * sum(a * (a - 1) // 2 for a in nodes[p]) * scale
            for img in images:
                dist = math.dist(n_pos[p], [n_pos[q][c] + img[c] for c in range(3)])
                if dist > 1e-6:
                    energy += (0.5 if p == q else 1.0) * J_SCREENED_RY * math.exp(-dist / LAMBDA_A) * scale
    out += [f"!    total energy              = {energy:17.8f} Ry", ""]
    force = 0.1 if "large_force" in modes else 0.00001
    out += ["     Forces acting on atoms (cartesian axes, Ry/au):", ""]
    out += [f"     atom {k + 1:4d} type  1   force = {force:14.8f}{0.0:14.8f}{0.0:14.8f}" for k in range(nat)]
    out += ["", f"     Total force = {force * nat:12.6f}     Total SCF correction =     0.000000", ""]
    out += ["     total   stress  (Ry/bohr**3)                   (kbar)     P=        0.10"]
    out += ["  " + "  ".join("%12.8f" % (0.00000068 if r == c else 0.0) for c in range(3)) + "   "
            + "  ".join("%10.2f" % (0.10 if r == c else 0.0) for c in range(3)) for r in range(3)] + [""]

    if calc in ("relax", "vc-relax"):
        final = []
        for el, r in atoms:
            if "topology_change" in modes and el == "N":
                r = [r[0] + 4.0, r[1], r[2]]
            final.append((el, r))
        if "lose_atom" in modes:
            final = final[:-1]
        if "unconverged_relax" not in modes:
            out += ["     bfgs converged in   3 scf cycles and   2 bfgs steps", "     End of BFGS Geometry Optimization", "",
                    "Begin final coordinates", "", "CELL_PARAMETERS (angstrom)"]
            out += ["   " + "  ".join(f"{x:.9f}" for x in v) for v in lattice]
            out += ["", "ATOMIC_POSITIONS (angstrom)"]
            out += [f"{el}  " + "  ".join(f"{x:.9f}" for x in r) for el, r in final]
            out += ["End final coordinates", ""]
        else:
            out += ["     The maximum number of steps has been reached.", ""]
    out += ["     JOB DONE."]
    print("\n".join(out))


if __name__ == "__main__":
    main()
