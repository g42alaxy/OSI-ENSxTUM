"""Mock finite-cluster engine for embedding plumbing tests. SYNTHETIC: never physical evidence.

Reads the cluster job (``input.json``) written by embedding.ClusterBackend and writes ``result.json`` in the
cluster_worker format. The energy is an explicit toy formula: per-element constants plus a screened pair term,
with a level-dependent constant and pair strength, so E_high - E_low depends on the cluster's composition and
geometry. The same level always gives the same energy, so high = low gives a zero correction.
MOCK_CLUSTER_MODE=fail_high makes every non-DFT level fail (a missing high-level term).
"""
import json
import math
import os
import sys

MU_HA = {"H": -0.5, "C": -37.8, "N": -54.5, "O": -75.0, "Zn": -1777.0}
SHIFT_HA = {"H": -0.02, "C": -0.15, "N": -0.18, "O": -0.21, "Zn": -0.9}  # high level only
PAIR_HA = {"DFT": 0.010, "high": 0.013}


def main():
    job = json.load(open(sys.argv[1]))
    level = job["level"]
    kind = "DFT" if level["method"] == "DFT" else "high"
    if kind == "high" and os.environ.get("MOCK_CLUSTER_MODE") == "fail_high":
        result = {"status": "FAILED", "error": {"code": "NOT_CONVERGED", "detail": "mock"}, "energy_Ha": None}
    else:
        els, xyz = job["elements"], job["coords_A"]
        energy = sum(MU_HA[e] + (SHIFT_HA[e] if kind == "high" else 0.0) for e in els)
        energy -= PAIR_HA[kind] * sum(math.exp(-math.dist(xyz[i], xyz[j]))
                                      for i in range(len(els)) for j in range(i + 1, len(els)))
        result = {"status": "SUCCEEDED", "energy_Ha": energy, "converged": True, "error": None,
                  "details": {"engine": "mock_cluster (SYNTHETIC)"}, "versions": {"mock_cluster": "1"}}
    json.dump(result, open("result.json", "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
