"""A2 design-Hamiltonian mode and the Kang et al. (2025) reproduction benchmark.

Inputs per material are what Kang et al. started from (ACS Cent. Sci. 2025, doi:10.1021/acscentsci.5c00918;
arXiv:2502.06339, Tables S2-S3): the net-derived site graph with edge distances and edge kinds (topological /
second-neighbour), the candidate linkers with their characteristic lengths, and the composition. No energies
exist. A2 (the LLM) composes the Hamiltonian from a term catalogue; code turns the choice into an exact binary
polynomial (x_i = 1 when site i carries the second linker), searches the fixed-composition states exhaustively,
and runs the QAOA statevector simulation (optionally a Qiskit fake device) on that Hamiltonian.

Catalogue (code computes every coefficient; the model never writes numbers other than alpha and relative weights):
  length_balance      sum_e w_e (l_i + l_j - Lbar)^2, Lbar = 2 x composition-weighted mean linker length
  unlike_pair_mixing  sum_e w_e [x_i == x_j]  (penalize_like_pairs)  or  sum_e w_e [x_i != x_j]  (penalize_unlike_pairs)
  w_e = relative_weight x (d_e | d_e^alpha | 1) for weighting (distance | distance_power | uniform)

The reference answer is held back from the model: the experimental arrangement as reported in the paper (the
ground state of their model corresponds to the experimental structure, Fig. S3). ``kang_control`` evaluates the
paper's own weighting choices through the same compiler, as a check of this code, not as a model output.

Spectrum agreement with the paper's numbers: the paper's Hamiltonian (Eq. 2, 5-8; Table S3 weights) is evaluated on
every fixed-composition state and checked against its published levels (Fig. S3). A designed Hamiltonian has its own
scale and offset, so its values are mapped onto the paper's by the least-squares affine map with a non-negative
scale (lower must stay lower); MAE is in the paper's units, NMAE divides by the paper's spectrum range so materials
can be averaged, and Spearman rho compares the orderings.
"""
from __future__ import annotations

import json
import math
import random
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

from ..contract import canonical_json, content_hash
from .client import AgentCallError, Cache, Ledger, OpenRouterClient, call_role
from .contracts import CONTRACT_VERSION, EDGE_KINDS

TOL = 1e-9
DEFAULT_QAOA = {"layers": 2, "seeds": [0, 1, 2], "restarts": 2, "max_evaluations": 200}


def load_materials(directory) -> list:
    return [json.loads(p.read_text()) for p in sorted(Path(directory).glob("*.json"))]


def build_packet(material: dict) -> dict:
    pub = material["public"]
    packet = {
        "run_id": f"kang2025:{material['id']}", "schema_version": CONTRACT_VERSION, "mode": "design_hamiltonian",
        "role": "A2", "material": {"name": material["name"], "description": pub["description"], "net": pub["net"]},
        "linkers": pub["linkers"], "composition": pub["composition"],
        "sites": {"n_sites": pub["n_sites"], "encoding": {pub["linkers"][0]["id"]: 0, pub["linkers"][1]["id"]: 1}},
        "edges": pub["edges"],
        "term_catalogue": [
            {"term_id": "length_balance", "formula": "sum over chosen edges of w_e * (l_i + l_j - Lbar)^2; l = characteristic "
             "linker length on each end, Lbar = twice the composition-weighted mean linker length (constant)",
             "meaning": "penalises edges whose combined linker length departs from the average (geometric strain)"},
            {"term_id": "unlike_pair_mixing", "formula": "sum over chosen edges of w_e * [same linker on both ends] "
             "(penalize_like_pairs) or w_e * [different linkers] (penalize_unlike_pairs)",
             "meaning": "favours mixing or segregation of the two linkers along the chosen edges"}],
        "weightings": {"distance": "w_e = relative_weight * d_e", "distance_power": "w_e = relative_weight * d_e^alpha",
                       "uniform": "w_e = relative_weight"},
        "fixed_by_code": ["composition (exact linker counts)", "one linker per site", "all coefficients"],
        "limits": {"max_terms": 6},
    }
    return {**packet, "input_hash": content_hash(packet)}


def _weight(term: dict, edge: dict) -> float:
    d = edge["distance_A"]
    base = {"distance": d, "distance_power": d ** term["alpha"], "uniform": 1.0}[term["weighting"]]
    return term["relative_weight"] * base


def compile_design(design: dict, packet: dict) -> dict:
    """Binary polynomial {sites tuple: coefficient} (x_i = 1: second linker) from the chosen terms."""
    (a, la), (b, lb) = [(l["id"], l["characteristic_length_A"]) for l in packet["linkers"]]
    n, counts = packet["sites"]["n_sites"], packet["composition"]
    lbar = 2 * (counts[a] * la + counts[b] * lb) / n
    poly: dict = {}

    def add(key, value):
        poly[key] = poly.get(key, 0.0) + value

    for term in design["terms"]:
        for e in packet["edges"]:
            if e["kind"] not in term["edge_kinds"]:
                continue
            w, i, j = _weight(term, e), e["i"], e["j"]
            if term["term_id"] == "length_balance":  # (2 la - lbar + dl x_i + dl x_j)^2 with x^2 = x
                c, dl = 2 * la - lbar, lb - la
                add((), w * c * c)
                add((i,), w * (2 * c * dl + dl * dl))
                add((j,), w * (2 * c * dl + dl * dl))
                add((min(i, j), max(i, j)), w * 2 * dl * dl)
            else:  # [x_i != x_j] = x_i + x_j - 2 x_i x_j
                sign = 1.0 if term["preference"] == "penalize_unlike_pairs" else -1.0
                if sign < 0:
                    add((), w)  # [x_i == x_j] = 1 - [x_i != x_j]
                add((i,), sign * w)
                add((j,), sign * w)
                add((min(i, j), max(i, j)), -2 * sign * w)
    return {k: v for k, v in poly.items() if abs(v) > 1e-15}


def feasible_states(packet: dict) -> list:
    n = packet["sites"]["n_sites"]
    k = packet["composition"][packet["linkers"][1]["id"]]
    return [tuple(int(i in ones) for i in range(n)) for ones in combinations(range(n), k)]


def evaluate_polynomial(poly: dict, x) -> float:
    return sum(c * all(x[i] for i in key) for key, c in poly.items())


def validate_design(output: dict, packet: dict) -> dict:
    if output["status"] != "OK":
        ok = bool(output["error_codes"] or output["missing_input_ids"])
        return {"status": output["status"] if ok else "REJECTED", "errors": [] if ok else [
            {"code": "EMPTY_ERROR_STATE", "detail": output["status"]}]}
    errors, seen = [], set()
    kinds = {e["kind"] for e in packet["edges"]}
    if not output["terms"]:
        errors.append({"code": "NO_TERMS", "detail": "at least one term is required"})
    for t in output["terms"]:
        key = (t["term_id"], tuple(sorted(t["edge_kinds"])))
        if key in seen:
            errors.append({"code": "DUPLICATE_TERM", "detail": str(key)})
        seen.add(key)
        if set(t["edge_kinds"]) - kinds:
            errors.append({"code": "UNKNOWN_EDGE_KIND", "detail": str(sorted(set(t["edge_kinds"]) - kinds))})
        if (t["term_id"] == "length_balance") != (t["preference"] == "not_applicable"):
            errors.append({"code": "PREFERENCE_MISMATCH", "detail": f"{t['term_id']}: {t['preference']}"})
        if t["relative_weight"] <= 0:
            errors.append({"code": "ZERO_WEIGHT", "detail": t["term_id"]})
    if output["error_codes"] or output["missing_input_ids"]:
        errors.append({"code": "OK_WITH_ERROR_CODES", "detail": str(output["error_codes"] + output["missing_input_ids"])})
    if not errors:
        poly = compile_design(output, packet)
        values = [evaluate_polynomial(poly, s) for s in feasible_states(packet)]
        if max(values) - min(values) < TOL:
            errors.append({"code": "CONSTANT_HAMILTONIAN", "detail": "every arrangement has the same cost"})
    return {"status": "VALID" if not errors else "REJECTED", "errors": errors}


def _bits(state) -> str:
    return "".join(map(str, state))


def qaoa_probabilities(packet: dict, poly: dict, reference: set, config: dict) -> dict:
    """QAOA (XY exchange mixer on every edge, Dicke start) on the compiled Hamiltonian within the feasible subspace."""
    from ..qaoa import evolve, exchange_pairs, nelder_mead

    states = feasible_states(packet)
    values = [evaluate_polynomial(poly, s) for s in states]
    mean = sum(values) / len(values)
    spread = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)) or 1.0
    cost = [(v - mean) / spread for v in values]
    edges = sorted({(min(e["i"], e["j"]), max(e["i"], e["j"])) for e in packet["edges"]})
    pairs = exchange_pairs(states, edges)
    minimum = min(values)
    ground = [u for u, v in enumerate(values) if v - minimum < TOL]
    ref = [u for u, s in enumerate(states) if _bits(s) in reference]
    runs = []
    for seed in config["seeds"]:
        rng = random.Random(seed)
        best_params, best_value = None, None
        for _ in range(config["restarts"]):
            x0 = [rng.uniform(0, math.pi / 2) for _ in range(2 * config["layers"])]
            params, value, _ = nelder_mead(lambda p: sum(abs(a) ** 2 * c for a, c in zip(evolve(p, cost, pairs), cost)),
                                           x0, 0.3, config["max_evaluations"])
            if best_value is None or value < best_value:
                best_params, best_value = params, value
        probs = [abs(a) ** 2 for a in evolve(best_params, cost, pairs)]
        runs.append({"seed": seed, "params": [round(p, 6) for p in best_params],
                     "p_ground_state": sum(probs[u] for u in ground), "p_reference": sum(probs[u] for u in ref)})
    best = max(runs, key=lambda r: r["p_ground_state"])
    return {"layers": config["layers"], "runs": runs, "edges": [list(e) for e in edges],
            "median_p_reference": sorted(r["p_reference"] for r in runs)[len(runs) // 2],
            "median_p_ground_state": sorted(r["p_ground_state"] for r in runs)[len(runs) // 2],
            "uniform_baseline_p_reference": len(ref) / len(states), "best_params": best["params"],
            "scale": {"mean": mean, "spread": spread}}


def fake_device(packet: dict, poly: dict, reference: set, qaoa: dict, qpu: dict, workdir: Path) -> dict:
    """Sample the best QAOA circuit on a Qiskit fake IBM device (qpu_worker.py); decode and score against the reference."""
    from ..qpu import WORKER

    k = packet["composition"][packet["linkers"][1]["id"]]
    terms = sorted([list(s), c / qaoa["scale"]["spread"]] for s, c in poly.items() if s)
    job = {"device": qpu["device"], "n_qubits": packet["sites"]["n_sites"], "n_ones": k, "edges": qaoa["edges"],
           "terms": terms, "optimization_level": qpu.get("optimization_level", 3),
           "runs": [{"seed": 0, "params": qaoa["best_params"], "shots": qpu["shots"]}]}
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "job.json").write_text(json.dumps(job))
    proc = subprocess.run([qpu["python"], str(WORKER), "job.json"], cwd=workdir, capture_output=True, text=True,
                          timeout=qpu.get("timeout_s", 3600))
    if proc.returncode != 0:
        return {"status": "FAILED", "error": proc.stderr[-300:]}
    run = json.loads((workdir / "result.json").read_text())["runs"][0]
    samples = [("".join(reversed(b)), n) for b, n in run["counts"].items()]
    shots = sum(n for _, n in samples)
    feasible = sum(n for b, n in samples if b.count("1") == k)
    hits = sum(n for b, n in samples if b in reference)
    return {"status": "COMPLETED", "device": qpu["device"], "shots": shots, "postselection_rate": feasible / shots,
            "p_reference_per_shot": hits / shots, "p_reference_per_feasible": hits / feasible if feasible else 0.0,
            "two_qubit_gates": run["transpiled"]["two_qubit_gates"], "depth": run["transpiled"]["depth"]}


def kang_spectrum(material: dict, packet: dict) -> dict:
    """The paper's Hamiltonian on every fixed-composition state (ratio/occupancy terms vanish there)."""
    ref = material["reference"]["kang_hamiltonian"]
    lengths = [l["characteristic_length_A"] for l in packet["linkers"]]
    out = {}
    for s in feasible_states(packet):
        Ls = [(ref["weights"][e["kind"]], 2 * (lengths[s[e["i"]]] + lengths[s[e["j"]]])) for e in packet["edges"]]
        lbar = sum(L for _, L in Ls) / len(Ls)
        out[_bits(s)] = sum(w * (L - lbar) ** 2 for w, L in Ls)
    return out


def _ranks(values: list) -> list:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and abs(values[order[j + 1]] - values[order[i]]) < TOL:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2
        i = j + 1
    return ranks


def spectrum_agreement(material: dict, packet: dict, poly: dict) -> dict:
    """MAE / NMAE after the best non-negative affine map onto the paper's values, and Spearman rho, over all states."""
    kang = kang_spectrum(material, packet)
    keys = sorted(kang)
    k = [kang[b] for b in keys]
    q = [evaluate_polynomial(poly, tuple(int(c) for c in b)) for b in keys]
    n = len(keys)
    mq, mk = sum(q) / n, sum(k) / n
    var = sum((x - mq) ** 2 for x in q)
    scale = max(sum((x - mq) * (y - mk) for x, y in zip(q, k)) / var, 0.0) if var > TOL else 0.0
    offset = mk - scale * mq
    mae = sum(abs(scale * x + offset - y) for x, y in zip(q, k)) / n
    rq, rk = _ranks(q), _ranks(k)
    mrq, mrk = sum(rq) / n, sum(rk) / n
    den = math.sqrt(sum((a - mrq) ** 2 for a in rq) * sum((b - mrk) ** 2 for b in rk))
    spread = max(k) - min(k)
    return {"mae": mae, "nmae": mae / spread if spread else None, "spearman": (sum((a - mrq) * (b - mrk)
            for a, b in zip(rq, rk)) / den) if den else None, "scale": scale, "offset": offset, "n_states": n,
            "units": "paper's Hamiltonian units (Fig. S3)"}


def assess(material: dict, packet: dict, design: dict, qaoa_cfg: dict, qpu: dict | None, workdir: Path) -> dict:
    reference = set(material["reference"]["ground_state_bitstrings"])
    poly = compile_design(design, packet)
    states = feasible_states(packet)
    values = {_bits(s): evaluate_polynomial(poly, s) for s in states}
    minimum = min(values.values())
    ground = {b for b, v in values.items() if v - minimum < TOL}
    levels = sorted({round(v, 9) for v in values.values()})
    ref_energy = min(values[b] for b in reference)
    outcome = ("REPRODUCED" if ground <= reference else "REFERENCE_AMONG_DEGENERATE_MINIMA" if reference & ground
               else "MISSED")
    qaoa = qaoa_probabilities(packet, poly, reference, qaoa_cfg)
    out = {"outcome": outcome, "n_states": len(states), "ground_states": sorted(ground),
           "reference_level_rank": levels.index(round(ref_energy, 9)) + 1, "n_levels": len(levels),
           "reference_gap_to_minimum": ref_energy - minimum,
           "spectrum": spectrum_agreement(material, packet, poly),
           "qaoa": {k: v for k, v in qaoa.items() if k not in ("best_params", "scale", "edges")}}
    if qpu:
        out["fake_device"] = fake_device(packet, poly, reference, qaoa, qpu, workdir / "qpu")
    return out


def _ask(packet: dict, client, controls, ledger: Ledger, cache: Cache) -> tuple[dict | None, dict, dict]:
    """One A2 design request (+ one repair on invalid output if the extra request is allowed)."""
    max_tokens = controls["openrouter"]["max_tokens"]["A2"]
    output, meta = call_role("A2_design", packet, client=client, ledger=ledger, cache=cache, max_tokens=max_tokens)
    validation = validate_design(output, packet)
    if validation["status"] == "REJECTED" and ledger.extra_available:
        detail = "; ".join(f"{e['code']}: {e['detail']}" for e in validation["errors"])
        repair = [{"role": "assistant", "content": canonical_json(output)},
                  {"role": "user", "content": f"Application validation rejected the output: {detail}. "
                                              "Return one corrected, complete JSON object."}]
        output, meta = call_role("A2_design", packet, client=client, ledger=ledger, cache=cache, max_tokens=max_tokens,
                                 purpose="repair", repair=repair)
        validation = validate_design(output, packet)
    return output, meta, validation


def run_one(material: dict, controls, model: str, repeat: int, out_dir: Path, client=None, qaoa_cfg=None,
            qpu=None) -> dict:
    run_dir = out_dir / material["id"] / model.replace("/", "_") / f"r{repeat}"
    run_dir.mkdir(parents=True, exist_ok=True)
    packet = build_packet(material)
    (run_dir / "packet.json").write_text(json.dumps(packet, indent=2))
    row = {"material": material["id"], "model": model, "repeat": repeat}
    t0 = time.time()
    if model == "kang_control":  # the paper's weighting choices through the same compiler (code check, no LLM)
        design, meta, validation = material["reference"]["kang_control"], {}, {"status": "VALID", "errors": []}
        ledger = None
    else:
        ledger = Ledger(run_dir / "ledger.json", controls["budget"]["allow_extra_request"])
        try:
            design, meta, validation = _ask(packet, client, controls, ledger, Cache(None))
        except AgentCallError as exc:
            design, meta, validation = None, {}, {"status": "ERROR", "errors": [exc.as_dict()]}
    row.update(validation=validation["status"], validation_errors=validation["errors"], design=design,
               llm_requests=ledger.requests if ledger else 0,
               usage=[e.get("usage") for e in (ledger.entries if ledger else []) if e.get("usage")],
               model_id=meta.get("model"), provider=meta.get("provider"), llm_seconds=round(time.time() - t0, 1))
    row["cost_usd"] = round(sum((u or {}).get("cost") or 0 for u in row["usage"]), 5)
    if validation["status"] == "VALID":
        row["assessment"] = assess(material, packet, design, qaoa_cfg or DEFAULT_QAOA, qpu, run_dir)
    (run_dir / "result.json").write_text(json.dumps(row, indent=2, default=str))
    return row


def run_benchmark(materials: list, controls_factory, models: list, repeats: int, out_dir, client_factory=None,
                  qaoa_cfg=None, qpu=None, workers: int = 4, control: bool = True) -> list:
    """controls_factory(model) -> Controls; client_factory(model) -> client (default: OpenRouter with that model)."""
    out_dir = Path(out_dir)
    jobs = [(m, "kang_control", 0) for m in materials] if control else []
    jobs += [(m, model, r) for m in materials for model in models for r in range(repeats)]

    def work(job):
        material, model, r = job
        controls = controls_factory(model) if model != "kang_control" else None
        client = None if model == "kang_control" else (client_factory(model) if client_factory else OpenRouterClient(controls))
        return run_one(material, controls, model, r, out_dir, client, qaoa_cfg, qpu)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(work, jobs))
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2, default=str))
    (out_dir / "results.md").write_text(render(rows, materials, random_baseline(materials)))
    return rows


def random_baseline(materials: list, n: int = 500, seed: int = 0) -> dict:
    """Seeded random valid catalogue designs: what the scores look like without any chemical judgement."""
    rng, out = random.Random(seed), {}
    for m in materials:
        packet, ref = build_packet(m), set(m["reference"]["ground_state_bitstrings"])
        scores, reproduced = [], 0
        while len(scores) < n:
            terms = []
            for _ in range(rng.randint(1, 3)):
                t = rng.choice(["length_balance", "unlike_pair_mixing"])
                terms.append({"term_id": t, "edge_kinds": rng.choice([[k] for k in EDGE_KINDS] + [list(EDGE_KINDS)]),
                              "weighting": rng.choice(["distance", "distance_power", "uniform"]), "alpha": rng.random(),
                              "relative_weight": rng.uniform(0.05, 1.0), "preference": "not_applicable"
                              if t == "length_balance" else rng.choice(["penalize_like_pairs", "penalize_unlike_pairs"])})
            design = {"status": "OK", "terms": terms, "missing_input_ids": [], "error_codes": []}
            if validate_design(design, packet)["status"] != "VALID":
                continue
            poly = compile_design(design, packet)
            values = {_bits(s): evaluate_polynomial(poly, s) for s in feasible_states(packet)}
            low = min(values.values())
            reproduced += {b for b, v in values.items() if v - low < TOL} <= ref
            scores.append(spectrum_agreement(m, packet, poly))
        out[m["id"]] = {"n": n, "reproduced_rate": reproduced / n, "nmae": _mean([x["nmae"] for x in scores]),
                        "spearman": _mean([x["spearman"] for x in scores]), "mae": _mean([x["mae"] for x in scores])}
    return out


def rescore(out_dir, materials: list) -> list:
    """Recompute the spectrum metrics of saved runs from their designs (no LLM requests) and re-render."""
    out_dir, by_id = Path(out_dir), {m["id"]: m for m in materials}
    rows = json.loads((out_dir / "results.json").read_text())
    for r in rows:
        if r.get("assessment") and r.get("design"):
            material = by_id[r["material"]]
            packet = build_packet(material)
            r["assessment"]["spectrum"] = spectrum_agreement(material, packet, compile_design(r["design"], packet))
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2, default=str))
    (out_dir / "results.md").write_text(render(rows, materials, random_baseline(materials)))
    return rows


def _mean(values: list):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _fmt(value, spec=".3f") -> str:
    return "–" if value is None else format(value, spec)


def render(rows: list, materials: list, baseline: dict | None = None) -> str:
    kang = {m["id"]: m["reference"].get("kang_vqe_ground_state_probability") for m in materials}
    groups = {}
    for r in rows:
        groups.setdefault((r["material"], r["model"]), []).append(r)
    lines = ["# Kang et al. (2025) design benchmark: LLM-designed Hamiltonians vs the paper", "",
             "Inputs as in the paper (net graph, edge distances, linker lengths, composition); A2 composes the "
             "Hamiltonian, code computes coefficients and solves. `kang_control` = the paper's weighting through the "
             "same compiler. MAE: designed spectrum vs the paper's Hamiltonian values over every fixed-composition state, "
             "after the best non-negative affine map (paper units); NMAE = MAE / paper spectrum range; rho = Spearman.",
             "QAOA here samples only fixed-composition states; the paper's VQE samples the full register, so the two "
             "probabilities are not directly comparable.", "",
             "## Per model (materials weighted equally)", "",
             "| model | runs | valid | reproduced | mean NMAE | mean rho | MAE per material | extra requests | cost $ |",
             "|---|---|---|---|---|---|---|---|---|"]
    for model in sorted({r["model"] for r in rows}):
        rs = [r for r in rows if r["model"] == model]
        per_mat = {mat: [r["assessment"]["spectrum"] for r in g if "assessment" in r]
                   for (mat, m), g in groups.items() if m == model}
        nmae = _mean([_mean([x["nmae"] for x in v]) for v in per_mat.values() if v])
        rho = _mean([_mean([x["spearman"] for x in v]) for v in per_mat.values() if v])
        maes = ", ".join(f"{mat} {_fmt(_mean([x['mae'] for x in v]), '.1f')}" for mat, v in sorted(per_mat.items()) if v)
        reproduced = sum((r.get("assessment") or {}).get("outcome") == "REPRODUCED" for r in rs)
        extra = sum(max(r.get("llm_requests", 0) - 1, 0) for r in rs)
        lines.append(f"| {model} | {len(rs)} | {sum('assessment' in r for r in rs)} | {reproduced}/{len(rs)} | "
                     f"{_fmt(nmae)} | {_fmt(rho)} | {maes} | {extra} | {sum(r['cost_usd'] for r in rs):.4f} |")
    if baseline:
        maes = ", ".join(f"{mat} {_fmt(b['mae'], '.1f')}" for mat, b in sorted(baseline.items()))
        lines.append(f"| random catalogue design (baseline, {next(iter(baseline.values()))['n']} per material) | – | – | "
                     f"{_fmt(_mean([b['reproduced_rate'] for b in baseline.values()]), '.0%')} | "
                     f"{_fmt(_mean([b['nmae'] for b in baseline.values()]))} | "
                     f"{_fmt(_mean([b['spearman'] for b in baseline.values()]))} | {maes} | – | – |")
    lines += ["", "## Per material and model", "",
              "| material | model | runs | reproduced | MAE | NMAE | rho | median QAOA p(ref) | uniform p(ref) | "
              "paper VQE p(ground) | cost $ |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (mat, model), rs in sorted(groups.items()):
        assessed = [r["assessment"] for r in rs if "assessment" in r]
        spec = [a["spectrum"] for a in assessed]
        p = sorted(a["qaoa"]["median_p_reference"] for a in assessed)
        lines.append(f"| {mat} | {model} | {len(rs)} | {sum(a['outcome'] == 'REPRODUCED' for a in assessed)}/{len(rs)} | "
                     f"{_fmt(_mean([x['mae'] for x in spec]), '.2f')} | {_fmt(_mean([x['nmae'] for x in spec]))} | "
                     f"{_fmt(_mean([x['spearman'] for x in spec]))} | {_fmt(p[len(p) // 2] if p else None)} | "
                     f"{_fmt(assessed[0]['qaoa']['uniform_baseline_p_reference'] if assessed else None)} | {kang.get(mat)} | "
                     f"{sum(r['cost_usd'] for r in rs):.4f} |")
    lines += ["", "## Designs chosen", ""]
    for r in rows:
        if r.get("design") and r["model"] != "kang_control":
            terms = "; ".join(f"{t['term_id']}[{'+'.join(t['edge_kinds'])}, {t['weighting']}"
                              + (f" a={t['alpha']}" if t["weighting"] == "distance_power" else "")
                              + f", w={t['relative_weight']}" + ("" if t["preference"] == "not_applicable" else f", {t['preference']}")
                              + "]" for t in r["design"].get("terms", []))
            sp = (r.get("assessment") or {}).get("spectrum") or {}
            lines.append(f"- {r['material']} / {r['model']} r{r['repeat']}: {r['validation']} — "
                         f"{(r.get('assessment') or {}).get('outcome', '-')}, NMAE {_fmt(sp.get('nmae'))}, "
                         f"rho {_fmt(sp.get('spearman'))} — {terms}")
    return "\n".join(lines) + "\n"
