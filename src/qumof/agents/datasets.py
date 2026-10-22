"""Reference datasets for A67: local, pinned copies of public MOF datasets, matched and compared by code.

Formats:
- ``materials_toolkits``: an ``*_index.parquet`` plus an HDF5 file of flat arrays, as published on Hugging Face by
  materials-toolkits for CoRE MOF 2024 (experimental structures with Zeo++ pore descriptors and topology labels) and
  QMOF v18 (PBE-D3(BJ)-relaxed structures; topology only inside ``mofid``).
- ``table``: a CSV / JSONL / Parquet table with a declared column mapping.

Matching is deterministic. ``pinned`` rows (record IDs or names such as a CSD refcode, declared in controls.yaml)
are the specific known material; other rows are family-level evidence scored by topology label (+2), identical
element set (+2) or shared metal (+1), kept at ``min_score`` or above. Metrics compare only intensive, cell-independent quantities with the same unit
(density, largest cavity and pore-limiting diameters). Dataset energies come from other protocols and are never
compared with pipeline energies. Files are verified against a declared sha256 when one is given.
"""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

from ..framework import METALS
from ..intake import FIXTURE_PROJECT_TYPE
from ..store import RunStore

# Dataset property name -> (pipeline quantity, unit). Only intensive, cell-independent quantities.
QUANTITIES = {"density": ("density_g_cm3", "g/cm3"), "lcd": ("lcd_A", "A"), "pld": ("pld_A", "A")}
MOFID_TOPOLOGY = re.compile(r"MOFid-v1\.([A-Za-z0-9\-]+)\.")
AMU_PER_A3_TO_G_CM3 = 1.66053906660


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _text(value):
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else str(value)


def _topology(row: dict, columns: dict):
    if columns.get("topology") and _text(row.get(columns["topology"])):
        label = _text(row[columns["topology"]]).lower()
        return None if label in ("unstable", "unnamed", "unknown", "error", "nan", "none", "") else label
    match = MOFID_TOPOLOGY.search(_text(row.get(columns.get("mofid", ""))) or "")
    return match.group(1).lower() if match and match.group(1).upper() not in ("ERROR", "UNKNOWN", "NA") else None


def _elements_from_formula(formula) -> set:
    return set(re.findall(r"[A-Z][a-z]?", formula or ""))


class ReferenceDataset:
    def __init__(self, cfg: dict):
        self.cfg, self.id = cfg, cfg["id"]
        self.columns = cfg.get("columns") or {}

    def files(self) -> list:
        return [Path(self.cfg[k]).expanduser() for k in ("index", "hdf5", "path") if self.cfg.get(k)]

    def verify(self) -> list:
        gaps = [{"code": "DATASET_FILE_MISSING", "detail": f"{self.id}: {p}"} for p in self.files() if not p.exists()]
        for name, expected in (self.cfg.get("sha256") or {}).items():
            path = Path(self.cfg[name]).expanduser()
            if path.exists() and _sha256(path) != expected:
                gaps.append({"code": "DATASET_CHECKSUM_MISMATCH", "detail": f"{self.id}: {name}"})
        return gaps

    def rows(self) -> list:
        """[{row_id, name, doi, topology, metals, elements, values}] with values keyed by pipeline quantity."""
        import pandas as pd

        cols = self.columns
        if self.cfg["format"] == "materials_toolkits":
            import h5py
            index = pd.read_parquet(Path(self.cfg["index"]).expanduser())
            with h5py.File(Path(self.cfg["hdf5"]).expanduser()) as h:
                ptr, num, z = h["ptr"][:], h["num_atoms"][:], h["z"][:]
                props = {q: h[f"properties/{name}"][:] for name, (q, _) in QUANTITIES.items() if f"properties/{name}" in h}
            from ase.data import chemical_symbols
            elements = [{chemical_symbols[int(a)] for a in set(z[p:p + n].tolist())} for p, n in zip(ptr, num)]
        else:
            path = Path(self.cfg["path"]).expanduser()
            reader = {".csv": pd.read_csv, ".parquet": pd.read_parquet, ".jsonl": lambda p: pd.read_json(p, lines=True)}
            index = reader[path.suffix.lower()](path)
            props = {q: index[cols[q]].to_numpy(dtype=float) for q in ("density_g_cm3", "lcd_A", "pld_A") if cols.get(q)}
            elements = [_elements_from_formula(_text(f)) for f in index[cols["formula"]]] if cols.get("formula") else None
        out = []
        for i, row in enumerate(index.to_dict("records")):
            els = elements[i] if elements else _elements_from_formula(_text(row.get(cols.get("formula", ""))))
            metals = set(re.findall(r"[A-Z][a-z]?", _text(row.get(cols.get("metals", ""))) or "")) or (els & METALS)
            out.append({"row_id": _text(row.get(cols.get("record_id", ""))) or str(i), "name": _text(row.get(cols.get("name", ""))),
                        "doi": _text(row.get(cols.get("doi", ""))), "topology": _topology(row, cols), "metals": metals,
                        "elements": els, "values": {q: float(v[i]) for q, v in props.items() if not math.isnan(float(v[i]))}})
        return out


def project_descriptors(spec) -> dict:
    nets = {t.crystallographic_net_name.lower() for t in spec.topologies.values() if t.crystallographic_net_name}
    elements = set(((spec.physical or {}).get("chemistry") or {}).get("target_elemental_inventory") or {})
    return {"nets": nets, "elements": elements, "metals": elements & METALS}


def match_score(row: dict, project: dict) -> tuple[int, list]:
    score, on = 0, []
    if row["topology"] and row["topology"] in project["nets"]:
        score, on = score + 2, on + ["topology"]
    if project["elements"] and row["elements"] == project["elements"]:
        score, on = score + 2, on + ["element_set"]
    elif project["metals"] & set(row["metals"]):
        score, on = score + 1, on + ["metal"]
    return score, on


def candidate_descriptors(report: dict, pipeline_dir, candidates: list) -> dict:
    """Code-measured intensive descriptors per final candidate: density from the relaxed geometry; LCD/PLD from Zeo++."""
    from ase.data import atomic_masses, atomic_numbers

    store, index, out = RunStore(pipeline_dir), report.get("artifact_index", {}), {}
    for cand in candidates:
        values = {}
        calc_id = index.get(f"calc/{cand.get('orbit_id')}")
        geometry = store.get(calc_id).get("relaxed_geometry") if calc_id else None
        if geometry:
            a, b, c = geometry["lattice_A"]
            volume = abs(a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]))
            mass = sum(atomic_masses[atomic_numbers[e]] for e in geometry["elements"])
            values["density_g_cm3"] = round(mass / volume * AMU_PER_A3_TO_G_CM3, 4)
        val_id = index.get(f"structure_validation/{cand.get('orbit_id')}")
        zeo = (store.get(val_id)["checks"].get("zeopp_pores") or {}).get("values") if val_id else None
        if zeo:
            values["lcd_A"], values["pld_A"] = zeo["largest_included_sphere_A"], zeo["largest_free_sphere_A"]
        out[cand["candidate_id"]] = values
    return out


def gather_datasets(spec, controls, report: dict, pipeline_dir, candidates: list) -> dict:
    """{"datasets", "records", "metrics", "gaps"} for the A67 packet."""
    configs = [d for d in controls["datasets"] if d.get("enabled", True)]
    if spec.project_type == FIXTURE_PROJECT_TYPE:
        return {"datasets": [], "records": [], "metrics": [], "gaps": [
            {"code": "DATASET_NOT_APPLICABLE", "detail": "synthetic fixture graphs have no counterpart in MOF datasets"}]}
    if not configs:
        return {"datasets": [], "records": [], "metrics": [], "gaps": [{"code": "NO_DATASETS_CONFIGURED", "detail": "controls"}]}
    project, gaps, summaries, scored = project_descriptors(spec), [], [], []
    if spec.provenance == "SYNTHETIC":
        gaps.append({"code": "SYNTHETIC_PROVENANCE", "detail": "toy frameworks; dataset matches are plumbing checks only"})
    for cfg in configs:
        ds = ReferenceDataset(cfg)
        problems = ds.verify()
        if problems:
            gaps += problems
            continue
        rows, pinned = ds.rows(), set(cfg.get("pinned") or [])
        is_pinned = lambda r: r["row_id"] in pinned or (r["name"] or "").split("_")[0] in pinned
        found = {p for r in rows if is_pinned(r) for p in pinned if p in (r["row_id"], (r["name"] or "").split("_")[0])}
        gaps += [{"code": "PINNED_RECORD_NOT_FOUND", "detail": f"{ds.id}: {p}"} for p in sorted(pinned - found)]
        matched = [((10, ["pinned_reference"]) if is_pinned(r) else match_score(r, project), r) for r in rows]
        matched = [(s, on, r) for (s, on), r in matched if s >= cfg.get("min_score", 3)]
        summaries.append({"dataset_id": ds.id, "version": cfg.get("version"), "kind": cfg.get("kind"),
                          "license": cfg.get("license"), "n_rows": len(rows), "n_matched": len(matched),
                          "checksum_verified": bool(cfg.get("sha256"))})
        best = sorted(matched, key=lambda x: (-x[0], x[2]["row_id"]))[: cfg.get("max_records", 5)]
        scored += [(cfg, s, on, r) for s, on, r in best]
        if not matched:
            gaps.append({"code": "NO_MATCHING_DATASET_RECORD", "detail": f"{ds.id}: nets {sorted(project['nets'])}, "
                                                                         f"elements {sorted(project['elements'])}"})
    records = [{"record_id": f"D{n:02d}", "dataset_id": cfg["id"], "dataset_kind": cfg.get("kind"), "row_id": r["row_id"],
                "name": r["name"], "doi": r["doi"], "match_score": s, "matched_on": on,
                "match_level": "specific_material" if on == ["pinned_reference"] else "same_family",
                "topology": r["topology"],
                "metals": sorted(r["metals"]), "elements": sorted(r["elements"]), "values": r["values"]}
               for n, (cfg, s, on, r) in enumerate(scored, 1)]
    descriptors = candidate_descriptors(report, pipeline_dir, candidates)
    metrics = []
    for cid, cvals in descriptors.items():
        for rec in records:
            for quantity, unit in QUANTITIES.values():
                if quantity in cvals and quantity in rec["values"]:
                    c, d = cvals[quantity], rec["values"][quantity]
                    metrics.append({"metric_id": f"M{len(metrics) + 1:02d}", "candidate_id": cid,
                                    "record_id": rec["record_id"], "quantity": quantity, "unit": unit,
                                    "candidate_value": c, "dataset_value": round(d, 4), "difference": round(c - d, 4),
                                    "relative_difference": round((c - d) / d, 4) if d else None,
                                    "basis": f"pipeline relaxed geometry vs {rec['dataset_id']} ({rec['dataset_kind']})"})
    if records and not metrics:
        gaps.append({"code": "NO_COMPARABLE_QUANTITY", "detail": "no shared intensive quantity between candidates and records"})
    return {"datasets": summaries, "records": records, "metrics": metrics, "gaps": gaps}


def fetch(cfg: dict) -> list:
    """Download a dataset declared with ``hf: {repo, revision, files}`` next to its configured index/hdf5/path.

    Pinned to a revision so the local copy is reproducible; record the sha256 of the files in controls.yaml.
    """
    from huggingface_hub import hf_hub_download

    hf = cfg["hf"]
    target = Path(cfg.get("index") or cfg.get("path")).expanduser().parent
    out = []
    for name in hf["files"]:
        path = Path(hf_hub_download(hf["repo"], name, repo_type="dataset", revision=hf["revision"], local_dir=target))
        out.append({"file": str(path), "sha256": _sha256(path)})
    return out
