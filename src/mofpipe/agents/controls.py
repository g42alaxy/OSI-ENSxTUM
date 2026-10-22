"""controls.yaml: API keys, model, budgets, cache, MCP servers and literature settings for ``mofpipe agents``.

Secrets live only here or in the environment (OPENROUTER_API_KEY, OPENROUTER_MODEL, VALENCY_API_TOKEN, which
fill empty fields). They are never written to prompts, ledgers, artifacts or logs: ``Controls.public()`` is the
redacted view that runs record.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

from ..contract import PipelineError

DEFAULTS = {
    "openrouter": {
        "api_key": "", "base_url": "https://openrouter.ai/api/v1", "model": "",
        "temperature": 0, "reasoning": None, "provider": {}, "app_url": "", "app_title": "mofpipe-agents",
        "timeout_s": [10, 300], "max_tokens": {"A1": 8000, "A2": 10000, "A67": 24000},
    },
    "budget": {"max_llm_requests": 3, "allow_extra_request": True, "allow_model_expansion": False},
    "cache": {"enabled": True, "dir": "runs/.agent_cache"},
    "orchestration": {"engine": "auto"},
    "mcp": {"valency": {"enabled": False, "url": "https://mcp.valency.io/", "api_token": "", "timeout_s": 60}},
    "literature": {"enabled": True, "local_papers_dir": "", "results_per_query": 5, "max_sources": 12,
                   "max_passages": 40, "max_passage_chars": 1200, "references": [], "extra_queries": []},
    "assessment": {"top_k_candidates": 5, "required_checks": None},
    "selection": {"allow_scope_exclusions": True},
    "datasets": [],
}
ENV = {("openrouter", "api_key"): "OPENROUTER_API_KEY", ("openrouter", "model"): "OPENROUTER_MODEL",
       ("mcp", "valency", "api_token"): "VALENCY_API_TOKEN"}
SECRET_KEYS = {"api_key", "api_token"}
ENGINES = ("auto", "langgraph", "sequential")


def _merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (extra or {}).items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def _redact(d):
    if isinstance(d, dict):
        return {k: ("<set>" if v else "<empty>") if k in SECRET_KEYS else _redact(v) for k, v in d.items()}
    return d


class Controls:
    def __init__(self, data: dict, path: Path | None = None):
        self.data, self.path = data, path

    def __getitem__(self, key):
        return self.data[key]

    @property
    def max_llm_requests(self) -> int:
        return self.data["budget"]["max_llm_requests"] + (1 if self.data["budget"]["allow_extra_request"] else 0)

    def public(self) -> dict:
        return _redact(self.data)

    def errors(self, need_llm: bool = True) -> list:
        """Configuration errors as (code, field); checked before any request is made."""
        o, b, errs = self.data["openrouter"], self.data["budget"], []
        if need_llm and not o["api_key"]:
            errs.append(("MISSING_API_KEY", "openrouter.api_key / OPENROUTER_API_KEY"))
        if need_llm and not o["model"]:
            errs.append(("MISSING_MODEL", "openrouter.model / OPENROUTER_MODEL"))
        if b["max_llm_requests"] != 3 or type(b["allow_extra_request"]) is not bool or type(b["allow_model_expansion"]) is not bool:
            errs.append(("INVALID_BUDGET", "budget: max_llm_requests is 3 (one per role); allow_extra_request and "
                                           "allow_model_expansion true/false"))
        for d in self.data["datasets"]:
            needed = ("index", "hdf5") if d.get("format") == "materials_toolkits" else ("path",)
            if d.get("format") not in ("materials_toolkits", "table") or not d.get("id") or not all(d.get(k) for k in needed):
                errs.append(("INVALID_DATASET", f"datasets[{d.get('id')}]: format materials_toolkits (index, hdf5) or table (path)"))
        if set(o["max_tokens"]) != {"A1", "A2", "A67"} or not all(type(v) is int and v > 0 for v in o["max_tokens"].values()):
            errs.append(("INVALID_MAX_TOKENS", "openrouter.max_tokens needs positive integers for A1, A2, A67"))
        if self.data["orchestration"]["engine"] not in ENGINES:
            errs.append(("INVALID_ENGINE", f"orchestration.engine must be one of {ENGINES}"))
        valency = self.data["mcp"]["valency"]
        if valency["enabled"] and not valency["api_token"]:
            errs.append(("MISSING_VALENCY_TOKEN", "mcp.valency.api_token / VALENCY_API_TOKEN"))
        return errs


def load_controls(path=None, overrides: dict | None = None) -> Controls:
    data = {}
    if path is not None:
        path = Path(path)
        if not path.exists():
            raise PipelineError("CONTROLS_NOT_FOUND", str(path))
        import yaml
        data = yaml.safe_load(path.read_text()) or {}
        if not isinstance(data, dict):
            raise PipelineError("MALFORMED_CONTROLS", str(path))
    data = _merge(_merge(DEFAULTS, data), overrides or {})
    for keys, var in ENV.items():
        node = data
        for k in keys[:-1]:
            node = node[k]
        if not node.get(keys[-1]) and os.environ.get(var):
            node[keys[-1]] = os.environ[var]
    nodes = [(data["literature"], "local_papers_dir"), (data["cache"], "dir")]
    nodes += [(d, k) for d in data["datasets"] for k in ("index", "hdf5", "path")]
    for node, key in nodes:  # relative to controls.yaml
        value = node.get(key)
        if value and path is not None and not Path(value).expanduser().is_absolute():
            node[key] = str((path.parent / value).resolve())
    return Controls(data, path)
