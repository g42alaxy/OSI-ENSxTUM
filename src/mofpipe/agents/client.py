"""One structured request per role through OpenRouter Chat Completions, with a global request budget and cache.

``OpenRouterClient.complete`` is the guide's ``call_agent`` transport (section 8): strict ``json_schema`` response
format, ``provider.require_parameters = true``, no streaming, no tool calls. ``call_role`` adds what the guide
requires around it: every attempt (including failures and retries) counts against the run's request ceiling,
transient faults and invalid output may use only the single optional extra request, authentication and
unsupported-parameter errors are never retried, and outputs are cached by input hash, prompt/contract version
and model configuration. Provenance (response/model/provider IDs, usage, hashes) is added by code, not the model.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from ..contract import PipelineError, canonical_json, content_hash
from .contracts import CONTRACT_VERSION, PROMPT_VERSION, PROMPTS, SCHEMAS, transport_schema

RETRY_TRANSIENT = {"API_TRANSIENT"}
REPAIRABLE = {"INVALID_JSON", "SCHEMA_VIOLATION", "OUTPUT_TRUNCATED"}


class AgentCallError(PipelineError):
    def __init__(self, code: str, detail: str = "", meta: dict | None = None):
        super().__init__(code, detail)
        self.meta = meta or {}  # provider metadata of a failed response (usage is billed even when truncated)


def check_schema(role: str, output) -> None:
    from jsonschema import Draft202012Validator
    errors = sorted(Draft202012Validator(SCHEMAS[role]).iter_errors(output), key=lambda e: list(e.absolute_path))
    if errors:
        raise AgentCallError("SCHEMA_VIOLATION", "; ".join(
            f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}" for e in errors[:8]))


class OpenRouterClient:
    def __init__(self, controls):
        self.cfg = controls["openrouter"]

    def model_config(self) -> dict:
        """Everything except secrets that changes what the model returns (part of the cache key)."""
        c = self.cfg
        return {"model": c["model"], "temperature": c["temperature"], "reasoning": c["reasoning"],
                "provider": {**(c["provider"] or {}), "require_parameters": True}, "max_tokens": c["max_tokens"]}

    def complete(self, role: str, messages: list, max_tokens: int) -> tuple[str, dict]:
        import requests

        c = self.cfg
        body = {"model": c["model"], "messages": messages, "max_tokens": max_tokens, "stream": False,
                "temperature": c["temperature"],
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": f"{role}_output", "strict": True, "schema": transport_schema(SCHEMAS[role])}},
                "provider": {**(c["provider"] or {}), "require_parameters": True}}
        if c["reasoning"] is not None:
            body["reasoning"] = c["reasoning"]
        headers = {"Authorization": f"Bearer {c['api_key']}", "Content-Type": "application/json"}
        if c["app_url"]:
            headers["HTTP-Referer"] = c["app_url"]
        if c["app_title"]:
            headers["X-Title"] = c["app_title"]
        # The body is read incrementally against a wall-clock deadline: OpenRouter keeps idle requests alive with
        # whitespace, so a per-read timeout alone never fires on a stalled generation.
        deadline = time.monotonic() + c.get("deadline_s", 600)
        try:
            response = requests.post(c["base_url"].rstrip("/") + "/chat/completions", headers=headers, json=body,
                                     timeout=tuple(c["timeout_s"]), stream=True)
            raw = bytearray()
            for chunk in response.iter_content(8192):
                raw += chunk
                if time.monotonic() > deadline:
                    response.close()
                    raise AgentCallError("API_TRANSIENT", f"deadline {c.get('deadline_s', 600)} s exceeded")
        except (requests.Timeout, requests.ConnectionError) as exc:
            raise AgentCallError("API_TRANSIENT", type(exc).__name__) from exc
        response._content = bytes(raw)  # json() / text below read the buffered body
        status = response.status_code
        if status in (401, 403):
            raise AgentCallError("API_AUTH_ERROR", str(status))
        if status == 402:
            raise AgentCallError("API_PAYMENT_REQUIRED", "insufficient OpenRouter credits")
        if status in (408, 429) or status >= 500:
            raise AgentCallError("API_TRANSIENT", str(status))
        if status == 404:
            raise AgentCallError("API_NO_COMPATIBLE_ENDPOINT", "no provider serves this model with the requested parameters")
        if status != 200:
            raise AgentCallError("API_BAD_REQUEST", f"{status}: {response.text[:300]}")
        data = response.json()
        if data.get("error") or not data.get("choices"):
            raise AgentCallError("API_RESPONSE_ERROR", str(data.get("error"))[:300])
        choice = data["choices"][0]
        message = choice.get("message") or {}
        meta = {"response_id": data.get("id"), "model": data.get("model"), "provider": data.get("provider"),
                "usage": data.get("usage"), "finish_reason": choice.get("finish_reason")}
        if message.get("refusal"):
            raise AgentCallError("MODEL_REFUSAL", str(message["refusal"])[:300], meta)
        if message.get("tool_calls"):
            raise AgentCallError("UNEXPECTED_TOOL_CALL", "the contract has no tools", meta)
        if choice.get("finish_reason") == "length":
            raise AgentCallError("OUTPUT_TRUNCATED", f"max_tokens={max_tokens}; truncated JSON is a failed response", meta)
        if choice.get("finish_reason") != "stop":
            raise AgentCallError("UNEXPECTED_FINISH_REASON", str(choice.get("finish_reason")), meta)
        return message.get("content") or "", meta


class ReplayClient:
    """Offline client returning scripted outputs per role (tests, demos, replay of recorded runs)."""

    def __init__(self, script: dict, model: str = "replay"):
        self.script = {role: list(outputs) for role, outputs in script.items()}
        self.model, self.requests = model, []

    def model_config(self) -> dict:
        return {"model": self.model}

    def complete(self, role: str, messages: list, max_tokens: int) -> tuple[str, dict]:
        self.requests.append({"role": role, "messages": messages})
        if not self.script.get(role):
            raise AgentCallError("REPLAY_EXHAUSTED", role)
        item = self.script[role].pop(0)
        if isinstance(item, Exception):
            raise item
        if callable(item):  # scripted answer computed from the packet (tests)
            item = item(json.loads(messages[1]["content"]))
        text = item if isinstance(item, str) else json.dumps(item)
        return text, {"response_id": f"replay-{len(self.requests)}", "model": self.model, "provider": "replay",
                      "usage": None, "finish_reason": "stop"}


class Ledger:
    """Global LLM request counter for one run: one normal request per role call (``initial``, and ``round_k`` for
    A67/A4 in the k-th optimisation round) plus the single optional extra request (repair, retry or expansion)."""

    def __init__(self, path: Path, allow_extra: bool):
        self.path, self.allow_extra = Path(path), allow_extra
        self.normal_used: set = set()
        self.extra_used, self.entries = False, []

    @property
    def requests(self) -> int:
        return sum(1 for e in self.entries if not e.get("cache_hit"))

    @property
    def extra_available(self) -> bool:
        return self.allow_extra and not self.extra_used

    def reserve(self, role: str, purpose: str) -> str:
        slot = (role, purpose)
        normal = purpose == "initial" or (purpose.startswith("round_") and role in ("A67", "A4"))
        if normal and slot not in self.normal_used:
            self.normal_used.add(slot)
            return "normal"
        if self.extra_available:
            self.extra_used = True
            return "extra"
        raise AgentCallError("LLM_BUDGET_EXHAUSTED", f"{role}/{purpose}: request ceiling reached")

    def record(self, entry: dict) -> None:
        self.entries.append(entry)
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"requests": self.requests, "extra_used": self.extra_used,
                                         "entries": self.entries}, indent=2) + "\n")


class Cache:
    def __init__(self, directory: Path | None):
        self.dir = Path(directory) if directory else None

    def get(self, key: str):
        path = self.dir / f"{key}.json" if self.dir else None
        return json.loads(path.read_text()) if path and path.exists() else None

    def put(self, key: str, output: dict, meta: dict) -> None:
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)
            (self.dir / f"{key}.json").write_text(json.dumps({"output": output, "meta": meta}, indent=2) + "\n")


def cache_key(role: str, packet: dict, client) -> str:
    model = client.model_config()
    if isinstance(model.get("max_tokens"), dict):  # only this role's limit changes this role's output
        model = {**model, "max_tokens": model["max_tokens"].get(role)}
    return content_hash({"role": role, "packet": packet, "prompt": PROMPTS[role], "prompt_version": PROMPT_VERSION,
                         "contract_version": CONTRACT_VERSION, "schema": SCHEMAS[role], "model": model})


def call_role(role: str, packet: dict, *, client, ledger: Ledger, cache: Cache, max_tokens: int,
              purpose: str = "initial", repair: list | None = None) -> tuple[dict, dict]:
    """Schema-valid output of one role for one packet. ``repair`` = prior [assistant, user] turns to correct.

    Returns (output, meta); meta["cache_key"] lets the caller cache the output once it also passes semantic
    validation. Raises AgentCallError when the budget is exhausted or the failure is not retryable.
    """
    user = canonical_json(packet)
    key = cache_key(role, packet, client)  # a repaired output is cached under its packet's key too
    hit = cache.get(key) if repair is None else None
    base = {"role": role, "purpose": purpose, "prompt_version": PROMPT_VERSION, "contract_version": CONTRACT_VERSION,
            "input_hash": content_hash(packet), "model_config": client.model_config()}
    if hit:
        ledger.record({**base, "cache_hit": True, "output_hash": content_hash(hit["output"]), "upstream": hit["meta"]})
        return hit["output"], {**hit["meta"], "cache_hit": True, "cache_key": key}
    messages = [{"role": "system", "content": PROMPTS[role]}, {"role": "user", "content": user}] + (repair or [])
    while True:
        kind = ledger.reserve(role, purpose)
        text, meta = None, {}
        try:
            text, meta = client.complete(role, messages, max_tokens)
            try:
                output = json.loads(text)
            except json.JSONDecodeError as exc:
                raise AgentCallError("INVALID_JSON", str(exc)) from exc
            check_schema(role, output)
        except AgentCallError as exc:
            meta = {**getattr(exc, "meta", {}), **meta}
            ledger.record({**base, "attempt": kind, "cache_hit": False, **meta, "outcome": "ERROR", "error": exc.as_dict()})
            retry = exc.code in RETRY_TRANSIENT or exc.code in REPAIRABLE
            if not (retry and ledger.extra_available):
                raise
            if exc.code in REPAIRABLE and text is not None:
                messages = messages + [{"role": "assistant", "content": text},
                                       {"role": "user", "content": f"The output was rejected ({exc.code}: {exc.detail}). "
                                                                   "Return one corrected, complete JSON object."}]
            purpose = "retry"
            continue
        meta = {**meta, "attempt": kind, "cache_hit": False, "cache_key": key, "output_hash": content_hash(output)}
        ledger.record({**base, **{k: v for k, v in meta.items() if k != "cache_key"}, "outcome": "SCHEMA_VALID"})
        return output, meta
