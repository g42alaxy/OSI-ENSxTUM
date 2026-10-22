"""Shared contract.

One status vocabulary, typed error codes, content hashing and record helpers
used across every module boundary.
"""
from __future__ import annotations

import hashlib
import json
from enum import Enum
from fractions import Fraction

SCHEMA_VERSION = "0.1"


class StageStatus(str, Enum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    UNSUPPORTED = "UNSUPPORTED"
    INCONCLUSIVE = "INCONCLUSIVE"
    INCOMPLETE = "INCOMPLETE"
    SKIPPED_NO_ATOMIC_GEOMETRY = "SKIPPED_NO_ATOMIC_GEOMETRY"
    DISABLED_BY_CONFIG = "DISABLED_BY_CONFIG"
    NOT_RUN = "NOT_RUN"


class JobState(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class StopReason(str, Enum):
    CONVERGED_WITHIN_SCOPE = "CONVERGED_WITHIN_SCOPE"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    MODEL_INADEQUATE = "MODEL_INADEQUATE"
    SCIENTIFIC_SPEC_MISSING = "SCIENTIFIC_SPEC_MISSING"
    INCONCLUSIVE = "INCONCLUSIVE"


class PipelineError(ValueError):
    """Error whose str() is a stable machine-readable code."""

    def __init__(self, code: str, detail: str = ""):
        super().__init__(code)
        self.code = code
        self.detail = detail

    def __str__(self) -> str:
        return self.code

    def as_dict(self) -> dict:
        return {"code": self.code, "detail": self.detail}


def exact(value) -> Fraction:
    """Exact rational from a JSON number or exact string; bool is rejected."""
    if isinstance(value, bool):
        raise PipelineError("INVALID_NUMBER", repr(value))
    if isinstance(value, Fraction):
        return value
    if isinstance(value, (int, str)):
        return Fraction(value)
    if isinstance(value, float):
        return Fraction(repr(value))
    raise PipelineError("INVALID_NUMBER", repr(value))


def _json_default(obj):
    if isinstance(obj, Fraction):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)


def content_hash(obj) -> str:
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


def make_record(record_type: str, provenance: str, **fields) -> dict:
    """Immutable artifact record; artifact_id is derived from its content."""
    body = {"record_type": record_type, "schema_version": SCHEMA_VERSION, "provenance": provenance, **fields}
    body = json.loads(canonical_json(body))  # normalise Fractions/sets to JSON types
    body["artifact_id"] = f"{record_type}:{content_hash(body)[:20]}"
    return body


def energy_fields(name: str, value: Fraction) -> dict:
    """Float for reading plus the exact rational string for audit."""
    return {f"{name}_eV": float(value), f"{name}_exact": str(value)}


def bit_string(bits) -> str:
    return "".join(str(int(b)) for b in bits)
