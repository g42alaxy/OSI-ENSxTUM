"""Test harness. Tests may read harness/ and reference_outputs/; the pipeline under test may not."""
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "mof_no_agent_fixture"
PROJECT = FIXTURE / "project.json"
CASES = ("core_pairwise", "higher_order")
MODES = ("none", "constant", "topology_dependent")

from mofpipe.intake import load_project  # noqa: E402
from mofpipe.orchestrator import run_pipeline  # noqa: E402


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def harness_json(rel):
    return json.loads((FIXTURE / rel).read_text())


def fixture_core():
    """The fixture's own reference utilities, imported harness-side only."""
    tools = str(FIXTURE / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import fixture_core as fc
    return fc


def copy_fixture(dst: Path, with_harness: bool = False) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURE / "project.json", dst / "project.json")
    for sub in ("data", "tools") + (("harness", "reference_outputs") if with_harness else ()):
        shutil.copytree(FIXTURE / sub, dst / sub, ignore=shutil.ignore_patterns("__pycache__"))
    return dst


@pytest.fixture(scope="session")
def runs(tmp_path_factory):
    out = {}
    for case in CASES:
        for mode in MODES:
            spec = load_project(PROJECT, case=case, correction_mode=mode)
            run_dir = tmp_path_factory.mktemp(f"{case}_{mode}")
            out[(case, mode)] = (spec, run_dir, run_pipeline(spec, run_dir))
    return out
