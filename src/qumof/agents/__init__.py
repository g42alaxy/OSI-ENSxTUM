"""Runtime agent layer: A1 (setup), A2 (Hamiltonian proposal) and A67 (final assessment).

Implements MOF_AI_Agent_Implementation_Guide-2 on top of the deterministic pipeline. Three structured requests
per run (one per role), plus at most one optional repair/expansion request; no model-driven tool loop. Literature
retrieval (Valency MCP, local papers) is ordinary software that runs before A67.

This package is imported only by ``python -m qumof agents`` and its tests: the pipeline core never imports it,
so deterministic runs stay offline (tests/test_isolation_and_resume.py audits sockets and imports).
Extra dependencies: ``pip install -e .[agents]``.
"""
