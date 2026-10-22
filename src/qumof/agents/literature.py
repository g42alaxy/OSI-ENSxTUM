"""Literature preparation for A67 without extra LLM turns (guide section 7).

1. Deterministic query templates from the project (material, linker, topology, method and benchmark terms).
2. Valency MCP (https://docs.valency.io/, streamable HTTP, bearer token): resolve declared references (arXiv/PMC IDs,
   titles with DOI checks) and run the template searches. A shorthand alias is accepted only on a DOI match or a
   near-identical title; otherwise it stays an UNRESOLVED_SOURCE_ALIAS gap.
3. Local papers (.pdf/.txt/.md, optional <name>.json metadata sidecar): BM25 passage retrieval over full text.
4. One bounded evidence packet with source/passage IDs, access level and URLs. Retrieved text is data: it is
   whitespace-normalized, truncated and stripped of citation markers before it reaches a prompt.
"""
from __future__ import annotations

import asyncio
import json
import math
import re
from collections import Counter
from pathlib import Path

from ..intake import FIXTURE_PROJECT_TYPE

TITLE_MATCH = 0.85
CHUNK_CHARS = 1500
STOP = set("a an and are as at be by for from in into is it of on or that the this to with we our via using".split())


def _clean(text, limit: int) -> str:
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", str(text or ""))
    text = re.sub(r"\s+", " ", text.replace("[[", "[").replace("]]", "]")).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _tokens(text: str) -> list:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP and len(t) > 1]


def title_similarity(a: str, b: str) -> float:
    x, y = set(_tokens(a)), set(_tokens(b))
    return len(x & y) / len(x | y) if x | y else 0.0


def query_templates(spec, extra=()) -> list:
    queries = ["multivariate metal-organic framework linker arrangement ground-state configuration prediction",
               "QUBO Hamiltonian optimization of mixed-linker metal-organic framework configurations"]
    if spec.project_type != FIXTURE_PROJECT_TYPE:
        blocks = [b.get("formula") or b.get("id") for b in (spec.physical.get("blocks") or {}).values()
                  if b.get("role") == "linker"]
        nets = sorted({t.crystallographic_net_name for t in spec.topologies.values() if t.crystallographic_net_name})
        if blocks:
            queries.append(f"mixed-linker metal-organic framework {' '.join(map(str, blocks))} linker distribution")
        for net in nets:
            queries.append(f"metal-organic framework {net} topology ligand functionalization topology competition")
        if (spec.physical.get("protocols") or {}).get("embedding"):
            queries.append("quantum embedding periodic metal-organic framework active space correction")
    return queries + [q for q in extra if q]


def references_from(spec, controls_refs) -> list:
    """Declared references to resolve: controls.yaml literature.references plus the project's literature manifest."""
    refs = [dict(r) for r in controls_refs or []]
    for e in (spec.literature or {}).get("entries", []):
        if e.get("title") and e.get("doi"):  # papers only; software repositories are not in a scholarly index
            refs.append({"alias": e.get("id"), "title": e["title"], "doi": e.get("doi")})
    return refs


def _paper_source(paper: dict, origin: str, alias=None) -> dict:
    year = ((paper.get("dates") or {}).get("first_available") or {}).get("value") or paper.get("datestamp") or ""
    authors = [a.get("name") if isinstance(a, dict) else str(a) for a in (paper.get("authors") or [])][:12]
    return {"title": paper.get("title"), "authors": authors, "year": str(year)[:4] or None,
            "doi": paper.get("doi") or None, "url": paper.get("url"), "external_id": paper.get("id"),
            "origin": origin, "alias": alias, "access_level": "abstract_only" if paper.get("abstract") else "metadata_only",
            "_abstract": paper.get("abstract") or ""}


class ValencyMCP:
    """Minimal MCP client for Valency: one session per batch of (tool, arguments) calls."""

    def __init__(self, url: str, token: str, timeout_s: float = 60):
        self.url, self.token, self.timeout = url, token, timeout_s

    def run(self, calls: list) -> list:
        try:
            return asyncio.run(self._run(calls))
        except BaseExceptionGroup as group:  # anyio task groups nest the real cause (e.g. "Invalid or revoked token")
            inner = group
            while isinstance(inner, BaseExceptionGroup):
                inner = inner.exceptions[0]
            raise RuntimeError(f"Valency MCP: {inner}") from group

    async def _run(self, calls: list) -> list:
        from mcp import ClientSession

        headers = {"Authorization": f"Bearer {self.token}"}
        try:  # mcp >= 2: headers go on the HTTP client
            from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client
            http = create_mcp_http_client(headers=headers)
            transport = streamable_http_client(self.url, http_client=http)
        except ImportError:  # mcp 1.x
            from mcp.client.streamable_http import streamablehttp_client
            http, transport = None, streamablehttp_client(self.url, headers=headers)
        out = []
        try:
            async with transport as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    await session.initialize()
                    for name, arguments in calls:
                        try:
                            result = await session.call_tool(name, arguments, read_timeout_seconds=self.timeout)
                            out.append(_parse_tool_result(result))
                        except Exception as exc:  # one failed call is a retrieval gap, not a run failure
                            out.append(exc)
        finally:
            if http is not None:
                await http.aclose()
        return out


def _parse_tool_result(result):
    if getattr(result, "is_error", getattr(result, "isError", False)):
        raise RuntimeError("tool error: " + " ".join(getattr(c, "text", "") for c in result.content)[:300])
    structured = getattr(result, "structured_content", getattr(result, "structuredContent", None))
    if isinstance(structured, dict):
        return structured
    text = "".join(getattr(c, "text", "") for c in result.content)
    return json.loads(text)


class LocalPapers:
    """Full-text passages from a local folder, ranked by BM25 against the query templates."""

    def __init__(self, directory):
        self.dir, self.docs, self.gaps = Path(directory).expanduser(), [], []
        for path in sorted(self.dir.rglob("*")):
            if path.suffix.lower() not in (".pdf", ".txt", ".md") or path.name.startswith("."):
                continue
            try:
                text = self._read(path)
            except Exception as exc:
                self.gaps.append({"code": "LOCAL_PAPER_UNREADABLE", "detail": f"{path.name}: {type(exc).__name__}"})
                continue
            meta_path = path.with_suffix(".json")
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            if not text.strip():
                self.gaps.append({"code": "LOCAL_PAPER_NO_TEXT", "detail": f"{path.name}: scanned PDF needs OCR"})
                continue
            self.docs.append({"path": path, "meta": meta, "chunks": self._chunks(text)})

    @staticmethod
    def _read(path: Path) -> str:
        if path.suffix.lower() == ".pdf":
            from pypdf import PdfReader
            return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
        return path.read_text(errors="replace")

    @staticmethod
    def _chunks(text: str) -> list:
        paragraphs, chunks, current = re.split(r"\n\s*\n", text), [], ""
        for p in paragraphs:
            p = re.sub(r"\s+", " ", p).strip()
            if current and len(current) + len(p) > CHUNK_CHARS:
                chunks.append(current)
                current = ""
            current = (current + " " + p).strip()
            while len(current) > 2 * CHUNK_CHARS:
                chunks.append(current[:CHUNK_CHARS])
                current = current[CHUNK_CHARS:]
        return chunks + ([current] if current else [])

    def search(self, queries: list, per_doc: int) -> list:
        """[(doc, [(score, chunk_index), ...])] best chunks per document, documents by best score."""
        all_chunks = [(d, i, _tokens(c)) for d in self.docs for i, c in enumerate(d["chunks"])]
        if not all_chunks:
            return []
        n, avg = len(all_chunks), sum(len(t) for _, _, t in all_chunks) / len(all_chunks)
        df = Counter(term for _, _, toks in all_chunks for term in set(toks))
        terms = Counter(t for q in queries for t in _tokens(q))
        scored = {}
        for doc, i, toks in all_chunks:
            tf = Counter(toks)
            score = sum(qn * math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) * tf[t] * 2.5
                        / (tf[t] + 1.5 * (0.25 + 0.75 * len(toks) / max(avg, 1))) for t, qn in terms.items() if tf[t])
            scored.setdefault(id(doc), (doc, []))[1].append((score, i))
        ranked = [(doc, sorted(s, reverse=True)[:per_doc]) for doc, s in scored.values()]
        return sorted(ranked, key=lambda x: -x[1][0][0])


def _resolve_calls(refs: list, rpq: int) -> list:
    calls = []
    for ref in refs:
        if ref.get("paper_id") or ref.get("arxiv_id"):
            calls.append(("get_paper_by_id", {"paper_id": ref.get("paper_id") or ref["arxiv_id"], "enrich_citations": False}))
        elif ref.get("title"):
            calls.append(("search_by_title", {"query": ref["title"][:1000], "limit": rpq, "enrich_citations": False}))
    return calls


def _accept(ref: dict, papers: list):
    for p in papers:
        doi_ok = ref.get("doi") and (p.get("doi") or "").lower() == ref["doi"].lower()
        if doi_ok or (ref.get("title") and title_similarity(ref["title"], p.get("title") or "") >= TITLE_MATCH):
            return p
    return None


def gather_evidence(spec, controls, valency=None, local=None) -> dict:
    """Bounded evidence packet: {"sources", "passages", "gaps", "retrieval_requests", "queries"}."""
    lit = controls["literature"]
    rpq, limit = lit["results_per_query"], lit["max_passage_chars"]
    queries = query_templates(spec, lit["extra_queries"])
    refs = references_from(spec, lit.get("references"))
    sources, gaps, n_requests = [], [], 0
    if not lit["enabled"]:
        return {"sources": [], "passages": [], "gaps": [{"code": "LITERATURE_DISABLED", "detail": "controls"}],
                "retrieval_requests": 0, "queries": []}
    if valency is None:
        gaps.append({"code": "VALENCY_NOT_CONFIGURED", "detail": "mcp.valency.enabled is false"})
    else:
        resolvable = [r for r in refs if r.get("paper_id") or r.get("arxiv_id") or r.get("title")]
        calls = _resolve_calls(resolvable, rpq) + [("semantic_search_papers", {"query": q, "limit": rpq,
                                                                               "enrich_citations": False}) for q in queries]
        results = valency.run(calls)
        n_requests += len(calls)
        fallback = []
        for ref, res in zip(resolvable, results[:len(resolvable)]):
            if isinstance(res, Exception):
                gaps.append({"code": "RETRIEVAL_ERROR", "detail": f"{ref.get('alias')}: {type(res).__name__}"})
                continue
            papers = [res["paper"]] if res.get("found") and res.get("paper") else res.get("papers") or []
            hit = papers[0] if (ref.get("paper_id") or ref.get("arxiv_id")) and papers else _accept(ref, papers)
            if hit:
                sources.append(_paper_source(hit, "valency:resolved_reference", ref.get("alias")))
            elif ref.get("title"):
                fallback.append(ref)
            else:
                gaps.append({"code": "UNRESOLVED_SOURCE_ALIAS", "detail": str(ref.get("alias") or ref)})
        if fallback:  # titles not found by keyword: one semantic attempt each, same acceptance rule
            again = valency.run([("semantic_search_papers", {"query": r["title"], "limit": rpq, "enrich_citations": False})
                                 for r in fallback])
            n_requests += len(fallback)
            for ref, res in zip(fallback, again):
                hit = None if isinstance(res, Exception) else _accept(ref, res.get("papers") or [])
                if hit:
                    sources.append(_paper_source(hit, "valency:resolved_reference", ref.get("alias")))
                else:
                    gaps.append({"code": "UNRESOLVED_SOURCE_ALIAS",
                                 "detail": f"{ref.get('alias')}: not found in Valency by DOI or title ({ref.get('doi')})"})
        for q, res in zip(queries, results[len(resolvable):]):
            if isinstance(res, Exception):
                gaps.append({"code": "RETRIEVAL_ERROR", "detail": f"query '{q}': {type(res).__name__}"})
                continue
            sources += [_paper_source(p, "valency:semantic_search_papers") for p in (res.get("papers") or [])]
    if local is not None:
        gaps += local.gaps
        for doc, best in local.search(queries, per_doc=3):
            meta = doc["meta"]
            sources.append({"title": meta.get("title") or doc["path"].stem, "authors": meta.get("authors") or [],
                            "year": str(meta["year"]) if meta.get("year") else None, "doi": meta.get("doi"),
                            "url": meta.get("url"), "external_id": doc["path"].name, "origin": "local_papers",
                            "alias": meta.get("alias"), "access_level": "full_text",
                            "_chunks": [(i, doc["chunks"][i]) for score, i in best if score > 0]})
    elif lit["local_papers_dir"]:
        gaps.append({"code": "LOCAL_PAPERS_UNAVAILABLE", "detail": lit["local_papers_dir"]})
    # Deduplicate by DOI and by normalized title (preprint and journal copies), resolved references first, then cap.
    seen, unique = set(), []
    for s in sorted(sources, key=lambda s: s["origin"] != "valency:resolved_reference"):
        keys = {k for k in ((s.get("doi") or "").lower(), " ".join(_tokens(s.get("title") or ""))) if k}
        if keys and not keys & seen:
            seen |= keys
            unique.append(s)
    unique = unique[: lit["max_sources"]]
    out_sources, passages = [], []
    for n, s in enumerate(unique, 1):
        sid = f"S{n:02d}"
        texts = ([("abstract", s["_abstract"])] if s.get("_abstract") else []) + [(f"p{i}", t) for i, t in s.get("_chunks", [])]
        for label, text in texts:
            passages.append({"passage_id": f"{sid}:{label}", "source_id": sid, "access_level": s["access_level"],
                             "text": _clean(text, limit)})
        out_sources.append({"source_id": sid, **{k: (_clean(v, 300) if isinstance(v, str) else v)
                                                 for k, v in s.items() if not k.startswith("_")}})
    if len(passages) > lit["max_passages"]:
        gaps.append({"code": "PASSAGES_TRUNCATED", "detail": f"{len(passages)} > {lit['max_passages']}"})
        passages = passages[: lit["max_passages"]]
    if not passages:
        gaps.append({"code": "MISSING_LITERATURE", "detail": "no passages retrieved"})
    return {"sources": out_sources, "passages": passages, "gaps": gaps, "retrieval_requests": n_requests,
            "queries": queries}
