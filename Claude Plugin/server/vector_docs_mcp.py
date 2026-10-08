r"""MCP server that exposes the Vector_Database reference-document store to Claude.

Tools:
    search_documents      semantic (+ optional keyword) search over the document store
    list_documents        the documents available to search
    read_document_pages   raw text of a page range from one completed document

The server runs over stdio. Where the project lives is set by start.bat (VECDB_HOME), from, in order:
    1. the VECDB_HOME environment variable
    2. %APPDATA%\vector-docs\project_home.txt, written once by set_project_home.bat
    3. two folders above this file (the plugin sits in "<project>/Claude Plugin/server")

stdout carries the MCP protocol, so everything the database code prints is redirected to stderr.
"""
import contextlib
import os
import sys
from pathlib import Path

HOME = Path(os.environ.get("VECDB_HOME") or Path(__file__).resolve().parents[2]).resolve()
if not (HOME / "Vector_Database_Ollama.py").is_file():
    sys.stderr.write(
        f"vector-docs: cannot find Vector_Database_Ollama.py in '{HOME}'. "
        "Run server\\set_project_home.bat from inside the Vector_Database project (or set VECDB_HOME).\n"
    )
    raise SystemExit(1)
sys.path.insert(0, str(HOME))

from mcp.server.fastmcp import FastMCP  # noqa: E402

with contextlib.redirect_stdout(sys.stderr):
    import Vector_Database_Ollama as vdb  # noqa: E402

MAX_PAGES_PER_READ = 20
MAX_TOP_K = 30

mcp = FastMCP(
    "vector-docs",
    instructions=(
        "Search an engineering reference library (electrical codes such as NFPA 70/NEC, product catalogs, "
        "industry standards, textbooks). Use search_documents for questions; cite the document name and "
        "pages it returns. Use read_document_pages to read around a hit."
    ),
)


def _documents():
    with contextlib.redirect_stdout(sys.stderr):
        return vdb.list_documents()


def _result_view(rank, r):
    meta = r.get("metadata") or {}
    start, end = meta.get("page_start"), meta.get("page_end")
    doc_path = r.get("doc_path") or ""

    tables = []
    for t in meta.get("table_data") or []:
        if isinstance(t, dict):
            tables.append(t.get("markdown") or t.get("rows"))

    return {
        "rank": rank,
        "score": round(float(r.get("score") or 0.0), 4),
        "document": Path(doc_path).stem,
        # Pages of the full PDF this section was taken from (a section may sit anywhere inside the range)
        "pages": f"{start}-{end}" if start and end else None,
        "headings": meta.get("headings") or [],
        "text": r.get("text") or "",
        "tables": tables,
        "images": [str(vdb.resolve_path(p["path"])) for p in (r.get("pictures") or []) if p.get("path")],
        "file": str(vdb.resolve_path(doc_path)) if doc_path else None,
    }


@mcp.tool()
def search_documents(
    query: str,
    keywords: str = "",
    documents: list[str] | None = None,
    top_k: int = 8,
) -> dict:
    """Search the reference-document library and return the most relevant passages.

    Args:
        query: What to look for, phrased naturally and keyword-dense (code/article numbers, equipment
            names). Put several alternative phrasings on separate lines to search them all at once.
        keywords: Optional comma-separated literal terms that should appear in the passage text
            (e.g. "250.122, grounding conductor"). Narrows and boosts the results.
        documents: Optional list of document names (from list_documents) to restrict the search to.
        top_k: How many passages to return (1-30, default 8).
    """
    queries = [q.strip() for q in query.splitlines() if q.strip()]
    if not queries:
        raise ValueError("query must not be empty")
    kw_list = [k.strip() for k in keywords.split(",") if k.strip()] or None
    top_k = max(1, min(int(top_k), MAX_TOP_K))

    doc_paths = None
    if documents:
        groups = {d["name"]: d["doc_paths"] for d in _documents()}
        unknown = [d for d in documents if d not in groups]
        if unknown:
            raise ValueError(f"Unknown document name(s): {unknown}. Call list_documents for valid names.")
        doc_paths = [p for d in documents for p in groups[d]]

    try:
        with contextlib.redirect_stdout(sys.stderr):
            if kw_list or doc_paths:
                results = vdb.search_vectors_keywords(
                    queries, top_k_per_query=top_k, top_k_overall=top_k,
                    deduplicate_per_query=True, keywords=kw_list, doc_paths=doc_paths,
                )
            else:
                results = vdb.search_vectors(
                    queries, top_k_per_query=top_k, top_k_overall=top_k, deduplicate_per_query=True,
                )
    except Exception as e:
        raise RuntimeError(
            f"Search failed ({type(e).__name__}: {e}). Searching needs the Ollama server running "
            "with the embedding model available."
        ) from e

    hits = results.get("overall_top_k", [])
    return {"count": len(hits), "results": [_result_view(i + 1, r) for i, r in enumerate(hits)]}


@mcp.tool()
def list_documents() -> list[dict]:
    """List every document in the library with its number of searchable sections."""
    return [
        {"name": d["name"], "sections": d["chunk_count"], "file": str(vdb.resolve_path(d["doc_paths"][0]))}
        for d in _documents()
    ]


@mcp.tool()
def read_document_pages(document: str, start_page: int, end_page: int | None = None) -> str:
    """Read the raw text of a page range from one completed PDF document.

    Use this to read around a search hit (pages come back with each result).

    Args:
        document: Document name exactly as returned by list_documents / search_documents.
        start_page: First page to read (1-based).
        end_page: Last page to read; defaults to start_page. At most 20 pages per call.
    """
    match = next((d for d in _documents() if d["name"] == document), None)
    if not match:
        raise ValueError(f"Unknown document '{document}'. Call list_documents for valid names.")
    pdf_path = vdb.resolve_path(match["doc_paths"][0])
    if not pdf_path or not Path(pdf_path).is_file() or not str(pdf_path).lower().endswith(".pdf"):
        raise ValueError(f"'{document}' has no readable PDF on disk.")

    import pymupdf  # PyMuPDF

    end_page = end_page or start_page
    if start_page < 1 or end_page < start_page:
        raise ValueError("Pages are 1-based and end_page must be >= start_page.")
    end_page = min(end_page, start_page + MAX_PAGES_PER_READ - 1)

    with pymupdf.open(pdf_path) as pdf:
        total = len(pdf)
        if start_page > total:
            raise ValueError(f"'{document}' has only {total} pages.")
        last = min(end_page, total)
        parts = [f"--- page {n} of {total} ---\n{pdf[n - 1].get_text().strip()}" for n in range(start_page, last + 1)]
    return "\n\n".join(parts)


if __name__ == "__main__":
    mcp.run(transport="stdio")
