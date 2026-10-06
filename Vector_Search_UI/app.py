"""
Local web UI for Vector_Database_Ollama.search_vectors_keywords / search_vectors.

Run with the project's venv, e.g. from the repo root:
    .venv\\Scripts\\python.exe Vector_Search_UI\\app.py

Then open http://127.0.0.1:5151 in a browser.
"""
import json
import mimetypes
import os
import re
import sys
import traceback

from urllib.parse import quote

from flask import Flask, jsonify, redirect, render_template, request, send_file, abort

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import Vector_Database_Ollama as vdb  # noqa: E402

app = Flask(__name__)

SUGGEST_SYSTEM_PROMPT = """You are a search-query rewriter for a semantic vector database of engineering \
reference documents (electrical codes such as NFPA 70/NEC, product catalogs like Appleton and Belden, \
industry standards, and textbooks). The user will give you a plain-language question. Rewrite it into \
the query and keywords that will retrieve the most relevant passages from embedding similarity search \
plus a keyword filter.

Respond with ONLY a JSON object, no markdown fences and no other text, in exactly this shape:
{"query": "<one concise, keyword-dense phrase or sentence optimized for embedding similarity search>", \
"keywords": ["<keyword1>", "<keyword2>"]}

Guidelines:
- "query" should read naturally, emphasizing the technical terms, code/article numbers, and equipment \
names implied by the question.
- "keywords" should be 3-6 short, literal terms (article numbers, product names, code words) likely to \
appear verbatim in the source text.
- Do not answer the user's question. Only produce the search query and keywords.
"""


def _parse_suggestion(raw_text, fallback_question):
    text = (raw_text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {"query": fallback_question, "keywords": []}

    query = str(data.get("query") or "").strip() or fallback_question
    keywords = data.get("keywords") or []
    if not isinstance(keywords, list):
        keywords = []
    keywords = [str(k).strip() for k in keywords if str(k).strip()]
    return {"query": query, "keywords": keywords}


@app.route("/")
def index():
    return render_template("index.html", store_dir=vdb.STORE_DIR)


@app.route("/api/documents")
def api_documents():
    docs = vdb.list_documents()
    return jsonify([{"name": d["name"], "chunk_count": d["chunk_count"]} for d in docs])


@app.route("/api/search", methods=["POST"])
def api_search():
    payload = request.get_json(force=True) or {}

    raw_queries = payload.get("query", "")
    queries = [q.strip() for q in raw_queries.splitlines() if q.strip()]
    if not queries:
        return jsonify({"error": "Enter at least one query."}), 400

    top_k_per_query = int(payload.get("top_k_per_query", 5))
    top_k_overall = int(payload.get("top_k_overall", 20))
    dedupe = bool(payload.get("dedupe", True))
    use_keywords = bool(payload.get("use_keywords", False))
    raw_keywords = payload.get("keywords", "")
    keywords = [k.strip() for k in raw_keywords.split(",") if k.strip()] or None

    raw_doc_names = payload.get("doc_names") or []
    doc_names = [d.strip() for d in raw_doc_names if isinstance(d, str) and d.strip()]

    doc_paths = None
    if doc_names:
        doc_groups = {d["name"]: d["doc_paths"] for d in vdb.list_documents()}
        doc_paths = [p for name in doc_names for p in doc_groups.get(name, [])]
        if not doc_paths:
            return jsonify({"error": "Selected document(s) not found in the store."}), 400

    try:
        # Document filtering is implemented via the keyword-search's exact-score
        # position-filter machinery, so route through it whenever a document
        # filter is active even if the "keyword-boosted search" checkbox is off.
        if use_keywords or doc_paths:
            results = vdb.search_vectors_keywords(
                queries,
                top_k_per_query=top_k_per_query,
                top_k_overall=top_k_overall,
                deduplicate_per_query=dedupe,
                keywords=keywords,
                doc_paths=doc_paths,
            )
        else:
            results = vdb.search_vectors(
                queries,
                top_k_per_query=top_k_per_query,
                top_k_overall=top_k_overall,
                deduplicate_per_query=dedupe,
                keyword=(keywords[0] if keywords else None),
            )
    except Exception as e:  # surface Ollama/FAISS errors to the UI instead of a 500 page
        return jsonify({"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}), 500

    overall = results.get("overall_top_k", [])
    for r in overall:
        r["doc_name"] = os.path.basename(r.get("doc_path", "") or "")
        r["pictures"] = r.get("pictures") or []

    return jsonify(
        {
            "query_count": results.get("query_count", 0),
            "overall_top_k": overall,
            "per_query_counts": [len(pq) for pq in results.get("per_query", [])],
        }
    )


@app.route("/api/suggest_query", methods=["POST"])
def api_suggest_query():
    payload = request.get_json(force=True) or {}
    question = (payload.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Enter a question first."}), 400

    try:
        from LLM_Summary_Ollama import summarize_text  # deferred: touches Ollama at import time
    except Exception as e:
        return jsonify({"error": f"Could not load LLM helper: {type(e).__name__}: {e}"}), 500

    try:
        raw = summarize_text(question, max_tokens=300, setup=SUGGEST_SYSTEM_PROMPT, temperature=0.2)
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

    if not raw:
        return jsonify({"error": "The model returned no suggestion."}), 500

    return jsonify(_parse_suggestion(raw, fallback_question=question))


SUMMARIZE_REFERENCES_SYSTEM_PROMPT = """You are helping an engineer decide which search results from a \
vector database of technical reference documents (electrical codes, product catalogs, standards) actually \
matter for answering their question.

You will be given the user's question and a numbered list of retrieved passages, each with its source \
document, section headings, and relevance score.

Respond in plain text (no markdown headers, no code fences), structured like this:
1. For each passage worth using, write its number, then a one-to-two sentence explanation of why it \
matters and what it tells the reader -- name the document and section, don't just say "reference 3".
2. Skip or briefly dismiss passages that are irrelevant or redundant; don't force relevance onto weak matches.
3. End with a short "Bottom line" paragraph naming the 2-4 references most worth reading in full.

Be concise and specific.
"""


def _format_results_for_prompt(question, results, max_chars_per_result=700, max_results=15):
    lines = [f"Question: {question}", ""]
    for i, r in enumerate(results[:max_results], 1):
        doc_name = r.get("doc_name") or ""
        headings = r.get("headings") or []
        score = r.get("score")
        text = (r.get("text") or "")[:max_chars_per_result]

        lines.append(f"[{i}] Document: {doc_name}")
        if headings:
            lines.append(f"    Section: {' > '.join(headings)}")
        if isinstance(score, (int, float)):
            lines.append(f"    Relevance score: {score:.3f}")
        lines.append(f"    Excerpt: {text}")
        lines.append("")
    return "\n".join(lines)


@app.route("/api/summarize_references", methods=["POST"])
def api_summarize_references():
    payload = request.get_json(force=True) or {}
    question = (payload.get("question") or "").strip()
    results = payload.get("results") or []

    if not question:
        return jsonify({"error": "No question available to summarize against."}), 400
    if not results:
        return jsonify({"error": "No search results to summarize. Run a search first."}), 400

    try:
        from LLM_Summary_Ollama import summarize_text  # deferred: touches Ollama at import time
    except Exception as e:
        return jsonify({"error": f"Could not load LLM helper: {type(e).__name__}: {e}"}), 500

    prompt = _format_results_for_prompt(question, results)

    try:
        raw = summarize_text(prompt, max_tokens=12800, setup=SUMMARIZE_REFERENCES_SYSTEM_PROMPT)
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

    if not raw:
        return jsonify({"error": "The model returned no summary."}), 500

    return jsonify({"summary": raw})


@app.route("/api/image")
def api_image():
    path = vdb.resolve_path(request.args.get("path", ""))  # stored paths are relative to the documents root
    if not path or not os.path.isfile(path):
        abort(404)
    mime, _ = mimetypes.guess_type(path)
    return send_file(path, mimetype=mime or "application/octet-stream")


@app.route("/api/document")
def api_document():
    """Serve a source document (the page-range chunk file) inline, only if it is in the vector store."""
    rel_path = request.args.get("path", "")
    path = vdb.resolve_path(rel_path)
    if not path or not os.path.isfile(path) or not vdb.document_exists(rel_path):
        abort(404)
    mime, _ = mimetypes.guess_type(path)
    return send_file(path, mimetype=mime or "application/octet-stream", as_attachment=False)


_page_cache = {}


def _words(text):
    return re.findall(r"\w+", (text or "").lower())


def _shingles(words, n=3):
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def _find_chunk_page(pdf_path, chunk_text, page_start=1, page_end=None):
    """Return the 1-based PDF page whose text best overlaps the chunk text.

    Only pages page_start..page_end are searched (the 10-page range the chunk came from);
    falls back to page_start if undeterminable.
    """
    import fitz  # PyMuPDF

    chunk_shingles = _shingles(_words(chunk_text))
    if not chunk_shingles:
        return page_start
    best_page, best_score = page_start, 0
    with fitz.open(pdf_path) as pdf:
        last = min(page_end or len(pdf), len(pdf))
        for i in range(page_start - 1, last):
            score = len(chunk_shingles & _shingles(_words(pdf[i].get_text())))
            if score > best_score:
                best_page, best_score = i + 1, score
    return best_page


@app.route("/api/open_chunk")
def api_open_chunk():
    """Redirect to the source document, jumping to the page that holds the given chunk."""
    import sqlite3

    vid = request.args.get("id", "")
    conn = sqlite3.connect(vdb.DB_PATH)
    try:
        row = conn.execute("SELECT doc_path, text, metadata FROM vectors WHERE id = ?", (vid,)).fetchone()
    finally:
        conn.close()
    full_path = vdb.resolve_path(row[0]) if row else None
    if not full_path or not os.path.isfile(full_path):
        abort(404)
    doc_path, text, metadata_json = row
    try:
        metadata = json.loads(metadata_json or "{}")
    except (TypeError, ValueError):
        metadata = {}
    page_start = int(metadata.get("page_start") or 1)
    page_end = metadata.get("page_end")

    page = page_start
    if doc_path.lower().endswith(".pdf"):
        if vid not in _page_cache:
            try:
                _page_cache[vid] = _find_chunk_page(full_path, text, page_start, page_end)
            except Exception:
                traceback.print_exc()
                _page_cache[vid] = page_start
        page = _page_cache[vid]
    return redirect(f"/api/document?path={quote(doc_path, safe='')}#page={page}")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5151, debug=False)
