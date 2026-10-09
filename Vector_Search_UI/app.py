"""
Local web UI for Vector_Database_Ollama.search_vectors_keywords / search_vectors.

Run with the project's venv, e.g. from the repo root:
    .venv\\Scripts\\python.exe Vector_Search_UI\\app.py

Then open http://127.0.0.1:5151 in a browser.
"""
import codecs
import itertools
import json
import mimetypes
import os
import re
import shlex
import subprocess
import sys
import threading
import traceback

from urllib.parse import quote

from flask import Flask, Response, jsonify, redirect, render_template, request, send_file, abort

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
    try:
        vdb.add_neighbor_chunks(overall)
    except Exception:  # context is a nicety; never fail the search over it
        traceback.print_exc()

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
document, document address (path), page location, section headings, relevance score, the matched excerpt, \
and the text of the chunks immediately before and after it as surrounding context.

Respond in plain text (no markdown headers, no code fences), structured like this:
1. For each passage worth using, write its number, then a one-to-two sentence explanation of why it \
matters and what it tells the reader -- name the document, page and section, don't just say "reference 3". \
Use the before/after text to understand the passage in context.
2. Skip or briefly dismiss passages that are irrelevant or redundant; don't force relevance onto weak matches.
3. End with a short "Bottom line" paragraph naming the 2-4 references most worth reading in full, with \
their document address and page.

Be concise and specific.
"""


def _page_label(r):
    """Best page description for a result: the exact PDF page when it can be found, else the chunk's page range."""
    start, end = r.get("page_start"), r.get("page_end")
    page = None
    full_path = vdb.resolve_path(r.get("doc_path") or "")
    if r.get("id") and full_path and full_path.lower().endswith(".pdf") and os.path.isfile(full_path):
        try:
            page = _exact_page(r["id"], full_path, r.get("text") or "", int(start or 1), end)
        except Exception:
            traceback.print_exc()
    if page:
        return f"page {page} (chunk file covers pages {start}-{end})" if start and end else f"page {page}"
    if start and end:
        return f"pages {start}-{end}"
    return None


def _format_results_for_prompt(question, results, max_chars_per_result=700, max_results=15):
    lines = [f"Question: {question}", ""]
    for i, r in enumerate(results[:max_results], 1):
        doc_name = r.get("doc_name") or ""
        headings = r.get("headings") or []
        score = r.get("score")
        text = (r.get("text") or "")[:max_chars_per_result]

        lines.append(f"[{i}] Document: {doc_name}")
        if r.get("doc_path"):
            lines.append(f"    Document address: {r['doc_path']}")
        page = _page_label(r)
        if page:
            lines.append(f"    Location: {page}")
        if headings:
            lines.append(f"    Section: {' > '.join(headings)}")
        if isinstance(score, (int, float)):
            lines.append(f"    Relevance score: {score:.3f}")
        prev_text = (r.get("prev_text") or "")[-max_chars_per_result:]  # the end of the preceding chunk is the nearest context
        next_text = (r.get("next_text") or "")[:max_chars_per_result]
        if prev_text:
            lines.append(f"    Text before (preceding chunk): {prev_text}")
        lines.append(f"    Excerpt (matched chunk): {text}")
        if next_text:
            lines.append(f"    Text after (following chunk): {next_text}")
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


@app.route("/api/document")
def api_document():
    """Serve a source document (the page-range chunk file) inline, only if it is in the vector store."""
    rel_path = request.args.get("path", "")
    path = vdb.resolve_path(rel_path)
    if not path or not os.path.isfile(path) or not vdb.document_exists(rel_path):
        abort(404)
    pages = request.args.get("pages", "")
    m = re.fullmatch(r"(\d+)-(\d+)", pages)
    if m and path.lower().endswith(".pdf"):
        # Huge PDFs (e.g. multi-GB scans) can't be opened by the browser's viewer, so serve a
        # small PDF holding just this page window; /api/open_chunk links here for those files.
        import fitz  # PyMuPDF
        first, last = int(m.group(1)), int(m.group(2))
        with fitz.open(path) as src, fitz.open() as out:
            first = max(1, min(first, src.page_count))
            last = max(first, min(last, src.page_count))
            out.insert_pdf(src, from_page=first - 1, to_page=last - 1)
            data = out.tobytes(garbage=3, deflate=True)
        return Response(data, mimetype="application/pdf", headers={"Content-Disposition": "inline"})
    mime, _ = mimetypes.guess_type(path)
    return send_file(path, mimetype=mime or "application/octet-stream", as_attachment=False)


_page_cache = {}
LARGE_PDF_BYTES = 200 * 1024 * 1024  # above this, open_chunk serves a page window instead of the whole file


def _exact_page(vid, pdf_path, text, page_start, page_end):
    """Cached vdb.find_chunk_page keyed by vector id (falls back to page_start on failure)."""
    if vid not in _page_cache:
        try:
            _page_cache[vid] = vdb.find_chunk_page(pdf_path, text, page_start, page_end)
        except Exception:
            traceback.print_exc()
            _page_cache[vid] = page_start
    return _page_cache[vid]


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
        page = _exact_page(vid, full_path, text, page_start, page_end)
    if doc_path.lower().endswith(".pdf") and os.path.getsize(full_path) > LARGE_PDF_BYTES:
        first, last = max(1, page - 2), page + 2  # api_document clamps to the real page count
        return redirect(f"/api/document?path={quote(doc_path, safe='')}&pages={first}-{last}#page={page - first + 1}")
    return redirect(f"/api/document?path={quote(doc_path, safe='')}#page={page}")


# ---------------------------------------------------------------------------
# Task runner: launch the repo's Run_*.bat files from the UI and stream their output.
# ---------------------------------------------------------------------------
# fields: "text" values become positional args (or "flag value" when `flag` is set), "checkbox"
# adds `flag` when ticked, "raw" is split shell-style into several args. actions are the buttons;
# each appends its `extra` args.
TASKS = [
    {
        "id": "pipeline", "title": "Ingest documents (pipeline)", "bat": "Run_Pipeline.bat",
        "description": "Drop documents onto the box to convert and ingest them into the vector database "
                       "(they are saved to Documents\\Inputs first). Needs a CUDA GPU.",
        "dropzone": True,
        "fields": [],
        "actions": [{"label": "Ingest everything already in Inputs", "extra": []}],
    },
    {
        "id": "pipeline_in_place", "title": "Ingest folder in place (no copies)", "bat": "Run_Pipeline_In_Place.bat",
        "description": "Ingests every document in a folder and its subfolders. Originals are not copied: the "
                       "database stores their original paths. Needs a CUDA GPU.",
        "fields": [{"name": "folder", "label": "Folder to scan", "type": "text", "required": True}],
        "actions": [{"label": "Run in-place ingest", "extra": []}],
    },
    {
        "id": "docling", "title": "Docling convert only", "bat": "Run_Docling_Convert.bat",
        "description": "Converts every document in Documents\\Inputs to Docling JSON/markdown in the Working "
                       "Directory. Nothing is added to the database. Needs a CUDA GPU.",
        "fields": [], "actions": [{"label": "Run conversion", "extra": []}],
    },
    {
        "id": "backfill", "title": "Backfill missing images", "bat": "Run_Backfill_Missing_Images.bat",
        "description": "Re-converts already-ingested PDFs whose picture crops were never saved to disk. "
                       "Needs a CUDA GPU.",
        "fields": [], "actions": [{"label": "Run backfill", "extra": []}],
    },
    {
        "id": "clean_pdf", "title": "Clean PDF for OCR", "bat": "Run_Clean_PDF_For_OCR.bat",
        "description": "Deskews/cleans a poor scan before ingesting it. Needs Tesseract on PATH (unpaper for Clean).",
        "fields": [
            {"name": "input", "label": "Input PDF", "type": "text", "required": True},
            {"name": "output", "label": "Output PDF", "type": "text", "required": True},
            {"name": "clean", "label": "Also clean with unpaper (--clean)", "type": "checkbox", "flag": "--clean"},
            {"name": "oversample", "label": "Oversample DPI (optional, e.g. 300)", "type": "text",
             "flag": "--oversample"},
        ],
        "actions": [{"label": "Clean PDF", "extra": []}],
    },
    {
        "id": "query", "title": "Ask Ollama with database context", "bat": "Run_Query.bat",
        "description": "Asks Ollama a question grounded in the database from a JSON prompt file "
                       "(blank = Testing\\Example_Prompt.md).",
        "fields": [
            {"name": "prompt", "label": "Prompt file (optional)", "type": "text"},
            {"name": "only", "label": "Print only the response (--only-response)", "type": "checkbox",
             "flag": "--only-response"},
        ],
        "actions": [{"label": "Run query", "extra": []}],
    },
    {
        "id": "db_cli", "title": "Database CLI", "bat": "Run_Database_CLI.bat",
        "description": "Command-line access to the vector store. Blank prints the help guide. Arguments are "
                       "split like a POSIX shell, e.g. search_vectors_keywords --json "
                       "\"[\\\"gearbox torque\\\", 5, 20, true, [\\\"gearbox\\\"]]\"",
        "fields": [{"name": "args", "label": "Arguments (e.g. list_documents)", "type": "raw"}],
        "actions": [{"label": "Run", "extra": []}],
    },
    {
        "id": "migrate", "title": "Migrate paths to relative", "bat": "Run_Migrate_Paths.bat",
        "description": "One-off: migrates vec_store\\meta.db paths to the portable relative layout "
                       "(meta.db is backed up first).",
        "fields": [],
        "actions": [{"label": "Preview (dry run)", "extra": ["--dry-run"]},
                    {"label": "Apply migration", "extra": [], "danger": True}],
    },
    {
        "id": "populate", "title": "Populate completed documents", "bat": "Run_Populate_Completed.bat",
        "description": "One-off: builds \"Completed Documents For Reference\" from the old Test_Documents layout "
                       "(copies only).",
        "fields": [
            {"name": "inputs", "label": "Legacy inputs folder (Test_Documents)", "type": "text",
             "flag": "--legacy-inputs", "required": True},
            {"name": "working", "label": "Legacy working folder (Test_Documents_Markdown)", "type": "text",
             "flag": "--legacy-working", "required": True},
        ],
        "actions": [{"label": "Preview (dry run)", "extra": ["--dry-run"]},
                    {"label": "Populate", "extra": [], "danger": True}],
    },
    {
        "id": "mcp", "title": "MCP server (test)", "bat": "Run_MCP_Server.bat",
        "description": "Starts the vector-docs MCP server over stdio. Claude normally launches it itself; with "
                       "no client attached it exits immediately.",
        "fields": [], "actions": [{"label": "Start server", "extra": []}],
    },
]
TASKS_BY_ID = {t["id"]: t for t in TASKS}

_job_counter = itertools.count(1)
_jobs = {}  # task id -> job dict (the most recent run of that task)
_jobs_lock = threading.Lock()
_CMD_METACHARS = set('&|<>^%\r\n')
_LOCAL_HOSTS = {"127.0.0.1:5151", "localhost:5151"}


def _request_allowed():
    """Only same-origin JSON posts to the local server may start/stop processes (blocks CSRF from other sites)."""
    # Also require a loopback client so that, in --lan mode, other machines can search but never run tasks.
    return request.is_json and request.host in _LOCAL_HOSTS and request.remote_addr in ("127.0.0.1", "::1")


def _build_args(task, values, action):
    args = []
    for f in task["fields"]:
        value = values.get(f["name"])
        if f["type"] == "checkbox":
            if value:
                args.append(f["flag"])
            continue
        value = str(value or "").strip()
        if not value:
            if f.get("required"):
                raise ValueError(f"\"{f['label']}\" is required.")
            continue
        if f["type"] == "raw":
            args.extend(shlex.split(value))
            continue
        if f.get("flag"):
            args.append(f["flag"])
        args.append(value)
    args.extend(action["extra"])
    for a in args:
        if any(c in _CMD_METACHARS for c in a):
            raise ValueError("Arguments may not contain & | < > ^ % or line breaks.")
    return args


def _pump_output(job):
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    stream = job["proc"].stdout
    while True:
        chunk = stream.read1(4096)
        if not chunk:
            break
        text = decoder.decode(chunk)
        with _jobs_lock:
            job["output"] += text
    with _jobs_lock:
        job["output"] += decoder.decode(b"", final=True)
        job["exit_code"] = job["proc"].wait()


def _job_state(task_id):
    job = _jobs.get(task_id)
    if not job:
        return {"running": False, "job_id": None, "exit_code": None}
    return {"running": job["exit_code"] is None, "job_id": job["id"], "exit_code": job["exit_code"],
            "command": job["command"]}


@app.route("/api/tasks")
def api_tasks():
    with _jobs_lock:
        return jsonify([{**t, "state": _job_state(t["id"])} for t in TASKS])


@app.route("/api/tasks/<task_id>/run", methods=["POST"])
def api_task_run(task_id):
    if not _request_allowed():
        abort(403)
    task = TASKS_BY_ID.get(task_id)
    if not task:
        abort(404)
    payload = request.get_json() or {}
    action_index = int(payload.get("action", 0))
    if not 0 <= action_index < len(task["actions"]):
        return jsonify({"error": "Unknown action."}), 400
    try:
        args = _build_args(task, payload.get("values") or {}, task["actions"][action_index])
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return _start_job(task_id, task, args)


def _task_running(task_id):
    with _jobs_lock:
        return task_id in _jobs and _jobs[task_id]["exit_code"] is None


def _start_job(task_id, task, args):
    with _jobs_lock:
        if task_id in _jobs and _jobs[task_id]["exit_code"] is None:
            return jsonify({"error": "This task is already running."}), 409
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        # stdin is closed so the trailing "pause" in each .bat returns immediately instead of hanging.
        proc = subprocess.Popen(
            ["cmd.exe", "/c", os.path.join(REPO_ROOT, task["bat"]), *args], cwd=REPO_ROOT, env=env,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        job = {"id": next(_job_counter), "proc": proc, "output": "", "exit_code": None,
               "command": subprocess.list2cmdline([task["bat"], *args])}
        _jobs[task_id] = job
    threading.Thread(target=_pump_output, args=(job,), daemon=True).start()
    return jsonify({"job_id": job["id"], "command": job["command"]})


UPLOAD_EXTENSIONS = {
    ".doc", ".docx", ".rtf", ".odt", ".xls", ".xlsx", ".ods", ".csv",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp",
    ".ppt", ".pptx", ".odp", ".pdf",
}  # mirrors Docling_Convert_Ollama.ALL_SUPPORTED_EXTENSIONS (not imported: it loads Docling/torch)


def _upload_allowed():
    """Multipart posts can't use the is_json CSRF guard, so require a custom header (forces a CORS preflight).

    Unlike the task runner, uploads are deliberately open to other machines on the LAN (--lan mode).
    """
    return request.headers.get("X-Requested-With") == "vector-ui"


def _unique_upload_path(dest_dir, filename, size):
    """Path in dest_dir for an uploaded file; reuses an identical-size file of the same name, else numbers it."""
    stem, ext = os.path.splitext(filename)
    candidate, n = os.path.join(dest_dir, filename), 0
    while os.path.exists(candidate):
        if os.path.getsize(candidate) == size:
            return candidate, True
        n += 1
        candidate = os.path.join(dest_dir, f"{stem}_{n}{ext}")
    return candidate, False


@app.route("/api/ingest", methods=["POST"])
def api_ingest():
    """Save dropped documents into Documents/Inputs and ingest exactly those files with the pipeline."""
    from werkzeug.utils import secure_filename

    if not _upload_allowed():
        abort(403)
    task = TASKS_BY_ID["pipeline"]
    files = [f for f in request.files.getlist("files") if f and f.filename]
    if not files:
        return jsonify({"error": "No files received."}), 400
    if _task_running("pipeline"):
        return jsonify({"error": "The pipeline is already running."}), 409

    rejected = [f.filename for f in files if os.path.splitext(f.filename)[1].lower() not in UPLOAD_EXTENSIONS]
    if rejected:
        return jsonify({"error": "Unsupported file type: " + ", ".join(rejected)}), 400

    dest_dir = os.path.join(vdb.DOCS_ROOT, vdb.INPUTS_DIR_NAME)
    os.makedirs(dest_dir, exist_ok=True)
    saved = []
    for f in files:
        name = secure_filename(f.filename)
        ext = os.path.splitext(f.filename)[1].lower()
        if not os.path.splitext(name)[0]:  # e.g. a name made only of non-ASCII characters
            name = "document" + ext
        f.save(tmp := os.path.join(dest_dir, f"~upload_{os.getpid()}_{len(saved)}"))
        size = os.path.getsize(tmp)
        target, exists = _unique_upload_path(dest_dir, name, size)
        if exists:
            os.remove(tmp)
        else:
            os.replace(tmp, target)
        saved.append(target)

    if any(any(c in _CMD_METACHARS for c in p) for p in saved):
        return jsonify({"error": "The Documents path contains characters that can't be passed to the pipeline."}), 400
    return _start_job("pipeline", task, saved)


@app.route("/api/tasks/<task_id>/output")
def api_task_output(task_id):
    offset = max(0, int(request.args.get("offset", 0)))
    with _jobs_lock:
        job = _jobs.get(task_id)
        if not job:
            return jsonify({"running": False, "job_id": None, "exit_code": None, "output": "", "next_offset": 0})
        return jsonify({**_job_state(task_id), "output": job["output"][offset:], "next_offset": len(job["output"])})


@app.route("/api/tasks/<task_id>/stop", methods=["POST"])
def api_task_stop(task_id):
    if not _request_allowed():
        abort(403)
    with _jobs_lock:
        job = _jobs.get(task_id)
        running = bool(job) and job["exit_code"] is None
        pid = job["proc"].pid if running else None
    if not running:
        return jsonify({"error": "This task is not running."}), 400
    # Kill the whole tree (cmd -> python), not just the cmd.exe wrapper.
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True,
                   creationflags=subprocess.CREATE_NO_WINDOW)
    return jsonify({"stopped": True})


if __name__ == "__main__":
    import socket
    import sys
    if "--lan" in sys.argv:
        try:
            lan_ip = socket.gethostbyname(socket.gethostname())
        except OSError:
            lan_ip = "<this-pc-ip>"
        print(f"LAN mode: other machines on the network can use http://{lan_ip}:5151 (search and "
              f"drag-and-drop ingest; other batch tasks remain restricted to this PC).")
        app.run(host="0.0.0.0", port=5151, debug=False, threaded=True)
    else:
        app.run(host="127.0.0.1", port=5151, debug=False)
