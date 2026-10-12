"""
Crash-proof driver for Document_To_Database_Pipeline_Ollama.py.

Runs the pipeline once per document, each in its own Python process, so a hard crash (CUDA/host out of
memory, std::bad_alloc, a model server dying, ...) only costs that one document: it is recorded as failed
and the run moves on to the next. Already-ingested documents are skipped by the pipeline itself, so the
run can be repeated safely.

State lives in Documents/Working Directory/_resilient_state.json:
  - failed documents are skipped on later runs (use --retry-failed to try them again)
  - if the whole run dies hard (power loss, killed console), the document that was in progress is marked
    failed on the next start instead of being retried forever

Usage:
    Run_Pipeline_Resilient.py                       everything in Documents/Inputs
    Run_Pipeline_Resilient.py --in-place FOLDER     scan FOLDER, reference the originals (no copies)
    Run_Pipeline_Resilient.py --retry-failed        also retry documents that failed before
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PIPELINE = REPO_ROOT / "Document_To_Database_Pipeline_Ollama.py"
WORKING_DIR = REPO_ROOT / "Documents" / "Working Directory"
STATE_FILE = WORKING_DIR / "_resilient_state.json"

# Mirrors Docling_Convert_Ollama.ALL_SUPPORTED_EXTENSIONS (not imported: it loads Docling/torch).
# Excel files are left out because the pipeline's own folder scan skips them too.
SUPPORTED = {
    ".doc", ".docx", ".rtf", ".odt", ".csv", ".ods",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp",
    ".ppt", ".pptx", ".odp", ".pdf",
}


def load_state():
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    state.setdefault("in_progress", None)
    state.setdefault("failed", {})
    return state


def save_state(state):
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, STATE_FILE)


def iter_documents(folder):
    """Yield supported documents lazily, top-down: a folder's own files first, then each subfolder in turn.

    Nothing is collected up front, so work starts at once and only one folder listing is held in memory.
    """
    try:
        with os.scandir(folder) as it:
            entries = sorted(it, key=lambda e: e.name.lower())
    except OSError as e:
        print(f"  (cannot read folder, skipping: {folder} - {e})", flush=True)
        return
    subfolders = []
    for entry in entries:
        try:
            if entry.is_dir(follow_symlinks=False):
                subfolders.append(entry.path)
            elif (entry.is_file() and not entry.name.startswith("~")
                  and os.path.splitext(entry.name)[1].lower() in SUPPORTED):
                yield Path(entry.path)
        except OSError:
            continue
    for sub in subfolders:
        yield from iter_documents(sub)


def main(argv):
    retry_failed = "--retry-failed" in argv
    in_place_dir = None
    if "--in-place" in argv:
        idx = argv.index("--in-place")
        if idx + 1 >= len(argv):
            print("--in-place needs a folder path.")
            return 2
        in_place_dir = Path(os.path.abspath(argv[idx + 1]))
        if not in_place_dir.is_dir():
            print(f"Folder not found: {in_place_dir}")
            return 2
    root = in_place_dir or (REPO_ROOT / "Documents" / "Inputs")
    if not root.is_dir():
        print(f"Input folder not found: {root}")
        return 2

    state = load_state()
    if state["in_progress"]:
        stuck = state["in_progress"]
        print(f"The previous run died while processing:\n  {stuck}\nMarking it as failed and skipping it.\n")
        state["failed"][stuck] = {"reason": "previous run died while processing this document",
                                  "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        state["in_progress"] = None
        save_state(state)
    if retry_failed:
        state["failed"] = {}
        save_state(state)

    print(f"Processing documents under {root}, one at a time, starting at the top folder.", flush=True)
    done = skipped = seen = 0
    failed_now = []
    for doc in iter_documents(root):
        seen += 1
        number = seen
        key = str(doc)
        if key in state["failed"]:
            print(f"[#{number}] Skipping (failed earlier: {state['failed'][key]['reason']}): {doc.name}")
            skipped += 1
            continue

        print(f"\n{'=' * 70}\n[#{number}] {doc}\n{'=' * 70}", flush=True)
        state["in_progress"] = key
        save_state(state)

        cmd = [sys.executable, str(PIPELINE)]
        if in_place_dir:
            cmd += ["--in-place", str(in_place_dir)]
        cmd.append(key)
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        try:
            code = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env).returncode
        except KeyboardInterrupt:
            # Deliberate stop by the user: not a crash, so the document is not marked failed.
            state["in_progress"] = None
            save_state(state)
            print("\nStopped by user.")
            return 130

        state["in_progress"] = None
        if code != 0:
            print(f"\n!!! Crashed (exit code {code}); skipping {doc.name} and continuing.", flush=True)
            state["failed"][key] = {"reason": f"exit code {code}", "time": time.strftime("%Y-%m-%d %H:%M:%S")}
            failed_now.append(doc)
        else:
            done += 1
        save_state(state)

    print(f"\n{'=' * 70}\nFinished. Documents seen: {seen}. Completed: {done}, skipped (failed earlier): {skipped}, crashed this run: {len(failed_now)}")
    if state["failed"]:
        print("Documents that failed (not ingested, or only partly):")
        for path, info in state["failed"].items():
            print(f"  - {path}  [{info['reason']}]")
        print("Run with --retry-failed to try them again.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
