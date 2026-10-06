"""
One-off backfill: reconvert every already-ingested PDF chunk whose vector-DB
metadata references pictures that were never actually saved to disk.

Root cause: Docling_Convert_Ollama.py had pdf_pipeline_options.generate_picture_images
set to False (to avoid an OOM risk on large-format engineering drawings), so no PDF
ever got its picture crops written, even though chunk metadata/pictures paths were
still recorded as if they had been. That flag is now True, and process_pdf() will
reconvert (bypassing the cached-JSON fast path) whenever it detects a document with
pictures but no saved PNGs -- this script just drives that reconversion for every
affected document already in the store.

Usage (from repo root, project venv):
    .venv\\Scripts\\python.exe Backfill_Missing_Images.py
"""
import sqlite3
import time
from pathlib import Path

from Docling_Convert_Ollama import process_file
from Vector_Database_Ollama import DB_PATH, resolve_path


def find_affected_doc_paths():
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT doc_path FROM vectors "
            "WHERE pictures IS NOT NULL AND pictures != '[]' AND pictures != ''"
        )
        return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


def main():
    doc_paths = find_affected_doc_paths()
    if any(p.startswith("Completed Documents For Reference/") for p in doc_paths):
        print(
            "This one-off tool predates the completed-documents layout: the database now points at full\n"
            "documents (Completed Documents For Reference/<doc>/<doc>.pdf), not 10-page working chunks,\n"
            "so there is no chunk PDF to re-convert. Re-run the pipeline on the input document instead."
        )
        return
    print(f"Found {len(doc_paths)} document(s) with picture references to backfill.\n")

    ok, missing, failed = 0, 0, 0
    start = time.time()

    for i, doc_path in enumerate(doc_paths, 1):
        p = Path(resolve_path(doc_path) or doc_path)  # stored paths are relative to DOCS_ROOT
        elapsed = time.time() - start
        print(f"[{i}/{len(doc_paths)}] ({elapsed:.0f}s elapsed) {p.name}")

        if not p.is_file():
            print(f"  SKIP: source file not found on disk: {doc_path}")
            missing += 1
            continue

        # doc_path is already the split-page-chunk PDF under
        # <output_root>/_pdf_chunks/<orig_stem>/<chunk>.pdf, and process_pdf's
        # doc_output_dir is <output_root>/<chunk_stem>, so output_root is three
        # levels up: chunk file -> orig-stem subfolder -> _pdf_chunks -> output_root.
        output_root = p.parents[2]

        try:
            process_file(str(p), output_root)
            ok += 1
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            failed += 1

    total_elapsed = time.time() - start
    print(
        f"\nDone in {total_elapsed / 60:.1f} min. "
        f"Reconverted: {ok}, missing source: {missing}, failed: {failed}, total: {len(doc_paths)}"
    )


if __name__ == "__main__":
    main()
