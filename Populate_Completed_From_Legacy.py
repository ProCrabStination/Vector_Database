"""Build 'Documents/Completed Documents For Reference' from the old Test_Documents layout.

For every document in the vector database this COPIES (the old folders are never modified):
  * the full original PDF         -> Completed.../<doc>/<doc>.pdf
  * each chunk's picture crops     -> Completed.../<doc>/images/pages_0001-0010_pictures_0.png

Usage:
    python Populate_Completed_From_Legacy.py --legacy-inputs <Test_Documents dir> \\
        --legacy-working <Test_Documents_Markdown dir> [--dry-run]

Existing destination files are skipped, so it can be re-run safely.
"""
import argparse
import os
import shutil
import sqlite3
import sys
from pathlib import Path

from Vector_Database_Ollama import DB_PATH, DOCS_ROOT, resolve_path


def find_original(legacy_inputs: Path, filename: str):
    """Locate <filename> anywhere under the legacy inputs folder."""
    for root, _dirs, files in os.walk(legacy_inputs):
        if filename in files:
            return Path(root) / filename
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--legacy-inputs", required=True, type=Path, help="old Test_Documents folder (original PDFs)")
    ap.add_argument("--legacy-working", required=True, type=Path, help="old Test_Documents_Markdown folder")
    ap.add_argument("--dry-run", action="store_true", help="report what would be copied, copy nothing")
    args = ap.parse_args()

    conn = sqlite3.connect(DB_PATH)
    try:
        doc_paths = [r[0] for r in conn.execute("SELECT DISTINCT doc_path FROM vectors")]
        # picture rows -> (doc_path, destination relative path)
        pictures = conn.execute("SELECT doc_path, pictures FROM vectors WHERE pictures IS NOT NULL AND pictures != '[]'").fetchall()
    finally:
        conn.close()

    import json
    copied_docs = copied_imgs = missing_docs = missing_imgs = skipped = 0
    total_bytes = 0

    def copy(src: Path, dest: Path):
        nonlocal skipped, total_bytes
        if dest.exists():
            skipped += 1
            return False
        total_bytes += src.stat().st_size
        if not args.dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        return True

    for doc_path in sorted(doc_paths):
        dest = Path(resolve_path(doc_path))
        src = find_original(args.legacy_inputs, dest.name)
        if src is None:
            print(f"MISSING original: {dest.name}")
            missing_docs += 1
            continue
        copied_docs += copy(src, dest)

    for doc_path, pictures_json in pictures:
        for pic in json.loads(pictures_json):
            rel = pic.get("path")
            if not rel:
                continue
            dest = Path(resolve_path(rel))
            # "<doc>/images/pages_0001-0010_pictures_0.png" <- "<doc>_pages_0001-0010/images/pictures_0.png"
            doc_name = dest.parent.parent.name
            label, _, picture_name = dest.name.partition("_pictures_")
            src = args.legacy_working / f"{doc_name}_{label}" / "images" / f"pictures_{picture_name}"
            if not src.is_file():
                missing_imgs += 1
                continue
            copied_imgs += copy(src, dest)

    verb = "Would copy" if args.dry_run else "Copied"
    print(f"{verb}: {copied_docs} document(s), {copied_imgs} image(s) (~{total_bytes / 1e9:.2f} GB); "
          f"already present: {skipped}")
    print(f"Missing originals: {missing_docs}, missing image files: {missing_imgs}")
    print(f"Destination root: {DOCS_ROOT}")
    return 1 if missing_docs else 0


if __name__ == "__main__":
    sys.exit(main())
