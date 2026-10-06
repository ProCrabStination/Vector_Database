"""Migrate vec_store/meta.db paths to the portable "completed documents" layout.

Usage:  python Migrate_Paths_To_Relative.py [--dry-run]

Before (old layouts, absolute or relative to the old Test_Documents_Markdown folder):
    doc_path  .../_pdf_chunks/<doc>/<doc>_pages_0001-0010.pdf        (a 10-page working chunk)
    picture   .../<doc>_pages_0001-0010/images/pictures_0.png

After (relative to the Documents folder, forward slashes):
    doc_path  Completed Documents For Reference/<doc>/<doc>.pdf      (the full document)
    picture   Completed Documents For Reference/<doc>/images/pages_0001-0010_pictures_0.png
    metadata  gains "source_chunk", "page_start", "page_end" so the UI can open the right page

A backup of meta.db is written first (meta.db.pre_completed_layout.bak). Safe to re-run:
rows already in the new layout are left alone.
"""
import json
import posixpath
import re
import shutil
import sqlite3
import sys

from Vector_Database_Ollama import COMPLETED_DIR_NAME, DB_PATH, parse_chunk_name, to_relative

# Folder names that were the document root on previous machines/layouts
LEGACY_ROOTS = ("Test_Documents_Markdown", "Working Directory", COMPLETED_DIR_NAME)
_LEGACY_RE = re.compile(r"^.*?[\\/](?:%s)[\\/]" % "|".join(map(re.escape, LEGACY_ROOTS)), re.IGNORECASE)
_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]|^[\\/]")


def relativize(path):
    """Absolute/legacy path -> forward-slash path relative to the old document root."""
    if not path:
        return path
    p = str(path)
    if _ABSOLUTE_RE.match(p):
        m = _LEGACY_RE.match(p)
        if m:
            p = p[m.end():]
        else:
            return to_relative(p)
    return p.replace("\\", "/")


def convert_doc_path(doc_path):
    """Return (new_doc_path, chunk_info). chunk_info is parse_chunk_name() output or None."""
    p = relativize(doc_path)
    if p.startswith(COMPLETED_DIR_NAME + "/"):
        return p, None  # already migrated
    parts = p.split("/")
    chunk = parse_chunk_name(posixpath.splitext(parts[-1])[0])
    if chunk:
        source = chunk[0]
        return f"{COMPLETED_DIR_NAME}/{source}/{source}{posixpath.splitext(parts[-1])[1]}", chunk
    # Not a page-range chunk: keep the file name under its own completed folder
    stem = posixpath.splitext(parts[-1])[0]
    return f"{COMPLETED_DIR_NAME}/{stem}/{parts[-1]}", None


def convert_picture_path(path):
    p = relativize(path)
    if p.startswith(COMPLETED_DIR_NAME + "/"):
        return p
    parts = p.split("/")
    # <doc>_pages_A-B/images/<file>
    if len(parts) >= 3 and parts[-2] == "images":
        chunk = parse_chunk_name(parts[-3])
        if chunk:
            return f"{COMPLETED_DIR_NAME}/{chunk[0]}/images/{chunk[1]}_{parts[-1]}"
        return f"{COMPLETED_DIR_NAME}/{parts[-3]}/images/{parts[-1]}"
    return p


def main():
    dry = "--dry-run" in sys.argv
    if not dry:
        backup = DB_PATH + ".pre_completed_layout.bak"
        shutil.copy2(DB_PATH, backup)
        print(f"Backup written: {backup}")

    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute("SELECT id, doc_path, metadata, pictures FROM vectors").fetchall()
        changed = 0
        for vid, doc_path, metadata_json, pictures_json in rows:
            new_doc, chunk = convert_doc_path(doc_path)

            new_meta = metadata_json
            if chunk:
                try:
                    meta = json.loads(metadata_json or "{}")
                    meta.update({"source_chunk": chunk[1], "page_start": chunk[2], "page_end": chunk[3]})
                    new_meta = json.dumps(meta)
                except (TypeError, ValueError):
                    print(f"  WARNING: unparseable metadata JSON for {vid}; left unchanged")

            new_pics = pictures_json
            if pictures_json:
                try:
                    pics = json.loads(pictures_json)
                    for pic in pics:
                        if isinstance(pic, dict) and pic.get("path"):
                            pic["path"] = convert_picture_path(pic["path"])
                    new_pics = json.dumps(pics)
                except (TypeError, ValueError):
                    print(f"  WARNING: unparseable pictures JSON for {vid}; left unchanged")

            if (new_doc, new_meta, new_pics) != (doc_path, metadata_json, pictures_json):
                changed += 1
                if not dry:
                    conn.execute(
                        "UPDATE vectors SET doc_path = ?, metadata = ?, pictures = ? WHERE id = ?",
                        (new_doc, new_meta, new_pics, vid),
                    )
        if not dry:
            conn.commit()
        print(f"{'Would update' if dry else 'Updated'} {changed} of {len(rows)} rows.")

        n_docs = conn.execute("SELECT COUNT(DISTINCT doc_path) FROM vectors").fetchone()[0]
        print(f"Distinct doc_path values: {n_docs}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
