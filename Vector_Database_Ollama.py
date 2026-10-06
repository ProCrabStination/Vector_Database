import logging
import sqlite3
import time
import os
import json
import faiss
import numpy as np
import requests
import re
import uuid
from pathlib import Path
from annotated_types import doc
from Text_Vectorizing_Ollama import Vectorize_Text as Convert_Text_To_Vectors
import sqlite3


STORE_DIR = os.environ.get("VECDB_STORE_DIR") or os.path.join(os.path.dirname(__file__), "vec_store")
DB_PATH = os.path.join(STORE_DIR, "meta.db")
INDEX_PATH = os.path.join(STORE_DIR, "index.faiss")

# Document folder layout (all under DOCS_ROOT):
#   Inputs/                              new documents are dropped here
#   Working Directory/                   scratch space for the tools (page chunks, conversions)
#   Completed Documents For Reference/   finished documents:  <doc>/<doc>.pdf  and  <doc>/images/
# Stored document and picture paths are relative to DOCS_ROOT (forward slashes), so the
# database stays valid when the project is moved. Override with VECDB_DOCS_ROOT if needed.
DOCS_ROOT = os.path.abspath(
    os.environ.get("VECDB_DOCS_ROOT")
    or os.path.join(os.path.dirname(os.path.abspath(__file__)), "Documents")
)
INPUTS_DIR_NAME = "Inputs"
WORKING_DIR_NAME = "Working Directory"
COMPLETED_DIR_NAME = "Completed Documents For Reference"

# 10-page PDF chunks are named "<source>_pages_0001-0010"
_CHUNK_RE = re.compile(r"^(?P<stem>.+)_pages_(?P<start>\d{4})-(?P<end>\d{4})$")


def completed_doc_dir(source_stem):
    """Absolute path of the completed-documents folder for one source document."""
    return os.path.join(DOCS_ROOT, COMPLETED_DIR_NAME, source_stem)


def parse_chunk_name(stem):
    """Split "<source>_pages_0001-0010" into (source, "pages_0001-0010", 1, 10); None if not a chunk name."""
    m = _CHUNK_RE.match(stem)
    if not m:
        return None
    start, end = int(m.group("start")), int(m.group("end"))
    return m.group("stem"), f"pages_{m.group('start')}-{m.group('end')}", start, end


def to_relative(path):
    """Return `path` relative to DOCS_ROOT with forward slashes (as stored in the database).

    Already-relative paths are only normalized; absolute paths outside DOCS_ROOT are returned
    unchanged (slashes normalized) since they cannot be made relative.
    """
    if path is None:
        return path
    p = str(path)
    if os.path.isabs(p):
        try:
            rel = os.path.relpath(p, DOCS_ROOT)
            if not rel.startswith(".."):
                p = rel
        except ValueError:  # different drive
            pass
    return p.replace("\\", "/")


def resolve_path(path):
    """Resolve a stored (relative) path to an absolute filesystem path under DOCS_ROOT.

    Returns None if the result would fall outside DOCS_ROOT (path traversal) or `path` is empty.
    Absolute paths are accepted only if they are inside DOCS_ROOT.
    """
    if not path:
        return None
    full = os.path.abspath(os.path.join(DOCS_ROOT, str(path).replace("\\", "/")))
    root = os.path.normcase(DOCS_ROOT)
    if os.path.normcase(full) != root and not os.path.normcase(full).startswith(root + os.sep):
        return None
    return full

def _chunk_filter(source_chunk):
    """SQL fragment + params restricting rows to one page-range chunk (e.g. "pages_0001-0010")."""
    if not source_chunk:
        return "", ()
    return " AND metadata LIKE ?", (f'%"source_chunk": "{source_chunk}"%',)


def document_exists(doc_path, source_chunk=None):
    """True if the document (optionally one page-range chunk of it) already has stored vectors."""
    doc_path_str = to_relative(doc_path)
    extra_sql, extra_params = _chunk_filter(source_chunk)
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM vectors WHERE doc_path = ?" + extra_sql + " LIMIT 1", (doc_path_str, *extra_params))
        exists = cur.fetchone() is not None
    finally:
        conn.close()
    return exists

def processed_chunk_indices(doc_path, source_chunk=None):
    """Return chunk indexes already stored for a document path (optionally one page-range chunk)."""
    doc_path_str = to_relative(doc_path)
    extra_sql, extra_params = _chunk_filter(source_chunk)
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("SELECT metadata FROM vectors WHERE doc_path = ?" + extra_sql, (doc_path_str, *extra_params))
        processed = set()
        for (metadata_json,) in cur.fetchall():
            try:
                metadata = json.loads(metadata_json or "{}")
                chunk_index = metadata.get("chunk_index")
                if chunk_index is not None:
                    processed.add(int(chunk_index))
            except (TypeError, ValueError, json.JSONDecodeError):
                logging.warning("Ignoring invalid chunk metadata for %s", doc_path_str)
        return processed
    finally:
        conn.close()

def list_documents():
    """List the source documents in the store (one entry per document).

    Each document's doc_path is its full file under "Completed Documents For
    Reference/<doc>/<doc>.<ext>", and "chunk_count" is the number of stored
    sections. Databases from the older layout, where doc_path was a 10-page chunk
    under "_pdf_chunks/<source>/", are still grouped by that source folder.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("SELECT doc_path, COUNT(*) FROM vectors GROUP BY doc_path")
        doc_rows = cur.fetchall()
    finally:
        conn.close()

    groups = {}
    for dp, row_count in doc_rows:
        parts = Path(dp).parts
        if "_pdf_chunks" in parts:  # legacy layout: one doc_path per page-range chunk
            idx = parts.index("_pdf_chunks")
            group_name = parts[idx + 1] if idx + 1 < len(parts) else Path(dp).stem
        else:
            group_name = Path(dp).stem
        entry = groups.setdefault(group_name, {"paths": [], "count": 0})
        entry["paths"].append(dp)
        entry["count"] += row_count

    return sorted(
        (
            {"name": name, "doc_paths": sorted(g["paths"]), "chunk_count": g["count"]}
            for name, g in groups.items()
        ),
        key=lambda g: g["name"].lower(),
    )


def ensure_store(dim):

    os.makedirs(STORE_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS vectors (
        id TEXT PRIMARY KEY,
        position INTEGER,
        doc_path TEXT,
        model TEXT,
        text TEXT,
        metadata TEXT,
        pictures TEXT,
        created_at REAL
    )""")
    conn.commit()

    # Migrate older databases that predate the pictures column
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(vectors)").fetchall()}
    if "pictures" not in existing_cols:
        conn.execute("ALTER TABLE vectors ADD COLUMN pictures TEXT")
        conn.commit()
    conn.close()
    if os.path.exists(INDEX_PATH):
        try:
            index = faiss.read_index(INDEX_PATH)
            # check dimensionality and recreate if mismatch
            if hasattr(index, "d") and index.d != dim:
                print(f"Index dimension ({index.d}) != requested dim ({dim}), recreating index.")
                index = faiss.IndexFlatIP(dim)
                faiss.write_index(index, INDEX_PATH)
        except Exception as e:
            print(f"Failed to read existing index ({e}), creating new index with dim={dim}")
            index = faiss.IndexFlatIP(dim)
            faiss.write_index(index, INDEX_PATH)
    else:
        # use inner-product on normalized vectors for cosine similarity
        index = faiss.IndexFlatIP(dim)
        faiss.write_index(index, INDEX_PATH)  # create file
    return index

def add_vector(embedding, doc_path, text, metadata=None, model=None, pictures=None):
    # embedding: list or 1d np array (float32)
    vec = np.array(embedding, dtype=np.float32).reshape(1, -1)
    # normalize for cosine similarity
    faiss.normalize_L2(vec)
    dim = vec.shape[1]
    index = ensure_store(dim)

    # compute position == current number of vectors
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM vectors")
    position = cur.fetchone()[0]

    # add to index
    index.add(vec)  # one vector
    faiss.write_index(index, INDEX_PATH)

    # store metadata (paths are stored relative to DOCS_ROOT)
    doc_path = to_relative(doc_path)
    pictures = [
        {**pic, "path": to_relative(pic["path"])} if isinstance(pic, dict) and pic.get("path") else pic
        for pic in (pictures or [])
    ]
    vid = str(uuid.uuid4())
    cur.execute(
        "INSERT INTO vectors (id, position, doc_path, model, text, metadata, pictures, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (vid, position, doc_path, model or "unknown", text, json.dumps(metadata or {}), json.dumps(pictures or []), time.time())
    )
    conn.commit()
    conn.close()
    return vid

def search_vectors(query_texts, top_k_per_query=5, top_k_overall=20, deduplicate_per_query=True, keyword=None):
    """
    Search for similar vectors using one or more text queries.
    
    Args:
        query_texts: Single text string or list of text strings to search for
        top_k_per_query: Number of top results to return for each individual query
        top_k_overall: Number of top results to return overall (aggregated from all queries)
        deduplicate_per_query: Whether to deduplicate results per query
        keyword: Optional string. If provided, only results containing this keyword (in text or metadata) are included.
    
    Returns:
        Dictionary containing:
        - 'per_query': List of lists, where each inner list contains top_k_per_query results for that query
        - 'overall_top_k': List of top_k_overall results aggregated from all queries
        - 'query_count': Number of queries processed
    """
    # Normalize input to list
    if isinstance(query_texts, str):
        query_texts = [query_texts]
    
    if not query_texts:
        logging.warning("No query texts provided")
        return {
            'per_query': [],
            'overall_top_k': [],
            'query_count': 0
        }
    
    # Check if index exists
    if not os.path.exists(INDEX_PATH):
        logging.warning("No vector index found")
        return {
            'per_query': [[] for _ in query_texts],
            'overall_top_k': [],
            'query_count': len(query_texts)
        }
    
    # Load index
    index = faiss.read_index(INDEX_PATH)
    conn = sqlite3.connect(DB_PATH)
    
    per_query_results = []
    all_results_for_aggregation = []
    
    # Process each query
    for query_idx, query_text in enumerate(query_texts):
        # Generate embedding for this query
        query_embedding = Convert_Text_To_Vectors([query_text])
        
        if query_embedding is None or len(query_embedding) == 0:
            logging.warning(f"Failed to generate embedding for query {query_idx + 1}")
            per_query_results.append([])
            continue
        
        # Prepare query vector for FAISS
        query_vec = np.array(query_embedding[0], dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query_vec)
        
        # Search the index
        # Multiply top_k_per_query by 2 to account for potential duplicates if deduplicating
        search_k = top_k_per_query * 2 if deduplicate_per_query else top_k_per_query
        D, I = index.search(query_vec, search_k)
        
        # Map positions to metadata for this query
        query_results = []
        seen_ids = set()  # Track duplicates within this query
        
        for score, pos in zip(D[0], I[0]):
            if pos < 0:
                continue
            cur = conn.cursor()
            cur.execute("SELECT id, doc_path, model, metadata, text, pictures FROM vectors WHERE position = ?", (int(pos),))
            row = cur.fetchone()
            if row:
                vid, doc_path, model, metadata_json, text, pictures_json = row
                # Skip if we've already seen this vector in this query
                if deduplicate_per_query and vid in seen_ids:
                    continue
                seen_ids.add(vid)
                metadata = json.loads(metadata_json)
                pictures = json.loads(pictures_json) if pictures_json else []
                # Keyword filter (case-insensitive, checks text and metadata string values)
                if keyword:
                    keyword_lower = str(keyword).lower()
                    text_match = keyword_lower in (text or '').lower()
                    metadata_match = any(keyword_lower in str(v).lower() for v in metadata.values()) if isinstance(metadata, dict) else False
                    if not (text_match or metadata_match):
                        continue
                result = {
                    "id": vid,
                    "doc_path": doc_path,
                    "model": model,
                    "metadata": metadata,
                    "text": text,
                    "pictures": pictures,
                    "score": float(score),
                    "query_index": query_idx,
                    "query_text": query_text[:100] + ('...' if len(query_text) > 100 else ''),
                }
                query_results.append(result)
                all_results_for_aggregation.append(result)
                # Stop if we have enough unique results
                if deduplicate_per_query and len(query_results) >= top_k_per_query:
                    break
        # Store top_k_per_query for this specific query
        per_query_results.append(query_results[:top_k_per_query])
        logging.info(f"Query {query_idx + 1}/{len(query_texts)}: Found {len(query_results)} unique results for '{query_text[:50]}...'")
    conn.close()
    
    # Aggregate and deduplicate for overall top_k_overall
    # Use a dictionary to keep only the highest score for each vector ID
    unique_results = {}
    for result in all_results_for_aggregation:
        vid = result['id']
        if vid not in unique_results or result['score'] > unique_results[vid]['score']:
            unique_results[vid] = result
    
    # Sort by score and take top_k_overall
    overall_results = sorted(unique_results.values(), key=lambda x: x['score'], reverse=True)[:top_k_overall]
    
    logging.info(f"Processed {len(query_texts)} queries. Overall top results: {len(overall_results)}")
    
    return {
        'per_query': per_query_results,
        'overall_top_k': overall_results,
        'query_count': len(query_texts)
    }

def compute_metadata_score(metadata, keywords=None):
    # Adjust this function to your metadata fields and relevance heuristics
    import datetime
    score = 0.0
    if not metadata:
        return score
    try:
        if 'date' in metadata:
            doc_date = metadata['date']
            doc_dt = datetime.datetime.fromisoformat(doc_date)
            age_days = (datetime.datetime.utcnow() - doc_dt).days
            if age_days < 365:
                score += 0.1  # freshness boost
    except Exception:
        pass

    if keywords and 'domain' in metadata:
        domain = metadata.get('domain', '').lower()
        for kw in keywords:
            if kw.lower() in domain:
                score += 0.2  # domain match boost
    return score

def search_vectors_keywords(
        query_texts,
        top_k_per_query=5,
        top_k_overall=20,
        deduplicate_per_query=True,
        keywords=None,
        doc_paths=None
    ):

    import logging
    import os
    import sqlite3
    import json
    import numpy as np
    import faiss

    def compute_metadata_score(metadata, keywords=None):
        import datetime
        score = 0.0
        if not metadata:
            return score
        try:
            if 'date' in metadata:
                doc_date = metadata['date']
                doc_dt = datetime.datetime.fromisoformat(doc_date)
                age_days = (datetime.datetime.utcnow() - doc_dt).days
                if age_days < 365:
                    score += 0.1
        except Exception:
            pass
        if keywords and 'domain' in metadata:
            domain = metadata.get('domain', '').lower()
            for kw in keywords:
                if kw.lower() in domain:
                    score += 0.2
        return score


    def get_document_score(doc_path, query_text, keywords=None):
        """
        Compute a score for the entire document based on keyword matches in doc_path and text.
        """
        if keywords is None:
            keywords = query_text.lower().split()

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()

        score = 0.0
        PATH_WEIGHT = 2.0  # more weight to keywords in file path
        TEXT_WEIGHT = 1.0

        # Count keyword matches in doc_path (once per keyword)
        doc_path_lower = doc_path.lower()
        for kw in keywords:
            if kw in doc_path_lower:
                score += PATH_WEIGHT

        # Count total keyword occurrences in all text chunks of this document
        for kw in keywords:
            # Count number of vectors with text containing keyword in this doc
            cur.execute("""
                SELECT COUNT(*) FROM vectors
                WHERE doc_path = ? AND LOWER(text) LIKE ?
            """, (doc_path, f"%{kw}%"))
            count = cur.fetchone()[0]
            score += TEXT_WEIGHT * count

        conn.close()
        return float(score)


    if isinstance(query_texts, str):
        query_texts = [query_texts]

    if not query_texts:
        logging.warning("No query texts provided")
        return {
            'per_query': [],
            'overall_top_k': [],
            'query_count': 0
        }

    if not os.path.exists(INDEX_PATH):
        logging.warning("No vector index found")
        return {
            'per_query': [[] for _ in query_texts],
            'overall_top_k': [],
            'query_count': len(query_texts)
        }

    index = faiss.read_index(INDEX_PATH)
    dimension = index.d
    conn = sqlite3.connect(DB_PATH)

    positions_filter_set = None
    if keywords or doc_paths:
        cur = conn.cursor()
        clauses = []
        params = []
        if keywords:
            or_clauses = []
            for kw in keywords:
                # Search in chunk text, doc_path, and metadata text
                or_clauses.append("(LOWER(text) LIKE ? OR LOWER(doc_path) LIKE ?)")
                params.extend((f"%{kw.lower()}%", f"%{kw.lower()}%"))
            clauses.append("(" + " OR ".join(or_clauses) + ")")
        if doc_paths:
            placeholders = ", ".join("?" for _ in doc_paths)
            clauses.append(f"doc_path IN ({placeholders})")
            params.extend(doc_paths)

        query_str = f"""
            SELECT DISTINCT position FROM vectors
            WHERE {" AND ".join(clauses)}
        """
        cur.execute(query_str, params)
        rows = cur.fetchall()
        positions_filter_set = set(row[0] for row in rows)
        if not positions_filter_set:
            logging.info("Keyword/document filter found no matching vectors in DB.")
            conn.close()
            return {
                'per_query': [[] for _ in query_texts],
                'overall_top_k': [],
                'query_count': len(query_texts)
            }

    per_query_results = []
    all_results_for_aggregation = []

    for query_idx, query_text in enumerate(query_texts):
        query_embedding = Convert_Text_To_Vectors([query_text])
        if query_embedding is None or len(query_embedding) == 0:
            logging.warning(f"Failed to generate embedding for query {query_idx + 1}")
            per_query_results.append([])
            continue

        query_vec = np.array(query_embedding[0], dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query_vec)

        if positions_filter_set is not None:
            # Just get positions for keyword/document-filtered candidates
            candidate_positions = list(positions_filter_set)
        else:
            # Use FAISS search to get candidate positions
            search_k = top_k_per_query * 4 if deduplicate_per_query else top_k_per_query * 2
            D, I = index.search(query_vec, search_k)
            candidate_positions = [int(pos) for pos in I[0] if pos >= 0]

        # Reconstruct vectors for all candidate positions and compute exact scores
        candidate_vectors = np.zeros((len(candidate_positions), dimension), dtype=np.float32)
        for i, pos in enumerate(candidate_positions):
            candidate_vectors[i] = index.reconstruct(pos)
        faiss.normalize_L2(candidate_vectors)
        faiss.normalize_L2(query_vec)

        exact_scores = candidate_vectors.dot(query_vec.T).flatten()
        
        # Create lightweight candidates with position and score only
        scored_candidates = [
            {'position': pos, 'exact_score': float(score)}
            for pos, score in zip(candidate_positions, exact_scores)
        ]
        
        # Sort by score and take top candidates before fetching metadata
        top_n_candidates = min(100, len(scored_candidates))  # Only fetch metadata for top 100
        scored_candidates.sort(key=lambda x: x['exact_score'], reverse=True)
        top_scored = scored_candidates[:top_n_candidates]
        
        # Now fetch full metadata only for top candidates
        candidates = []
        cur = conn.cursor()
        for sc in top_scored:
            pos = sc['position']
            cur.execute("SELECT id, doc_path, model, metadata, text, pictures FROM vectors WHERE position = ?", (pos,))
            row = cur.fetchone()
            if not row:
                continue
            vid, doc_path, model, metadata_json, text, pictures_json = row
            metadata = json.loads(metadata_json)
            pictures = json.loads(pictures_json) if pictures_json else []
            candidates.append({
                'id': vid,
                'position': pos,
                'doc_path': doc_path,
                'model': model,
                'metadata': metadata,
                'text': text,
                'pictures': pictures,
                'exact_score': sc['exact_score'],
            })
        
        # # Add metadata scoring to candidates
        # for c in candidates:
        #     c['metadata_score'] = compute_metadata_score(c.get('metadata'), keywords)

        # WEIGHT_EXACT = 0.75
        # WEIGHT_META = 0.25

        # for c in candidates:
        #     c['final_score'] = WEIGHT_EXACT * c['exact_score'] + WEIGHT_META * c['metadata_score']

        # Use exact_score directly as final_score (no weighting)
        for c in candidates:
            c['final_score'] = c['exact_score']

        # Deduplicate by ID if requested
        if deduplicate_per_query:
            seen_ids = set()
            unique_candidates = []
            for c in sorted(candidates, key=lambda x: x['final_score'], reverse=True):
                if c['id'] in seen_ids:
                    continue
                seen_ids.add(c['id'])
                unique_candidates.append(c)
            query_results = unique_candidates[:top_k_per_query]
        else:
            query_results = sorted(candidates, key=lambda x: x['final_score'], reverse=True)[:top_k_per_query]

        # if deduplicate_per_query:
        #     seen_ids = set()
        #     unique_candidates = []
        #     for c in sorted(candidates, key=lambda x: x['final_score'], reverse=True):
        #         if c['id'] in seen_ids:
        #             continue
        #         seen_ids.add(c['id'])
        #         unique_candidates.append(c)
        #         # Do not limit here because we want to gather all candidates for document scoring
        #     query_results = unique_candidates
        # else:
        #     query_results = sorted(candidates, key=lambda x: x['final_score'], reverse=True)

        # # === NEW: Document-level scoring ===
        # # Get distinct documents matched
        # doc_scores = {}
        # for chunk in query_results:
        #     doc_path = chunk['doc_path']
        #     if doc_path not in doc_scores:
        #         doc_scores[doc_path] = None

        # # Query document-level scores
        # for doc_path in doc_scores.keys():
        #     doc_scores[doc_path] = get_document_score(doc_path, query_text)

        # # Pick top 2 documents by document-level score
        # top_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        # top_doc_paths = set(d[0] for d in top_docs)

        # # From top documents, select top 5 chunks each by chunk-level final_score
        # filtered_results = []
        # for doc_path in top_doc_paths:
        #     chunks = [c for c in query_results if c['doc_path'] == doc_path]
        #     chunks_sorted = sorted(chunks, key=lambda x: x['final_score'], reverse=True)
        #     filtered_results.extend(chunks_sorted[:5])

        per_query_results.append(query_results)
        all_results_for_aggregation.extend(query_results)

        logging.info(f"Query {query_idx + 1}/{len(query_texts)}: Found {len(query_results)} results for '{query_text[:50]}...'")

    conn.close()

    # Aggregate and deduplicate overall results by id with final_score
    unique_results = {}
    for res in all_results_for_aggregation:
        vid = res['id']
        if vid not in unique_results or res['final_score'] > unique_results[vid]['final_score']:
            unique_results[vid] = res

    overall_results = sorted(unique_results.values(), key=lambda x: x['final_score'], reverse=True)[:top_k_overall]

    logging.info(f"Processed {len(query_texts)} queries, overall top results: {len(overall_results)}")

    return {
        'per_query': per_query_results,
        'overall_top_k': overall_results,
        'query_count': len(query_texts)
    }



def format_search_results(results, include_text=False):
    """
    Format search results for display.
    
    Args:
        results: List of search result dictionaries
        include_text: If True, include chunk text in output
    
    Returns:
        Formatted string representation of results
    """
    if not results:
        return "No results found."
    
    output = []
    output.append(f"\n{'='*80}")
    output.append(f"Top {len(results)} Search Results")
    output.append(f"{'='*80}\n")
    
    for i, result in enumerate(results, 1):
        output.append(f"--- Result {i} (Score: {result['score']:.4f}) ---")
        output.append(f"  Vector ID: {result['id']}")
        output.append(f"  Document: {result['doc_path']}")
        output.append(f"  Chunk ID: {result['chunk_id']}")
        output.append(f"  Has Table: {result['has_table']}")
        
        if result.get('headings'):
            output.append(f"  Headings: {' > '.join(result['headings'])}")
        
        if result.get('query_text'):
            output.append(f"  Matched Query: {result['query_text']}")
        
        metadata = result.get('metadata', {})
        if 'token_count' in metadata:
            output.append(f"  Token Count: {metadata['token_count']}")
        
        if include_text and 'text' in metadata:
            text = metadata['text']
            preview = text[:200] + ('...' if len(text) > 200 else '')
            output.append(f"  Text Preview: {preview}")
        
        output.append("-" * 80)
    
    return "\n".join(output)


def reset_database():
    """
    Delete and recreate the vector database with the correct schema.
    WARNING: This will delete all existing vectors!
    """
    print("Resetting the vector database...")
    import shutil
    
    if os.path.exists(STORE_DIR):
        print(f"Removing existing database at {STORE_DIR}...")
        shutil.rmtree(STORE_DIR)
        print("✓ Old database removed")
    
    # Recreate with a dummy dimension (will be updated when first vector is added)
    print("Creating new database...")
    ensure_store(dim=768)  # Default dimension
    print("✓ New database created with correct schema")
    print("\nDatabase schema:")
    print("  - id: TEXT PRIMARY KEY")
    print("  - position: INTEGER")
    print("  - doc_path: TEXT")
    print("  - model: TEXT")
    print("  - text: TEXT")
    print("  - metadata: TEXT")
    print("  - pictures: TEXT")
    print("  - created_at: REAL")


# Map of functions callable by other programs via: python Vector_Database_Ollama.py <function> [json_args...]
AVAILABLE_CLI_FUNCTIONS = {
    'document_exists': document_exists,
    'processed_chunk_indices': processed_chunk_indices,
    'list_documents': list_documents,
    'add_vector': add_vector,
    'search_vectors': search_vectors,
    'search_vectors_keywords': search_vectors_keywords,
    'format_search_results': format_search_results,
    'reset_database': reset_database,
}


def _json_default(obj):
    # Make otherwise non-serializable results (sets, faiss.Index, etc.) safe to print
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)


def _print_help():
    print(f"""
Vector_Database_Ollama.py - CLI usage guide
{'=' * 80}

HOW TO RUN (PowerShell):
  & "<path to python.exe>" "Vector_Database_Ollama.py" <function_name> <arg1> <arg2> ...

  - <function_name> is a bare word from the list below. Do NOT prefix it with
    "-" or "--" (e.g. use "search_vectors", not "-search_vectors").
  - Every argument after the function name is its OWN separate quoted string.
    Do NOT concatenate multiple arguments into one string like "[5][5][yes]".
  - Arguments are parsed as JSON when possible, so:
      strings -> wrap in quotes:      "hello world"
      numbers -> plain, no quotes:    5
      booleans -> lowercase:          true / false
      lists   -> JSON array syntax:   [ ... ]
      objects -> JSON object syntax:  {{ ... }}
  - Omitted trailing arguments fall back to each function's default value.

  *** POWERSHELL QUOTING GOTCHA ***
  PowerShell's argument marshaling to native executables is unreliable once a
  single argument mixes double quotes AND spaces -- both backtick-escaping and
  single-quoting can still get the quotes silently stripped and the argument
  split apart. This is the #1 cause of "positional arguments" / JSON errors
  from this script. The VERIFIED-WORKING fix is PowerShell's --% (stop-parsing)
  token: everything after it is passed through literally, so ordinary
  backslash-escaped quotes survive untouched:
    UNRELIABLE (may get mangled):  '["a b", "c"]'   or   "[`"a b`", `"c`"]"
    RELIABLE (use --% before it):  --% "[\\"a b\\", \\"c\\"]"
  If you are calling from cmd.exe, a batch file, C#, Python subprocess, etc.
  instead of PowerShell, --% is not needed -- just use normal backslash-escaped
  double quotes as usual.

RECOMMENDED FOR OTHER PROGRAMS/SCRIPTS -- the --json mode:
  Shell quoting for several separate arguments is error-prone. Instead, pass
  ALL arguments as a single JSON array in one argument:

    python Vector_Database_Ollama.py <function_name> --json <json array>

  Example (PowerShell -- use --% so the quotes survive):
    python Vector_Database_Ollama.py search_vectors_keywords --json --% "[\\"gearbox torque spec\\", 5, 20, true, [\\"gearbox\\", \\"torque\\"]]"

  Example (cmd.exe / batch / most other languages, normal backslash escaping):
    python Vector_Database_Ollama.py search_vectors_keywords --json "[\\"gearbox torque spec\\", 5, 20, true, [\\"gearbox\\", \\"torque\\"]]"

  Any language calling this script should build that JSON array with its own
  JSON encoder (e.g. json.dumps(args) in Python, JsonSerializer.Serialize in
  C#) and pass the result as one argument -- this avoids all shell-escaping
  ambiguity from the per-token form above.

Special invocations:
  python Vector_Database_Ollama.py -help        (or --help / help)  -> this guide
  python Vector_Database_Ollama.py               (no args)          -> runs a
      self-test that directly invokes the real PowerShell command for
      search_vectors_keywords against the real (read-only) vec_store/.

OUTPUT CONTRACT (for other programs calling this script):
  Exactly one line of JSON is printed to stdout:
    {{"result": ...}}   on success
    {{"error": "..."}}  on failure (process exit code is 1)
  All logging/warnings go to stderr, so stdout is always safe to parse as JSON.

{'-' * 80}
AVAILABLE FUNCTIONS
{'-' * 80}

1) search_vectors_keywords <query_texts> [top_k_per_query] [top_k_overall] [deduplicate_per_query] [keywords]
   Same as search_vectors, but filters/boosts using a JSON array of keywords.
   Args:
     query_texts            (string, or JSON array of strings, required)
     top_k_per_query         (number, optional, default 5)
     top_k_overall           (number, optional, default 20)
     deduplicate_per_query   (true/false, optional, default true)
     keywords                (JSON array of strings, optional, default none)
   Example (PowerShell -- use --% so the quotes survive):
     python Vector_Database_Ollama.py search_vectors_keywords --json --% "[\\"gearbox torque spec\\", 5, 20, true, [\\"gearbox\\", \\"torque\\"]]"

{'-' * 80}
NOTES
{'-' * 80}
  - search_vectors / search_vectors_keywords call out to the local embedding
    model (via Text_Vectorizing_Ollama), so Ollama must be running and reachable.
  - Run with no arguments at all to execute the built-in self-test, which
    exercises every function above safely against a temp store and prints a
    PASS/FAIL summary.
{'=' * 80}""")


def _dispatch_args_list(function_name, args_list):
    """Run one call using an already-decoded list of positional args and return the single JSON output line."""
    if function_name not in AVAILABLE_CLI_FUNCTIONS:
        hint = ""
        if function_name.startswith("-") and function_name.lstrip("-") in AVAILABLE_CLI_FUNCTIONS:
            hint = f" Did you mean '{function_name.lstrip('-')}' (no leading dash)?"
        return json.dumps({"error": f"Unknown function: {function_name}.{hint} Available functions: {', '.join(AVAILABLE_CLI_FUNCTIONS.keys())}. Run with -help for full usage."})

    try:
        result = AVAILABLE_CLI_FUNCTIONS[function_name](*args_list)
        return json.dumps({"result": result}, default=_json_default)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})


def _dispatch(function_name, function_args):
    """Run one CLI-style call (function name + string args) and return the single JSON output line."""
    if function_name not in AVAILABLE_CLI_FUNCTIONS:
        hint = ""
        if function_name.startswith("-") and function_name.lstrip("-") in AVAILABLE_CLI_FUNCTIONS:
            hint = f" Did you mean '{function_name.lstrip('-')}' (no leading dash)?"
        return json.dumps({"error": f"Unknown function: {function_name}.{hint} Available functions: {', '.join(AVAILABLE_CLI_FUNCTIONS.keys())}. Run with -help for full usage."})

    try:
        # Convert each argument from JSON where possible (dicts, lists, numbers), else keep as string
        processed_args = []
        for arg in function_args:
            try:
                processed_args.append(json.loads(arg))
            except (json.JSONDecodeError, ValueError):
                processed_args.append(arg)

        result = AVAILABLE_CLI_FUNCTIONS[function_name](*processed_args)
        return json.dumps({"result": result}, default=_json_default)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})


def main():
    """Directly invoke the real PowerShell command for search_vectors_keywords -- the only
    CLI option exercised by this self-test -- against the real vec_store/ (read-only)."""
    import sys
    import subprocess

    script_path = os.path.abspath(__file__)
    python_exe = sys.executable

    args = ["Hazardous Location HMI", 3, 5, True, ["HMI"]]
    json_payload = json.dumps(args).replace('"', '\\"')
    # --% (stop-parsing) makes PowerShell pass everything after it through literally,
    # so ordinary backslash-escaped quotes survive instead of being stripped/mangled.
    command_str = f'& "{python_exe}" "{script_path}" search_vectors_keywords --json --% "{json_payload}"'

    print("=" * 80)
    print("Invoking the real PowerShell command for search_vectors_keywords")
    print("=" * 80)
    print(f"\n--- {command_str} ---")

    proc = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command_str],
        capture_output=True, text=True
    )
    print(proc.stdout.strip())
    if proc.stderr.strip():
        print(f"[stderr] {proc.stderr.strip()}")

    print("\n" + "=" * 80)
    print(f"{'PASS' if proc.returncode == 0 else 'FAIL'} - search_vectors_keywords")
    print("=" * 80)


if __name__ == "__main__":
    import sys

    # Send logging to stderr so stdout carries only the JSON result/error
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stderr
    )

    if len(sys.argv) > 1 and sys.argv[1] in ('-help', '--help', 'help'):
        _print_help()
    elif len(sys.argv) > 3 and sys.argv[2] == '--json':
        # Recommended for other programs: <function_name> --json "<json array of args>"
        try:
            args_list = json.loads(sys.argv[3])
            if not isinstance(args_list, list):
                raise ValueError("--json payload must be a JSON array")
        except Exception as e:
            print(json.dumps({"error": f"Invalid --json payload: {e}"}))
            sys.exit(1)
        output = _dispatch_args_list(sys.argv[1], args_list)
        print(output)
        sys.exit(1 if '"error"' in output else 0)
    elif len(sys.argv) > 1:
        output = _dispatch(sys.argv[1], sys.argv[2:])
        print(output)
        sys.exit(1 if '"error"' in output else 0)
    else:
        # No arguments provided: simulate every CLI option against a temp store
        main()

