import logging
import sqlite3
import time
import os
import json
import faiss
import numpy as np
import requests
import uuid
from pathlib import Path
from annotated_types import doc
from Text_Vectorizing import Vectorize_Text as Convert_Text_To_Vectors
import sqlite3


STORE_DIR = os.path.join(os.path.dirname(__file__), "vec_store")
DB_PATH = os.path.join(STORE_DIR, "meta.db")
INDEX_PATH = os.path.join(STORE_DIR, "index.faiss")

def document_exists(doc_path):
    from Vector_Database import DB_PATH
    # Ensure doc_path is a string
    doc_path_str = str(doc_path)
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM vectors WHERE doc_path = ? LIMIT 1", (doc_path_str,))
        exists = cur.fetchone() is not None
    finally:
        conn.close()
    return exists

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
        created_at REAL
    )""")
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

def add_vector(embedding, doc_path, text, metadata=None, model=None):
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

    # store metadata
    vid = str(uuid.uuid4())
    cur.execute(
        "INSERT INTO vectors (id, position, doc_path, model, text, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (vid, position, doc_path, model or "unknown", text, json.dumps(metadata or {}), time.time())
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
            cur.execute("SELECT id, doc_path, model, metadata, text FROM vectors WHERE position = ?", (int(pos),))
            row = cur.fetchone()
            if row:
                vid, doc_path, model, metadata_json, text = row
                # Skip if we've already seen this vector in this query
                if deduplicate_per_query and vid in seen_ids:
                    continue
                seen_ids.add(vid)
                metadata = json.loads(metadata_json)
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
        keywords=None
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
    if keywords:
        cur = conn.cursor()
        or_clauses = []
        params = []
        for kw in keywords:
            # Search in chunk text, doc_path, and metadata text
            or_clauses.append("(LOWER(text) LIKE ? OR LOWER(doc_path) LIKE ?)")
            params.extend((f"%{kw.lower()}%", f"%{kw.lower()}%"))
        query_str = f"""
            SELECT DISTINCT position FROM vectors
            WHERE {" OR ".join(or_clauses)}
        """
        cur.execute(query_str, params)
        rows = cur.fetchall()
        positions_filter_set = set(row[0] for row in rows)
        if not positions_filter_set:
            logging.info("Keywords filter found no matching vectors in DB.")
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

        if keywords:
            # Just get positions for keyword-filtered candidates
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
            cur.execute("SELECT id, doc_path, model, metadata, text FROM vectors WHERE position = ?", (pos,))
            row = cur.fetchone()
            if not row:
                continue
            vid, doc_path, model, metadata_json, text = row
            metadata = json.loads(metadata_json)
            candidates.append({
                'id': vid,
                'position': pos,
                'doc_path': doc_path,
                'model': model,
                'metadata': metadata,
                'text': text,
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


def main():
    """
    Test function for all vector database operations.
    Demonstrates adding vectors, searching, and formatting results.
    """
    print("\n" + "="*80)
    print("Testing Vector Database Functions")
    print("="*80 + "\n")
    
    # Test 1: Setup store with a specific dimension
    print("Test 1: Setting up vector store...")
    dim = 1024  # Common embedding dimension
    try:
        index = ensure_store(dim)
        print(f"✓ Store initialized successfully with dimension {dim}")
        print(f"  - Database path: {DB_PATH}")
        print(f"  - Index path: {INDEX_PATH}")
        print(f"  - Current vector count: {index.ntotal}")
    except Exception as e:
        print(f"✗ Failed to initialize store: {e}")
        return
    
    # Test 2: Add some sample vectors
    print("\nTest 2: Adding sample vectors...")
    sample_vectors = [
        {
            "embedding": np.random.randn(dim).astype(np.float32).tolist(),
            "doc_path": "test_doc_1.pdf",
            "model": "cohere.embed-english-v3",
            "metadata": {
                "chunk_id": "chunk_001",
                "text": "This is a test document about machine learning and AI.",
                "has_table": False,
                "headings": ["Introduction", "Machine Learning"],
                "token_count": 150
            }
        },
        {
            "embedding": np.random.randn(dim).astype(np.float32).tolist(),
            "doc_path": "test_doc_2.pdf",
            "model": "cohere.embed-english-v3",
            "metadata": {
                "chunk_id": "chunk_002",
                "text": "Database systems and SQL queries are fundamental to data management.",
                "has_table": True,
                "headings": ["Chapter 2", "Database Fundamentals"],
                "token_count": 200
            }
        },
        {
            "embedding": np.random.randn(dim).astype(np.float32).tolist(),
            "doc_path": "test_doc_3.pdf",
            "model": "cohere.embed-english-v3",
            "metadata": {
                "chunk_id": "chunk_003",
                "text": "Python programming language is widely used for data science and machine learning.",
                "has_table": False,
                "headings": ["Programming", "Python Basics"],
                "token_count": 180
            }
        }
    ]
    
    added_ids = []
    for i, sample in enumerate(sample_vectors, 1):
        try:
            vid = add_vector(
                embedding=sample["embedding"],
                doc_path=sample["doc_path"],
                text=sample["metadata"]["text"],  # Extract text from metadata
                model=sample["model"],
                metadata=sample["metadata"]
            )
            added_ids.append(vid)
            print(f"✓ Added vector {i}/3 - ID: {vid[:8]}... - Doc: {sample['doc_path']}")
        except Exception as e:
            print(f"✗ Failed to add vector {i}: {e}")
    
    # Test 3: Search with a single query
    # Test 3: Search with actual queries
    print("\nTest 3: Searching with real queries...")
    try:
        # Test 3a: Search without keywords
        print("  Test 3a: Vector search without keywords")
        test_query = "Cummins engine test specification"
        results = search_vectors_keywords(
            query_texts=test_query,
            top_k_per_query=3,
            top_k_overall=5,
            deduplicate_per_query=True,
            keywords=None
        )
        
        print(f"    Query: '{test_query}'")
        print(f"    ✓ Found {len(results['overall_top_k'])} results")
        for i, res in enumerate(results['overall_top_k'][:3], 1):
            doc_name = res['doc_path'].split('\\')[-1] if '\\' in res['doc_path'] else res['doc_path']
            print(f"      {i}. {doc_name} (score: {res['final_score']:.4f})")
        
        # Test 3b: Search with keywords
        print("\n  Test 3b: Vector search with keywords")
        test_query_2 = "analyzer training course"
        keywords = ["analyzer", "training"]
        results_kw = search_vectors_keywords(
            query_texts=test_query_2,
            top_k_per_query=3,
            top_k_overall=5,
            deduplicate_per_query=True,
            keywords=keywords
        )
        
        print(f"    Query: '{test_query_2}'")
        print(f"    Keywords: {keywords}")
        print(f"    ✓ Found {len(results_kw['overall_top_k'])} results")
        for i, res in enumerate(results_kw['overall_top_k'][:3], 1):
            doc_name = res['doc_path'].split('\\')[-1] if '\\' in res['doc_path'] else res['doc_path']
            print(f"      {i}. {doc_name} (score: {res['final_score']:.4f})")
        
        # Test 3c: Multiple queries
        print("\n  Test 3c: Multiple query search")
        queries = ["MR2 test specification", "Daimler loadbox"]
        results_multi = search_vectors_keywords(
            query_texts=queries,
            top_k_per_query=2,
            top_k_overall=5,
            deduplicate_per_query=True,
            keywords=None
        )
        
        print(f"    Queries: {queries}")
        print(f"    ✓ Found {len(results_multi['overall_top_k'])} overall results")
        print(f"    Per-query results: {[len(pq) for pq in results_multi['per_query']]}")
        
    except Exception as e:
        print(f"  ✗ Search test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Demonstrate search_multiple_queries (conceptual)
    print("\nTest 4: API usage examples...")
    print("  Example usage:")
    print("    # Simple search")
    print("    results = search_vectors_keywords('your query', top_k_per_query=5)")
    print("    ")
    print("    # Search with keywords filter")
    print("    results = search_vectors_keywords('query', keywords=['Cummins', 'CM2150'])")
    print("    ")
    print("    # Multiple queries")
    print("    results = search_vectors_keywords(['query1', 'query2'], top_k_overall=10)")

    
    # Test 5: Format mock results
    print("\nTest 5: Testing result formatting...")
    mock_results = [
        {
            "id": added_ids[0] if added_ids else "mock_id_1",
            "doc_path": "test_doc_1.pdf",
            "model": "cohere.embed-english-v3",
            "metadata": {
                "chunk_id": "chunk_001",
                "text": "This is a test document about machine learning and AI.",
                "token_count": 150
            },
            "score": 0.8765,
            "chunk_id": "chunk_001",
            "has_table": False,
            "headings": ["Introduction", "Machine Learning"],
            "query_text": "machine learning"
        },
        {
            "id": added_ids[2] if len(added_ids) > 2 else "mock_id_2",
            "doc_path": "test_doc_3.pdf",
            "model": "cohere.embed-english-v3",
            "metadata": {
                "chunk_id": "chunk_003",
                "text": "Python programming language is widely used for data science.",
                "token_count": 180
            },
            "score": 0.7543,
            "chunk_id": "chunk_003",
            "has_table": False,
            "headings": ["Programming", "Python Basics"],
            "query_text": "python programming"
        }
    ]
    
    formatted = format_search_results(mock_results, include_text=True)
    print(formatted)
    
    # Test 6: Check database state
    print("\nTest 6: Checking database state...")
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM vectors")
        total_vectors = cur.fetchone()[0]
        
        cur.execute("SELECT doc_path, COUNT(*) FROM vectors GROUP BY doc_path")
        doc_counts = cur.fetchall()
        
        print(f"✓ Database contains {total_vectors} vectors")
        print("  Vectors per document:")
        for doc_path, count in doc_counts:
            print(f"    - {doc_path}: {count} vector(s)")
        
        conn.close()
    except Exception as e:
        print(f"✗ Failed to query database: {e}")
    
    print("\n" + "="*80)
    print("Testing Complete!")
    print("="*80 + "\n")
    
    print("Next steps:")
    print("  1. To test actual search, ensure Convert_Text_To_Vectors() is working")
    print("  2. Use search_vectors() with real queries")
    print("  3. Use search_multiple_queries() for complex searches")
    print("  4. Check vec_store/ directory for database and index files")

def reset_database():
    """
    Delete and recreate the vector database with the correct schema.
    WARNING: This will delete all existing vectors!
    """
    import shutil
    
    if os.path.exists(STORE_DIR):
        print(f"Removing existing database at {STORE_DIR}...")
        shutil.rmtree(STORE_DIR)
        print("✓ Old database removed")
    
    # Recreate with a dummy dimension (will be updated when first vector is added)
    print("Creating new database...")
    ensure_store(dim=1024)  # Default dimension
    print("✓ New database created with correct schema")
    print("\nDatabase schema:")
    print("  - id: TEXT PRIMARY KEY")
    print("  - position: INTEGER")
    print("  - doc_path: TEXT")
    print("  - model: TEXT")
    print("  - text: TEXT")
    print("  - metadata: TEXT")
    print("  - created_at: REAL")


if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Check if arguments are provided
    if len(sys.argv) > 1:
        function_name = sys.argv[1]
        function_args = sys.argv[2:]
        
        # Map of available functions
        available_functions = {
            'reset_database': reset_database,
            'ensure_store': ensure_store,
            'add_vector': add_vector,
            'search_vectors': search_vectors,
            'format_search_results': format_search_results,
        }
        
        if function_name in available_functions:
            try:
                # Call the function with the provided arguments
                # Convert arguments appropriately (e.g., JSON strings to dicts/lists)
                processed_args = []
                for arg in function_args:
                    try:
                        # Try to parse as JSON first (for dicts, lists, numbers)
                        processed_args.append(json.loads(arg))
                    except (json.JSONDecodeError, ValueError):
                        # If not JSON, keep as string
                        processed_args.append(arg)
                
                result = available_functions[function_name](*processed_args)
                print(json.dumps(result, indent=2) if result is not None else "Function executed successfully")
            except Exception as e:
                print(f"Error executing {function_name}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Unknown function: {function_name}")
            print(f"Available functions: {', '.join(available_functions.keys())}")
    else:
        # No arguments provided, run the main test function
        main()

