
# Example: simple local store using FAISS + SQLite
import os
import sqlite3
from typing import List
import uuid
import time
import numpy as np
import faiss
import json
import requests

STORE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vec_store")
DB_PATH = os.path.join(STORE_DIR, "meta.db")
INDEX_PATH = os.path.join(STORE_DIR, "index.faiss")

def ensure_store(dim):
    os.makedirs(STORE_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS vectors (
        id TEXT PRIMARY KEY,
        position INTEGER,
        doc_path TEXT,
        model TEXT,
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

def add_vector(embedding, doc_path, model, metadata=None):
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
        "INSERT INTO vectors (id, position, doc_path, model, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (vid, position, doc_path, model, json.dumps(metadata or {}), time.time())
    )
    conn.commit()
    conn.close()
    return vid

def search(query_embedding, top_k=5):
    q = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(q)
    if not os.path.exists(INDEX_PATH):
        return []
    index = faiss.read_index(INDEX_PATH)
    D, I = index.search(q, top_k)
    # map positions to ids/metadata
    conn = sqlite3.connect(DB_PATH)
    results = []
    for score, pos in zip(D[0], I[0]):
        if pos < 0:
            continue
        cur = conn.cursor()
        cur.execute("SELECT id, doc_path, model, metadata FROM vectors WHERE position = ?", (int(pos),))
        row = cur.fetchone()
        if row:
            vid, doc_path, model, metadata_json = row
            results.append({"id": vid, "doc_path": doc_path, "model": model, "metadata": json.loads(metadata_json), "score": float(score)})
    conn.close()
    return results

def _extract_text_from_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".txt", ".md"):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

def chunk_text_by_chars(text: str, max_chunk_chars: int = 2000, overlap: int = 200) -> List[str]:
    """
    Split text into chunks <= max_chunk_chars. Try to split on whitespace and keep small overlap.
    """
    if len(text) <= max_chunk_chars:
        return [text]
    words = text.split()
    chunks = []
    cur = []
    cur_len = 0
    for w in words:
        if cur_len + len(w) + (1 if cur else 0) > max_chunk_chars:
            chunks.append(" ".join(cur))
            # start new chunk with overlap
            if overlap > 0:
                overlap_words = " ".join(cur)[-overlap:]
                # simple overlap: re-add last few words (best-effort)
                cur = cur[-min(len(cur),  max(1, overlap // max(1, len(w)))):]
                cur_len = sum(len(x) + 1 for x in cur) - 1 if cur else 0
            else:
                cur = []
                cur_len = 0
        cur.append(w)
        cur_len += len(w) + (1 if cur_len else 0)
    if cur:
        chunks.append(" ".join(cur))
    # fallback: ensure none exceed limit (rare)
    final = []
    for c in chunks:
        if len(c) <= max_chunk_chars:
            final.append(c)
        else:
            # hard split
            for i in range(0, len(c), max_chunk_chars):
                final.append(c[i:i+max_chunk_chars])
    return final

def _mean_embedding(embs: List[List[float]]) -> List[float]:
    if not embs:
        return []
    dim = len(embs[0])
    # ensure all same length
    for e in embs:
        if len(e) != dim:
            raise ValueError("embeddings have different dimensions")
    mean = [0.0] * dim
    for e in embs:
        for i, v in enumerate(e):
            mean[i] += float(v)
    n = len(embs)
    return [x / n for x in mean]

def Convert_Markdown_To_Vectors(document_path=None, api_key=None, api_host=None, model=None):


    # document_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Documents", "Working Directory")
    # api_key = os.getenv("AI_GATEWAY_API_KEY", "")
    # api_host = "https://ai-gateway.vitesco.io/v1"
    # model = "cohere.embed-multilingual-v3"

    # Recursively go through a folder of documents, chunk, and vectorize each chunk
    for root, dirs, files in os.walk(document_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            # Only process .txt files (markdown converted)
            if filename.lower().endswith(".md"):
                print(f"Vectorizing document: {file_path}")

                text = _extract_text_from_file(file_path)
                if not text:
                    print("Error: failed to extract text from document.")
                    continue

                # If model has a known max context, chunk accordingly. Cohere example expects ~2048 chars.
                max_chunk_chars = 2000

                chunks = chunk_text_by_chars(text, max_chunk_chars=max_chunk_chars, overlap=200)
                # limit number of chunks to avoid very large requests
                max_chunks = 10

                #batch chunks in groups of max_chunks
                chunk_batches = [chunks[i:i + max_chunks] for i in range(0, len(chunks), max_chunks)]
                for batch_start, batch in enumerate(chunk_batches):
                    print(f"Processing batch of {len(batch)} chunks...")
                    chunks = batch

                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    }

                    payload = {
                        "model": model,
                        "input": chunks, 
                    }

                    url = api_host.rstrip('/') + "/embeddings"

                    try:
                        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
                        resp.raise_for_status()
                    except requests.exceptions.HTTPError as e:
                        body = resp.text if 'resp' in locals() and resp is not None else str(e)
                        print(f"Error: request failed: {e}\nResponse body:\n{body}")
                        return
                    except Exception as e:
                        print(f"Error: request failed: {e}")
                        return

                    try:
                        resp_json = resp.json()
                    except Exception as e:
                        print(f"Error: failed to parse response JSON: {e}\nRaw body:\n{resp.text}")
                        return

                    data = resp_json.get("data")
                    if not data or not isinstance(data, list):
                        print(f"Error: unexpected response shape: {json.dumps(resp_json)[:1000]}")
                        return

                    # data should contain one embedding per input chunk in same order
                    embeddings = []
                    for item in data:
                        emb = item.get("embedding")
                        if emb is None:
                            print("Error: missing embedding in response item.")
                            return
                        try:
                            embeddings.append([float(x) for x in emb])
                        except Exception:
                            embeddings.append(list(emb))

                    try:
                        doc_embedding = _mean_embedding(embeddings)
                    except Exception as e:
                        print(f"Error combining embeddings: {e}")
                        return
                    
                    # Store each chunk's vector with chunk index metadata
                    for chunk_idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
                        metadata = {
                            "chunk_index": batch_start * max_chunks + chunk_idx,
                        }
                        vid = add_vector(embedding, file_path, model, metadata)
                        print(f"Added vector for chunk {metadata['chunk_index']} with id: {vid}")

                    print(f"Chunks sent: {len(chunks)}. Embeddings received: {len(embeddings)}")

