
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import os
import sqlite3
import sys
from typing import List
import uuid
import time
import numpy as np
import faiss
import json
import requests

STORE_DIR = os.path.join(os.path.dirname(__file__), "vec_store")
DB_PATH = os.path.join(STORE_DIR, "meta.db")
INDEX_PATH = os.path.join(STORE_DIR, "index.faiss")

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

import os
def get_chunk_text(doc_path, chunk_index, chunk_size=10):
    """
    Retrieve the text for a specific chunk index from a document.
    """
    if not os.path.exists(doc_path):
        return f"File not found: {doc_path}"
    with open(doc_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    start = chunk_index * chunk_size
    end = start + chunk_size
    chunk_lines = lines[start:end]
    return ''.join(chunk_lines)

def print_search_result_chunks(results, chunk_size=10):
    for r in results:
        doc_path = r['doc_path']
        chunk_index = r['metadata']['chunk_index']
        text = get_chunk_text(doc_path, chunk_index, chunk_size)
        print(f"Chunk from {doc_path} (index {chunk_index}):\n{text}\n{'-'*40}")


def keyword_search_in_chunks(chunks, keyword):
    """
    Search for the exact keyword in all chunks. Returns list of (chunk_index, chunk_text) where found.
    """
    results = []
    for idx, chunk in enumerate(chunks):
        if keyword in chunk:
            results.append((idx, chunk))
    return results

# Example usage:
if __name__ == "__main__":
    # Example search results
    results = [
        {'id': 'aecfbcfd-b190-4726-a79b-b4c7c0b9ba7d', 'doc_path': 'C:\\Users\\uiv11567\\source\\repos\\MESSelfService\\Python\\Test_Documents_Markdown\\Daimler\\MR2\\10308111_SPE_000_AD_MR2TestSpec (2)\\10308111_SPE_000_AD_MR2TestSpec (2).md', 'model': 'cohere.embed-multilingual-v3', 'metadata': {'chunk_index': 122}, 'score': 0.7096729278564453},
        {'id': 'afe7379c-0396-4869-9895-2254bf297ab6', 'doc_path': 'C:\\Users\\uiv11567\\source\\repos\\MESSelfService\\Python\\Test_Documents_Markdown\\Daimler\\MR2\\10308111_SPE_000_AD_MR2TestSpec (2)\\10308111_SPE_000_AD_MR2TestSpec (2).md', 'model': 'cohere.embed-multilingual-v3', 'metadata': {'chunk_index': 197}, 'score': 0.707927942276001},
        {'id': 'b0e35366-1653-483f-833f-5090b4144f84', 'doc_path': 'C:\\Users\\uiv11567\\source\\repos\\MESSelfService\\Python\\Test_Documents_Markdown\\Daimler\\MR2\\MR2_EoL_TestSpec_Bgl\\MR2_EoL_TestSpec_Bgl.md', 'model': 'cohere.embed-multilingual-v3', 'metadata': {'chunk_index': 0}, 'score': 0.7054145336151123},
        {'id': '99abedf2-248d-42c7-b440-4fdb67496a28', 'doc_path': 'C:\\Users\\uiv11567\\source\\repos\\MESSelfService\\Python\\Test_Documents_Markdown\\Daimler\\MR2\\10308111_SPE_000_AD_MR2TestSpec (2)\\10308111_SPE_000_AD_MR2TestSpec (2).md', 'model': 'cohere.embed-multilingual-v3', 'metadata': {'chunk_index': 208}, 'score': 0.701673150062561},
        {'id': '6bcf8dab-27cd-40b6-a581-6237a1af090d', 'doc_path': 'C:\\Users\\uiv11567\\source\\repos\\MESSelfService\\Python\\Test_Documents_Markdown\\Daimler\\MR2\\10308111_SPE_000_AD_MR2TestSpec (2)\\10308111_SPE_000_AD_MR2TestSpec (2).md', 'model': 'cohere.embed-multilingual-v3', 'metadata': {'chunk_index': 0}, 'score': 0.7009074687957764}
    ]
    print_search_result_chunks(results, chunk_size=10)

    # Keyword search example
    keyword = "71010 LS-Switch of Bank 2 measurement"
    for r in results:
        doc_path = r['doc_path']
        chunk_index = r['metadata']['chunk_index']
        chunk_size = 10
        if not os.path.exists(doc_path):
            continue
        with open(doc_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Chunk the document
        chunks = ["".join(lines[i:i+chunk_size]) for i in range(0, len(lines), chunk_size)]
        matches = keyword_search_in_chunks(chunks, keyword)
        for idx, chunk in matches:
            print(f"Keyword found in chunk {idx} of {doc_path}:")
            print(chunk)
            print("-"*40)

def main():

    # take the argument to this file as the text to vectorize
    if len(sys.argv) == 2:
        text = sys.argv[1]
    else:
        # os.error("Usage: python Test_Vector_Search.py 'text to vectorize'")
        # return

        # print("Reading text from text file example_vector_input.txt...")
        with open("example_vector_input.txt", "r", encoding="utf-8") as f:
            text = f.read()

    api_key = "REDACTED_API_KEY"
    api_host = "https://ai-gateway.vitesco.io/v1"
    model = "cohere.embed-multilingual-v3"

    # If model has a known max context, chunk accordingly. Cohere example expects ~2048 chars.
    max_chunk_chars = 2000

    chunks = chunk_text_by_chars(text, max_chunk_chars=max_chunk_chars, overlap=200)
    # limit number of chunks to avoid very large requests
    max_chunks = 50

    #batch chunks in groups of max_chunks
    chunk_batches = [chunks[i:i + max_chunks] for i in range(0, len(chunks), max_chunks)]
    for batch_start, batch in enumerate(chunk_batches):
        # print(f"Processing batch of {len(batch)} chunks...")
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
            # print(f"Error: request failed: {e}\nResponse body:\n{body}")
            return
        except Exception as e:
            # print(f"Error: request failed: {e}")
            return

        try:
            resp_json = resp.json()
        except Exception as e:
            # print(f"Error: failed to parse response JSON: {e}\nRaw body:\n{resp.text}")
            return

        data = resp_json.get("data")
        if not data or not isinstance(data, list):
            # print(f"Error: unexpected response shape: {json.dumps(resp_json)[:1000]}")
            return

        # data should contain one embedding per input chunk in same order
        embeddings = []
        for item in data:
            emb = item.get("embedding")
            if emb is None:
                # print("Error: missing embedding in response item.")
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
        
        # print(f"Chunks sent: {len(chunks)}. Embeddings received: {len(embeddings)}")

        #search
        results = search(doc_embedding, top_k=5)
        # print("Search results:")
        # for result in results:
        #     print(f" - {result}")

        print_search_result_chunks(results, chunk_size=10)

if __name__ == "__main__":
    main()
