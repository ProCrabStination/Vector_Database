# ...existing code...
import requests
import sys
import json
import os
from typing import List

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

def VectorizeDocument(document_path = r"C:\Users\uiv11567\source\repos\docling\tests\data\test documents\pptx\16x9 PowerPointTemplate1.pptx"
                      , api_key = "REDACTED_API_KEY"
                      , api_host = "https://ai-gateway.vitesco.io/v1"
                      , model = "cohere.embed-multilingual-v3"):

    if not os.path.exists(document_path):
        print(f"Error: document not found: {document_path}")
        return

    text = _extract_text_from_file(document_path) #replace with DOCLING markdowns + chunking
    if not text:
        print("Error: failed to extract text from document (pptx requires python-pptx).")
        return

    # If model has a known max context, chunk accordingly. Cohere example expects ~2048 chars.
    max_chunk_chars = 2000

    chunks = chunk_text_by_chars(text, max_chunk_chars=max_chunk_chars, overlap=200)
    # limit number of chunks to avoid very large requests
    max_chunks = 128
    if len(chunks) > max_chunks:
        print(f"Warning: document produced {len(chunks)} chunks; trimming to {max_chunks} chunks.")
        chunks = chunks[:max_chunks]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "input": chunks  # array of strings as required by the gateway
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

    print(f"Chunks sent: {len(chunks)}. Embeddings received: {len(embeddings)}")
    print(f"Document embedding length: {len(doc_embedding)}")
    print(f"First 10 values: {doc_embedding[:10]}")
    
    return doc_embedding