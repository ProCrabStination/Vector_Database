
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


def generate_progressive_keywords(original_keyword):
    """
    Automatically generate progressive keywords from a full phrase.
    Starts with the full phrase and gradually breaks it down into smaller components.
    Prioritizes meaningful phrases over individual words.
    """
    # Normalize the keyword
    words = original_keyword.strip().split()
    
    if len(words) == 0:
        return [original_keyword]
    
    progressive_keywords = []
    
    # 1. Add full phrase variations (with common punctuation fixes)
    full_phrase = " ".join(words)
    progressive_keywords.append(full_phrase)
    
    # Add hyphen/space variations for the full phrase
    if "-" in full_phrase:
        progressive_keywords.append(full_phrase.replace("-", " "))
    
    # 2. Generate meaningful combinations by removing words (prioritize longer phrases)
    # Start with removing 1 word, then 2 words, etc.
    for words_to_remove in range(1, len(words)):
        # Remove from start
        for start_remove in range(words_to_remove + 1):
            end_remove = words_to_remove - start_remove
            
            if start_remove + end_remove >= len(words):
                continue
                
            remaining_words = words[start_remove:len(words)-end_remove] if end_remove > 0 else words[start_remove:]
            
            if len(remaining_words) < 2:  # Skip single words in this phase
                continue
                
            phrase = " ".join(remaining_words)
            if phrase not in progressive_keywords:
                progressive_keywords.append(phrase)
                
                # Add hyphen/space variation
                if "-" in phrase:
                    space_version = phrase.replace("-", " ")
                    if space_version not in progressive_keywords:
                        progressive_keywords.append(space_version)
    
    # 3. Add meaningful two-word combinations
    for i in range(len(words) - 1):
        two_word = f"{words[i]} {words[i+1]}"
        if two_word not in progressive_keywords and len(two_word) > 4:
            progressive_keywords.append(two_word)
            
            # Add hyphen version if it makes sense
            if any(len(w) <= 4 for w in [words[i], words[i+1]]):
                hyphen_version = f"{words[i]}-{words[i+1]}"
                if hyphen_version not in progressive_keywords:
                    progressive_keywords.append(hyphen_version)
    
    # 4. Add individual significant words (length > 2 and not common words)
    common_words = {'of', 'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'a', 'an'}
    for word in words:
        if len(word) > 2 and word.lower() not in common_words and word not in progressive_keywords:
            progressive_keywords.append(word)
    
    # 5. Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for keyword in progressive_keywords:
        if keyword not in seen and len(keyword.strip()) > 1:
            seen.add(keyword)
            unique_keywords.append(keyword)
    
    return unique_keywords

def keyword_search_in_markdown_files(directory_path, keyword, max_chunk_chars=2000):
    """
    Search for the exact keyword in all markdown files in a directory.
    Returns list of (file_path, chunk_index, chunk_text) where found.
    """
    results = []
    
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return results
    
    # Find all markdown and text files in the directory
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(('.md', '.markdown', '.txt')):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Split content into chunks
                    chunks = chunk_text_by_chars(content, max_chunk_chars=max_chunk_chars, overlap=200)
                    
                    # Search for keyword in each chunk
                    for idx, chunk in enumerate(chunks):
                        if keyword.lower() in chunk.lower():  # Case-insensitive search
                            results.append((file_path, idx, chunk))
                            
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
    
    return results

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

    text = "71010 LS-Switch of Bank 2 measurement"

    api_key = "REDACTED_API_KEY"
    api_host = "https://ai-gateway.vitesco.io/v1"
    model = "cohere.embed-multilingual-v3"

    # # If model has a known max context, chunk accordingly. Cohere example expects ~2048 chars.
    # max_chunk_chars = 2000

    # chunks = chunk_text_by_chars(text, max_chunk_chars=max_chunk_chars, overlap=200)
    # # limit number of chunks to avoid very large requests
    # max_chunks = 50

    # #batch chunks in groups of max_chunks
    # chunk_batches = [chunks[i:i + max_chunks] for i in range(0, len(chunks), max_chunks)]
    # for batch_start, batch in enumerate(chunk_batches):
    #     # print(f"Processing batch of {len(batch)} chunks...")
    #     chunks = batch

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        # "input": chunks
        "input": text
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

    # try:
    #     doc_embedding = _mean_embedding(embeddings)
    # except Exception as e:
    #     print(f"Error combining embeddings: {e}")
    #     return
    
    doc_embedding = embeddings

    # print(f"Chunks sent: {len(chunks)}. Embeddings received: {len(embeddings)}")

    #Vector search
    results = search(doc_embedding, top_k=5)


    #Keyword search 
    # Progressive keyword search - start with full phrase, then break down gradually
    original_keyword = "71010 LS-Switch of Bank 2 measurement"
    
    # Generate progressive keyword list automatically
    progressive_keywords = generate_progressive_keywords(original_keyword)
    
    markdown_dir = "Test_Documents_Markdown"
    
    print("=" * 80)
    print("PROGRESSIVE KEYWORD SEARCH")
    print(f"Original keyword: '{original_keyword}'")
    print("=" * 80)
    print(f"Generated {len(progressive_keywords)} progressive keywords:")
    for i, kw in enumerate(progressive_keywords, 1):
        print(f"  {i:2d}. '{kw}'")
    print("=" * 80)
    
    # Find the first (most specific) keyword that has matches
    best_matches = []
    best_keyword = None
    
    for keyword in progressive_keywords:
        print(f"\nTrying keyword: '{keyword}'...")
        keyword_results = keyword_search_in_markdown_files(markdown_dir, keyword)
        
        if keyword_results:
            print(f"✅ Found {len(keyword_results)} matches for '{keyword}'!")
            best_matches = keyword_results
            best_keyword = keyword
            break  # Stop at first successful match (most specific)
        else:
            print(f"❌ No matches found for '{keyword}'")
    
    if best_matches:
        print(f"\n{'='*80}")
        print(f"BEST MATCH RESULTS")
        print(f"Using keyword: '{best_keyword}'")
        print(f"Found {len(best_matches)} matches")
        print(f"{'='*80}")
        
        # Show all matches for the best keyword
        for i, (file_path, chunk_idx, chunk_text) in enumerate(best_matches):
            print(f"\n--- Match {i+1} of {len(best_matches)} ---")
            print(f"File: {file_path}")
            print(f"Chunk {chunk_idx}:")
            
            # Highlight the matching keyword in the text (case-insensitive)
            highlighted_text = chunk_text
            keyword_lower = best_keyword.lower()
            text_lower = chunk_text.lower()
            
            if keyword_lower in text_lower:
                start_idx = text_lower.find(keyword_lower)
                end_idx = start_idx + len(keyword_lower)
                highlighted_text = (
                    chunk_text[:start_idx] + 
                    f"***{chunk_text[start_idx:end_idx]}***" + 
                    chunk_text[end_idx:]
                )
            
            print(f"{highlighted_text[:800]}...")  # Show more content
            print("-" * 60)
    else:
        print(f"\n{'='*80}")
        print("❌ NO MATCHES FOUND")
        print("None of the progressive keywords returned any results.")
        print(f"{'='*80}")

    # print("Search results:")
    # for result in results:
    #     print(f" - {result}")

    print_search_result_chunks(results, chunk_size=10)

if __name__ == "__main__":
    main()
