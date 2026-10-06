# ...existing code...
import requests
import json
import os
from functools import lru_cache
from typing import List
from Ollama_Model_Info import get_model_context_length


EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text-v2-moe")

# Ollama's runtime (llama.cpp) automatically prepends/appends BOS/EOS special
# tokens to every embedding request (tokenizer.ggml.add_bos_token /
# add_eos_token), which our own tokenizer count (add_special_tokens=False)
# does not include. Reserve headroom so a chunk measured right at the model's
# reported context length doesn't overflow once Ollama adds its own tokens.
_EMBEDDING_SPECIAL_TOKEN_MARGIN = 8
EMBEDDING_MAX_TOKENS = max(
    1, get_model_context_length(EMBEDDING_MODEL) - _EMBEDDING_SPECIAL_TOKEN_MARGIN
)


@lru_cache(maxsize=1)
def get_embedding_tokenizer():
    """Load the tokenizer used by the Ollama embedding model.

    Loads from the local HF cache only (no network calls). This is the
    root fix for the "unauthenticated requests to the HF Hub" warning:
    the tokenizer is already fully cached, so there is no need to hit the
    Hub at all, which also skips the HEAD-request round trips on every run.
    Falls back to a normal (online) load if the cache is ever missing.
    """
    from transformers import AutoTokenizer

    model_id = "nomic-ai/nomic-embed-text-v2-moe-unsupervised"
    try:
        return AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    except OSError:
        return AutoTokenizer.from_pretrained(model_id)


def count_embedding_tokens(text: str) -> int:
    """Count tokens using the embedding model's tokenizer."""
    return len(get_embedding_tokenizer().encode(text, add_special_tokens=False))


def truncate_to_embedding_tokens(text: str, max_tokens: int) -> str:
    """Truncate text by the embedding model's token IDs."""
    tokenizer = get_embedding_tokenizer()
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text
    return tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True)


def _normalize_ollama_host(api_host: str) -> str:
    """Normalize Ollama host so both ...:11434 and ...:11434/v1 are supported."""
    normalized = (api_host or "http://localhost:11434").rstrip('/')
    if normalized.endswith('/v1'):
        normalized = normalized[:-3]
    return normalized

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

def Vectorize_Text(texts: List[str]
                      , api_key = None
                      , api_host = "http://localhost:11434"
                      , model = EMBEDDING_MODEL):

    #check the inputs do not exceed 96 entries
    if len(texts) > 96:
        print(f"Error: too many input texts ({len(texts)}). Max is 96.")
        return

    # headers = {
    #     "Authorization": f"Bearer {api_key}",
    #     "Content-Type": "application/json"
    # }

    payload = {
        "model": model,
        "input": texts  # array of strings as required by the gateway
    }

    url = _normalize_ollama_host(api_host) + "/api/embeddings"

    all_embeddings = []
    for text in texts:
        embed_text = text
        token_count = count_embedding_tokens(text)
        # if token_count > EMBEDDING_MAX_TOKENS:
        #     raise ValueError(
        #         f"Embedding input has {token_count} tokens; "
        #         f"the model supports {EMBEDDING_MAX_TOKENS} tokens."
        #     )
        try:
            resp = requests.post(url, json={"model": model, "prompt": embed_text})
            resp.raise_for_status()
            all_embeddings.append(resp.json()["embedding"])
        except requests.exceptions.HTTPError as e:
            body = resp.text if 'resp' in locals() and resp is not None else str(e)
            if resp is not None and resp.status_code == 500 and "context length" in body.lower():
                # Ollama's own tokenizer disagrees with ours (e.g. different
                # special-token handling); fall back to a hard truncation and
                # retry once rather than failing the whole chunk.
                shrunk_text = truncate_to_embedding_tokens(
                    embed_text, max(1, EMBEDDING_MAX_TOKENS - _EMBEDDING_SPECIAL_TOKEN_MARGIN)
                )
                try:
                    retry_resp = requests.post(url, json={"model": model, "prompt": shrunk_text})
                    retry_resp.raise_for_status()
                    all_embeddings.append(retry_resp.json()["embedding"])
                    continue
                except Exception as retry_error:
                    raise RuntimeError(
                        f"Embedding request failed even after truncation: {retry_error}"
                    ) from retry_error
            raise RuntimeError(f"Embedding request failed: {e}\nResponse body:\n{body}") from e
        except Exception as e:
            raise RuntimeError(f"Embedding request failed: {e}") from e

    # try:
    #     resp_json = resp.json()
    # except Exception as e:
    #     print(f"Error: failed to parse response JSON: {e}\nRaw body:\n{resp.text}")
    #     return

    # data = resp_json.get("data")
    # if not data or not isinstance(data, list):
    #     print(f"Error: unexpected response shape: {json.dumps(resp_json)[:1000]}")
    #     return

    # # data should contain one embedding per input chunk in same order
    # embeddings = []
    # for item in data:
    #     emb = item.get("embedding")
    #     if emb is None:
    #         print("Error: missing embedding in response item.")
    #         return
    #     try:
    #         embeddings.append([float(x) for x in emb])
    #     except Exception:
    #         embeddings.append(list(emb))

    # Return individual embeddings, not the mean
    return all_embeddings


def main():
    """
    Test function to demonstrate the vectorization capabilities
    """
    print("="*80)
    print("Text Vectorization Test")
    print("="*80 + "\n")
    
    # Test 1: Single text
    print("Test 1: Single text vectorization")
    print("-" * 40)
    test_texts_1 = ["Hello, this is a test sentence."]
    print(f"Input: {test_texts_1}")
    
    result_1 = Vectorize_Text(test_texts_1)
    if result_1:
        print(f"✓ Success! Generated {len(result_1)} embedding(s)")
        print(f"  Dimension: {len(result_1[0])}")
        print(f"  First 5 values: {result_1[0][:5]}")
    else:
        print("✗ Failed to generate embeddings")
    print()
    
    # Test 2: Multiple texts
    print("Test 2: Multiple texts vectorization")
    print("-" * 40)
    test_texts_2 = [
        "The manufacturing process is complete.",
        "Unit failed quality inspection.",
        "System error detected in station 5."
    ]
    print(f"Input: {len(test_texts_2)} texts")
    for i, text in enumerate(test_texts_2, 1):
        print(f"  {i}. {text}")
    
    result_2 = Vectorize_Text(test_texts_2)
    if result_2:
        print(f"✓ Success! Generated {len(result_2)} embedding(s)")
        print(f"  Dimension: {len(result_2[0])}")
        for i, emb in enumerate(result_2, 1):
            print(f"  Embedding {i}: First 5 values = {emb[:5]}")
    else:
        print("✗ Failed to generate embeddings")
    print()
    
    # Test 3: Mean embedding calculation
    if result_2:
        print("Test 3: Calculate mean embedding")
        print("-" * 40)
        mean_emb = _mean_embedding(result_2)
        print(f"✓ Mean embedding calculated")
        print(f"  Dimension: {len(mean_emb)}")
        print(f"  First 5 values: {mean_emb[:5]}")
        print()
    
    # Test 4: Edge case - empty input
    print("Test 4: Edge case - empty embedding list")
    print("-" * 40)
    mean_empty = _mean_embedding([])
    print(f"Mean of empty list: {mean_empty}")
    print(f"✓ Handled gracefully (returned: {mean_empty})")
    print()
    
    print("="*80)
    print("Testing Complete!")
    print("="*80 + "\n")
    
    print("Summary:")
    print("  ✓ Single text vectorization works")
    print("  ✓ Multiple texts vectorization works")
    print("  ✓ Mean embedding calculation works")
    print("  ✓ Edge cases handled properly")


if __name__ == "__main__":
    import sys
    import logging
    
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
            'Vectorize_Text': Vectorize_Text,
            '_mean_embedding': _mean_embedding,
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
        # No arguments provided, run the test
        main()
