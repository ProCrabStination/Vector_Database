import requests
import sys
import json
import base64
import os
from pathlib import Path
from Ollama_Model_Info import get_model_context_length
# from ragas import evaluate
# from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

# Optional: your provided glossary/setup string
DEFAULT_SETUP = (""
    # "Glossary:\n"
    # "- AOI: Automated Optical Inspection (relevant only for real defects routed through repair)\n"
    # "- NTF: No Trouble Found\n"
    # "- ICT: In-Circuit Test\n"
    # "- FT/FIN: Functional Test (runs all functions and measures results)\n"
    # "- QC/QCC/Quality Cold: Functional test at cold temperature\n"
    # "- QH/QCH/Quality Hot: Functional test at hot temperature\n"
    # "- Strat/Strategy: Strategy test (checks memory/software discrepancies)\n"
    # "- RI: Run-In functional Test (extended high-temp stress functional test)\n"
    # "- X-Ray: Non-destructive test for hidden defects (relevant only for real defects routed through repair)\n"
    # "- Red boards: Boards designed to fail tests for bay verification\n"
    # "- Gold boards: Known good boards for verifying test bay functionality\n"
)

SUMMARY_MODEL = os.getenv("AI_SUMMARY_MODEL", "gemma3:4b")
SUMMARY_MODEL_MAX_INPUT_TOKENS = get_model_context_length(SUMMARY_MODEL)
# None = never time out (requests waits indefinitely for Ollama).
SUMMARY_REQUEST_TIMEOUT = None


def _normalize_ollama_host(api_host: str) -> str:
    """Normalize Ollama host so both ...:11434 and ...:11434/v1 are supported."""
    normalized = (api_host or "http://localhost:11434").rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3]
    return normalized


def _truncate_prompt(prompt: str, max_chars: int) -> str:
    """Trim very large prompts to reduce Ollama split-input scheduler pressure."""
    if prompt is None:
        return ""
    if max_chars <= 0 or len(prompt) <= max_chars:
        return prompt

    keep_head = int(max_chars * 0.7)
    keep_tail = max_chars - keep_head
    marker = "\n\n...[truncated for model stability]...\n\n"

    if keep_tail <= len(marker):
        return prompt[:max_chars]

    return prompt[:keep_head] + marker + prompt[-(keep_tail - len(marker)):]

# Usage: python Test_Script.py "your prompt" "API_KEY" "API_HOST" "MODEL" "ROUTING_HISTORY" "TEST_FAILURES" "UNIT_SUMMARY" "UNIT_ID"
def summarize_text(prompt, max_tokens=128000, image_paths=None, setup=None, temperature=0.2):
    """
    Summarize text (and optionally images) with a domain-specific system setup.

    Args:
        prompt (str): Text to summarize.
        max_tokens (int): Max tokens for the summary.
        image_paths (list[str] | None): Optional list of image file paths.
        setup (str | None): System/setup string. Defaults to DEFAULT_SETUP if None.
        temperature (float): Decoding temperature for the LLM.

    Returns: 
        str | None: The summary text, or None on error.
    """

    api_key = "None"
    api_host = _normalize_ollama_host(os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    model = SUMMARY_MODEL

    # Use caller-provided setup or default glossary
    setup = setup or DEFAULT_SETUP

    full_prompt = prompt

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Build image payload if provided (Ollama /api/chat expects base64 strings in message.images)
    message_images = []

    # Add images first if provided
    if image_paths:
        for image_path in image_paths:
            try:
                img_file = Path(image_path)
                if img_file.exists():
                    with open(img_file, 'rb') as f:
                        image_data = base64.b64encode(f.read()).decode('utf-8')

                    media_type_map = {
                        '.png': 'image/png',
                        '.jpg': 'image/jpeg',
                        '.jpeg': 'image/jpeg',
                        '.gif': 'image/gif',
                        '.webp': 'image/webp'
                    }
                    media_type = media_type_map.get(img_file.suffix.lower(), 'image/png')

                    message_images.append(image_data)
                    print(f"Added image to request: {image_path}")
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")

    try:
        num_predict = max(32, min(int(max_tokens), SUMMARY_MODEL_MAX_INPUT_TOKENS))

        user_message = {
            "role": "user",
            "content": full_prompt,
        }
        if message_images:
            user_message["images"] = message_images

        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": setup},
                user_message,
            ],
            "stream": False,
            "think": False,
            "keep_alive": "30m",
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
                "num_ctx": SUMMARY_MODEL_MAX_INPUT_TOKENS,
            }
        }

        response = requests.post(
            f"{api_host}/api/chat",
            headers=headers,
            data=json.dumps(data),
            timeout=SUMMARY_REQUEST_TIMEOUT
        )
        response.raise_for_status()

        response_data = response.json()
        message = response_data.get("message", {})
        chunk = message.get("content", "")
        done_reason = response_data.get("done_reason", "")
        if chunk and chunk.strip():
            return chunk.strip()

        # Some local model builds can return empty output with done_reason=length.
        if done_reason == "length":
            retry_prompt = _truncate_prompt(full_prompt, max(1200, max_tokens // 2))
            retry_data = {
                "model": model,
                "messages": [
                    {"role": "system", "content": setup},
                    {"role": "user", "content": retry_prompt},
                ],
                "stream": False,
                "think": False,
                "options": {
                    "temperature": max(0.0, min(temperature, 0.2)),
                    "num_predict": min(max(64, num_predict), 256),
                    "num_ctx": SUMMARY_MODEL_MAX_INPUT_TOKENS,
                }
            }
            retry_response = requests.post(
                f"{api_host}/api/chat",
                headers=headers,
                data=json.dumps(retry_data),
                timeout=SUMMARY_REQUEST_TIMEOUT
            )
            retry_response.raise_for_status()
            retry_json = retry_response.json()
            retry_message = retry_json.get("message", {})
            retry_chunk = retry_message.get("content", "")
            return retry_chunk.strip() if retry_chunk and retry_chunk.strip() else None

        return None

    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        if hasattr(e.response, "text"):
            body = e.response.text
            print(f"Response body: {body}")

            if "GGML_ASSERT(n_inputs < GGML_SCHED_MAX_SPLIT_INPUTS)" in body and len(full_prompt) > 1000:
                try:
                    safe_prompt = _truncate_prompt(full_prompt, max(1000, len(full_prompt) // 2))
                    retry_data = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": setup},
                            {"role": "user", "content": safe_prompt},
                        ],
                        "stream": False,
                        "think": False,
                        "options": {
                            "temperature": max(0.0, min(temperature, 0.2)),
                            "num_predict": min(num_predict, 256),
                            "num_ctx": SUMMARY_MODEL_MAX_INPUT_TOKENS,
                                    }
                    }
                    retry_response = requests.post(
                        f"{api_host}/api/chat",
                        headers=headers,
                        data=json.dumps(retry_data),
                        timeout=SUMMARY_REQUEST_TIMEOUT
                    )
                    retry_response.raise_for_status()
                    retry_json = retry_response.json()
                    retry_message = retry_json.get("message", {})
                    retry_chunk = retry_message.get("content", "")
                    return retry_chunk.strip() if retry_chunk else None
                except Exception as retry_exc:
                    print(f"Retry after GGML_ASSERT failed: {retry_exc}")
        return None
    except requests.exceptions.Timeout:
        print("Error: Ollama request timed out")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    """Test the summarize_text function with example inputs."""
    
    print("="*60)
    print("Testing LLM Summary Function")
    print("="*60)
    
    # Test 2: Markdown table summary
    print("\n" + "="*60)
    print("--- Test 2: Markdown Table Summary ---")
    table_markdown = """
| Test ID | Test Name | Status | Duration (ms) | Notes |
| --- | --- | --- | --- | --- |
| TC-001 | Login Validation | Passed | 1250 | All credentials validated correctly |
| TC-002 | Data Export | Failed | 3400 | Timeout occurred during large export |
| TC-003 | User Profile Update | Passed | 890 | Profile fields updated successfully |
| TC-004 | Payment Processing | Passed | 2100 | Transaction completed without errors |
| TC-005 | Report Generation | Failed | 5200 | Memory allocation error |
    """

    
    
    print("\nInput table:")
    print(table_markdown)
    print("\nGenerating summary...")
    table_summary = summarize_text(table_markdown, max_tokens=128000)
    print(f"\nTable summary result: {table_summary}")
    
    print("\n" + "="*60)
    print("Testing Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
