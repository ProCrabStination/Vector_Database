import requests
import sys
import json
import base64
from pathlib import Path
import os

# Optional: your provided glossary/setup string
DEFAULT_SETUP = (
    "Glossary:\n"
    "- AOI: Automated Optical Inspection (relevant only for real defects routed through repair)\n"
    "- NTF: No Trouble Found\n"
    "- ICT: In-Circuit Test\n"
    "- FT/FIN: Functional Test (runs all functions and measures results)\n"
    "- QC/QCC/Quality Cold: Functional test at cold temperature\n"
    "- QH/QCH/Quality Hot: Functional test at hot temperature\n"
    "- Strat/Strategy: Strategy test (checks memory/software discrepancies)\n"
    "- RI: Run-In functional Test (extended high-temp stress functional test)\n"
    "- X-Ray: Non-destructive test for hidden defects (relevant only for real defects routed through repair)\n"
    "- Red boards: Boards designed to fail tests for bay verification\n"
    "- Gold boards: Known good boards for verifying test bay functionality\n"
)

# Usage: python Test_Script.py "your prompt" "API_KEY" "API_HOST" "MODEL" "ROUTING_HISTORY" "TEST_FAILURES" "UNIT_SUMMARY" "UNIT_ID"
def summarize_text(prompt, max_tokens=512, image_paths=None, setup=None, temperature=0.2):
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

    # Prefer environment variables or caller-provided values over hard-coded secrets
    api_key = os.getenv("AI_GATEWAY_API_KEY") or "REDACTED_API_KEY"
    api_host = os.getenv("AI_GATEWAY_HOST", "https://ai-gateway.vitesco.io/v1")
    model = os.getenv("AI_GATEWAY_MODEL", "anthropic.claude-3-haiku-20240307-v1:0")

    if not api_key:
        print("Error: API key not set. Set AI_GATEWAY_API_KEY environment variable.")
        return None

    # Use caller-provided setup or default glossary
    setup = setup or DEFAULT_SETUP

    full_prompt = prompt

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Build message content (multimodal support for the user message)
    message_content = []

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

                    message_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data
                        }
                    })
                    print(f"Added image to request: {image_path}")
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")

    # Add text prompt
    message_content.append({
        "type": "text",
        "text": full_prompt
    })

    # Assemble messages with system setup + policy first, then user multimodal content
    data = {
        "model": model,
        "messages": [
            # System message: glossary + strict summarization policy
            {"role": "system", "content": setup},
            # User message: multimodal content (images + text)
            {"role": "user", "content": message_content}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    try:
        response = requests.post(
            f"{api_host}/chat/completions",
            headers=headers,
            data=json.dumps(data)
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response body: {e.response.text}")
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
    table_summary = summarize_text(table_markdown, max_tokens=150)
    print(f"\nTable summary result: {table_summary}")
    
    print("\n" + "="*60)
    print("Testing Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
