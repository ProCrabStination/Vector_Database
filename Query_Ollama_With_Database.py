import requests
import sys
import json
from LLM_Summary_Ollama import summarize_text
from Text_Vectorizing_Ollama import Vectorize_Text
from Vector_Database_Ollama import search_vectors, search_vectors_keywords
import os
import spacy
import nltk
# nltk.download('punkt')
# nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

def _normalize_ollama_host(api_host: str) -> str:
    """Normalize Ollama host so both ...:11434 and ...:11434/v1 are supported."""
    normalized = (api_host or "http://localhost:11434").rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3]
    return normalized

# Prefer environment variables or caller-provided values over hard-coded secrets
api_key = os.getenv("AI_GATEWAY_API_KEY") or "None"
api_host = _normalize_ollama_host(os.getenv("AI_GATEWAY_HOST", "http://localhost:11434"))
model = "gemma3:4b"

def get_model_context_length(model_name, api_host):
    try:
        response = requests.post(
            f"{api_host}/api/show",
            json={"model": model_name},
            timeout=30
        )
        response.raise_for_status()

        model_info = response.json().get("model_info", {})

        # Find the context_length field regardless of architecture
        for key, value in model_info.items():
            if key.endswith(".context_length"):
                return int(value)

    except Exception as e:
        print(f"Warning: Could not determine context length: {e}")

    # Fallback
    return 128000

context_token_limit = get_model_context_length(model, api_host)

# Usage: python MESSS_Query.py "path/to/input.md"
# Input markdown file should contain JSON: {"prompt": "...", "routing_history": "...", "test_failures": "...", "unit_summary": "...", "unit_id": "...", "chat_history": [...]}
def main():
    # Default values
    setup = ""
    prompt = ""
    chat_history = []

    # A bare "--only-response" flag (in any position) trims the JSON output
    # down to just {"ai_response": ...}, dropping extracted_text/tables/etc.
    only_response = "--only-response" in sys.argv
    positional_args = [a for a in sys.argv[1:] if a != "--only-response"]

    # Parse arguments if provided
    if positional_args:
        input_path = positional_args[0]

    else:
        # Use the example prompt next to this script so it works on any machine
        repo_root = os.path.dirname(os.path.abspath(__file__))
        input_path = os.path.join(repo_root, "Testing", "Example_Prompt.md")

    try:
        # Read JSON from markdown file
        with open(input_path, "r", encoding="utf-8") as f:
            file_content = f.read()
            data = json.loads(file_content)
        
        prompt = data.get('prompt', '')
        setup = data.get('setup', '')
        chat_history = data.get('chat_history', [])

        # gather keywords for vector search from prompt
        # Extract keywords from unit_summary markdown table
        import re

        prompt_keywords = []
        nlp = spacy.load("en_core_web_sm")

        doc = nlp(prompt)
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop:
                prompt_keywords.append(token.lemma_.lower())

    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit()

    # search over the faiss database with the prompt and keywords
    search_results_dict = search_vectors_keywords([prompt], top_k_per_query=15, keywords=prompt_keywords)
    search_results = search_results_dict.get('overall_top_k', [])

    
    # Extract results for both searches
    def extract_results(search_results):
        text_with_docs = []
        tables_with_docs = []
        pictures_with_docs = []
        extracted_text = []
        extracted_tables = []
        extracted_pictures = []
        extracted_filepaths = []
        for result in search_results:
            doc_path = result.get('doc_path', 'Unknown Document')
            headings = result.get('headings', [])
            if 'text' in result:
                text_with_docs.append({
                    'text': result['text'],
                    'doc_path': doc_path,
                    'headings': headings
                })
                extracted_text.append(result['text'])
            if 'metadata' in result and 'table_data' in result['metadata'] and result['metadata']['table_data']:
                for table in result['metadata']['table_data']:
                    table_with_doc = table.copy() if isinstance(table, dict) else {}
                    table_with_doc['doc_path'] = doc_path
                    table_with_doc['headings'] = headings
                    tables_with_docs.append(table_with_doc)
                extracted_tables.append(result['metadata']['table_data'])
            if result.get('pictures'):
                pic_paths = [pic['path'] for pic in result['pictures']]
                for pic_path in pic_paths:
                    pictures_with_docs.append({
                        'path': pic_path,
                        'doc_path': doc_path,
                        'headings': headings
                    })
                extracted_pictures.append(pic_paths)
            if 'doc_path' in result:
                extracted_filepaths.append(result['doc_path'])
        return {
            'text_with_docs': text_with_docs,
            'tables_with_docs': tables_with_docs,
            'pictures_with_docs': pictures_with_docs,
            'extracted_text': extracted_text,
            'extracted_tables': extracted_tables,
            'extracted_pictures': extracted_pictures,
            'extracted_filepaths': list(set(extracted_filepaths))
        }


    # results_1 = extract_results(search_results_1)
    # results_2 = extract_results(search_results_2)

    results = extract_results(search_results)

    # Combine results, deduplicating by doc_path and text
    def dedup_results(list1, list2, key_fields):
        seen = set()
        combined = []
        for item in list1 + list2:
            key = []
            for field in key_fields:
                val = item.get(field, None)
                if isinstance(val, list):
                    val = tuple(val)
                key.append(val)
            key = tuple(key)
            if key not in seen:
                seen.add(key)
                combined.append(item)
        return combined

    combined_text_with_docs = dedup_results(results['text_with_docs'], results['text_with_docs'], ['doc_path', 'text'])
    combined_tables_with_docs = dedup_results(results['tables_with_docs'], results['tables_with_docs'], ['doc_path', 'headings'])
    combined_pictures_with_docs = dedup_results(results['pictures_with_docs'], results['pictures_with_docs'], ['doc_path', 'path'])
    combined_filepaths = list(set(results['extracted_filepaths'] + results['extracted_filepaths']))

    # Limit total prompt size to ~680,000 unicode characters
    # MAX_PROMPT_CHARS = 680_000
    # Estimate static context size (prompt, summaries, etc.)
    static_context = (
        setup + "\n\n" + prompt + "\n\n" + "Extracted relevant information from the database:\n\n" 
    )
    # static_len = len(static_context)
    
#     # Convert texts and tables to markdown strings first
#     formatted_texts = []
#     for text in combined_text_with_docs:
#         headings_str = ' > '.join(text.get('headings', []))
#         formatted_texts.append(f"### Document {text['doc_path']}\n\n### Headings: {headings_str}\n\n### Content:\n{text['text']}\n")
    
#     formatted_tables = []
#     for table in combined_tables_with_docs:
#         # Use full markdown content instead of truncated summary
#         formatted_tables.append(f"### Document {table['doc_path']}\n\n### Table Content:\n{table.get('markdown', '')}\n")
    
#    # Calculate available budget
#     available = context_token_limit - len(word_tokenize(static_context))

#     # Calculate current lengths (token counts)
#     texts_len = sum(len(word_tokenize(t)) for t in formatted_texts)
#     tables_len = sum(len(word_tokenize(t)) for t in formatted_tables)
#     chat_history_len = sum(
#         len(word_tokenize(msg.get("content", "")))
#         for msg in chat_history
#     )

#     # Track which items are kept
#     texts_to_keep = list(range(len(formatted_texts)))
#     tables_to_keep = list(range(len(formatted_tables)))
#     chat_history_to_keep = list(range(len(chat_history)))

#     total_len = texts_len + tables_len + chat_history_len

#     if total_len > available:

#         # Always trim from the largest bucket first
#         while total_len > available:

#             lengths = {
#                 "texts": texts_len if texts_to_keep else 0,
#                 "tables": tables_len if tables_to_keep else 0,
#                 "chat": chat_history_len if chat_history_to_keep else 0,
#             }

#             largest = max(lengths, key=lengths.get)

#             if lengths[largest] == 0:
#                 break

#             if largest == "texts":
#                 texts_to_keep.pop()
#                 removed = formatted_texts.pop()
#                 removed_tokens = len(word_tokenize(removed))
#                 texts_len -= removed_tokens
#                 total_len -= removed_tokens

#             elif largest == "tables":
#                 tables_to_keep.pop()
#                 removed = formatted_tables.pop()
#                 removed_tokens = len(word_tokenize(removed))
#                 tables_len -= removed_tokens
#                 total_len -= removed_tokens

#             else:  # chat
#                 chat_history_to_keep.pop(0)
#                 removed = chat_history.pop(0)
#                 removed_tokens = len(
#                     word_tokenize(removed.get("content", ""))
#                 )
#                 chat_history_len -= removed_tokens
#                 total_len -= removed_tokens

#     limited_texts = formatted_texts
#     limited_tables = formatted_tables
#     limited_chat_history = chat_history

#     # Get the actual table objects that were kept
#     limited_table_objects = [combined_tables_with_docs[i] for i in sorted(tables_to_keep)]

#     #inverse the retrieved context order so that the most relevant is at the top
#     limited_texts.reverse()
#     limited_tables.reverse()
#     limited_chat_history.reverse()

#     # mix the texts and tables together
#     limited_context = []
#     for i in range(max(len(limited_texts), len(limited_tables))):
#         if i < len(limited_texts):
#             limited_context.append(limited_texts[i])
#         if i < len(limited_tables):
#             limited_context.append(limited_tables[i])

    # Calculate available budget
    available = context_token_limit - len(word_tokenize(static_context))

    # Reserve room for chat history
    chat_history_len = sum(
        len(word_tokenize(msg.get("content", "")))
        for msg in chat_history
    )

    available -= chat_history_len

    context_items = []

    for result in search_results:

        score = float(
            result.get("final_score",
                    result.get("exact_score", 0))
        )

        doc_path = result.get("doc_path", "Unknown Document")

        metadata = result.get("metadata") or {}

        headings = (
            metadata.get("headings")
            or result.get("headings")
            or []
        )

        headings_str = " > ".join(headings)

        # Text chunk
        if result.get("text"):

            content = (
                f"### Document {doc_path}\n"
                f"### Score: {score:.4f}\n"
                f"### Headings: {headings_str}\n\n"
                f"### Content:\n"
                f"{result['text']}\n"
            )

            context_items.append({
                "score": score,
                "type": "text",
                "content": content,
                "tokens": len(word_tokenize(content))
            })

        # Table chunks
        if metadata.get("has_table"):

            for table in (metadata.get("table_data") or []):

                content = (
                    f"### Document {doc_path}\n"
                    f"### Score: {score:.4f}\n\n"
                    f"### Table Content:\n"
                    f"{table.get('markdown', '')}\n"
                )

                context_items.append({
                    "score": score,
                    "type": "table",
                    "content": content,
                    "table_obj": table,
                    "tokens": len(word_tokenize(content))
                })

    # Highest score first
    context_items.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # Keep best chunks until budget reached
    kept_context = []
    used_tokens = 0

    for item in context_items:

        if used_tokens + item["tokens"] > available:
            continue

        kept_context.append(item)
        used_tokens += item["tokens"]

    # Put the BEST chunk closest to the user question
    kept_context.reverse()

    limited_context = [
        x["content"]
        for x in kept_context
    ]

    limited_texts = [
        x["content"]
        for x in kept_context
        if x["type"] == "text"
    ]

    limited_tables = [
        x["content"]
        for x in kept_context
        if x["type"] == "table"
    ]

    limited_table_objects = [
        x["table_obj"]
        for x in kept_context
        if x["type"] == "table"
    ]

    limited_chat_history = chat_history

    # print(
    #     f"Context kept: {len(kept_context)} chunks "
    #     f"({used_tokens:,}/{available:,} tokens)"
    # )

    # print("\nTop kept scores:")
    # for item in kept_context[-10:]:
    #     print(
    #         f"{item['score']:.4f} "
    #         f"{item['type']}"
    #     )

    full_prompt = (
        f"""
        CHAT HISTORY:
        =========================
        {chr(10).join([f'{msg['role']}: {msg['content']}' for msg in limited_chat_history])}
        =========================

        RETRIEVED CONTEXT:
        =========================
        \n{chr(10).join(limited_context)}
        =========================

        FINAL TASK:
        Answer this users question using the retrieved context and your knowledge of engineering documents. Provide relevant information and paths to the relevant documents.:

        USER QUESTION:
        =========================
        {prompt}
        """
    )
   
    # Save full prompt to text file in Temp_Inputs folder
    from datetime import datetime
    temp_inputs_folder = os.path.join(os.path.dirname(__file__), "Temp_Inputs")
    os.makedirs(temp_inputs_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_file_path = os.path.join(temp_inputs_folder, f"full_prompt_{timestamp}.txt")
    with open(prompt_file_path, "w", encoding="utf-8") as f:
        f.write(f"=== FULL PROMPT ===\n\n{full_prompt}\n\n")
        f.write(f"=== PROMPT LENGTH ===\n{len(full_prompt)} characters\n")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        num_ctx = context_token_limit
        num_predict = max(32, min(int(context_token_limit), 8192))
        temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.0"))

        user_message = {
            "role": "user",
            "content": full_prompt,
        }
        # if message_images:
        #     user_message["images"] = message_images

        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": setup},
                user_message,
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": 1.0,
                "num_predict": num_predict,
                "num_ctx": num_ctx,
                "top_k": 64,
                "top_p": 0.95,
            }
        }

        response = requests.post(
            f"{api_host}/api/chat",
            headers=headers,
            data=json.dumps(data)
        )
        response.raise_for_status()

        result = response.json()
        message = result.get("message", {})
        if message.get("thinking"):
            ai_response = message["thinking"] + "\n\n" + message["content"]
        else:
            ai_response = message["content"]

        # Prepare the output as a JSON structure
        if only_response:
            output = {"ai_response": ai_response}
        else:
            output = {
                "ai_response": ai_response,
                "extracted_text": limited_texts,  # Text with document paths and headings
                "extracted_filepaths": combined_filepaths,  # Remove duplicates
                "extracted_tables": limited_table_objects,  # Return actual table objects with all properties
                "extracted_pictures": combined_pictures_with_docs  # Pictures with document paths and headings
            }

        if positional_args:
            # Print as JSON for the C# controller to parse
            print(json.dumps(output))
        else:
            # Print nicely for the console
            print("\nExtracted Texts with Document Info:")
            for text_info in limited_texts:
                print(f" - {text_info}")

            print("\nExtracted Filepaths:")
            for filepath in combined_filepaths:
                print(f" - {filepath}")

            print("\nExtracted Tables:")
            for i, table_info in enumerate(limited_tables):
                print(f"\n=== Table {i+1} ({len(table_info)} chars) ===")
                print(table_info)

            print("\nExtracted Pictures:")
            for picture_info in combined_pictures_with_docs:
                print(f" - {picture_info}")

            print("AI Response:\n", ai_response)

    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        if hasattr(e.response, "text"):
            body = e.response.text
            print(f"Response body: {body}")
        return None
    except requests.exceptions.Timeout:
        print("Error: Ollama request timed out")
        return None
    except Exception as e:
        if only_response:
            error_output = {"ai_response": f"Error: {e}"}
        else:
            error_output = {
                "ai_response": f"Error: {e}",
                "extracted_text": [],
                "extracted_filepaths": [],
                "extracted_tables": [],
                "extracted_pictures": []
            }
        print(json.dumps(error_output))

if __name__ == "__main__":
    main()