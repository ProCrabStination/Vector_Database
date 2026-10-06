import os
import sys
import json
import logging
import pandas as pd
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
from docling_core.types.doc.document import DoclingDocument, PictureItem, TableItem, RefItem
from docling_core.types.doc.labels import DocItemLabel
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE


def _excel_safe(value):
    """Strip control characters openpyxl/Excel XML cannot store (e.g. from OCR'd text)."""
    if isinstance(value, str):
        return ILLEGAL_CHARACTERS_RE.sub("", value)
    return value

# Import custom modules
from Docling_Convert_Ollama import process_file, is_supported_file
from Hybrid_Chunking_Ollama import (
    chunk_doclingdocument,
    EmbeddingTokenizer,
    HybridChunker,
    PicturePathSerializerProvider,
    TablePlaceholderSerializer,
)
from Text_Vectorizing_Ollama import (
    EMBEDDING_MAX_TOKENS,
    Vectorize_Text,
    count_embedding_tokens,
)
from Vector_Database_Ollama import (
    DB_PATH,
    add_vector,
    document_exists,
    processed_chunk_indices,
)
from LLM_Summary_Ollama import summarize_text


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

import subprocess
import time
import os

OLLAMA_APP = r"C:\Users\rperry\AppData\Local\Programs\Ollama\ollama app.exe"
PDF_PAGES_PER_CHUNK = 10
SUMMARY_REQUEST_MAX_CHARS = 10000


def split_pdf_into_page_chunks(pdf_path: Path, chunk_output_dir: Path, pages_per_chunk: int = PDF_PAGES_PER_CHUNK) -> List[Path]:
    """Split a PDF into fixed-size page chunks and return the chunk file paths."""
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError as e:
        raise ImportError(
            "PDF chunking requires pypdf. Install it with: pip install pypdf"
        ) from e

    chunk_output_dir.mkdir(parents=True, exist_ok=True)

    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    if total_pages == 0:
        logging.warning(f"PDF has no pages, skipping split: {pdf_path}")
        return []

    chunk_files: List[Path] = []
    for start_idx in range(0, total_pages, pages_per_chunk):
        end_idx = min(start_idx + pages_per_chunk, total_pages)
        writer = PdfWriter()

        for page_idx in range(start_idx, end_idx):
            writer.add_page(reader.pages[page_idx])

        chunk_name = f"{pdf_path.stem}_pages_{start_idx + 1:04d}-{end_idx:04d}.pdf"
        chunk_path = chunk_output_dir / chunk_name
        with chunk_path.open("wb") as chunk_file:
            writer.write(chunk_file)

        chunk_files.append(chunk_path)

    logging.info(
        f"Split PDF into {len(chunk_files)} file(s) of up to {pages_per_chunk} pages: {pdf_path.name}"
    )
    return chunk_files

def restart_ollama():
    # Kill any existing Ollama processes
    subprocess.run(
        ["taskkill", "/F", "/IM", "ollama.exe"],
        capture_output=True,
        text=True
    )

    subprocess.run(
        ["taskkill", "/F", "/IM", "ollama app.exe"],
        capture_output=True,
        text=True
    )

    time.sleep(5)

    # Verify executable exists
    if not os.path.exists(OLLAMA_APP):
        raise FileNotFoundError(f"Not found: {OLLAMA_APP}")

    # Start Ollama App
    subprocess.Popen([OLLAMA_APP])

    # Wait for startup
    time.sleep(30)

    # Verify process is running
    for _ in range(12):
        result = subprocess.run(
            ["tasklist"],
            capture_output=True,
            text=True
        )

        if ("ollama.exe" in result.stdout.lower()
                or "ollama app.exe" in result.stdout.lower()):
            print("Ollama restarted")
            return

        time.sleep(5)

    raise RuntimeError("Ollama failed to start")

# Helper function to process one document fully
def process_single_document(file_path: Path, output_root: Path, input_root: Path):
    if not file_path.exists():
        logging.error(f"File not found: {file_path}")
        return
    
    if not is_supported_file(file_path):
        logging.error(f"File type not supported: {file_path.suffix}")
        return
    
    # Determine output subdirectory maintaining structure relative to input_root
    try:
        relative_path = file_path.relative_to(input_root).parent
    except ValueError:
        relative_path = Path(".")
    output_subdir = output_root / relative_path

    files_to_process: List[Path] = [file_path]
    if file_path.suffix.lower() == ".pdf":
        chunk_output_dir = output_subdir / "_pdf_chunks" / file_path.stem
        try:
            files_to_process = split_pdf_into_page_chunks(file_path, chunk_output_dir)
        except Exception as e:
            logging.error(f"Error splitting PDF into {PDF_PAGES_PER_CHUNK}-page chunks: {file_path} ({e})")
            return

        if not files_to_process:
            logging.error(f"No PDF chunks produced for: {file_path}")
            return

        unprocessed_files = []
        for file_unit in files_to_process:
            if document_exists(file_unit):
                print(f"Skipping processed 10-page PDF unit before conversion: {file_unit}")
                continue
            unprocessed_files.append(file_unit)
        files_to_process = unprocessed_files

        if not files_to_process:
            print(f"Skipping {file_path}: all 10-page PDF units are already processed.")
            return

    for file_unit in files_to_process:
        logging.info(f"\nProcessing file: {file_unit}")

        # Step 1: Convert the document
        converted = None
        try:
            converted = process_file(file_unit, output_subdir)
        except Exception as e:
            logging.error(f"Error converting file {file_unit}: {e}")
            continue

        if converted is None:
            logging.error(f"Error: Document conversion failed: {file_unit.name}")
            continue

        doc = converted.document
        doc_full_path = str(file_unit)

        logging.info(f"✓ Document converted: {file_unit.name}")

        # Step 2 & 3: Chunk, vectorize and store

        print(f"\n{'-'*60}")
        print(f"Chunking and vectorizing document: {doc.name if hasattr(doc, 'name') else 'Unknown'}")
        print(f"Full path: {doc_full_path}")
        print(f"{'-'*60}")

        try:
            doc_name = doc.name if hasattr(doc, 'name') else 'Unknown'
            image_base_path = output_subdir / doc_name / "images"
            
            chunks = chunk_doclingdocument(doc, image_base_path=image_base_path)
            print(f"Total chunks created: {len(chunks)}")

            processed_indices = processed_chunk_indices(doc_full_path)
            chunk_indices = list(range(len(chunks)))
            skipped_count = sum(1 for idx in chunk_indices if idx in processed_indices)
            if skipped_count:
                remaining_chunks = [
                    (idx, chunk) for idx, chunk in zip(chunk_indices, chunks)
                    if idx not in processed_indices
                ]
                chunk_indices = [idx for idx, _ in remaining_chunks]
                chunks = [chunk for _, chunk in remaining_chunks]
                print(
                    f"Skipping {skipped_count} already processed section(s) "
                    f"for {doc_full_path}"
                )
            if not chunks:
                print(f"Skipping {doc_full_path}: all sections are already processed.")
                continue

            MAX_CHUNKS = 1000
            if len(chunks) > MAX_CHUNKS:
                logging.warning(f"Skipping {file_unit} because it has {len(chunks)} chunks (>{MAX_CHUNKS})")
                continue

            tokenizer = EmbeddingTokenizer()
            chunker = HybridChunker(
                tokenizer=tokenizer,
                serializer_provider=PicturePathSerializerProvider(base_path=image_base_path),
            )

            chunks_to_vectorize = []

            for position, chunk in enumerate(chunks):
                chunk_index = chunk_indices[position]
                ctx_text = chunker.contextualize(chunk=chunk)
                num_tokens = tokenizer.count_tokens(text=ctx_text)
                doc_items_refs = [it.self_ref for it in chunk.meta.doc_items] if hasattr(chunk.meta, 'doc_items') else []
                headings = getattr(chunk.meta, 'headings', None) or []

                # Extract and process tables
                has_table = False
                table_data = []
                table_markdown = []
                table_summary = None

                if hasattr(chunk.meta, 'doc_items'):
                    for raw_item in chunk.meta.doc_items:
                        # HybridChunker's merge step can hand back doc_items that lost
                        # their concrete subclass (e.g. a table item typed as the
                        # generic DocItem instead of TableItem, even though its label
                        # says TABLE). isinstance() then misses it, and even if it
                        # didn't, that generic object has no .data.grid. Resolve the
                        # real item from the source document by self_ref instead.
                        item = raw_item
                        if not isinstance(item, TableItem) and getattr(item, 'label', None) == DocItemLabel.TABLE:
                            try:
                                item = RefItem(cref=item.self_ref).resolve(doc)
                            except Exception as e:
                                logging.warning(f"Could not resolve table item {item.self_ref}: {e}")

                        if isinstance(item, TableItem):
                            has_table = True

                            try:
                                # Extract table data and convert to markdown
                                table_rows = []
                                if hasattr(item, 'data') and item.data:
                                    grid = item.data.grid

                                    for row in grid:
                                        row_data = []
                                        for cell in row:
                                            cell_text = cell.text if hasattr(cell, 'text') else str(cell)
                                            # Clean up cell text
                                            cell_text = cell_text.strip().replace('\n', ' ')
                                            row_data.append(cell_text)
                                        table_rows.append(row_data)

                                # Convert to markdown table format
                                if table_rows:
                                    markdown_table = []
                                    # Header row
                                    markdown_table.append("| " + " | ".join(table_rows[0]) + " |")
                                    # Separator row
                                    markdown_table.append("| " + " | ".join(["---"] * len(table_rows[0])) + " |")
                                    # Data rows
                                    for row in table_rows[1:]:
                                        markdown_table.append("| " + " | ".join(row) + " |")

                                    markdown_str = "\n".join(markdown_table)
                                    table_markdown.append(markdown_str)

                                    # Store table data as JSON (rows format)
                                    table_data.append({
                                        "ref": item.self_ref,
                                        "rows": table_rows,
                                        "markdown": markdown_str,
                                        # The chunker's table serializer emits this exact
                                        # placeholder into ctx_text (see TablePlaceholderSerializer),
                                        # so it can be located and replaced reliably.
                                        "placeholder": TablePlaceholderSerializer.placeholder_for(item),
                                    })

                            except Exception as e:
                                logging.warning(f"Could not extract table data from {item.self_ref}: {e}")

                # has_table must reflect whether we actually extracted usable
                # rows, not just whether a table-labeled item was seen -
                # otherwise a table with an empty/unresolvable grid leaves
                # has_table=True but table_data=[], crashing table_data[0] below.
                has_table = bool(table_data)

                if has_table:
                    print(f"  ✓ Chunk {chunk_index} contains {len(table_data)} table(s)")

                    #create the directory if it doesn't exist
                    table_base_path = output_subdir / doc_name / "tables"
                    table_base_path.mkdir(parents=True, exist_ok=True)

                    with pd.ExcelWriter(
                        f"{table_base_path}/{file_unit.stem}_table{chunk_index}.xlsx",
                        engine="openpyxl"
                    ) as writer:
                        # One sheet per table so a multi-table chunk doesn't
                        # lose every table after the first.
                        for t_idx, tbl in enumerate(table_data):
                            sheet_name = f"Table{t_idx + 1}"
                            table_df = pd.DataFrame(tbl["rows"]).map(_excel_safe)

                            # Write context in A1
                            pd.DataFrame([[_excel_safe(ctx_text)]]).to_excel(
                                writer,
                                sheet_name=sheet_name,
                                header=False,
                                index=False,
                                startrow=0
                            )

                            # Write table starting at row 3 (blank row 2)
                            table_df.to_excel(
                                writer,
                                sheet_name=sheet_name,
                                index=False,
                                startrow=2
                            )

                    # Summarize each table using LLM and replace tables in text with summaries
                    vectorize_text = ctx_text

                    current_chunk_tokens = count_embedding_tokens(ctx_text)

                    for i, tbl in enumerate(table_data):
                        try:
                            placeholder = tbl["placeholder"]
                            table_pos = vectorize_text.find(placeholder)
                            if table_pos == -1:
                                raise ValueError(
                                    f"Table {i+1} placeholder not found in chunk {chunk_index}; "
                                    "the chunker may not be using TablePlaceholderSerializer."
                                )

                            table_token_count = count_embedding_tokens(placeholder)

                            # Calculate how much space we can use for the summary
                            # Space freed by removing table + any remaining budget in chunk
                            chunk_remaining = EMBEDDING_MAX_TOKENS - current_chunk_tokens
                            available_for_summary = table_token_count + chunk_remaining

                            # Reserve tokens for the wrapper text "[Table N Summary: ]"
                            wrapper_overhead = count_embedding_tokens(f"[Table {i+1} Summary: ]")
                            max_summary_tokens = max(16, available_for_summary - wrapper_overhead)

                            # Give the summary model a generous character target; validate the
                            # returned summary against the embedding model's token limit below.
                            prompt_prefix = (
                                f"Summarize this table and its purpose in no more than {SUMMARY_REQUEST_MAX_CHARS} characters. "
                                "Preserve the important specifications, identifiers, quantities, ranges, and units. "
                            )

                            # Use the previous/next chunk's text as surrounding context
                            context_before = chunker.contextualize(chunk=chunks[position - 1]) if position > 0 else ""
                            context_after = chunker.contextualize(chunk=chunks[position + 1]) if position < len(chunks) - 1 else ""
                            prompt_prefix = f"{prompt_prefix}\n Section Headings: {', '.join(headings)}\n Chunk Before Table: {context_before}\n Chunk After Table: {context_after}\n Table To Summarize:\n"

                            table_markdown = tbl["markdown"]

                            empty_summary_marker = f"[Table {i+1} Summary: ]"
                            payload_without_summary = vectorize_text.replace(
                                placeholder,
                                empty_summary_marker,
                            )
                            payload_without_summary_tokens = count_embedding_tokens(
                                payload_without_summary
                            )
                            effective_embedding_limit = max(
                                EMBEDDING_MAX_TOKENS,
                                payload_without_summary_tokens,
                            )
                            if effective_embedding_limit > EMBEDDING_MAX_TOKENS:
                                print(
                                    f"Chunk {chunk_index} contextualized payload is "
                                    f"{payload_without_summary_tokens} tokens; using "
                                    f"{effective_embedding_limit} as the working summary limit."
                                )

                            # print(f"{prompt_prefix + table_markdown}")

                            summary_prompt = prompt_prefix + table_markdown
                            print("\n" + "=" * 100)
                            print(f"TABLE SUMMARY - chunk {chunk_index}, table {i + 1}")
                            print("PROMPT SENT TO SUMMARY MODEL:")
                            print(summary_prompt)
                            print("=" * 100)
                            summary = summarize_text(summary_prompt)
                            print("SUMMARY MODEL RESPONSE:")
                            print(summary if summary else "<EMPTY RESPONSE>")
                            if not summary:
                                raise RuntimeError(
                                    f"Failed to summarize table {i+1} in chunk {chunk_index}."
                                )
                            print(
                                f"SUMMARY TOKEN COUNT: {count_embedding_tokens(summary)}"
                            )

                            print(f"  ✓ Table {i+1} summary generated.")

                            # Replace the table placeholder with the summary
                            table_summary = f"[Table {i+1} Summary: {summary}]"
                            vectorize_text = vectorize_text.replace(
                                placeholder,
                                table_summary,
                            )

                            current_chunk_tokens = (
                                current_chunk_tokens
                                - table_token_count
                                + count_embedding_tokens(table_summary)
                            )
                            tbl["summary"] = summary
                            print(f"    Table {i+1}: {table_token_count} tokens → {count_embedding_tokens(table_summary)} tokens (limit: {max_summary_tokens})")

                        except Exception as e:
                            raise RuntimeError(
                                f"Error summarizing table {i+1} in chunk {chunk_index}: {e}"
                            ) from e

                else:
                    vectorize_text = ctx_text

                # Validate the exact text that will be sent to the embedding model.
                # This must happen after table replacement, not on the original context.
                token_count = count_embedding_tokens(vectorize_text)
                # if token_count > EMBEDDING_MAX_TOKENS:
                #     raise ValueError(
                #         f"Chunk {chunk_index} embedding payload has {token_count} tokens; "
                #         f"the model supports {EMBEDDING_MAX_TOKENS} tokens."
                #     )
                print(
                    f"    Final embedding payload: {token_count} tokens "
                    f"(max: {EMBEDDING_MAX_TOKENS})"
                )

                # Check if chunk contains a picture and collect picture references
                has_picture = False
                picture_refs = []
                picture_paths = []
                if hasattr(chunk.meta, 'doc_items'):
                    for item in chunk.meta.doc_items:
                        # Debug: Check item type
                        item_type = type(item).__name__
                        if isinstance(item, PictureItem):
                            has_picture = True
                            # Add the picture's self_ref if not already in doc_items_refs
                            if item.self_ref not in doc_items_refs:
                                doc_items_refs.append(item.self_ref)
                            picture_refs.append(item.self_ref)

                            # Extract picture path from the markdown if present
                            # Format: ![ref](path)
                            safe_ref = item.self_ref.replace('#/', '').replace('/', '_').replace('\\', '_').replace(':', '_')
                            picture_path = str(image_base_path / f"{safe_ref}.png") if image_base_path else f"images/{safe_ref}.png"
                            picture_paths.append(picture_path)
                        elif hasattr(item, 'self_ref') and item.self_ref.startswith('#/pictures/'):
                            # Fallback: detect by self_ref pattern if isinstance doesn't work
                            has_picture = True
                            if item.self_ref not in doc_items_refs:
                                doc_items_refs.append(item.self_ref)
                            picture_refs.append(item.self_ref)

                            safe_ref = item.self_ref.replace('#/', '').replace('/', '_').replace('\\', '_').replace(':', '_')
                            picture_path = str(image_base_path / f"{safe_ref}.png") if image_base_path else f"images/{safe_ref}.png"
                            picture_paths.append(picture_path)

                pictures = [
                    {"ref": ref, "path": path}
                    for ref, path in zip(picture_refs, picture_paths)
                ]

                if has_picture:
                    print(f"  ✓ Chunk {chunk_index} contains {len(picture_refs)} picture(s): {picture_refs}")

                # Prepare metadata for database
                metadata = {
                        "chunk_index": chunk_index,
                        "chunk_id": f"chunk_{chunk_index:04d}",
                    "token_count": token_count,
                    "has_table": has_table,
                    "table_data": table_data if has_table else None,  # Table as JSON with rows and markdown
                    "table_summary": table_summary if has_table else None,  # LLM-generated summary
                    "has_picture": has_picture,
                    "doc_items_refs": doc_items_refs,
                    "headings": headings,
                }

                chunks_to_vectorize.append({
                    "index": chunk_index,
                    "text": vectorize_text,  # Use summary + text for tables, plain text otherwise
                    "metadata": metadata,
                    "pictures": pictures,
                })

            # After chunks_to_vectorize prepared
            print(f"Total chunks prepared for vectorization: {len(chunks_to_vectorize)}")

            BATCH_SIZE = 20
            total_vectors_added = 0

            def print_failed_chunk(item, error):
                print("\n" + "=" * 80)
                print(f"FAILED CHUNK: {item['index']}")
                print(f"Token count: {count_embedding_tokens(item['text'])}")
                print(f"Metadata: {json.dumps(item['metadata'], indent=2, default=str)}")
                print("Text:")
                print(item["text"])
                print(f"Error: {error}")
                print("=" * 80 + "\n")

            def store_vector(item, embedding):
                nonlocal total_vectors_added
                try:
                    add_vector(
                        embedding=embedding,
                        doc_path=doc_full_path,
                        text=item["text"],
                        model="nomic-embed-text-v2-moe",
                        metadata=item["metadata"],
                        pictures=item["pictures"]
                    )
                    total_vectors_added += 1

                    if total_vectors_added % 20 == 0:
                        print(f"  ✓ Added {total_vectors_added} vectors...")
                except Exception as e:
                    print_failed_chunk(item, e)
                    raise RuntimeError(
                        f"Error adding vector for chunk {item['index']}: {e}"
                    ) from e

            for batch_start in range(0, len(chunks_to_vectorize), BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, len(chunks_to_vectorize))
                batch = chunks_to_vectorize[batch_start:batch_end]

                print(f"\nVectorizing batch {batch_start//BATCH_SIZE + 1}: chunks {batch_start} to {batch_end - 1} ({len(batch)} chunks)")

                texts = [item["text"] for item in batch]

                try:
                    embeddings = Vectorize_Text(texts)

                    if len(embeddings) != len(batch):
                        raise RuntimeError(
                            f"Embedding count mismatch: got {len(embeddings)}, expected {len(batch)}"
                        )

                    for item, embedding in zip(batch, embeddings):
                        store_vector(item, embedding)

                    print(f"✓ Batch complete: {len(batch)} vectors added")
                except Exception as e:
                    print(
                        f"Batch {batch_start//BATCH_SIZE + 1} failed ({e}); "
                        "retrying each chunk individually."
                    )

                    for item in batch:
                        try:
                            individual_embeddings = Vectorize_Text([item["text"]])
                            if len(individual_embeddings) != 1:
                                raise RuntimeError(
                                    f"Expected one embedding, got {len(individual_embeddings)}"
                                )
                            store_vector(item, individual_embeddings[0])
                        except Exception as individual_error:
                            print_failed_chunk(item, individual_error)
                            raise RuntimeError(
                                f"Error vectorizing chunk {item['index']} individually: "
                                f"{individual_error}"
                            ) from individual_error

            print(f"\n✓ Document complete: {total_vectors_added} vectors added to database")

        except Exception as e:
            raise RuntimeError(f"Error processing document chunks for {file_unit}: {e}") from e
    


def can_write_db():
    try:
        conn = sqlite3.connect(DB_PATH, timeout=1)
        cur = conn.cursor()

        cur.execute("BEGIN IMMEDIATE")
        conn.rollback()
        conn.close()

        return True

    except sqlite3.Error as e:
        print(f"DB not writable: {e}")
        return False

                
def main():
    args = sys.argv
    
    # Set up paths
    input_root = Path(__file__).parent / "Testing//Test_Documents"
    output_root = Path(__file__).parent / "Testing//Test_Documents_Markdown"
    
    # check that the database is writable
    if not can_write_db():
        logging.error(f"Database {DB_PATH} is not writable. Please check permissions.")
        return
    
    if len(args) >= 2:
        # Process single file
        single_doc_path = Path(args[1])
        process_single_document(single_doc_path, output_root, input_root)
        
    else:
        # Process all files recursively in input directory, one at a time
        if not input_root.exists():
            logging.error(f"Input directory not found: {input_root}")
            return
        
        processed_count = 0
        
        for file_path in input_root.rglob('*'):
            if (
                file_path.is_file()
                and not file_path.name.startswith('~$')
                and is_supported_file(file_path)
                and file_path.suffix.lower() not in ['.xls', '.xlsx', '.xlsm']
            ):
                print(f"\n{'='*60}")
                print(f"Processing file: {file_path}")
                print(f"{'='*60}")

                process_single_document(file_path, output_root, input_root)

                processed_count += 1
        
        logging.info(f"\nTotal files processed: {processed_count}")
        
    print(f"\n{'='*60}")
    print(f"Pipeline Complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()