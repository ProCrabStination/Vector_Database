import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from docling_core.types.doc.document import DoclingDocument, PictureItem, TableItem

# Import custom modules
from Docling_Convert import process_file, is_supported_file
from Hybrid_Chunking import chunk_doclingdocument, EmbeddingTokenizer, HybridChunker
from Text_Vectorizing import Vectorize_Text
from Vector_Database import add_vector, document_exists
from LLM_Summary import summarize_text


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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
    
    logging.info(f"\nProcessing file: {file_path}")
    
    # Step 1: Convert the document
    converted = None
    try:
        converted = process_file(file_path, output_subdir)
    except Exception as e:
        logging.error(f"Error converting file {file_path}: {e}")
        return
    
    if converted is None:
        logging.error(f"Error: Document conversion failed: {file_path.name}")
        return
    
    doc = converted.document
    doc_full_path = str(file_path)
    
    logging.info(f"✓ Document converted: {file_path.name}")
    
    # Step 2 & 3: Chunk, vectorize and store
    
    print(f"\n{'-'*60}")
    print(f"Chunking and vectorizing document: {doc.name if hasattr(doc, 'name') else 'Unknown'}")
    print(f"Full path: {doc_full_path}")
    print(f"{'-'*60}")
    
    try:
        # Uncomment if you want to skip documents already in vector db
        # if document_exists(doc_full_path):
        #     print(f"Skipping {doc_full_path}: already exists in vector database.")
        #     return
        
        doc_name = doc.name if hasattr(doc, 'name') else 'Unknown'
        image_base_path = output_subdir / doc_name / "images"
        
        chunks = chunk_doclingdocument(doc, image_base_path=image_base_path)
        print(f"Total chunks created: {len(chunks)}")

        MAX_CHUNKS = 1000
        if len(chunks) > MAX_CHUNKS:
            logging.warning(f"Skipping {file_path} because it has {len(chunks)} chunks (>{MAX_CHUNKS})")
            return
        
        tokenizer = EmbeddingTokenizer()
        chunker = HybridChunker(tokenizer=tokenizer)
        
        chunks_to_vectorize = []
        
        for idx, chunk in enumerate(chunks):
            ctx_text = chunker.contextualize(chunk=chunk)
            num_tokens = tokenizer.count_tokens(text=ctx_text)
            doc_items_refs = [it.self_ref for it in chunk.meta.doc_items] if hasattr(chunk.meta, 'doc_items') else []
            
            # Extract and process tables
            has_table = False
            table_data = []
            table_markdown = []
            table_summary = None
            
            if hasattr(chunk.meta, 'doc_items'):
                for item in chunk.meta.doc_items:
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
                                    "markdown": markdown_str
                                })
                                
                        except Exception as e:
                            logging.warning(f"Could not extract table data from {item.self_ref}: {e}")
            
            if has_table:
                print(f"  ✓ Chunk {idx} contains {len(table_data)} table(s)")
                
                # Summarize each table using LLM and replace tables in text with summaries
                vectorize_text = ctx_text
                
                # Calculate available character budget
                MAX_CHUNK_SIZE = 2048  # Must stay under 2048 for embedding API cohere.embed-multilingual-v3
                current_chunk_size = len(ctx_text)
                
                for i, tbl in enumerate(table_data):
                    try:
                        table_char_count = len(tbl["markdown"])
                        
                        # Calculate how much space we can use for the summary
                        # Space freed by removing table + any remaining budget in chunk
                        chunk_remaining = MAX_CHUNK_SIZE - current_chunk_size
                        available_for_summary = table_char_count + chunk_remaining
                        
                        # Reserve some characters for the wrapper text "[Table N Summary: ]"
                        wrapper_overhead = len(f"[Table {i+1} Summary: ]")
                        max_summary_chars = max(50, available_for_summary - wrapper_overhead)
                        
                        # Create prompt with character limit
                        prompt_prefix = f"Summarize this table in no more than {max_summary_chars} characters. Be concise and capture key information:\n\n"
                        
                        # Pass markdown table with instruction to summarize_text
                        summary = summarize_text(prompt_prefix + tbl["markdown"])
                        
                        if summary:
                            # Enforce the character limit
                            if len(summary) > max_summary_chars:
                                summary = summary[:max_summary_chars-3] + "..."
                            
                            # Replace the table markdown with the summary
                            table_summary = f"[Table {i+1} Summary: {summary}]"
                            vectorize_text = vectorize_text.replace(tbl["markdown"], table_summary)
                            
                            # Update current chunk size calculation
                            current_chunk_size = current_chunk_size - table_char_count + len(table_summary)
                            
                            # Add summary to table data for metadata
                            tbl["summary"] = summary
                            
                            print(f"    Table {i+1}: {table_char_count} chars → {len(table_summary)} chars (limit: {max_summary_chars})")
                        else:
                            logging.warning(f"No summary returned for table {i+1} in chunk {idx}")
                            # If no summary, still replace table to reduce size
                            replacement = f"[Table {i+1}]"
                            vectorize_text = vectorize_text.replace(tbl["markdown"], replacement)
                            current_chunk_size = current_chunk_size - table_char_count + len(replacement)
                    except Exception as e:
                        logging.error(f"Error summarizing table {i+1} in chunk {idx}: {e}")
                        # On error, replace table with placeholder to reduce size
                        replacement = f"[Table {i+1}]"
                        vectorize_text = vectorize_text.replace(tbl["markdown"], replacement)
                        current_chunk_size = current_chunk_size - table_char_count + len(replacement)
                
                print(f"    Final chunk size: {len(vectorize_text)} chars (max: {MAX_CHUNK_SIZE})")
            else:
                vectorize_text = ctx_text
            
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
                        print(f"    Picture path created: {picture_path}")
                        picture_paths.append(picture_path)
                    elif hasattr(item, 'self_ref') and item.self_ref.startswith('#/pictures/'):
                        # Fallback: detect by self_ref pattern if isinstance doesn't work
                        has_picture = True
                        if item.self_ref not in doc_items_refs:
                            doc_items_refs.append(item.self_ref)
                        picture_refs.append(item.self_ref)
                        
                        safe_ref = item.self_ref.replace('#/', '').replace('/', '_').replace('\\', '_').replace(':', '_')
                        picture_path = str(image_base_path / f"{safe_ref}.png") if image_base_path else f"images/{safe_ref}.png"
                        print(f"    Picture path created: {picture_path}")
                        picture_paths.append(picture_path)
                        print(f"  ✓ Found picture via self_ref pattern: {item.self_ref} (type: {item_type})")
            
            if has_picture:
                print(f"  ✓ Chunk {idx} contains {len(picture_refs)} picture(s): {picture_refs}")
                
            
            # Get headings
            headings = chunk.meta.headings if hasattr(chunk.meta, 'headings') else []
            
            # Prepare metadata for database
            metadata = {
                "chunk_index": idx,
                "chunk_id": f"chunk_{idx:04d}",
                "token_count": num_tokens,
                "has_table": has_table,
                "table_data": table_data if has_table else None,  # Table as JSON with rows and markdown
                "table_summary": table_summary if has_table else None,  # LLM-generated summary
                "has_picture": has_picture,
                "picture_refs": picture_refs,
                "picture_paths": picture_paths,
                "doc_items_refs": doc_items_refs,
                "headings": headings,
            }
            
            chunks_to_vectorize.append({
                "index": idx,
                "text": vectorize_text,  # Use summary + text for tables, plain text otherwise
                "metadata": metadata
            })
            
        # After chunks_to_vectorize prepared
        print(f"Total chunks prepared for vectorization: {len(chunks_to_vectorize)}")
        
        BATCH_SIZE = 20
        total_vectors_added = 0
        
        for batch_start in range(0, len(chunks_to_vectorize), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(chunks_to_vectorize))
            batch = chunks_to_vectorize[batch_start:batch_end]
            
            print(f"\nVectorizing batch {batch_start//BATCH_SIZE + 1}: chunks {batch_start} to {batch_end - 1} ({len(batch)} chunks)")
            
            texts = [item["text"] for item in batch]
            
            try:
                embeddings = Vectorize_Text(texts)
                
                if embeddings is None:
                    logging.error(f"Failed to get embeddings for batch starting at {batch_start}")
                    continue
                
                if len(embeddings) != len(batch):
                    logging.error(f"Embedding count mismatch: got {len(embeddings)}, expected {len(batch)}")
                    continue
                
                for item, embedding in zip(batch, embeddings):
                    try:
                        vid = add_vector(
                            embedding=embedding,
                            doc_path=doc_full_path,
                            text=item["text"],
                            model="cohere.embed-multilingual-v3",
                            metadata=item["metadata"]
                        )
                        total_vectors_added += 1
                        
                        if total_vectors_added % 20 == 0:
                            print(f"  ✓ Added {total_vectors_added} vectors...")
                            
                    except Exception as e:
                        logging.error(f"Error adding vector for chunk {item['index']}: {e}")
                
                print(f"✓ Batch complete: {len(batch)} vectors added")
            except Exception as e:
                logging.error(f"Error vectorizing batch: {e}")
                continue
        
        print(f"\n✓ Document complete: {total_vectors_added} vectors added to database")
    
    except Exception as e:
        logging.error(f"Error processing document chunks: {e}")
        return
                
def main():
    args = sys.argv
    
    # Set up paths
    input_root = Path(__file__).parent / "Testing" / "Test_Documents"
    output_root = Path(__file__).parent / "Testing" / "Test_Documents_Markdown"
      
    
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
                and not document_exists(file_path)
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