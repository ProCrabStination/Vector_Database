# The code here was assembled from the Docling library example advanced_chunking_and_serialization.ipynb

from docling_core.types.doc.document import DoclingDocument
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from typing import Iterable, Optional, List, Any
from docling_core.transforms.chunker.base import BaseChunk
from docling_core.transforms.chunker.hierarchical_chunker import DocChunk
from docling_core.types.doc.labels import DocItemLabel
from docling_core.types.doc.document import PictureItem, TableItem
from docling_core.transforms.serializer.base import (
    BaseDocSerializer,
    SerializationResult,
)
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import (
    MarkdownPictureSerializer,
    MarkdownTableSerializer,
)
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingSerializerProvider, 
    ChunkingDocSerializer
)
from rich.console import Console
from rich.panel import Panel
from pathlib import Path
from Text_Vectorizing_Ollama import (
    EMBEDDING_MAX_TOKENS,
    count_embedding_tokens,
    get_embedding_tokenizer,
)

class EmbeddingTokenizer(BaseTokenizer):
    """Tokenizer aligned with the Ollama embedding model."""
    
    def count_tokens(self, text: str) -> int:
        return count_embedding_tokens(text)
    
    def get_max_tokens(self) -> int:
        return EMBEDDING_MAX_TOKENS

    def get_tokenizer(self):
        return get_embedding_tokenizer()


class PicturePathSerializer(MarkdownPictureSerializer):
    """Picture serializer that omits pictures from the embedding text entirely.

    Pictures are tracked separately (see picture_refs/picture_paths in the
    pipeline) and stored in their own database column, so they must not
    contribute any tokens to the text that gets embedded.
    """

    def __init__(self, base_path: Optional[Path] = None):
        super().__init__()
        self.base_path = base_path or Path("images")

    def serialize(
        self,
        *,
        item: PictureItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        return create_ser_result(text="", span_source=item)


class TablePlaceholderSerializer(MarkdownTableSerializer):
    """Table serializer that emits a stable placeholder instead of the table body.

    The pipeline needs to find each table's exact position in the embedding
    text so it can swap it for an LLM-generated summary. Reconstructing the
    table markdown independently (manual grid formatting, item.export_to_markdown,
    etc.) does not reliably match what docling's own MarkdownTableSerializer
    produces (it runs cells through tabulate() with github-flavored formatting
    and escapes "|" as "&#124;"), so instead both sides agree on this literal
    placeholder string.
    """

    @staticmethod
    def placeholder_for(item: TableItem) -> str:
        return f"[[TABLE_PLACEHOLDER:{item.self_ref}]]"

    def serialize(
        self,
        *,
        item: TableItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        return create_ser_result(text=self.placeholder_for(item), span_source=item)


class PicturePathSerializerProvider(ChunkingSerializerProvider):
    """Serializer provider used when building/merging chunks.

    IMPORTANT: HierarchicalChunker.contextualize() (what chunker.contextualize()
    calls) does NOT re-run serialization - it just returns the chunk's already
    -computed .text plus some JSON-dumped meta fields. That means whatever this
    provider produces at build time (chunker.chunk(dl_doc=doc)) is what ends up
    baked into chunk.text permanently. A separate serializer_provider used only
    for .contextualize() calls later has no effect on the text at all, so the
    table placeholder MUST be configured here, not on a separate "embedding
    time" provider.
    """

    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path

    def get_serializer(self, doc: DoclingDocument):
        return ChunkingDocSerializer(
            doc=doc,
            picture_serializer=PicturePathSerializer(base_path=self.base_path),
            table_serializer=TablePlaceholderSerializer(),
        )
    
def find_n_th_chunk_with_label(
    iter: Iterable[BaseChunk], n: int, label: DocItemLabel
) -> Optional[DocChunk]:
    num_found = -1
    for i, chunk in enumerate(iter):
        doc_chunk = DocChunk.model_validate(chunk)
        for it in doc_chunk.meta.doc_items:
            if it.label == label:
                num_found += 1
                if num_found == n:
                    return i, chunk
    return None, None

def print_chunk(chunks, chunk_pos, chunker, tokenizer):
    """Print a chunk with its context and metadata."""
    console = Console(width=200)  # for getting Markdown tables rendered nicely
    chunk = chunks[chunk_pos]
    ctx_text = chunker.contextualize(chunk=chunk)
    num_tokens = tokenizer.count_tokens(text=ctx_text)
    doc_items_refs = [it.self_ref for it in chunk.meta.doc_items]
    title = f"{chunk_pos=} {num_tokens=} {doc_items_refs=}"
    console.print(Panel(ctx_text, title=title))

def chunk_doclingdocument(doc: DoclingDocument, image_base_path: Optional[Path] = None) -> List[BaseChunk]:
    """
    Chunk a DoclingDocument with custom picture serialization.
    
    Args:
        doc: The DoclingDocument to chunk
        image_base_path: Base path where images are saved (e.g., Path("document_name/images"))
    
    Returns:
        List of chunks with picture references including file paths
    """
    tokenizer = EmbeddingTokenizer()
    
    # Use custom serializer that includes image file paths
    serializer_provider = PicturePathSerializerProvider(base_path=image_base_path)
    
    chunker = HybridChunker(
        tokenizer=tokenizer,
        serializer_provider=serializer_provider
    )
    
    chunk_iter = chunker.chunk(dl_doc=doc)
    chunks = list(chunk_iter)

    MAX_CHUNKS = 1000
    if len(chunks) > MAX_CHUNKS:
        print(f"Shortening document because it has {len(chunks)} chunks (>{MAX_CHUNKS})")
        return chunks[1000:]  # Skip this document
    
    # Post-process chunks to add back picture references that may have been filtered out
    # Build a map of all pictures in the document
    picture_map = {}
    picture_positions = {}  # Track picture positions in document
    
    position = 0
    for item, level in doc.iterate_items():
        if isinstance(item, PictureItem):
            picture_map[item.self_ref] = item
            picture_positions[item.self_ref] = position
        position += 1
    
    print(f"Found {len(picture_map)} pictures in document: {list(picture_map.keys())}")
    
    # For each chunk, add all pictures from the document
    # Since the chunker might filter them out, we'll add them back based on proximity
    pictures_added = 0
    
    for idx, chunk in enumerate(chunks):
        if not hasattr(chunk.meta, 'doc_items'):
            continue
            
        # Get existing refs
        chunk_refs = {it.self_ref for it in chunk.meta.doc_items}
        
        # Get the position range of items in this chunk
        chunk_positions = []
        for doc_item in chunk.meta.doc_items:
            # Find this item's position in the document
            for pos, (item, _) in enumerate(doc.iterate_items()):
                if hasattr(item, 'self_ref') and item.self_ref == doc_item.self_ref:
                    chunk_positions.append(pos)
                    break
        
        if not chunk_positions:
            continue
        
        min_pos = min(chunk_positions)
        max_pos = max(chunk_positions)
        
        # Add pictures that are within or near this chunk's position range
        # Using a window approach: pictures within ±5 positions
        window = 10
        
        for pic_ref, pic_pos in picture_positions.items():
            if pic_ref not in chunk_refs:
                # Check if picture is near this chunk
                if min_pos - window <= pic_pos <= max_pos + window:
                    pic_item = picture_map[pic_ref]
                    chunk.meta.doc_items.append(pic_item)
                    chunk_refs.add(pic_ref)
                    pictures_added += 1
                    # print(f"  Added picture {pic_ref} to chunk {idx} (position {pic_pos} near chunk range {min_pos}-{max_pos})")

        final_token_count = count_embedding_tokens(chunker.contextualize(chunk=chunk))
        # if final_token_count > EMBEDDING_MAX_TOKENS:
        #     raise ValueError(
        #         f"Chunk {idx} exceeds the embedding model limit after picture "
        #         f"injection: {final_token_count} tokens > {EMBEDDING_MAX_TOKENS}."
        #     )
    
    print(f"Total pictures added to chunks: {pictures_added}")
    return chunks

def main():
    """Example usage of the chunking functionality."""
    SOURCE = str(Path(__file__).resolve().parent / "Testing" / "example_converted_document.json")

    doc = DoclingDocument.load_from_json(SOURCE)
    chunks = chunk_doclingdocument(doc)

    tokenizer = EmbeddingTokenizer()
    chunker = HybridChunker(tokenizer=tokenizer)
    
    print(f"{tokenizer.get_max_tokens()=}")
    print(f"{tokenizer.get_max_tokens()=}")
    print(f"Total chunks created: {len(chunks)}")
    
    # Find and print the first table chunk
    i, chunk = find_n_th_chunk_with_label(chunks, n=0, label=DocItemLabel.TABLE)
    if i is not None:
        print_chunk(
            chunks=chunks,
            chunk_pos=i,
            chunker=chunker,
            tokenizer=tokenizer,
        )

    # Find and print the first picture chunk
    i, chunk = find_n_th_chunk_with_label(chunks, n=0, label=DocItemLabel.PICTURE)
    if i is not None:
        print_chunk(
            chunks=chunks,
            chunk_pos=i,
            chunker=chunker,
            tokenizer=tokenizer,
        )

if __name__ == "__main__":
    main()