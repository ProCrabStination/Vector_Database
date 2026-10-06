# The code here was assembled from the Docling library example advanced_chunking_and_serialization.ipynb

from docling_core.types.doc.document import DoclingDocument
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from typing import Iterable, Optional, List, Any
from docling_core.transforms.chunker.base import BaseChunk
from docling_core.transforms.chunker.hierarchical_chunker import DocChunk
from docling_core.types.doc.labels import DocItemLabel
from docling_core.types.doc.document import PictureItem
from docling_core.transforms.serializer.base import (
    BaseDocSerializer,
    SerializationResult,
)
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import MarkdownPictureSerializer
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingSerializerProvider, 
    ChunkingDocSerializer
)
from rich.console import Console
from rich.panel import Panel
from pathlib import Path
from functools import lru_cache


COHERE_EMBEDDING_MAX_TOKENS = 512


@lru_cache(maxsize=1)
def get_embedding_tokenizer():
    """Load a tokenizer from Cohere's tokenizer family."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("CohereForAI/c4ai-command-r-v01")

class EmbeddingTokenizer(BaseTokenizer):
    """Tokenizer aligned with the Cohere embedding model family."""
    
    def count_tokens(self, text: str) -> int:
        return len(get_embedding_tokenizer().encode(text, add_special_tokens=False))
    
    def get_max_tokens(self) -> int:
        return COHERE_EMBEDDING_MAX_TOKENS

    def get_tokenizer(self):
        return get_embedding_tokenizer()


class PicturePathSerializer(MarkdownPictureSerializer):
    """Custom picture serializer that includes file paths in the serialization."""
    
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
        # Sanitize the picture reference for use as filename
        safe_ref = item.self_ref.replace('#/', '').replace('/', '_').replace('\\', '_').replace(':', '_')
        image_path = self.base_path / f"{safe_ref}.png"
        
        # Create markdown with image path reference
        text_res = f"![{item.self_ref}]({image_path})"
        text_res = doc_serializer.post_process(text=text_res)
        
        return create_ser_result(text=text_res, span_source=item)


class PicturePathSerializerProvider(ChunkingSerializerProvider):
    """Serializer provider that uses PicturePathSerializer."""
    
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path
    
    def get_serializer(self, doc: DoclingDocument):
        return ChunkingDocSerializer(
            doc=doc,
            picture_serializer=PicturePathSerializer(base_path=self.base_path),
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
    
    print(f"Total pictures added to chunks: {pictures_added}")
    return chunks

def main():
    """Example usage of the chunking functionality."""
    SOURCE = "C:\\Users\\uiv11567\\source\\repos\\MESSelfService\\Python\\Analysis_Assistant\\Test_Documents_Markdown\\Daimler\\MR2\\MR2_FT_TestSpec_Nürnberg_english\\MR2_FT_TestSpec_Nürnberg_english_DoclingDocument.json"

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