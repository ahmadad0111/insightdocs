"""Split page records into overlapping chunks that keep page metadata."""
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.config import Config


class TextChunker:
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size or Config.CHUNK_SIZE,
            chunk_overlap=chunk_overlap or Config.CHUNK_OVERLAP,
        )

    def chunk_pages(self, pages):
        chunks = []
        chunk_id = 0
        for page in pages:
            for text in self.splitter.split_text(page["text"]):
                if not text.strip():
                    continue
                chunks.append({
                    "chunk_id": chunk_id,
                    "page_number": page["page_number"],
                    "text": text,
                })
                chunk_id += 1
        return chunks

    # backwards-compatible alias
    def chunk(self, pages):
        return self.chunk_pages(pages)


# Alias kept so older imports (``Chunker``) keep working.
Chunker = TextChunker
