"""Ingest a PDF: load -> chunk -> embed -> store. (foundation version)"""
import os

from src.rag.ingestion.pdf_loader import PDFLoader
from src.rag.ingestion.chunker import TextChunker
from src.rag.embeddings.embedder import Embedder
from src.rag.retrieval.vector_store import VectorStore
from src.core.logging import logger


class DocumentIngestionService:
    def __init__(self, embedder: Embedder = None, vector_store: VectorStore = None):
        self.chunker = TextChunker()
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or VectorStore()

    def ingest(self, pdf_path: str) -> dict:
        filename = os.path.basename(pdf_path)
        logger.info(f"Ingesting {filename}")

        pages = PDFLoader(pdf_path).load()
        chunks = self.chunker.chunk_pages(pages)
        embedded = self.embedder.embed_chunks(chunks)
        n = self.vector_store.add_chunks(embedded)

        logger.info(f"Indexed {n} chunks from {filename}")
        return {"filename": filename, "num_chunks": n}
