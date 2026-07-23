"""Ingest a PDF as a managed document: load -> chunk -> embed -> upsert.

Re-ingesting the same filename updates the existing document in place
(same stable document_id) instead of creating duplicates.
"""
import os

from src.rag.ingestion.pdf_loader import PDFLoader
from src.rag.ingestion.chunker import TextChunker
from src.rag.embeddings.embedder import Embedder
from src.rag.retrieval.vector_store import VectorStore, make_document_id
from src.core.logging import logger


class DocumentIngestionService:
    def __init__(self, embedder: Embedder = None, vector_store: VectorStore = None):
        self.chunker = TextChunker()
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or VectorStore()

    def ingest(self, pdf_path: str, document_id: str = None) -> dict:
        filename = os.path.basename(pdf_path)
        document_id = document_id or make_document_id(filename)
        logger.info(f"Ingesting {filename} (document_id={document_id})")

        pages = PDFLoader(pdf_path).load()
        chunks = self.chunker.chunk_pages(pages)
        embedded = self.embedder.embed_chunks(chunks)
        n = self.vector_store.upsert_document(document_id, filename, embedded)

        return {"document_id": document_id, "filename": filename,
                "num_chunks": n, "status": "indexed"}

    def delete(self, document_id: str) -> dict:
        self.vector_store.delete_document(document_id)
        return {"document_id": document_id, "status": "deleted"}

    def list_documents(self):
        return self.vector_store.list_documents()
