"""Builds the RAG pipeline once and shares it across requests."""
from functools import lru_cache

from src.core.config import Config
from src.core.logging import logger
from src.rag.embeddings.embedder import Embedder
from src.rag.retrieval.vector_store import VectorStore
from src.rag.generation.llm import get_llm
from src.rag.generation.rag_chain import RAGChain
from src.rag.memory.conversation_memory import ConversationMemory
from src.services.rag_service import RAGService
from src.services.document_ingestion_service import DocumentIngestionService


@lru_cache(maxsize=1)
def build_pipeline():
    logger.info(f"Building pipeline: {Config.summary()}")
    embedder = Embedder()
    vector_store = VectorStore()
    llm = get_llm()
    memory = ConversationMemory()
    chain = RAGChain(embedder, vector_store, llm, memory)
    service = RAGService(chain)
    ingestion = DocumentIngestionService(embedder=embedder, vector_store=vector_store)
    return {"service": service, "ingestion": ingestion, "vector_store": vector_store}


def get_service() -> RAGService:
    return build_pipeline()["service"]


def get_ingestion() -> DocumentIngestionService:
    return build_pipeline()["ingestion"]


def get_vector_store() -> VectorStore:
    return build_pipeline()["vector_store"]
