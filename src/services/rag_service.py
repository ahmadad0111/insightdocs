import time
from src.core.logging import logger


class RAGService:
    def __init__(self, rag_chain):
        self.rag_chain = rag_chain

    def query(self, question: str, document_ids=None) -> dict:
        start = time.time()
        result = self.rag_chain.generate(question, document_ids=document_ids)
        result["latency_ms"] = round((time.time() - start) * 1000, 1)
        logger.info(f"Query done in {result['latency_ms']} ms")
        return result

    def stream(self, question: str, document_ids=None):
        return self.rag_chain.stream(question, document_ids=document_ids)
