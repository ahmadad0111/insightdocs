from src.core.logging import logger
import time
class RAGService:

    def __init__(self, rag_chain):
        self.rag_chain = rag_chain

    def query(self, question: str):
        start = time.time()

        logger.info(f"Query received: {question}")

        result = self.rag_chain.generate(question)
        elapsed = time.time() - start

        logger.info(f"Query completed in {elapsed:.2f}s")

        return result

    def stream(self, question: str):

        logger.info(f"Streaming query received: {question}")

        return self.rag_chain.stream(question)