"""Cross-encoder reranker.

Reorders candidate chunks by a supervised query-document relevance score,
which is far more accurate than cosine similarity alone. The model is
loaded lazily so importing this module never pulls in torch unless used.
"""
from src.core.config import Config
from src.core.logging import logger


class CrossEncoderReranker:
    def __init__(self, model_name: str = None):
        from sentence_transformers import CrossEncoder
        self.model_name = model_name or Config.RERANKER_MODEL
        self.model = CrossEncoder(self.model_name)
        logger.info(f"Reranker loaded: {self.model_name}")

    def rerank(self, query, candidates, top_k: int = None):
        top_k = top_k or Config.TOP_K
        if not candidates:
            return []
        pairs = [(query, c.payload["text"]) for c in candidates]
        scores = self.model.predict(pairs)
        for c, s in zip(candidates, scores):
            c.rerank_score = float(s)
            c.score = float(s)
        ranked = sorted(candidates, key=lambda c: c.rerank_score, reverse=True)
        return ranked[:top_k]
