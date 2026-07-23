"""Cross-encoder reranker.

Reorders candidate chunks by a supervised query-document relevance score,
which is far more accurate than cosine similarity alone. The model is
loaded lazily so importing this module never pulls in torch unless used.

Note: candidates may be Qdrant ``ScoredPoint`` objects (pydantic models that
reject unknown attributes), so we must NOT set new attributes on them. Instead
we wrap the reranked results in a uniform ``Candidate`` carrying the new score.
"""
from src.core.config import Config
from src.core.logging import logger
from src.rag.retrieval.hybrid import Candidate


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
        ranked = sorted(zip(candidates, scores), key=lambda cs: cs[1], reverse=True)
        # wrap in a uniform Candidate so we never mutate Qdrant's ScoredPoint
        return [
            Candidate(getattr(c, "id", None), c.payload, float(s))
            for c, s in ranked[:top_k]
        ]
