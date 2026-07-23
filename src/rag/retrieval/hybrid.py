"""Hybrid retriever: dense (Qdrant) + sparse (BM25), fused, then reranked.

Pipeline:
    1. Dense vector search  -> CANDIDATE_K candidates
    2. BM25 keyword search  -> CANDIDATE_K candidates
    3. Reciprocal-rank fusion of the two lists
    4. Optional cross-encoder rerank -> TOP_K final chunks

The BM25 index is built from the current collection and cached; call
``invalidate()`` after ingesting/deleting documents to rebuild it.
"""
import re

from src.core.config import Config
from src.core.logging import logger


def _tokenize(text: str):
    return re.findall(r"\w+", text.lower())


class Candidate:
    """Stand-in matching the Qdrant point interface (.payload / .score)."""
    def __init__(self, point_id, payload, score=0.0):
        self.id = point_id
        self.payload = payload
        self.score = score


def reciprocal_rank_fusion(result_lists, k: int = 60):
    """Fuse ranked lists with Reciprocal Rank Fusion. Pure function -> testable."""
    scores, holder = {}, {}

    def key_of(cand):
        return (cand.payload.get("chunk_id"),
                cand.payload.get("document_id"),
                cand.payload["text"][:60])

    for lst in result_lists:
        for rank, cand in enumerate(lst):
            key = key_of(cand)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            holder[key] = cand
    return sorted(holder.values(), key=lambda c: scores[key_of(c)], reverse=True)


class HybridRetriever:
    def __init__(self, embedder, vector_store, reranker=None):
        self.embedder = embedder
        self.vector_store = vector_store
        self.reranker = reranker
        self._bm25 = None
        self._bm25_docs = []

    def invalidate(self):
        self._bm25 = None
        self._bm25_docs = []

    def _ensure_bm25(self, document_ids=None):
        if self._bm25 is not None:
            return
        from rank_bm25 import BM25Okapi
        self._bm25_docs = list(self.vector_store.iter_chunks(document_ids=document_ids))
        corpus = [_tokenize(d["text"]) for d in self._bm25_docs]
        self._bm25 = BM25Okapi(corpus) if corpus else None
        logger.info(f"BM25 index built over {len(self._bm25_docs)} chunks")

    def _bm25_search(self, query, top_k, document_ids=None):
        self._ensure_bm25(document_ids=document_ids)
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in order[:top_k]:
            d = self._bm25_docs[i]
            payload = {k: v for k, v in d.items() if k != "id"}
            out.append(Candidate(d["id"], payload, float(scores[i])))
        return out

    def retrieve(self, query, document_ids=None, top_k: int = None):
        top_k = top_k or Config.TOP_K
        candidate_k = Config.CANDIDATE_K

        query_vec = self.embedder.embed_query(query)
        dense = list(self.vector_store.search(query_vec, top_k=candidate_k, document_ids=document_ids))

        if Config.USE_HYBRID:
            sparse = self._bm25_search(query, candidate_k, document_ids=document_ids)
            candidates = reciprocal_rank_fusion([dense, sparse])
        else:
            candidates = dense
        candidates = candidates[:candidate_k]

        if Config.USE_RERANKER and self.reranker is not None:
            return self.reranker.rerank(query, candidates, top_k=top_k)
        return candidates[:top_k]
