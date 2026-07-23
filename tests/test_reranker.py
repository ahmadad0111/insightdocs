"""Regression test: reranker must not mutate pydantic-style candidates.

Qdrant ScoredPoint rejects unknown attributes, so setting `.rerank_score` on it
raised in production. This uses a stand-in that mimics that behaviour.
"""
from src.rag.retrieval.reranker import CrossEncoderReranker
from src.rag.retrieval.hybrid import Candidate


class StrictPoint:
    """Mimics Qdrant ScoredPoint: has payload/score/id but forbids new fields."""
    __slots__ = ("id", "payload", "score")

    def __init__(self, pid, text, page, score):
        self.id = pid
        self.payload = {"text": text, "page": page, "document_id": "d1"}
        self.score = score


class FakeModel:
    # returns higher relevance for the second candidate
    def predict(self, pairs):
        return [0.1 * (i + 1) for i in range(len(pairs))]


def _reranker_with_fake_model():
    r = CrossEncoderReranker.__new__(CrossEncoderReranker)  # skip heavy __init__
    r.model = FakeModel()
    return r


def test_rerank_does_not_mutate_strict_points_and_orders_by_score():
    r = _reranker_with_fake_model()
    cands = [
        StrictPoint("a", "first chunk", 1, 0.9),
        StrictPoint("b", "second chunk", 2, 0.5),
    ]
    out = r.rerank("q", cands, top_k=2)          # must not raise
    assert all(isinstance(c, Candidate) for c in out)
    # FakeModel gave the 2nd candidate the higher score -> it ranks first
    assert out[0].payload["text"] == "second chunk"
    assert out[0].score == 0.2
    # original objects untouched (no rerank_score attribute leaked in)
    assert not hasattr(cands[0], "rerank_score")


def test_rerank_empty():
    assert _reranker_with_fake_model().rerank("q", [], top_k=3) == []
