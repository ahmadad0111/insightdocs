"""Reciprocal-rank fusion is a pure function -> unit-testable with fakes."""
from src.rag.retrieval.hybrid import reciprocal_rank_fusion, Candidate


def _mk(chunk_id, text):
    return Candidate(chunk_id, {"chunk_id": chunk_id, "document_id": "d1", "text": text})


def test_rrf_rewards_items_ranked_high_in_both_lists():
    a = _mk(1, "alpha chunk one")
    b = _mk(2, "beta chunk two")
    c = _mk(3, "gamma chunk three")
    dense = [a, b, c]
    sparse = [b, a, c]  # b is top in sparse, a second
    fused = reciprocal_rank_fusion([dense, sparse])
    ids = [x.payload["chunk_id"] for x in fused]
    # a (1,2) and b (2,1) both beat c (3,3); order among a,b is stable but c is last
    assert ids[-1] == 3
    assert set(ids[:2]) == {1, 2}


def test_rrf_dedupes_same_chunk():
    a = _mk(1, "same text here")
    fused = reciprocal_rank_fusion([[a], [a]])
    assert len(fused) == 1
