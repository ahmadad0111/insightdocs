"""document_id helpers are pure-python and testable without a running Qdrant."""
from src.rag.retrieval.vector_store import make_document_id


def test_document_id_is_stable_and_short():
    a = make_document_id("federated_learning.pdf")
    b = make_document_id("federated_learning.pdf")
    c = make_document_id("other.pdf")
    assert a == b            # same filename -> same id (re-upload updates in place)
    assert a != c            # different filename -> different id
    assert len(a) == 16
