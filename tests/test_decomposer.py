from src.rag.agentic.decomposer import (
    needs_decomposition, parse_subquestions, QueryDecomposer,
)


def test_needs_decomposition():
    assert needs_decomposition("What is FL?") is False           # short/simple
    assert needs_decomposition("Compare FedAvg and the proposed method in detail") is True
    assert needs_decomposition(
        "What is federated learning and how does it reduce communication cost?"
    ) is True
    assert needs_decomposition("What is the capital of France today please?") is False


def test_parse_subquestions_cleans_formatting():
    raw = "1. What is FL?\n2) How does it reduce cost?\n- What is FL?\n\n* Extra one"
    subs = parse_subquestions(raw, max_sub=4)
    assert subs[0] == "What is FL?"
    assert "How does it reduce cost?" in subs
    # duplicate ("What is FL?") removed
    assert subs.count("What is FL?") == 1


def test_decomposer_returns_single_for_simple():
    d = QueryDecomposer(llm=lambda p: "irrelevant")
    assert d.decompose("What is FL?") == ["What is FL?"]


def test_decomposer_splits_complex_with_llm():
    llm = lambda p: "What is federated learning?\nHow does it reduce communication cost?"
    d = QueryDecomposer(llm=llm)
    subs = d.decompose("What is federated learning and how does it reduce communication cost?")
    assert len(subs) == 2


def test_decomposer_without_llm_returns_original():
    d = QueryDecomposer(llm=None)
    q = "What is federated learning and how does it reduce communication cost?"
    assert d.decompose(q) == [q]
