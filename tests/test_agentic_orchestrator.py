"""End-to-end agentic flow with fakes (no Qdrant / LLM needed)."""
from src.rag.agentic.agent import AgenticRAGChain
from src.rag.memory.conversation_memory import ConversationMemory


class FakePoint:
    def __init__(self, text, page, score=0.9):
        self.payload = {"text": text, "page": page,
                        "document_id": "d1", "filename": "f.pdf"}
        self.score = score


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def retrieve(self, q, document_ids=None, top_k=None):
        self.calls.append(q)
        return [FakePoint(f"context for {q[:15]}", 1)]


class FakeLLM:
    def __call__(self, prompt):
        if "Sub-questions:" in prompt:
            return "What is FL?\nHow does it cut cost?"
        if "Label:" in prompt:
            return "RETRIEVE"
        if "Reply:" in prompt:
            return "Hi! Upload a PDF and ask away."
        return "Grounded answer [Source 1]."

    def stream(self, prompt):
        for t in ["Grounded ", "answer."]:
            yield t


def _chain():
    return AgenticRAGChain(None, None, FakeLLM(), ConversationMemory(),
                           retriever=FakeRetriever())


def test_direct_route_skips_retrieval():
    out = _chain().generate("hi")
    assert out["route"] == "direct"
    assert out["sources"] == []
    assert "Upload" in out["answer"]


def test_complex_query_is_decomposed_and_retrieved_per_subquestion():
    chain = _chain()
    out = chain.generate(
        "What is federated learning and how does it reduce communication cost?")
    assert out["route"] == "retrieve"
    assert len(out["sub_questions"]) == 2
    assert len(chain.retriever.calls) >= 2
    assert out["sources"]


def test_simple_query_not_decomposed():
    out = _chain().generate("What is the contribution?")
    assert out["route"] == "retrieve"
    assert out["sub_questions"] == []


def test_streaming_direct_and_retrieve():
    chain = _chain()
    direct = list(chain.stream("thanks"))
    assert direct[0]["route"] == "direct" and direct[-1]["done"]
    retr = list(chain.stream("What is FL and how does it reduce cost in detail?"))
    assert retr[0]["route"] == "retrieve" and retr[-1]["done"] and retr[-1]["sources"]
