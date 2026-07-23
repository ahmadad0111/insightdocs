from src.rag.generation.rag_chain import RAGChain
from src.rag.memory.conversation_memory import ConversationMemory


class _Payload(dict):
    pass


class _Point:
    def __init__(self, text, page, score=0.9):
        self.payload = {"text": text, "page": page}
        self.score = score


def test_build_prompt_includes_context_and_citations_rule():
    chain = RAGChain(embedder=None, vector_store=None, llm=None,
                     memory=ConversationMemory())
    chunks = [_Point("Federated learning trains without sharing data.", 3)]
    prompt = chain.build_prompt("What is FL?", chunks)
    assert "Federated learning" in prompt
    assert "[Source 1" in prompt
    assert "Page 3" in prompt
    assert "Not found in the document" in prompt


def test_sources_shape():
    chunks = [_Point("abc", 1, score=0.8)]
    src = RAGChain._sources(chunks)
    assert src[0]["page"] == 1
    assert src[0]["score"] == 0.8
    assert src[0]["text"] == "abc"
