"""Agentic RAG orchestrator.

Extends the standard RAGChain with two agentic behaviours:

    1. Routing      - a greeting/meta message is answered directly (no retrieval).
    2. Decomposition - a complex question is split into sub-questions, each
                       retrieved separately, and their contexts merged before a
                       single grounded answer is generated.

It exposes the same interface as RAGChain (`generate`, `stream`,
`_prepare_chunks`) so it drops straight into RAGService.
"""
from src.core.config import Config
from src.core.logging import logger
from src.rag.generation.rag_chain import RAGChain
from src.rag.agentic.router import QueryRouter
from src.rag.agentic.decomposer import QueryDecomposer

DIRECT_PROMPT = """You are InsightDocs, a helpful assistant that answers
questions about the user's uploaded documents. Respond briefly and naturally
to this message. If they are asking what you can do, explain that they can
upload PDFs and ask questions and you will answer with cited sources.

Message: {query}

Reply:"""


class AgenticRAGChain(RAGChain):
    def __init__(self, embedder, vector_store, llm, memory, retriever=None,
                 router=None, decomposer=None):
        super().__init__(embedder, vector_store, llm, memory, retriever=retriever)
        self.router = router or QueryRouter(llm=llm)
        self.decomposer = decomposer or QueryDecomposer(llm=llm)

    # ---- multi-query retrieval ----
    def _agentic_chunks(self, query, document_ids=None):
        sub_questions = self.decomposer.decompose(query)
        merged, seen = [], set()
        per_q = max(2, Config.TOP_K // max(1, len(sub_questions)) + 1)
        for sub in sub_questions:
            for c in self.retrieve(sub, document_ids=document_ids, top_k=per_q):
                text = c.payload["text"]
                if text not in seen:
                    seen.add(text)
                    merged.append(c)
        # keep the best-scored merged context, capped
        merged.sort(key=lambda c: getattr(c, "score", 0.0) or 0.0, reverse=True)
        return sub_questions, merged[: Config.TOP_K + 2]

    def _direct_answer(self, query):
        return self.llm(DIRECT_PROMPT.format(query=query))

    # ---- public API (same shape as RAGChain) ----
    def generate(self, query, document_ids=None):
        route = self.router.route(query)
        if route == "direct":
            answer = self._direct_answer(query)
            self.memory.add(query, answer)
            return {"answer": answer, "sources": [], "route": "direct",
                    "sub_questions": []}

        sub_questions, chunks = self._agentic_chunks(query, document_ids=document_ids)
        answer = self.llm(self.build_prompt(query, chunks))
        self.memory.add(query, answer)
        return {
            "answer": answer,
            "sources": self._sources(chunks),
            "route": "retrieve",
            "sub_questions": sub_questions if len(sub_questions) > 1 else [],
        }

    def stream(self, query, document_ids=None):
        route = self.router.route(query)
        if route == "direct":
            collected = []
            yield {"route": "direct"}
            for token in self.llm.stream(DIRECT_PROMPT.format(query=query)):
                collected.append(token)
                yield {"token": token}
            self.memory.add(query, "".join(collected))
            yield {"done": True, "sources": [], "route": "direct", "sub_questions": []}
            return

        sub_questions, chunks = self._agentic_chunks(query, document_ids=document_ids)
        yield {"route": "retrieve", "sub_questions": sub_questions if len(sub_questions) > 1 else []}
        prompt = self.build_prompt(query, chunks)
        collected = []
        for token in self.llm.stream(prompt):
            collected.append(token)
            yield {"token": token}
        self.memory.add(query, "".join(collected))
        yield {"done": True, "sources": self._sources(chunks),
               "route": "retrieve",
               "sub_questions": sub_questions if len(sub_questions) > 1 else []}
