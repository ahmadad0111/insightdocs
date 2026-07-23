"""Retrieve -> build prompt -> generate, with optional document scoping."""
from src.core.config import Config
from src.core.logging import logger

PROMPT_TEMPLATE = """You are a precise RAG assistant.

Rules:
- Answer ONLY using the provided context.
- If the answer is not in the context, reply exactly: "Not found in the document."
- Summarise in your own words; do not copy whole sentences.
- Keep the answer under 7 lines.
- Cite every claim with its source marker, e.g. [Source 2].

Conversation history:
{history}

Context:
{context}

Question: {question}

Answer:"""


class RAGChain:
    def __init__(self, embedder, vector_store, llm, memory, retriever=None):
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm
        self.memory = memory
        self.retriever = retriever  # optional hybrid/rerank retriever (added later)

    def retrieve(self, query, document_ids=None, top_k: int = None):
        top_k = top_k or Config.TOP_K
        if self.retriever is not None:
            return self.retriever.retrieve(query, document_ids=document_ids, top_k=top_k)
        query_vec = self.embedder.embed_query(query)
        results = self.vector_store.search(query_vec, top_k=top_k, document_ids=document_ids)
        logger.info(f"Retrieved {len(results)} chunks (dense)")
        return results

    def _prepare_chunks(self, query, document_ids=None):
        expanded = (self.memory.last_questions() + "\n" + query).strip()
        chunks = self.retrieve(expanded, document_ids=document_ids)
        seen, unique = set(), []
        for c in chunks:
            text = c.payload["text"]
            if text not in seen:
                seen.add(text)
                unique.append(c)
        return unique

    def build_prompt(self, query, chunks):
        blocks = [
            f"[Source {i + 1} | Page {c.payload.get('page')}]\n{c.payload['text']}"
            for i, c in enumerate(chunks)
        ]
        return PROMPT_TEMPLATE.format(
            history=self.memory.get_context() or "(none)",
            context="\n\n".join(blocks) or "(no context retrieved)",
            question=query,
        )

    @staticmethod
    def _sources(chunks):
        return [
            {
                "document_id": c.payload.get("document_id"),
                "filename": c.payload.get("filename"),
                "page": c.payload.get("page"),
                "score": getattr(c, "score", None),
                "text": c.payload["text"][:160],
            }
            for c in chunks
        ]

    def generate(self, query, document_ids=None):
        chunks = self._prepare_chunks(query, document_ids=document_ids)
        answer = self.llm(self.build_prompt(query, chunks))
        self.memory.add(query, answer)
        return {"answer": answer, "sources": self._sources(chunks)}

    def stream(self, query, document_ids=None):
        chunks = self._prepare_chunks(query, document_ids=document_ids)
        prompt = self.build_prompt(query, chunks)
        collected = []
        for token in self.llm.stream(prompt):
            collected.append(token)
            yield {"token": token}
        self.memory.add(query, "".join(collected))
        yield {"done": True, "sources": self._sources(chunks)}
