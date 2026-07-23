from src.core.logging import logger
class RAGChain:

    def __init__(self, embedder, vector_store, llm,memory):
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm
        self.memory = memory

    def retrieve(self, query, top_k=5):
        logger.info("Generating query embedding")
        query_vec = self.embedder.model.encode(query).tolist()
        logger.info("Searching vector database")
        results = self.vector_store.search(query_vec, top_k=top_k)
        logger.info(f"Retrieved {len(results)} chunks")

        return results



    def build_prompt(self, query, chunks):

        context_blocks = []

        for i, chunk in enumerate(chunks):
            context_blocks.append(
                f"[Source {i+1} | Page {chunk.payload.get('page')}]\n"
                f"{chunk.payload['text']}"
            )

        context = "\n\n".join(context_blocks)

        history = self.memory.get_context()

        prompt = f"""
            You are a precise RAG-based assistant.

            Rules:
            - Answer ONLY using provided context
            - If not in context, say "Not found in document"
            - Do NOT paraphrase outside context
            - Do NOT copy full sentences from context
            - Keep answer 5–7 lines max
            - Summarize in your own words
            - Always cite sources [Source X]

            Conversation History:
            {history}

            Context:
            {context}

            Question:
            {query}

            Answer:
            """

        return prompt


    def generate(self, query):

        chunks = self._prepare_chunks(query)

        prompt = self.build_prompt(query, chunks)

        response = self.llm(prompt)

        self.memory.add(query,response)

        return {
            "answer": response,
            "sources": [
                {
                    "page": c.payload.get("page"),
                    "text": c.payload["text"][:120]
                }
                for c in chunks
            ]
        }
    def _prepare_chunks(self, query):

        
        retrieval_query = (
            self.memory.get_context()
            + "\n"
            + query
        )

        chunks = self.retrieve(retrieval_query)

        # filter
        filtered = []
        for c in chunks:
            score = getattr(c, "score", None)
            if score is None or score >= 0.78:
                filtered.append(c)

        chunks = filtered[:5]

        # deduplicate
        seen = set()
        unique_chunks = []

        for c in chunks:
            text = c.payload["text"]

            if text not in seen:
                unique_chunks.append(c)
                seen.add(text)

        return unique_chunks


    def stream(self, query):

        chunks = self._prepare_chunks(query)
        prompt = self.build_prompt(query, chunks)

        for chunk in self.llm.stream(prompt):
            if chunk:
                yield chunk