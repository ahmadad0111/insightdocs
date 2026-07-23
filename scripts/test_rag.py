from src.rag.embeddings.embedder import Embedder
from src.rag.retrieval.vector_store import VectorStore
# from src.rag.generation.llm import OpenAILLM
from src.rag.generation.llm import LocalLLM
from src.rag.generation.rag_chain import RAGChain

import os

embedder = Embedder()
vector_store = VectorStore()

# llm = OpenAILLM(
#     api_key=os.getenv("OPENAI_API_KEY")
# )

llm = LocalLLM(model="llama3") # or mistral

rag = RAGChain(embedder, vector_store, llm)

query = "What is federated learning?"

result = rag.generate(query)

print("\nFINAL ANSWER:\n")
print(result["answer"])

print("\nSOURCES:\n")

for s in result["sources"]:
    print(f"Page {s['page']}")
    print(s["text"])
    print("---")