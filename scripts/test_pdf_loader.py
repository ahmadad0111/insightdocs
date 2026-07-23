from src.rag.ingestion.pdf_loader import PDFLoader
import json
from src.rag.ingestion.chunker import TextChunker
from src.rag.embeddings.embedder import Embedder
from src.rag.retrieval.vector_store import VectorStore


loader = PDFLoader(
    "data/raw/federated_learning.pdf"
)

pages = loader.load()


# with open(
#     "data/processed/pages.json",
#     "w",
#     encoding="utf-8"
# ) as f:
#     json.dump(
#         pages,
#         f,
#         indent=2,
#         ensure_ascii=False
#     )

print(f"Total pages: {len(pages)}")

# print("\nFirst page preview:\n")
# print(pages[0]["text"][:1000])

chunker = TextChunker()

chunks = chunker.chunk_pages(pages)

with open(
    "data/processed/chunks.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        chunks,
        f,
        indent=2,
        ensure_ascii=False
    )


print(f"Total chunks: {len(chunks)}")

print(f"Total chunks: {len(chunks)}")

# for i in range(5):
#     print("\n" + "="*50)
#     print(f"Chunk {i}")
#     print("="*50)
#     print(chunks[i]["text"])

embedder = Embedder()

# embeddings = embedder.embed_chunks(
#     chunks[:10]
# )

# print(type(embeddings))
# print(embeddings.shape)

# print(embeddings[0][:5])

vector_store = VectorStore()

# Step 1: embed chunks
embedded_chunks = embedder.embed_chunks(chunks)

# Step 2: store in Qdrant
vector_store.add_chunks(embedded_chunks)

print("Chunks stored in Qdrant")

## query
query = "What is federated learning?"

query_embedding = embedder.model.encode(query)

results = vector_store.search(query_embedding)

for r in results:
    print("\nScore:", r.score)
    print(r.payload["text"][:300])