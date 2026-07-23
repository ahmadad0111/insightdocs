"""Sentence-Transformers embedding wrapper with batching."""
from sentence_transformers import SentenceTransformer

from src.core.config import Config


class Embedder:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or Config.EMBEDDING_MODEL
        self.model = SentenceTransformer(self.model_name)

    def embed_texts(self, texts, batch_size: int = 64):
        embeddings = self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=False,
            normalize_embeddings=True,
        )
        return [e.tolist() for e in embeddings]

    def embed_query(self, text: str):
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_chunks(self, chunks):
        texts = [c["text"] for c in chunks]
        vectors = self.embed_texts(texts)
        out = []
        for chunk, vec in zip(chunks, vectors):
            enriched = dict(chunk)
            enriched["embedding"] = vec
            out.append(enriched)
        return out
