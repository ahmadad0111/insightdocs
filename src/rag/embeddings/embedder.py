from sentence_transformers import SentenceTransformer


class Embedder:

    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.model = SentenceTransformer(model_name)

    def embed_chunks(self, chunks):

        texts = [c["text"] for c in chunks]

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True
        )

        # IMPORTANT: keep metadata + attach embedding
        embedded_chunks = []

        for chunk, emb in zip(chunks, embeddings):

            embedded_chunks.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "page_number": chunk["page_number"],
                    "text": chunk["text"],
                    "embedding": emb.tolist()
                }
            )

        return embedded_chunks