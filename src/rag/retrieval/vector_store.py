"""Thin Qdrant wrapper (foundation version).

Document-level management (upsert/delete/list by document_id) is added on
the doc-management branch.
"""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from src.core.config import Config
from src.core.logging import logger


class VectorStore:
    def __init__(self, url: str = None, collection_name: str = None, vector_size: int = None):
        self.client = QdrantClient(url=url or Config.QDRANT_URL)
        self.collection_name = collection_name or Config.COLLECTION_NAME
        self.vector_size = vector_size or Config.EMBEDDING_DIM
        self.ensure_collection()
        logger.info(f"VectorStore ready (collection={self.collection_name})")

    def ensure_collection(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )

    def add_chunks(self, embedded_chunks) -> int:
        points = [
            PointStruct(
                id=i,
                vector=list(c["embedding"]),
                payload={"text": c["text"], "page": c["page_number"]},
            )
            for i, c in enumerate(embedded_chunks)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        return len(points)

    def search(self, query_vector, top_k: int = 5):
        return self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector, limit=top_k, with_payload=True,
        ).points
