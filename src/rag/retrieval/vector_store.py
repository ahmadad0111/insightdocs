"""Qdrant wrapper with document-level management.

A single collection holds all chunks. Every point carries a
``document_id`` so a document can be updated (upsert) or deleted as a
unit, and searches can be scoped to specific documents.
"""
import uuid
import hashlib

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition,
    MatchValue, MatchAny, PayloadSchemaType,
)

from src.core.config import Config
from src.core.logging import logger


def make_document_id(filename: str) -> str:
    """Stable id derived from the filename so re-uploads update in place."""
    return hashlib.sha1(filename.encode("utf-8")).hexdigest()[:16]


class VectorStore:
    def __init__(self, url: str = None, collection_name: str = None, vector_size: int = None):
        self.client = QdrantClient(url=url or Config.QDRANT_URL)
        self.collection_name = collection_name or Config.COLLECTION_NAME
        self.vector_size = vector_size or Config.EMBEDDING_DIM
        self.ensure_collection()
        logger.info(f"VectorStore ready (collection={self.collection_name})")

    # ---- collection lifecycle ----
    def ensure_collection(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )
            logger.info(f"Created collection {self.collection_name}")
        # index document_id so filtering/deleting by document is fast
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="document_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass  # already exists

    def reset_collection(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.ensure_collection()
        logger.info(f"Reset collection {self.collection_name}")

    # ---- write ----
    def upsert_document(self, document_id: str, filename: str, embedded_chunks) -> int:
        """Replace all chunks for a document (delete-then-insert = update)."""
        self.delete_document(document_id)
        points = []
        for c in embedded_chunks:
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=list(c["embedding"]),
                payload={
                    "text": c["text"],
                    "page": c["page_number"],
                    "chunk_id": c["chunk_id"],
                    "document_id": document_id,
                    "filename": filename,
                },
            ))
        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info(f"Upserted {len(points)} chunks for document {document_id}")
        return len(points)

    def delete_document(self, document_id: str) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(must=[
                FieldCondition(key="document_id", match=MatchValue(value=document_id))
            ]),
        )

    # ---- read ----
    def _doc_filter(self, document_ids):
        if not document_ids:
            return None
        return Filter(must=[FieldCondition(key="document_id", match=MatchAny(any=list(document_ids)))])

    def search(self, query_vector, top_k: int = 5, document_ids=None):
        return self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector, limit=top_k, with_payload=True,
            query_filter=self._doc_filter(document_ids),
        ).points

    def iter_chunks(self, document_ids=None, batch: int = 256):
        """Yield every stored chunk payload (used to build the BM25 index)."""
        offset = None
        flt = self._doc_filter(document_ids)
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=flt, with_payload=True, with_vectors=False,
                limit=batch, offset=offset,
            )
            for p in points:
                yield {"id": p.id, **p.payload}
            if offset is None:
                break

    def list_documents(self):
        counts = {}
        names = {}
        for ch in self.iter_chunks():
            doc = ch.get("document_id")
            counts[doc] = counts.get(doc, 0) + 1
            names[doc] = ch.get("filename")
        return [
            {"document_id": d, "filename": names.get(d), "num_chunks": n}
            for d, n in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        ]
