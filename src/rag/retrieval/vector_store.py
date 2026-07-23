from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, SearchRequest
from src.core.config import Config
from src.core.logging import logger
class VectorStore:

    def __init__(
        self,
        url="http://localhost:6333",
        collection_name="rag_chunks",
        vector_size=384
    ):

        self.client = QdrantClient(url=Config.QDRANT_URL)

        self.collection_name = collection_name
        logger.info(f"Using collection: {self.collection_name}")
        self.vector_size = vector_size

        self._create_collection()

    def _create_collection(self):

        collections = self.client.get_collections().collections
        names = [c.name for c in collections]

        if self.collection_name not in names:

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )

    def add_chunks(self, chunks):

        points = []

        for i, chunk in enumerate(chunks):

            points.append(
                PointStruct(
                    id=chunk["chunk_id"],
                    vector=list(chunk["embedding"]),
                    payload={
                        "text": chunk["text"],
                        "page": chunk["page_number"]
                    }
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(self, query_vector, top_k=3):

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True
        )

        return results.points