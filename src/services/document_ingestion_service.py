from src.rag.ingestion.pdf_loader import PDFLoader
from src.rag.ingestion.chunker import Chunker
from src.rag.embeddings.embedder import Embedder
from src.rag.retrieval.vector_store import VectorStore

class DocumentIngestionService:

    def __init__(self):

        self.chunker = Chunker()
        self.embedder = Embedder()
        self.vector_store = VectorStore()

    def ingest(self, pdf_path):

        loader = PDFLoader(pdf_path)

        pages = loader.load()

        chunks = self.chunker.chunk(pages)

        for chunk in chunks:

            embedding = self.embedder.model.encode(
                chunk["text"]
            ).tolist()

            self.vector_store.add(
                embedding,
                payload=chunk
            )

        return len(chunks)