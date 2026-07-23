# import os

# class Config:
#     QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
#     COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_chunks")
#     EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
#     LLM_MODEL = "llama3:latest"
import os

class Config:
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_chunks")

    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "llama3"