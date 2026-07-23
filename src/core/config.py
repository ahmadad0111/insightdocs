"""Central configuration, driven entirely by environment variables.

Nothing here imports heavy libraries, so it is safe to import anywhere
(including tests) without pulling in torch / qdrant.
"""
import os


def _as_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).lower() in ("1", "true", "yes", "on")


class Config:
    # --- Vector database ---
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "insightdocs")

    # --- Embeddings ---
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

    # --- Retrieval quality ---
    USE_HYBRID = _as_bool("USE_HYBRID", True)
    USE_RERANKER = _as_bool("USE_RERANKER", True)
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    TOP_K = int(os.getenv("TOP_K", "5"))              # final chunks sent to the LLM
    CANDIDATE_K = int(os.getenv("CANDIDATE_K", "20"))  # candidates before reranking
    SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.0"))

    # --- LLM (provider-switchable) ---
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()  # ollama | openai | anthropic
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")

    # --- Chunking ---
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

    # --- Conversation ---
    MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "5"))

    @classmethod
    def summary(cls) -> dict:
        return {
            "qdrant_url": cls.QDRANT_URL,
            "collection": cls.COLLECTION_NAME,
            "embedding_model": cls.EMBEDDING_MODEL,
            "use_hybrid": cls.USE_HYBRID,
            "use_reranker": cls.USE_RERANKER,
            "llm_provider": cls.LLM_PROVIDER,
        }


config = Config()
