from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import QueryRequest
from src.services.rag_service import RAGService

from src.rag.embeddings.embedder import Embedder
from src.rag.retrieval.vector_store import VectorStore
from src.rag.generation.llm import LocalLLM
from src.rag.generation.rag_chain import RAGChain
from fastapi.responses import StreamingResponse
import json
from src.core.config import Config
from src.rag.memory.conversation_memory import ConversationMemory
from src.core.logging import logger

from fastapi import UploadFile, File
import os
import shutil
from src.services.document_ingestion_service import DocumentIngestionService

ingestion_service = DocumentIngestionService()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- build pipeline ---
embedder = Embedder(model_name=Config.EMBEDDING_MODEL )
vector_store = VectorStore(
    url=Config.QDRANT_URL,
    collection_name=Config.COLLECTION_NAME
)
llm = LocalLLM(
    model=Config.LLM_MODEL
)
memory = ConversationMemory()
rag_chain = RAGChain(embedder, vector_store, llm,memory)

service = RAGService(rag_chain)



UPLOAD_DIR = "data/uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    filepath = os.path.join(
        UPLOAD_DIR,
        file.filename
    )

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    num_chunks = ingestion_service.ingest(filepath)

    return {
        "message": "Document indexed",
        "filename": file.filename,
        "chunks": num_chunks
    }


@app.post("/query")
def query(request: QueryRequest):
    logger.info("/query endpoint called")
    return service.query(request.query)


@app.post("/stream")
def stream(request: QueryRequest):

    def event_generator():

        chunks = service.rag_chain._prepare_chunks(request.query)

        for token in service.stream(request.query):
            yield f"data: {json.dumps({'token': token})}\n\n"

        payload = {
            "done": True,
            "sources": [
                {
                    "page": c.payload.get("page"),
                    "text": c.payload["text"][:120]
                }
                for c in chunks
            ]
        }

        yield f"data: {json.dumps(payload)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

@app.get("/version")
def version():
    return {
        "service": "production-rag-system",
        "version": "0.1.0"
    }