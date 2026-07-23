"""Document management: upload/index, list, delete, and re-index."""
import os
import shutil

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException

from src.api.schemas import IngestResponse, DocumentList
from src.api.dependencies import get_ingestion, get_vector_store
from src.services.document_ingestion_service import DocumentIngestionService
from src.rag.retrieval.vector_store import VectorStore
from src.core.logging import logger

router = APIRouter(prefix="/documents", tags=["documents"])

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "data/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _save(file: UploadFile) -> str:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    path = os.path.join(UPLOAD_DIR, file.filename)
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return path


@router.post("", response_model=IngestResponse)
async def upload_document(
    file: UploadFile = File(...),
    ingestion: DocumentIngestionService = Depends(get_ingestion),
):
    """Upload and index a PDF. Re-uploading the same file updates it in place."""
    path = _save(file)
    try:
        return ingestion.ingest(path)
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")


@router.get("", response_model=DocumentList)
def list_documents(ingestion: DocumentIngestionService = Depends(get_ingestion)):
    docs = ingestion.list_documents()
    return {"documents": docs, "total": len(docs)}


@router.delete("/{document_id}")
def delete_document(
    document_id: str,
    ingestion: DocumentIngestionService = Depends(get_ingestion),
):
    return ingestion.delete(document_id)


@router.post("/reindex", response_model=IngestResponse)
async def reindex_document(
    file: UploadFile = File(...),
    ingestion: DocumentIngestionService = Depends(get_ingestion),
):
    """Explicit re-index (same as upload; kept for a clear, intentional API)."""
    path = _save(file)
    return ingestion.ingest(path)


@router.delete("")
def reset_collection(vector_store: VectorStore = Depends(get_vector_store)):
    """Danger: wipes the entire vector collection."""
    vector_store.reset_collection()
    return {"status": "collection reset"}
