import json
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from src.api.schemas import QueryRequest, QueryResponse
from src.api.dependencies import get_service
from src.services.rag_service import RAGService

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, service: RAGService = Depends(get_service)):
    return service.query(request.query, document_ids=request.document_ids)


@router.post("/stream")
def stream(request: QueryRequest, service: RAGService = Depends(get_service)):
    def event_generator():
        for event in service.stream(request.query, document_ids=request.document_ids):
            yield f"data: {json.dumps(event)}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")
