from typing import List, Optional
from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    document_ids: Optional[List[str]] = None  # restrict search to these docs


class Source(BaseModel):
    document_id: Optional[str] = None
    filename: Optional[str] = None
    page: Optional[int] = None
    score: Optional[float] = None
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    latency_ms: Optional[float] = None
    route: Optional[str] = None            # "retrieve" | "direct"
    sub_questions: Optional[List[str]] = None


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    num_chunks: int


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    num_chunks: int
    status: str = "indexed"


class DocumentList(BaseModel):
    documents: List[DocumentInfo]
    total: int
