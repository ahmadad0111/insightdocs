from pydantic import BaseModel
from typing import List, Dict, Any


class QueryRequest(BaseModel):
    query: str


class Source(BaseModel):
    page: int | None = None
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]