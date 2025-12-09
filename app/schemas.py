from pydantic import BaseModel
from typing import List, Optional

class UploadResponse(BaseModel):
    success: bool
    message: str
    uploaded_file: Optional[str]

class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    filename: str
    score: float

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    confidence: float
    retrieved: List[RetrievedChunk]
    safe: bool
    message: Optional[str]