"""
Pydantic models for request/response schemas.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# Document Models 

class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    num_chunks: int
    file_size_kb: float
    upload_time: str
    file_type: str


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total: int


class DeleteResponse(BaseModel):
    success: bool
    message: str
    doc_id: str


# Citation Models

class Citation(BaseModel):
    doc_id: str
    filename: str
    chunk_index: int
    page: Optional[int] = None
    score: float
    excerpt: str = Field(..., description="Short excerpt from the source chunk")


# Query Models

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    doc_ids: Optional[List[str]] = Field(
        default=None,
        description="Limit retrieval to specific document IDs. None = search all."
    )
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    multi_step: bool = Field(
        default=True,
        description="Enable multi-step reasoning for complex questions"
    )
    model: Optional[str] = Field(
        default=None,
        description="LLM model override: 'claude-opus-4-6' or 'claude-haiku-4-5-20251001'"
    )


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    reasoning_steps: Optional[List[str]] = None
    model_used: str
    tokens_used: Optional[int] = None


# Summarization Models 

class SummarizeRequest(BaseModel):
    doc_id: str
    focus: Optional[str] = Field(
        default=None,
        description="Optional topic/aspect to focus the summary on"
    )
    length: str = Field(
        default="medium",
        pattern="^(short|medium|long)$",
        description="Summary length: short (~100w), medium (~250w), long (~500w)"
    )


class SummarizeResponse(BaseModel):
    doc_id: str
    filename: str
    summary: str
    key_points: List[str]
    model_used: str


# Comparison Models 

class CompareResponse(BaseModel):
    model_a_name: str
    model_b_name: str
    model_a: QueryResponse
    model_b: QueryResponse


# Keyword Models 

class KeywordItem(BaseModel):
    phrase: str
    score: float


class KeywordRequest(BaseModel):
    doc_id: str
    top_n: int = Field(default=10, ge=3, le=30, description="Number of keywords to extract")


class KeywordResponse(BaseModel):
    doc_id: str
    filename: str
    keywords: List[KeywordItem]
    model_used: str


# Speech-to-Text Models

class TranscribeResponse(BaseModel):
    transcript: str
    model_used: str
    saved_filename: Optional[str] = None   # filename inside audio_recordings/


class RecordingInfo(BaseModel):
    filename: str
    timestamp: str
    transcript_slug: str
    size_kb: float


class RecordingListResponse(BaseModel):
    recordings: List[RecordingInfo]
    total: int


# Image Generation Models

class ImageGenRequest(BaseModel):
    doc_id: str
    user_prompt: Optional[str] = Field(
        default=None,
        description="Optional natural-language instruction. When provided, drives the image; the document is used only for context."
    )


class ImageGenResponse(BaseModel):
    doc_id: str
    filename: str
    image_base64: str
    prompt_used: str
    model_used: str


# Question Generation Models 

class QuestionGenRequest(BaseModel):
    doc_id: str
    num_questions: int = Field(default=5, ge=3, le=10, description="Number of questions to generate")


class QuestionGenResponse(BaseModel):
    doc_id: str
    filename: str
    questions: List[str]
    model_used: str


# Health Models

class HealthResponse(BaseModel):
    status: str
    version: str
    documents_indexed: int
    vector_store: str
    embedding_model: str
    llm_model: str
    available_models: Optional[Dict[str, str]] = None
