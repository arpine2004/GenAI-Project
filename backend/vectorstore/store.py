"""
Vector store manager - wraps ChromaDB with LangChain's Chroma integration.

Responsibilities:
  - Add / delete documents
  - Similarity search with score filtering
  - Doc-scoped filtering via metadata
  - Persistence across restarts (chromadb on-disk)
"""

import json
import os
from typing import List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from ..config import CHROMA_DIR, EMBEDDING_MODEL, RETRIEVAL_TOP_K, RETRIEVAL_SCORE_THRESHOLD

_embedding_fn: Optional[HuggingFaceEmbeddings] = None
_vector_store: Optional[Chroma] = None

_DOC_REGISTRY_PATH = os.path.join(CHROMA_DIR, "doc_registry.json")
_doc_registry: dict = {}


def _load_registry() -> None:
    global _doc_registry
    if os.path.exists(_DOC_REGISTRY_PATH):
        with open(_DOC_REGISTRY_PATH, "r", encoding="utf-8") as f:
            try:
                _doc_registry = json.load(f)
            except json.JSONDecodeError:
                _doc_registry = {}
                _save_registry()


def _save_registry() -> None:
    with open(_DOC_REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(_doc_registry, f, indent=2, ensure_ascii=False)


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embedding_fn
    if _embedding_fn is None:
        _embedding_fn = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedding_fn


def get_vector_store() -> Chroma:
    global _vector_store
    if _vector_store is None:
        _load_registry()
        _vector_store = Chroma(
            collection_name="knowledge_base",
            embedding_function=get_embeddings(),
            persist_directory=CHROMA_DIR,
            collection_metadata={"hnsw:space": "cosine"},
        )
    return _vector_store


def add_documents(chunks: List[Document], doc_metadata: dict) -> int:
    doc_id = doc_metadata["doc_id"]

    _load_registry()
    if doc_id in _doc_registry:
        store = get_vector_store()
        raw = store._collection.get(where={"doc_id": doc_id})
        chunk_ids = raw.get("ids", [])
        if chunk_ids:
            store._collection.delete(ids=chunk_ids)

    store = get_vector_store()
    store.add_documents(chunks)

    _doc_registry[doc_id] = {
        **doc_metadata,
        "num_chunks": len(chunks),
    }
    _save_registry()

    return len(chunks)


def delete_document(doc_id: str) -> bool:
    if doc_id not in _doc_registry:
        return False

    store = get_vector_store()
    raw = store._collection.get(where={"doc_id": doc_id})
    chunk_ids = raw.get("ids", [])
    if chunk_ids:
        store._collection.delete(ids=chunk_ids)

    del _doc_registry[doc_id]
    _save_registry()
    return True


def list_documents() -> List[dict]:
    _load_registry()
    return list(_doc_registry.values())


def document_exists(doc_id: str) -> bool:
    _load_registry()
    return doc_id in _doc_registry


def get_document_info(doc_id: str) -> Optional[dict]:
    _load_registry()
    return _doc_registry.get(doc_id)


def similarity_search(
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    doc_ids: Optional[List[str]] = None,
    score_threshold: float = RETRIEVAL_SCORE_THRESHOLD,
) -> List[Tuple[Document, float]]:

    store = get_vector_store()

    filter_dict = None
    if doc_ids:
        if len(doc_ids) == 1:
            filter_dict = {"doc_id": doc_ids[0]}
        else:
            filter_dict = {"doc_id": {"$in": doc_ids}}

    try:
        if filter_dict:
            raw = store._collection.get(where=filter_dict)
            available = len(raw.get("ids", []))
        else:
            available = store._collection.count()
    except Exception:
        available = top_k
    effective_k = max(1, min(top_k, available)) if available else 1

    try:
        results = store.similarity_search_with_relevance_scores(
            query=query,
            k=effective_k,
            filter=filter_dict,
        )
    except RuntimeError:
        results = store.similarity_search_with_relevance_scores(
            query=query,
            k=1,
            filter=filter_dict,
        )

    filtered = [(doc, score) for doc, score in results if score >= score_threshold]
    return filtered


def get_all_document_chunks(doc_id: str) -> List[Document]:
    store = get_vector_store()
    raw = store.get(where={"doc_id": doc_id})

    documents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []

    chunks: List[Document] = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(documents, metadatas)
    ]
    chunks.sort(key=lambda d: d.metadata.get("chunk_index", 0))
    return chunks
