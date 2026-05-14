"""
Document loader — supports PDF, DOCX, TXT, MD, and CSV files.
Returns a list of LangChain Documents with rich metadata.
"""
import os
import uuid
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
)

SUPPORTED_EXTENSIONS = {
    ".pdf": "PDF",
    ".docx": "Word Document",
    ".txt": "Plain Text",
    ".md": "Markdown",
    ".csv": "CSV",
}

def generate_doc_id(filepath: str) -> str:
    with open(filepath, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()[:12]
    return f"doc_{file_hash}"

def load_document(filepath: str) -> Tuple[List[Document], dict]:
    path = Path(filepath)
    ext = path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported types: {', '.join(SUPPORTED_EXTENSIONS.keys())}"
        )

    doc_id = generate_doc_id(filepath)
    file_size_kb = round(os.path.getsize(filepath) / 1024, 2)
    upload_time = datetime.now(timezone.utc).isoformat()
    file_type = SUPPORTED_EXTENSIONS[ext]

    if ext == ".pdf":
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        for i, doc in enumerate(docs):
            doc.metadata.update({
                "doc_id": doc_id,
                "filename": path.name,
                "file_type": file_type,
                "chunk_index": i,
                "upload_time": upload_time,
            })

    elif ext == ".docx":
        loader = Docx2txtLoader(filepath)
        docs = loader.load()
        for i, doc in enumerate(docs):
            doc.metadata.update({
                "doc_id": doc_id,
                "filename": path.name,
                "file_type": file_type,
                "chunk_index": i,
                "upload_time": upload_time,
            })

    elif ext == ".txt":
        loader = TextLoader(filepath, autodetect_encoding=True)
        docs = loader.load()
        for i, doc in enumerate(docs):
            doc.metadata.update({
                "doc_id": doc_id,
                "filename": path.name,
                "file_type": file_type,
                "chunk_index": i,
                "upload_time": upload_time,
            })

    elif ext == ".md":
        loader = UnstructuredMarkdownLoader(filepath)
        docs = loader.load()
        for i, doc in enumerate(docs):
            doc.metadata.update({
                "doc_id": doc_id,
                "filename": path.name,
                "file_type": file_type,
                "chunk_index": i,
                "upload_time": upload_time,
            })

    elif ext == ".csv":
        loader = CSVLoader(filepath)
        docs = loader.load()
        for i, doc in enumerate(docs):
            doc.metadata.update({
                "doc_id": doc_id,
                "filename": path.name,
                "file_type": file_type,
                "chunk_index": i,
                "upload_time": upload_time,
            })

    total_chars = sum(len((d.page_content or "").strip()) for d in docs)
    if not docs or total_chars == 0:
        raise ValueError(
            f"No extractable text found in '{path.name}'. "
            "If this is a scanned PDF or image, OCR it before uploading."
        )

    doc_metadata = {
        "doc_id": doc_id,
        "filename": path.name,
        "file_type": file_type,
        "file_size_kb": file_size_kb,
        "upload_time": upload_time,
        "num_pages": len(docs),
    }

    return docs, doc_metadata
