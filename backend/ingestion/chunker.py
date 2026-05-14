"""
Text chunking - splits loaded documents into overlapping chunks
suitable for embedding and retrieval process.
"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(docs)

    doc_chunk_counters: dict = {}
    for chunk in chunks:
        doc_id = chunk.metadata.get("doc_id", "unknown")
        idx = doc_chunk_counters.get(doc_id, 0)
        chunk.metadata["chunk_index"] = idx
        doc_chunk_counters[doc_id] = idx + 1

    return chunks
