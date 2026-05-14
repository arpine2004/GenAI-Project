"""
Summarization Agent — Claude (Anthropic)
Map-reduce summarization over full documents using Claude.
  1. Map:    Each chunk is summarized individually by Claude.
  2. Reduce: All chunk summaries are combined into one coherent executive summary.
"""
from __future__ import annotations
from typing import List, Optional
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from ..config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    CLAUDE_MAX_TOKENS,
    CLAUDE_TEMPERATURE,
    SUMMARIZE_MAX_CHUNKS,
)
from ..models import SummarizeResponse
from ..vectorstore.store import get_all_document_chunks, get_document_info

_CHUNK_SYSTEM = """You are a document summarizer.
Summarize the following text excerpt concisely, preserving key facts, decisions, and insights.
Focus on substance, not structure."""

_FINAL_SYSTEM = """You are a senior analyst preparing an executive summary.
Given a collection of partial summaries from different sections of the same document,
produce a single coherent summary. Include a "Key Points" section at the end
(prefix each point with '• ')."""

def _get_llm() -> ChatAnthropic:
    kwargs = dict(
        model=CLAUDE_MODEL,
        anthropic_api_key=ANTHROPIC_API_KEY,
        max_tokens=CLAUDE_MAX_TOKENS,
    )
    if not CLAUDE_MODEL.startswith("claude-opus-4-6"):
        kwargs["temperature"] = 0.2
    return ChatAnthropic(**kwargs)

def _content_to_text(content) -> str:
    if isinstance(content, list):
        return "".join(
            (b.get("text", "") if isinstance(b, dict) else getattr(b, "text", str(b)))
            for b in content
        )
    return str(content)

def _length_to_words(length: str) -> str:
    return {"short": "~100 words", "medium": "~250 words", "long": "~500 words"}.get(length, "~250 words")

def summarize_document(
    doc_id: str,
    focus: Optional[str] = None,
    length: str = "medium",
) -> SummarizeResponse:
    doc_info = get_document_info(doc_id)
    if not doc_info:
        raise ValueError(f"Document '{doc_id}' not found in registry.")
    filename   = doc_info["filename"]
    all_chunks = get_all_document_chunks(doc_id)
    if not all_chunks:
        raise ValueError(f"No content found for document '{doc_id}'.")

    all_chunks  = all_chunks[:SUMMARIZE_MAX_CHUNKS]
    llm         = _get_llm()
    target_len  = _length_to_words(length)
    focus_note  = f" Focus especially on: {focus}." if focus else ""

    chunk_summaries: List[str] = []
    for doc in all_chunks:
        text = doc.page_content.strip()
        if len(text) < 50:
            continue
        resp = llm.invoke([
            SystemMessage(content=_CHUNK_SYSTEM),
            HumanMessage(content=f"Summarize this excerpt from '{filename}'.{focus_note}\n\n{text}"),
        ])
        chunk_summaries.append(_content_to_text(resp.content).strip())

    if not chunk_summaries:
        raise ValueError("Document has no substantive content to summarize.")

    combined = "\n\n---\n\n".join(
        f"Section {i+1}:\n{s}" for i, s in enumerate(chunk_summaries)
    )
    focus_line   = f"Pay special attention to: {focus}.\n" if focus else ""
    reduce_prompt = (
        f"Document: '{filename}'.\n{focus_line}"
        f"Target length: {target_len}.\n\n"
        f"Below are section summaries. Write a coherent executive summary, "
        f"then list 3–6 key points (prefix each with '• ').\n\n{combined}"
    )

    final_resp = llm.invoke([
        SystemMessage(content=_FINAL_SYSTEM),
        HumanMessage(content=reduce_prompt),
    ])
    raw = _content_to_text(final_resp.content).strip()

    key_points, summary_lines = [], []
    for line in raw.splitlines():
        s = line.strip()
        if s.startswith("•"):
            key_points.append(s.lstrip("•").strip())
        else:
            summary_lines.append(line)

    return SummarizeResponse(
        doc_id=doc_id,
        filename=filename,
        summary="\n".join(summary_lines).strip(),
        key_points=key_points,
        model_used=CLAUDE_MODEL,
    )
