"""
Question Generation Agent — Claude (Anthropic)

Uses Claude to generate high-quality study questions from document chunks.
Claude produces contextually accurate, pedagogically sound questions that
directly reflect the document content — a significant quality improvement
over small local models like Flan-T5 base.
"""
from __future__ import annotations
import re
from typing import List
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from ..config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    CLAUDE_MAX_TOKENS,
    CLAUDE_TEMPERATURE,
    QG_MODEL,
)
from ..vectorstore.store import get_all_document_chunks, get_document_info

_QG_SYSTEM = """You are an expert educator creating study questions from document content.
Generate clear, specific questions that test genuine understanding of the material.
Output ONLY a numbered list of questions — no preamble, no explanations."""

def generate_questions(doc_id: str, num_questions: int = 5) -> dict:
    doc_info = get_document_info(doc_id)
    if not doc_info:
        raise ValueError(f"Document '{doc_id}' not found in registry.")
    filename = doc_info["filename"]
    chunks = get_all_document_chunks(doc_id)
    if not chunks:
        raise ValueError(f"No content found for document '{doc_id}'.")
    excerpt = "\n\n".join(c.page_content.strip() for c in chunks[:8])
    excerpt = excerpt[:4000]  
    llm = ChatAnthropic(
        model=CLAUDE_MODEL,
        anthropic_api_key=ANTHROPIC_API_KEY,
        max_tokens=CLAUDE_MAX_TOKENS,
        temperature=CLAUDE_TEMPERATURE,
    )
    prompt = (
        f"Read the following excerpt from '{filename}' and generate "
        f"{num_questions} study questions that test understanding of its "
        f"key concepts. Number each question.\n\n"
        f"Document excerpt:\n{excerpt}"
    )

    response = llm.invoke([
        SystemMessage(content=_QG_SYSTEM),
        HumanMessage(content=prompt),
    ])
    raw = response.content if isinstance(response.content, str) else "".join(
        b.get("text", "") if isinstance(b, dict) else getattr(b, "text", "")
        for b in response.content
    )

    questions = _parse_questions(raw.strip(), num_questions)
    return {
        "doc_id":     doc_id,
        "filename":   filename,
        "questions":  questions,
        "model_used": f"Claude ({CLAUDE_MODEL})",
    }

_QUESTION_STARTERS = (
    "what", "why", "how", "when", "where", "who", "which",
    "is", "are", "does", "do", "did", "can", "could", "should",
    "would", "will", "explain", "describe", "list", "name",
)

def _looks_like_question(line: str) -> bool:
    if line.endswith('?'):
        return True
    first = line.split(None, 1)[0].lower() if line else ""
    return first in _QUESTION_STARTERS

def _split_inline_questions(text: str) -> List[str]:
    parts = re.split(r'(?<=\?)\s+', text)
    return [p.strip() for p in parts if p.strip()]

def _parse_questions(raw: str, max_n: int) -> List[str]:
    questions: List[str] = []
    raw_lines = raw.splitlines() if "\n" in raw else _split_inline_questions(raw)
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'^[Qq]?\d+[\.\)\:]\s*', '', line).strip()
        line = re.sub(r'^[-•*]\s*', '', line).strip()
        if len(line) < 8:
            continue
        if not line.endswith('?'):
            line = line + '?'
        questions.append(line)

    if not questions and raw.strip():
        fallback = raw.strip()
        if not fallback.endswith('?'):
            fallback += '?'
        questions.append(fallback)
    seen: set = set()
    unique: List[str] = []
    for q in questions:
        key = q.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(q)
    return unique[:max_n]
