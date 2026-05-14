"""
RAG Agent - the core intelligence layer.

Capabilities:
  1. Document Retrieval - semantic search over the knowledge base
  2. Multi-Step Reasoning - decomposes complex questions into sub-questions
  3. Citation/Source Tracking - every answer includes grounded citations
  4. Summarization - map-reduce summarization over full documents
  5. Keyword Extraction - embedding-based keyphrase extraction 

Uses Anthropic Claude and Sentence-Transformers.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import re
import textwrap
from typing import List, Optional, Tuple

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from ..config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    CLAUDE_ALT_MODEL,
    CLAUDE_MAX_TOKENS,
    CLAUDE_TEMPERATURE,
    EMBEDDING_MODEL,
    GROQ_API_KEY,
    LLAMA_MODEL,
    RETRIEVAL_TOP_K,
)
from ..models import Citation, KeywordItem, KeywordResponse, QueryResponse
from ..vectorstore.store import get_all_document_chunks, get_document_info, similarity_search


def _get_llm(
    temperature: float = CLAUDE_TEMPERATURE,
    model: Optional[str] = None,
):
    active = model or CLAUDE_MODEL
    name = active.lower()

    if name.startswith("claude"):
        kwargs = dict(
            model=active,
            anthropic_api_key=ANTHROPIC_API_KEY,
            max_tokens=CLAUDE_MAX_TOKENS,
        )
        if not active.startswith("claude-opus-4-6"):
            kwargs["temperature"] = temperature
        return ChatAnthropic(**kwargs)

    if name.startswith("llama") or "llama" in name:
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not configured. Sign up at https://console.groq.com, "
                "create an API key, and add GROQ_API_KEY=gsk_... to your .env file."
            )
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=active,
            api_key=GROQ_API_KEY,
            max_tokens=CLAUDE_MAX_TOKENS,
            temperature=temperature,
        )

    raise ValueError(f"Unknown model: {active}")


_ST_MODEL = None

def _get_st_model():
    global _ST_MODEL
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _ST_MODEL = SentenceTransformer(EMBEDDING_MODEL)
    return _ST_MODEL


# System Prompts

_QA_SYSTEM = """You are a precise document-grounded question-answering assistant.

ABSOLUTE RULES (violating any of these is a failure):
1. The user's message lists EXACTLY which sources exist. You may ONLY cite source
   numbers that appear in that list. If only Source 1 is listed, you may write
   [Source 1] or [Source 1, Chunk N] — never [Source 2], [Source 3], etc.
2. NEVER invent document names, filenames, source numbers, or chunk content that
   was not in the provided excerpts.
3. NEVER echo the context layout in your answer. Do NOT output lines starting
   with `===` or block headers like `[Source N, Chunk M] (relevance ...)`.
   Citations belong INLINE inside prose only, e.g. "Tumanyan was a poet [Source 1, Chunk 2]."
4. If the provided context does not contain enough information, say so plainly
   ("The provided documents do not contain information about X.") instead of
   filling the gap with invented content or outside knowledge.
5. Do not use any knowledge beyond what the excerpts provide.

Format: plain natural-language prose answering the question, with [Source N] or
[Source N, Chunk M] citations attached to specific claims."""

_DECOMPOSE_SYSTEM = """You are a reasoning planner. Given a complex question, break it into
2–4 simpler sub-questions that together fully answer the original question.

Output ONLY a numbered list of sub-questions, one per line. No preamble, no explanation."""



def _content_to_text(content) -> str:
    if isinstance(content, list):
        return "".join(
            (block.get("text", "") if isinstance(block, dict) else getattr(block, "text", str(block)))
            for block in content
        )
    return str(content)


def _format_context(results: List[Tuple[Document, float]]):
    from collections import OrderedDict
    grouped: "OrderedDict[str, list]" = OrderedDict()
    for doc, score in results:
        doc_id = doc.metadata.get("doc_id", "unknown")
        grouped.setdefault(doc_id, []).append((doc, score))

    parts: List[str] = []
    source_filenames: List[str] = []
    for source_idx, (doc_id, items) in enumerate(grouped.items(), start=1):
        filename = items[0][0].metadata.get("filename", "Unknown")
        source_filenames.append(filename)
        parts.append(f"--- Source {source_idx} ({filename}) ---")
        for chunk_idx, (doc, score) in enumerate(items, start=1):
            meta = doc.metadata
            page = meta.get("page", meta.get("chunk_index", "?"))
            excerpt = doc.page_content.strip()[:800]
            parts.append(
                f"[Source {source_idx}, Chunk {chunk_idx}] (relevance {score:.2f}, page/chunk {page})\n{excerpt}"
            )
    return "\n\n".join(parts), source_filenames


def _build_citations(results: List[Tuple[Document, float]]) -> List[Citation]:
    citations = []
    for doc, score in results:
        meta = doc.metadata
        doc_id = meta.get("doc_id", "unknown")
        citations.append(
            Citation(
                doc_id=doc_id,
                filename=meta.get("filename", "Unknown"),
                chunk_index=meta.get("chunk_index", 0),
                page=meta.get("page"),
                score=round(score, 3),
                excerpt=doc.page_content.strip()[:200] + "…",
            )
        )
    return citations


def _decompose_question(question: str, llm) -> List[str]:
    response = llm.invoke([
        SystemMessage(content=_DECOMPOSE_SYSTEM),
        HumanMessage(content=f"Complex question: {question}"),
    ])
    raw = _content_to_text(response.content).strip()
    sub_questions = []
    for line in raw.splitlines():
        line = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
        if line:
            sub_questions.append(line)
    return sub_questions[:4]


def _is_complex_question(question: str) -> bool:
    complexity_signals = [
        " and ", " compare", "difference between", "how does", "why does",
        "what are the", "list all", "explain", "steps to", " vs ", "both",
    ]
    q_lower = question.lower()
    return any(sig in q_lower for sig in complexity_signals)


def answer_question(
    question: str,
    doc_ids: Optional[List[str]] = None,
    top_k: int = RETRIEVAL_TOP_K,
    multi_step: bool = True,
    model: Optional[str] = None,
) -> QueryResponse:
    active_model = model or CLAUDE_MODEL
    llm = _get_llm(model=active_model)
    reasoning_steps: List[str] = [f"Using model: {active_model}"]
    all_results: List[Tuple[Document, float]] = []

    if multi_step and _is_complex_question(question):
        sub_questions = _decompose_question(question, llm)
        reasoning_steps.append(
            f"Decomposed into {len(sub_questions)} sub-questions: "
            + " | ".join(f'"{q}"' for q in sub_questions)
        )

        seen_ids = set()
        for sub_q in sub_questions:
            results = similarity_search(sub_q, top_k=max(2, top_k // 2), doc_ids=doc_ids)
            for doc, score in results:
                chunk_key = (doc.metadata.get("doc_id"), doc.metadata.get("chunk_index"))
                if chunk_key not in seen_ids:
                    seen_ids.add(chunk_key)
                    all_results.append((doc, score))

        all_results.sort(key=lambda x: x[1], reverse=True)
        all_results = all_results[:top_k]
        reasoning_steps.append(f"Retrieved {len(all_results)} unique chunks across all sub-questions.")
    else:
        all_results = similarity_search(question, top_k=top_k, doc_ids=doc_ids)
        reasoning_steps.append(f"Retrieved {len(all_results)} relevant chunks.")

    if not all_results:
        return QueryResponse(
            answer=(
                "I could not find relevant information in the knowledge base to answer "
                "your question. Please make sure related documents have been uploaded, "
                "or try rephrasing your question."
            ),
            citations=[],
            reasoning_steps=reasoning_steps,
            model_used=active_model,
        )

    context, source_filenames = _format_context(all_results)
    citations = _build_citations(all_results)

    num_sources = len(source_filenames)
    if num_sources == 1:
        sources_line = f"You have exactly 1 source: [Source 1] = \"{source_filenames[0]}\". You may ONLY cite [Source 1] or [Source 1, Chunk M]."
    else:
        listed = "; ".join(f"[Source {i+1}] = \"{n}\"" for i, n in enumerate(source_filenames))
        sources_line = (
            f"You have exactly {num_sources} sources: {listed}. "
            f"You may ONLY cite [Source 1] through [Source {num_sources}] — no other source numbers exist."
        )

    user_prompt = f"""{sources_line}

Question: {question}

Document Excerpts (do not echo this structure in your answer — only cite inline):
{context}

Write a natural-prose answer with inline [Source N] or [Source N, Chunk M] citations. Do not invent sources that are not listed above."""

    response = llm.invoke([
        SystemMessage(content=_QA_SYSTEM),
        HumanMessage(content=user_prompt),
    ])

    answer = _content_to_text(response.content).strip()

    def _scrub_invalid_sources(text: str, n_real: int) -> str:
        def repl(m):
            try:
                src_num = int(m.group(1))
            except ValueError:
                return m.group(0)
            if 1 <= src_num <= n_real:
                return m.group(0)
            return "[invalid citation removed]"
        return re.sub(r"\[Source\s+(\d+)(?:,\s*Chunk\s+\d+)?\]", repl, text)

    answer = _scrub_invalid_sources(answer, num_sources)

    usage = getattr(response, "usage_metadata", None) or {}
    total_tokens = None
    if usage:
        input_t = usage.get("input_tokens", 0) or 0
        output_t = usage.get("output_tokens", 0) or 0
        if input_t or output_t:
            total_tokens = input_t + output_t

    reasoning_steps.append("Generated answer grounded in retrieved context.")

    return QueryResponse(
        answer=answer,
        citations=citations,
        reasoning_steps=reasoning_steps,
        model_used=active_model,
        tokens_used=total_tokens,
    )

_STOP_WORDS = {
    'the','a','an','and','or','but','in','on','at','to','for','of','with','by',
    'from','as','is','are','was','were','be','been','have','has','had','do',
    'does','did','will','would','can','could','should','may','might','must',
    'this','that','these','those','it','its','they','them','their','we','our',
    'you','your','he','she','his','her','not','also','more','than','such',
    'when','which','who','how','what','where','there','here','some','any',
    'all','each','every','both','few','most','other','into','about','after',
    'before','between','through','during','against','above','below','since',
    'without','within','along','across','just','very','so','then','now','up',
    'out','no','only','same','because','while','although','however','therefore',
}


def extract_keywords(doc_id: str, top_n: int = 10) -> KeywordResponse:
    doc_info = get_document_info(doc_id)
    if not doc_info:
        raise ValueError(f"Document '{doc_id}' not found in registry.")

    filename = doc_info["filename"]

    chunks = get_all_document_chunks(doc_id)
    if not chunks:
        raise ValueError(f"No content found for document '{doc_id}'.")

    chunk_texts = [c.page_content.strip() for c in chunks if c.page_content.strip()]
    full_text = " ".join(chunk_texts)

    tokens = re.findall(r'\b[a-zA-Z][a-zA-Z]{2,}\b', full_text.lower())
    tokens = [t for t in tokens if t not in _STOP_WORDS]

    seen: set = set()
    candidates: List[str] = []
    for i, tok in enumerate(tokens):
        if tok not in seen:
            seen.add(tok)
            candidates.append(tok)
        if i + 1 < len(tokens):
            bigram = f"{tok} {tokens[i + 1]}"
            if bigram not in seen:
                seen.add(bigram)
                candidates.append(bigram)

    candidates = candidates[:600]
    if not candidates:
        raise ValueError("Document has no extractable content.")

    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    st = _get_st_model()
    chunk_embs = st.encode(
        chunk_texts,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=False,
    )
    doc_vec = np.mean(chunk_embs, axis=0)
    norm = np.linalg.norm(doc_vec)
    if norm > 0:
        doc_vec = doc_vec / norm
    doc_emb = doc_vec.reshape(1, -1)

    cand_embs = st.encode(
        candidates,
        normalize_embeddings=True,
        batch_size=128,
        show_progress_bar=False,
    )

    scores = cosine_similarity(doc_emb, cand_embs)[0]
    ranked = sorted(zip(candidates, scores.tolist()), key=lambda x: x[1], reverse=True)

    selected: List[KeywordItem] = []
    selected_word_sets: List[set] = []
    for phrase, score in ranked:
        phrase_words = set(phrase.split())
        if any(phrase_words <= s for s in selected_word_sets):
            continue
        selected.append(KeywordItem(phrase=phrase, score=round(score, 3)))
        selected_word_sets.append(phrase_words)
        if len(selected) >= top_n:
            break

    return KeywordResponse(
        doc_id=doc_id,
        filename=filename,
        keywords=selected,
        model_used=f"Sentence-Transformers ({EMBEDDING_MODEL})",
    )
