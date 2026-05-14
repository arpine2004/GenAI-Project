"""
Image Generation Agent — FLUX.1-schnell via HuggingFace InferenceClient

Two-stage pipeline:
  1. Claude reads the document and writes an optimised image prompt.
  2. The prompt is sent to the HF InferenceClient which routes to a supported
     provider and returns a PNG image.

Uses huggingface_hub.InferenceClient (recommended modern approach) which handles
provider routing automatically — avoids hardcoding a provider that may not
support the selected model.

This is the only model in the project that generates a non-text output (image),
making it architecturally distinct from Claude (text→text) and Whisper (audio→text).
"""
from __future__ import annotations
import base64
import io
from huggingface_hub import InferenceClient
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from ..config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    CLAUDE_MAX_TOKENS,
    HF_TOKEN,
    IMAGE_GEN_MODEL,
)
from ..vectorstore.store import get_all_document_chunks, get_document_info

_PROMPT_SYSTEM = """You are an expert at writing prompts for AI image generation models like FLUX.
Given a document excerpt (and optionally a user instruction), write a single vivid, detailed
visual prompt (max 70 words) that an image model can use.

Rules:
- Describe a concrete visual scene, not abstract concepts.
- Include art style, lighting, and composition details.
- Do NOT include words like "document", "text", "paper", or "summary".
- If a user instruction is given, it OVERRIDES the document — the image should depict
  what the user asked for, with the document only providing thematic context (subject,
  setting, mood, named entities) where it doesn't conflict with the user's request.
- If no user instruction is given, generate an image representing the document's core theme.
- Output ONLY the prompt — no explanation, no preamble."""

def _build_image_prompt(doc_id: str, user_prompt: str = "") -> str:
    chunks = get_all_document_chunks(doc_id)
    excerpt = "\n\n".join(c.page_content.strip() for c in chunks[:6])[:3000]
    user_prompt = (user_prompt or "").strip()
    if user_prompt:
        human_content = (
            f"User instruction (drives the image):\n{user_prompt}\n\n"
            f"Document excerpt (use as context only — only borrow from it where it fits "
            f"the user's instruction):\n{excerpt}"
        )
    else:
        human_content = f"Document excerpt:\n{excerpt}"
    llm = ChatAnthropic(
        model=CLAUDE_MODEL,
        anthropic_api_key=ANTHROPIC_API_KEY,
        max_tokens=150,
    )
    response = llm.invoke([
        SystemMessage(content=_PROMPT_SYSTEM),
        HumanMessage(content=human_content),
    ])
    content = response.content
    if isinstance(content, list):
        text = "".join(
            b.get("text", "") if isinstance(b, dict) else getattr(b, "text", str(b))
            for b in content
        )
    else:
        text = str(content)
    return text.strip()

def _call_hf(prompt: str) -> bytes:
    if not HF_TOKEN:
        raise ValueError(
            "HF_TOKEN is not set. Add HF_TOKEN=<your_token> to your .env file. "
            "Get a free token at https://huggingface.co/settings/tokens"
        )

    client = InferenceClient(
        provider="auto",  
        api_key=HF_TOKEN,
    )
    image = client.text_to_image(prompt, model=IMAGE_GEN_MODEL)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()

def generate_image(doc_id: str, user_prompt: str = "") -> dict:
    doc_info = get_document_info(doc_id)
    if not doc_info:
        raise ValueError(f"Document '{doc_id}' not found in registry.")
    image_prompt = _build_image_prompt(doc_id, user_prompt=user_prompt)
    image_bytes  = _call_hf(image_prompt)
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    return {
        "doc_id":       doc_id,
        "filename":     doc_info["filename"],
        "image_base64": image_base64,
        "prompt_used":  image_prompt,
        "model_used":   f"{IMAGE_GEN_MODEL} via HuggingFace Inference API",
    }
