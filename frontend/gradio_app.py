"""
Gradio frontend for the RAG Q&A Assistant.

Mounted onto the FastAPI app at /app via gr.mount_gradio_app. Calls the backend
agent modules directly (no HTTP self-calls).
"""

from __future__ import annotations

import base64
import os
import shutil
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr

from backend import config
from backend import chat_history as ch
from backend.agents.image_agent import generate_image as image_generate
from backend.agents.qg_agent import generate_questions as qg_generate
from backend.agents.rag_agent import answer_question, extract_keywords
from backend.agents.stt_agent import transcribe_audio
from backend.agents.summarize_agent import summarize_document
from backend.ingestion.chunker import chunk_documents
from backend.ingestion.loader import SUPPORTED_EXTENSIONS, load_document
from backend.vectorstore.store import (
    add_documents,
    delete_document,
    document_exists,
    list_documents,
)


# Helpers

def _doc_table_rows():
    docs = list_documents()
    if not docs:
        return [["-", "(no documents indexed yet)", 0, 0.0, "", ""]]
    rows = []
    for d in docs:
        rows.append([
            d["doc_id"][:12] + "...",
            d["filename"],
            d.get("num_chunks", 0),
            round(d.get("file_size_kb", 0.0), 1),
            d.get("file_type", ""),
            d.get("upload_time", ""),
        ])
    return rows


def _doc_choices():
    docs = list_documents()
    return [(f"{d['filename']}  [{d['doc_id'][:8]}]", d["doc_id"]) for d in docs]


def _refresh_outputs():
    """Returns 7 values: table, delete_dd, query_dd, summarize_dd, kw_dd, qg_dd, img_dd."""
    rows = _doc_table_rows()
    choices = _doc_choices()
    return (
        rows,
        gr.update(choices=choices, value=None),
        gr.update(choices=choices, value=[]),
        gr.update(choices=choices, value=None),
        gr.update(choices=choices, value=None),
        gr.update(choices=choices, value=None),
        gr.update(choices=choices, value=None),
    )


# Document handlers 

def handle_upload(file_path):
    print(f"[gradio] upload click: file_path={file_path!r}", flush=True)
    if not file_path:
        return ("Please choose a file first.",) + _refresh_outputs()

    save_path = None
    try:
        src = Path(file_path)
        safe_name = src.name
        ext = src.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            msg = f"Unsupported file type {ext}. Supported: {', '.join(SUPPORTED_EXTENSIONS.keys())}"
            return (msg,) + _refresh_outputs()

        save_path = os.path.join(config.UPLOAD_DIR, safe_name)
        print(f"[gradio] copying to {save_path}", flush=True)
        shutil.copyfile(src, save_path)

        print("[gradio] loading & chunking...", flush=True)
        docs, doc_metadata = load_document(save_path)
        chunks = chunk_documents(docs)
        print(f"[gradio] embedding {len(chunks)} chunks...", flush=True)
        num_chunks = add_documents(chunks, doc_metadata)
        print(f"[gradio] indexed {num_chunks} chunks for {safe_name}", flush=True)

        if num_chunks == 0:
            if save_path and os.path.exists(save_path):
                os.remove(save_path)
            return ("Document produced no indexable chunks.",) + _refresh_outputs()

        return (f"Indexed '{safe_name}' - {num_chunks} chunks.",) + _refresh_outputs()
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            if save_path and os.path.exists(save_path):
                os.remove(save_path)
        except Exception:
            pass
        return (f"Upload failed: {type(e).__name__}: {e}",) + _refresh_outputs()


def handle_delete(doc_id):
    print(f"[gradio] delete click: doc_id={doc_id!r}", flush=True)
    if not doc_id:
        return ("Select a document first.",) + _refresh_outputs()
    try:
        success = delete_document(doc_id)
    except Exception as e:
        return (f"Delete failed: {e}",) + _refresh_outputs()
    if not success:
        return (f"Document '{doc_id}' not found.",) + _refresh_outputs()
    try:
        from backend.vectorstore.store import get_document_info
        info = get_document_info(doc_id)
        if info:
            path = os.path.join(config.UPLOAD_DIR, info["filename"])
            if os.path.exists(path):
                os.remove(path)
    except Exception:
        pass
    return (f"Deleted '{doc_id}'.",) + _refresh_outputs()


def handle_refresh():
    print("[gradio] refresh click", flush=True)
    return _refresh_outputs()


def upload_pending():
    return "Uploading... (check terminal for [gradio] log lines)"


# Q&A (chat)

def _format_assistant_message(resp) -> str:
    """Render answer + grouped citations + metadata as a single markdown blob."""
    parts = [resp.answer or ""]

    if resp.citations:
        groups = OrderedDict()
        for c in resp.citations:
            if c.doc_id not in groups:
                groups[c.doc_id] = {"filename": c.filename, "items": []}
            groups[c.doc_id]["items"].append(c)

        parts.append("\n---")
        parts.append("**Citations**")
        for i, group in enumerate(groups.values(), 1):
            parts.append(f"\n**[{i}] {group['filename']}**")
            for c in group["items"]:
                page = f", p. {c.page}" if c.page else ""
                excerpt = " ".join(c.excerpt.split())  
                parts.append(
                    f"> _chunk {c.chunk_index}{page}_  \n> {excerpt}"
                )

    meta_bits = [f"`{resp.model_used}`"]
    if resp.tokens_used:
        meta_bits.append(f"{resp.tokens_used} tokens")
    parts.append(f"\n_Model: {' - '.join(meta_bits)}_")

    if resp.reasoning_steps:
        parts.append("\n<details><summary>Reasoning steps</summary>\n")
        for i, s in enumerate(resp.reasoning_steps, 1):
            parts.append(f"{i}. {s}")
        parts.append("\n</details>")

    return "\n".join(parts)


def handle_new_chat():
    chat_id = ch.create_chat()
    return (
        gr.update(choices=ch.get_chat_choices(), value=chat_id),
        [],
        chat_id,
    )


def handle_select_chat(chat_id):
    if not chat_id:
        return [], None
    return ch.get_messages(chat_id), chat_id


def handle_delete_chat(chat_id):
    if chat_id:
        ch.delete_chat(chat_id)
    chats = ch.list_chats()
    new_active = chats[0]["id"] if chats else None
    msgs = ch.get_messages(new_active) if new_active else []
    return (
        gr.update(choices=ch.get_chat_choices(), value=new_active),
        msgs,
        new_active,
    )


def handle_ask(question, audio_path, chat_id, doc_ids, top_k, multi_step, model):
    question = (question or "").strip()
    voice_used = False

    if not question and audio_path:
        try:
            result = transcribe_audio(audio_path)
            question = (result.get("transcript") or "").strip()
            voice_used = True
        except Exception as e:
            if not chat_id or not ch.chat_exists(chat_id):
                chat_id = ch.create_chat()
            ch.add_message(chat_id, "user", "_(voice message)_")
            ch.add_message(chat_id, "assistant", f"_Transcription failed: {type(e).__name__}: {e}_")
            return (
                gr.update(choices=ch.get_chat_choices(), value=chat_id),
                ch.get_messages(chat_id),
                chat_id,
                "",
                None,
            )

    if not question:
        return gr.update(), gr.update(), chat_id, question, None

    if not chat_id or not ch.chat_exists(chat_id):
        chat_id = ch.create_chat()

    display_question = ("🎤 " + question) if voice_used else question

    if not list_documents():
        ch.add_message(chat_id, "user", display_question)
        ch.add_message(chat_id, "assistant", "_Knowledge base is empty. Upload a document first._")
    elif not config.ANTHROPIC_API_KEY:
        ch.add_message(chat_id, "user", display_question)
        ch.add_message(chat_id, "assistant", "_ANTHROPIC_API_KEY is not configured._")
    else:
        ch.add_message(chat_id, "user", display_question)
        model_choice = None if not model or model == "auto" else model
        try:
            resp = answer_question(
                question=question,
                doc_ids=doc_ids or None,
                top_k=int(top_k),
                multi_step=bool(multi_step),
                model=model_choice,
            )
            assistant_md = _format_assistant_message(resp)
        except Exception as e:
            assistant_md = f"_Error: {type(e).__name__}: {e}_"
        ch.add_message(chat_id, "assistant", assistant_md)

    return (
        gr.update(choices=ch.get_chat_choices(), value=chat_id),
        ch.get_messages(chat_id),
        chat_id,
        "",
        None,
    )


# Summarize/Keywords/QG/Image/Transcribe/Compare


def handle_summarize(doc_id, focus, length):
    if not doc_id:
        return "Select a document first.", "", ""
    if not config.ANTHROPIC_API_KEY:
        return "ANTHROPIC_API_KEY is not configured.", "", ""
    if not document_exists(doc_id):
        return f"Document '{doc_id}' not found.", "", ""
    try:
        resp = summarize_document(
            doc_id=doc_id,
            focus=(focus or "").strip() or None,
            length=length,
        )
    except Exception as e:
        return f"Error: {e}", "", ""
    kp = "\n".join(f"- {p}" for p in resp.key_points)
    return resp.summary, kp, f"**Model:** `{resp.model_used}`"


def handle_keywords(doc_id, top_n):
    if not doc_id:
        return [], ""
    if not document_exists(doc_id):
        return [], f"Document '{doc_id}' not found."
    try:
        resp = extract_keywords(doc_id=doc_id, top_n=int(top_n))
    except Exception as e:
        return [], f"Error: {e}"
    rows = [[k.phrase, round(k.score, 3)] for k in resp.keywords]
    return rows, f"**Model:** `{resp.model_used}`"


def handle_gen_questions(doc_id, num_questions):
    if not doc_id:
        return "Select a document first.", ""
    if not document_exists(doc_id):
        return f"Document '{doc_id}' not found.", ""
    try:
        result = qg_generate(doc_id=doc_id, num_questions=int(num_questions))
    except Exception as e:
        return f"Error: {e}", ""
    qs = "\n".join(f"{i}. {q}" for i, q in enumerate(result["questions"], 1))
    return qs, f"**Model:** `{result['model_used']}`"


def handle_gen_image(doc_id, user_prompt):
    if not doc_id:
        return None, "Select a document first."
    if not config.HF_TOKEN:
        return None, "HF_TOKEN is not configured. Add it to your .env file."
    if not document_exists(doc_id):
        return None, f"Document '{doc_id}' not found."
    try:
        result = image_generate(doc_id, user_prompt=(user_prompt or "").strip())
    except Exception as e:
        return None, f"Error: {e}"
    img_bytes = base64.b64decode(result["image_base64"])
    out_path = os.path.join(tempfile.gettempdir(), f"ragqa_{doc_id[:8]}.png")
    with open(out_path, "wb") as f:
        f.write(img_bytes)
    info = f"**Model:** `{result['model_used']}`\n\n**Prompt used:** {result['prompt_used']}"
    return out_path, info


def handle_compare(question, top_k, multi_step):
    if not (question or "").strip():
        return "Please enter a question.", "", "", ""
    if not list_documents():
        return "Knowledge base is empty. Upload documents first.", "", "", ""
    if not config.ANTHROPIC_API_KEY:
        return "ANTHROPIC_API_KEY is not configured.", "", "", ""
    try:
        ra = answer_question(question, None, int(top_k), bool(multi_step), config.CLAUDE_MODEL)
        rb = answer_question(question, None, int(top_k), bool(multi_step), config.CLAUDE_ALT_MODEL)
    except Exception as e:
        return f"Error: {e}", "", "", ""
    return (
        ra.answer,
        f"**{config.CLAUDE_MODEL}** - {ra.tokens_used or 0} tokens",
        rb.answer,
        f"**{config.CLAUDE_ALT_MODEL}** - {rb.tokens_used or 0} tokens",
    )


# UI

DOC_TABLE_HEADERS = ["Doc ID", "Filename", "Chunks", "Size (KB)", "Type", "Uploaded"]


def build_demo():
    initial_choices = _doc_choices()
    initial_chat_choices = ch.get_chat_choices()
    initial_chat_id = initial_chat_choices[0][1] if initial_chat_choices else None
    initial_messages = ch.get_messages(initial_chat_id) if initial_chat_id else []

    with gr.Blocks(title="RAG Q&A Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# RAG-Powered Document Q&A Assistant\n"
            "Upload documents, ask grounded questions (by text or voice), "
            "summarize, extract keywords, generate study questions, transcribe "
            "audio, compare models, and more."
        )

        with gr.Tabs():
            with gr.Tab("Documents"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Upload")
                        upload_file = gr.File(
                            label=f"Supported: {', '.join(SUPPORTED_EXTENSIONS.keys())}",
                            file_count="single",
                            type="filepath",
                        )
                        upload_btn = gr.Button("Upload & Index", variant="primary")
                        upload_status = gr.Textbox(
                            label="Upload status",
                            value="(no upload yet)",
                            interactive=False,
                            lines=2,
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("### Delete")
                        delete_dropdown = gr.Dropdown(
                            label="Select a document",
                            choices=initial_choices,
                            interactive=True,
                        )
                        delete_btn = gr.Button("Delete", variant="stop")
                        delete_status = gr.Textbox(
                            label="Delete status",
                            value="",
                            interactive=False,
                            lines=2,
                        )

                gr.Markdown("### Indexed documents")
                doc_table = gr.Dataframe(
                    headers=DOC_TABLE_HEADERS,
                    value=_doc_table_rows(),
                    interactive=False,
                    wrap=True,
                )
                refresh_btn = gr.Button("Refresh list")

            with gr.Tab("Q&A"):
                active_chat_state = gr.State(value=initial_chat_id)
                with gr.Row():
                    with gr.Column(scale=1, min_width=240):
                        gr.Markdown("### Chats")
                        new_chat_btn = gr.Button("+ New chat", variant="primary")
                        chat_list = gr.Radio(
                            label="Conversations",
                            choices=initial_chat_choices,
                            value=initial_chat_id,
                            interactive=True,
                        )
                        delete_chat_btn = gr.Button("Delete this chat", variant="stop")

                    with gr.Column(scale=3):
                        with gr.Row():
                            model_choice = gr.Dropdown(
                                label="Model",
                                choices=["auto", config.CLAUDE_MODEL, config.CLAUDE_ALT_MODEL],
                                value="auto",
                                interactive=True,
                                scale=1,
                            )
                            query_doc_filter = gr.Dropdown(
                                label="Ask about (leave empty to search all documents)",
                                choices=initial_choices,
                                multiselect=True,
                                interactive=True,
                                scale=2,
                            )
                        chatbot = gr.Chatbot(
                            value=initial_messages,
                            type="messages",
                            label="Conversation",
                            height=460,
                            show_copy_button=True,
                        )
                        with gr.Row():
                            question_box = gr.Textbox(
                                label="",
                                placeholder="Type a question, or record / upload audio below...",
                                lines=2,
                                scale=5,
                            )
                            ask_btn = gr.Button("Ask", variant="primary", scale=1)
                        qa_audio = gr.Audio(
                            sources=["microphone", "upload"],
                            type="filepath",
                            label="Speak your question",
                        )
                        with gr.Accordion("Advanced options", open=False):
                            top_k_slider = gr.Slider(1, 20, value=config.RETRIEVAL_TOP_K, step=1, label="Top-K chunks")
                            multi_step_box = gr.Checkbox(label="Multi-step reasoning", value=True)

            with gr.Tab("Summarize"):
                with gr.Row():
                    summarize_doc = gr.Dropdown(label="Document", choices=initial_choices)
                    summarize_focus = gr.Textbox(label="Focus (optional)", placeholder="e.g. methodology")
                    summarize_length = gr.Radio(["short", "medium", "long"], value="medium", label="Length")
                summarize_btn = gr.Button("Summarize", variant="primary")
                summary_box = gr.Markdown()
                key_points_box = gr.Markdown()
                summarize_meta = gr.Markdown()

            with gr.Tab("Keywords"):
                with gr.Row():
                    kw_doc = gr.Dropdown(label="Document", choices=initial_choices)
                    kw_n = gr.Slider(3, 30, value=10, step=1, label="Number of keywords")
                kw_btn = gr.Button("Extract", variant="primary")
                kw_table = gr.Dataframe(headers=["Phrase", "Score"], interactive=False)
                kw_meta = gr.Markdown()

            with gr.Tab("Generate questions"):
                with gr.Row():
                    qg_doc = gr.Dropdown(label="Document", choices=initial_choices)
                    qg_n = gr.Slider(3, 10, value=5, step=1, label="Number of questions")
                qg_btn = gr.Button("Generate", variant="primary")
                qg_box = gr.Markdown()
                qg_meta = gr.Markdown()

            with gr.Tab("Image"):
                gr.Markdown(
                    "Pick a source document for context, then describe the image you want. "
                    "If you leave the prompt empty, the image is generated from the document's core theme. "
                    "Requires HF_TOKEN."
                )
                with gr.Row():
                    img_doc = gr.Dropdown(label="Source document", choices=initial_choices)
                img_user_prompt = gr.Textbox(
                    label="Your image prompt (optional)",
                    placeholder="e.g. dark watercolor portrait of the main subject, candle-lit, dramatic shadows",
                    lines=2,
                )
                img_btn = gr.Button("Generate image", variant="primary")
                img_output = gr.Image(label="Generated image", type="filepath")
                img_meta = gr.Markdown()

            with gr.Tab("Compare models"):
                gr.Markdown(f"Compare **{config.CLAUDE_MODEL}** vs **{config.CLAUDE_ALT_MODEL}** side-by-side.")
                cmp_question = gr.Textbox(label="Question", lines=2)
                with gr.Row():
                    cmp_top_k = gr.Slider(1, 20, value=config.RETRIEVAL_TOP_K, step=1, label="Top-K")
                    cmp_multi = gr.Checkbox(label="Multi-step reasoning", value=True)
                cmp_btn = gr.Button("Compare", variant="primary")
                with gr.Row():
                    with gr.Column():
                        cmp_a_meta = gr.Markdown()
                        cmp_a_answer = gr.Markdown()
                    with gr.Column():
                        cmp_b_meta = gr.Markdown()
                        cmp_b_answer = gr.Markdown()

        refresh_outputs = [
            doc_table,
            delete_dropdown,
            query_doc_filter,
            summarize_doc,
            kw_doc,
            qg_doc,
            img_doc,
        ]

        # Documents
        upload_btn.click(
            upload_pending,
            outputs=[upload_status],
        ).then(
            handle_upload,
            inputs=[upload_file],
            outputs=[upload_status] + refresh_outputs,
        )
        delete_btn.click(
            handle_delete,
            inputs=[delete_dropdown],
            outputs=[delete_status] + refresh_outputs,
        )
        refresh_btn.click(handle_refresh, outputs=refresh_outputs)

        # Chat sidebar
        new_chat_btn.click(handle_new_chat, outputs=[chat_list, chatbot, active_chat_state])
        chat_list.change(handle_select_chat, inputs=[chat_list], outputs=[chatbot, active_chat_state])
        delete_chat_btn.click(
            handle_delete_chat,
            inputs=[active_chat_state],
            outputs=[chat_list, chatbot, active_chat_state],
        )

        # Ask: accepts text OR audio (audio is transcribed first if text is empty)
        ask_event_inputs = [
            question_box,
            qa_audio,
            active_chat_state,
            query_doc_filter,
            top_k_slider,
            multi_step_box,
            model_choice,
        ]
        ask_event_outputs = [chat_list, chatbot, active_chat_state, question_box, qa_audio]
        ask_btn.click(handle_ask, inputs=ask_event_inputs, outputs=ask_event_outputs)
        question_box.submit(handle_ask, inputs=ask_event_inputs, outputs=ask_event_outputs)

        # Other tabs
        summarize_btn.click(
            handle_summarize,
            inputs=[summarize_doc, summarize_focus, summarize_length],
            outputs=[summary_box, key_points_box, summarize_meta],
        )
        kw_btn.click(handle_keywords, inputs=[kw_doc, kw_n], outputs=[kw_table, kw_meta])
        qg_btn.click(handle_gen_questions, inputs=[qg_doc, qg_n], outputs=[qg_box, qg_meta])
        img_btn.click(handle_gen_image, inputs=[img_doc, img_user_prompt], outputs=[img_output, img_meta])
        cmp_btn.click(
            handle_compare,
            inputs=[cmp_question, cmp_top_k, cmp_multi],
            outputs=[cmp_a_answer, cmp_a_meta, cmp_b_answer, cmp_b_meta],
        )

    return demo


if __name__ == "__main__":
    build_demo().launch()
