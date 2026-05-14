"""
RAG-Powered Document Q&A Assistant — FastAPI Backend

Endpoints:
  GET  /health                 — health check & system status
  GET  /documents              — list all indexed documents
  POST /documents/upload       — upload & ingest a document
  DELETE /documents/{doc_id}   — remove a document from the index
  POST /query                  — ask a question (RAG + agents)
  POST /summarize              — summarize a document
"""
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import gradio as gr
from . import config
from .agents.image_agent import generate_image as image_generate
from .agents.rag_agent import answer_question, extract_keywords
from .agents.summarize_agent import summarize_document
from .agents.stt_agent import transcribe_audio
from .agents.qg_agent import generate_questions as qg_generate
from .ingestion.chunker import chunk_documents
from .ingestion.loader import SUPPORTED_EXTENSIONS, load_document
from .models import (
    CompareResponse,
    DeleteResponse,
    DocumentInfo,
    DocumentListResponse,
    HealthResponse,
    ImageGenRequest,
    ImageGenResponse,
    KeywordRequest,
    KeywordResponse,
    QueryRequest,
    QueryResponse,
    QuestionGenRequest,
    QuestionGenResponse,
    RecordingInfo,
    RecordingListResponse,
    SummarizeRequest,
    SummarizeResponse,
    TranscribeResponse,
)
from .vectorstore.store import (
    add_documents,
    delete_document,
    document_exists,
    get_vector_store,
    list_documents,
)

def _ensure_nltk_data() -> None:
    try:
        import nltk
    except ImportError:
        return
    for pkg in ("punkt", "punkt_tab", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"):
        try:
            nltk.data.find(pkg)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception as e:
                print(f"[startup] NLTK download warning ({pkg}): {e}")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    print(f"[startup] Loading embedding model: {config.EMBEDDING_MODEL}")
    get_vector_store()
    _ensure_nltk_data()
    try:
        from . import chat_history
        chat_history.clear_all()
        print("[startup] Cleared previous chat history.")
    except Exception as e:
        print(f"[startup] Could not clear chat history: {e}")
    print("[startup] Vector store ready.")
    yield


app = FastAPI(
    title=config.APP_TITLE,
    version=config.APP_VERSION,
    description=(
        "A RAG-powered Q&A assistant that indexes your internal documents "
        "and answers questions with grounded citations."
    ),
    lifespan=_lifespan,
)

_cors_origins = [o.strip() for o in config.CORS_ORIGINS if o.strip()]
_cors_credentials = bool(_cors_origins) and "*" not in _cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins or ["*"],
    allow_credentials=_cors_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/audio", StaticFiles(directory=config.AUDIO_DIR), name="audio")

try:
    from frontend.gradio_app import build_demo as _build_gradio_demo
    _gradio_demo = _build_gradio_demo()
    app = gr.mount_gradio_app(app, _gradio_demo, path="/app")
    print("[startup] Gradio UI mounted at /app")
except Exception as _e:
    import traceback as _tb
    print(f"[startup] Could not mount Gradio UI: {_e!r}")
    _tb.print_exc()



@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    docs = list_documents()
    return HealthResponse(
        status="ok",
        version=config.APP_VERSION,
        documents_indexed=len(docs),
        vector_store="ChromaDB",
        embedding_model=config.EMBEDDING_MODEL,
        llm_model=config.CLAUDE_MODEL,
        available_models=config.AVAILABLE_MODELS,
    )



@app.get("/documents", response_model=DocumentListResponse, tags=["Documents"])
async def list_all_documents():
    docs = list_documents()
    doc_infos = [
        DocumentInfo(
            doc_id=d["doc_id"],
            filename=d["filename"],
            num_chunks=d.get("num_chunks", 0),
            file_size_kb=d.get("file_size_kb", 0.0),
            upload_time=d.get("upload_time", ""),
            file_type=d.get("file_type", ""),
        )
        for d in docs
    ]
    return DocumentListResponse(documents=doc_infos, total=len(doc_infos))


@app.post("/documents/upload", response_model=DocumentInfo, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")
    safe_name = Path(file.filename).name
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    ext = Path(safe_name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Supported: {', '.join(SUPPORTED_EXTENSIONS.keys())}"
            ),
        )

    save_path = os.path.join(config.UPLOAD_DIR, safe_name)
    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    try:
        docs, doc_metadata = load_document(save_path)
        chunks = chunk_documents(docs)
        num_chunks = add_documents(chunks, doc_metadata)
    except ValueError as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise HTTPException(status_code=500, detail=f"Failed to index document: {e}")

    if num_chunks == 0:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise HTTPException(
            status_code=400,
            detail="Document produced no indexable chunks (file may be empty or unreadable).",
        )

    return DocumentInfo(
        doc_id=doc_metadata["doc_id"],
        filename=doc_metadata["filename"],
        num_chunks=num_chunks,
        file_size_kb=doc_metadata["file_size_kb"],
        upload_time=doc_metadata["upload_time"],
        file_type=doc_metadata["file_type"],
    )


@app.delete("/documents/{doc_id}", response_model=DeleteResponse, tags=["Documents"])
async def remove_document(doc_id: str):
    from .vectorstore.store import get_document_info
    doc_info = get_document_info(doc_id)
    success = delete_document(doc_id)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{doc_id}' not found in the knowledge base.",
        )
    if doc_info:
        file_path = os.path.join(config.UPLOAD_DIR, doc_info["filename"])
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass
    return DeleteResponse(
        success=True,
        message=f"Document '{doc_id}' has been removed from the knowledge base.",
        doc_id=doc_id,
    )



@app.get("/recordings", response_model=RecordingListResponse, tags=["Speech-to-Text"])
async def list_recordings():
    from datetime import datetime, timezone

    recs = []
    try:
        entries = sorted(Path(config.AUDIO_DIR).glob("*.wav"), reverse=True)
    except Exception:
        entries = []

    for entry in entries:
        stat = entry.stat()
        name = entry.name
        try:
            ts = datetime.strptime(name[:15], "%Y%m%d_%H%M%S").replace(
                tzinfo=timezone.utc
            ).isoformat()
        except ValueError:
            ts = ""
        slug = name[16:].removesuffix(".wav").replace("_", " ") if len(name) > 16 else name
        recs.append(RecordingInfo(
            filename=name,
            timestamp=ts,
            transcript_slug=slug,
            size_kb=round(stat.st_size / 1024, 1),
        ))

    return RecordingListResponse(recordings=recs, total=len(recs))


@app.post("/query", response_model=QueryResponse, tags=["Q&A"])
async def query(request: QueryRequest):
    if not config.ANTHROPIC_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="ANTHROPIC_API_KEY is not configured. Please set it in your .env file.",
        )

    docs = list_documents()
    if not docs:
        raise HTTPException(
            status_code=404,
            detail="The knowledge base is empty. Please upload documents first.",
        )

    if request.doc_ids:
        all_ids = {d["doc_id"] for d in docs}
        invalid = [did for did in request.doc_ids if did not in all_ids]
        if invalid:
            raise HTTPException(
                status_code=404,
                detail=f"Document IDs not found: {invalid}",
            )

    if request.model and request.model not in (config.CLAUDE_MODEL, config.CLAUDE_ALT_MODEL):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{request.model}'. Choose from: {config.CLAUDE_MODEL}, {config.CLAUDE_ALT_MODEL}",
        )

    try:
        response = answer_question(
            question=request.question,
            doc_ids=request.doc_ids,
            top_k=request.top_k or config.RETRIEVAL_TOP_K,
            multi_step=request.multi_step,
            model=request.model,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

    return response



@app.post("/summarize", response_model=SummarizeResponse, tags=["Summarization"])
async def summarize(request: SummarizeRequest):
    if not config.ANTHROPIC_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="ANTHROPIC_API_KEY is not configured.",
        )

    if not document_exists(request.doc_id):
        raise HTTPException(
            status_code=404,
            detail=f"Document '{request.doc_id}' not found. Upload it first.",
        )

    try:
        response = summarize_document(
            doc_id=request.doc_id,
            focus=request.focus,
            length=request.length,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {e}")

    return response

@app.post("/generate-image", response_model=ImageGenResponse, tags=["Image Generation"])
async def generate_image_endpoint(request: ImageGenRequest):
    if not config.HF_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="HF_TOKEN is not configured. Add HF_TOKEN=<your_token> to your .env file.",
        )

    docs = list_documents()
    doc_ids = [d["doc_id"] for d in docs]
    if request.doc_id not in doc_ids:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{request.doc_id}' not found. Upload it first.",
        )

    try:
        result = image_generate(request.doc_id, user_prompt=request.user_prompt or "")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {e}")

    return ImageGenResponse(**result)

@app.post("/compare", response_model=CompareResponse, tags=["Comparison"])
async def compare_models(request: QueryRequest):
    if not config.ANTHROPIC_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="ANTHROPIC_API_KEY is not configured.",
        )

    docs = list_documents()
    if not docs:
        raise HTTPException(
            status_code=404,
            detail="The knowledge base is empty. Please upload documents first.",
        )

    try:
        response_a = answer_question(
            question=request.question,
            doc_ids=request.doc_ids,
            top_k=request.top_k or config.RETRIEVAL_TOP_K,
            multi_step=request.multi_step,
            model=config.CLAUDE_MODEL,
        )
        response_b = answer_question(
            question=request.question,
            doc_ids=request.doc_ids,
            top_k=request.top_k or config.RETRIEVAL_TOP_K,
            multi_step=request.multi_step,
            model=config.CLAUDE_ALT_MODEL,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {e}")

    return CompareResponse(
        model_a_name=config.CLAUDE_MODEL,
        model_b_name=config.CLAUDE_ALT_MODEL,
        model_a=response_a,
        model_b=response_b,
    )

@app.post("/keywords", response_model=KeywordResponse, tags=["Keyword Extraction"])
async def keywords(request: KeywordRequest):
    if not document_exists(request.doc_id):
        raise HTTPException(
            status_code=404,
            detail=f"Document '{request.doc_id}' not found.",
        )

    try:
        response = extract_keywords(doc_id=request.doc_id, top_n=request.top_n)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Keyword extraction failed: {e}")

    return response

@app.post("/transcribe", response_model=TranscribeResponse, tags=["Speech-to-Text"])
async def transcribe(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=400,
            detail="Only .wav files are supported. Please convert your audio to WAV format.",
        )

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio_path = tmp.name
    try:
        try:
            shutil.copyfileobj(file.file, tmp)
        finally:
            tmp.close()
        result = transcribe_audio(audio_path)
    except ValueError as e:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    from datetime import datetime, timezone
    ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug = "".join(
        c if c.isalnum() else "_"
        for c in result.get("transcript", "")[:40]
    ).strip("_") or "recording"
    save_name = f"{ts}_{slug}.wav"
    save_path = os.path.join(config.AUDIO_DIR, save_name)
    try:
        shutil.move(audio_path, save_path)
    except Exception:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        save_name = None

    return TranscribeResponse(**result, saved_filename=save_name)

@app.post("/generate-questions", response_model=QuestionGenResponse, tags=["Question Generation"])
async def generate_questions_endpoint(request: QuestionGenRequest):
    if not document_exists(request.doc_id):
        raise HTTPException(
            status_code=404,
            detail=f"Document '{request.doc_id}' not found.",
        )

    try:
        result = qg_generate(doc_id=request.doc_id, num_questions=request.num_questions)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Question generation failed: {e}")

    return QuestionGenResponse(**result)

@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse(
        content={
            "message": f"{config.APP_TITLE} is running.",
            "ui": "/app",
            "docs": "/docs",
            "health": "/health",
        }
    )
