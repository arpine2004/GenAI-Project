"""
Configuration settings for the RAG Q&A Assistant.
All settings are loaded from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
CHROMA_DIR = os.getenv("CHROMA_DIR", str(BASE_DIR / "chroma_db"))
UPLOAD_DIR = os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploads"))
AUDIO_DIR = os.getenv("AUDIO_DIR", str(BASE_DIR / "audio_recordings"))

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-4-6")
CLAUDE_MAX_TOKENS = int(os.getenv("CLAUDE_MAX_TOKENS", "4096"))
CLAUDE_TEMPERATURE = float(os.getenv("CLAUDE_TEMPERATURE", "0.1"))


GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "llama-3.3-70b-versatile")

CLAUDE_ALT_MODEL = LLAMA_MODEL

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "openai/whisper-base")

SUMMARIZE_MODEL = os.getenv("SUMMARIZE_MODEL", "sshleifer/distilbart-cnn-12-6")

IMAGE_GEN_MODEL = os.getenv("IMAGE_GEN_MODEL", "black-forest-labs/FLUX.1-schnell")
HF_TOKEN = os.getenv("HF_TOKEN", "")

QG_MODEL = os.getenv("QG_MODEL", "google/flan-t5-base")

AVAILABLE_MODELS = {
    CLAUDE_MODEL: f"Anthropic {CLAUDE_MODEL} — primary RAG Q&A LLM",
    LLAMA_MODEL: f"Meta {LLAMA_MODEL} via Groq — alternative LLM for comparison",
    EMBEDDING_MODEL: "Sentence-Transformers all-MiniLM-L6-v2 — semantic embedding & keyword extraction",
    WHISPER_MODEL: "OpenAI Whisper Base — speech-to-text (audio encoder-decoder)",
    SUMMARIZE_MODEL: "DistilBART CNN 12-6 (sshleifer) — abstractive summarization (local seq2seq)",
    IMAGE_GEN_MODEL: "FLUX.1-schnell (black-forest-labs) — text-to-image generation (HF Inference API)",
}


CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
RETRIEVAL_SCORE_THRESHOLD = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.3"))

SUMMARIZE_CHUNK_SIZE = int(os.getenv("SUMMARIZE_CHUNK_SIZE", "3000"))
SUMMARIZE_MAX_CHUNKS = int(os.getenv("SUMMARIZE_MAX_CHUNKS", "500"))

APP_TITLE = "RAG-Powered Document Q&A Assistant"
APP_VERSION = "1.0.0"
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
