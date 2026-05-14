"""
Speech-to-Text Agent - Model 2: OpenAI Whisper (via HuggingFace Transformers)

Model:  openai/whisper-base
Task:   Automatic Speech Recognition - audio (.wav) -> transcribed text
Audio loading: uses scipy.io.wavfile
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from ..config import WHISPER_MODEL

_whisper_pipeline = None


def _get_whisper():
    """Load the Whisper ASR pipeline on first call and cache it."""
    global _whisper_pipeline
    if _whisper_pipeline is None:
        import os, torch
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        from transformers import pipeline
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = 0
        else:
            device = -1
        _whisper_pipeline = pipeline(
            task="automatic-speech-recognition",
            model=WHISPER_MODEL,
            device=device,
            chunk_length_s=30,
            stride_length_s=5,
        )
    return _whisper_pipeline


# Main functions 

def transcribe_audio(audio_path: str) -> dict:
    """
    Transcribe a WAV audio file to text using Whisper.
    """
    import scipy.io.wavfile as wavfile

    # Load WAV
    try:
        sr, data = wavfile.read(audio_path)
    except Exception as e:
        raise ValueError(f"Could not read audio file: {e}")
 
    if data.ndim > 1:                        
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    if np.abs(data).max() > 1.0:            
        data = data / 32768.0

    target_sr = 16_000
    if sr != target_sr:
        import scipy.signal as sig
        data = sig.resample_poly(data, target_sr, sr).astype(np.float32)
        sr = target_sr

    pipe = _get_whisper()
    result = pipe({"array": data, "sampling_rate": sr})
    transcript = result.get("text", "").strip()

    if not transcript:
        raise ValueError("Whisper returned an empty transcription. "
                         "Check that the audio contains speech.")

    return {
        "transcript": transcript,
        "model_used": f"Whisper ({WHISPER_MODEL})",
    }
