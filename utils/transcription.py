# utils/transcription.py
from faster_whisper import WhisperModel
from langdetect import detect

# Configure device: "cpu" by default. Set WHISPER_DEVICE env var to "cuda" if you have GPU.
import os
_device = os.environ.get("WHISPER_DEVICE", "cpu")
_compute_type = os.environ.get("WHISPER_COMPUTE", "int8")  # int8 works on CPU, float16 for GPU

# load once
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")
_whisper = WhisperModel(MODEL_SIZE, device=_device, compute_type=_compute_type)

def transcribe_audio(file_path, beam_size=5):
    """
    Returns:
       segments: list of dicts {start, end, text}
       full_text: concatenated text
       detected_language: iso code (from langdetect on text)
    """
    segments_raw, info = _whisper.transcribe(file_path, beam_size=beam_size)
    segments = []
    parts = []
    for seg in segments_raw:
        segments.append({"start": float(seg.start), "end": float(seg.end), "text": seg.text.strip()})
        parts.append(seg.text.strip())
    full_text = " ".join(parts).strip()
    try:
        lang = detect(full_text) if full_text.strip() else "unknown"
    except:
        lang = "unknown"
    return segments, full_text, lang
