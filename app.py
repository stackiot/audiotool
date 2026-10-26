"""Push-to-talk Hinglish STT comparison.

Run:  .venv/bin/uvicorn app:app --port 8000   then open http://localhost:8000
"""
import io
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from faster_whisper import WhisperModel

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"

# id -> (label, local dir, description). A model shows as "not downloaded"
# until its directory contains model.bin.
MODELS = {
    "swift": ("Oriserve Swift", "swift-ct2", "Whisper base · int8 · 78 MB"),
    "apex": ("Oriserve Apex", "apex-ct2", "Whisper large-v3-turbo · int8 · 781 MB"),
    "prime": ("Oriserve Prime", "prime-ct2", "Whisper large-v3 · int8 · 1.5 GB"),
}

_loaded: dict[str, WhisperModel] = {}
# CTranslate2 already uses all CPU cores; running models one at a time keeps
# per-model timings honest and memory bounded on an 8 GB machine.
_lock = threading.Lock()

app = FastAPI()


def _available(model_id: str) -> bool:
    return (MODELS_DIR / MODELS[model_id][1] / "model.bin").exists()


def _get_model(model_id: str) -> WhisperModel:
    if model_id not in _loaded:
        _loaded[model_id] = WhisperModel(
            str(MODELS_DIR / MODELS[model_id][1]), device="cpu", compute_type="int8"
        )
    return _loaded[model_id]


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/models")
def list_models():
    return [
        {"id": k, "label": v[0], "desc": v[2], "available": _available(k)}
        for k, v in MODELS.items()
    ]


@app.post("/transcribe/{model_id}")
def transcribe(model_id: str, audio: UploadFile = File(...)):
    if model_id not in MODELS:
        raise HTTPException(404, "unknown model")
    if not _available(model_id):
        raise HTTPException(409, "model not downloaded")
    data = audio.file.read()
    with _lock:
        t0 = time.perf_counter()
        model = _get_model(model_id)
        load_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        # Oriserve models emit Roman-script Hinglish when decoded as "en";
        # "hi" makes them fall back to Devanagari.
        segments, info = model.transcribe(
            io.BytesIO(data), language="en", task="transcribe", beam_size=5,
            temperature=0.0, condition_on_previous_text=False,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        infer_s = time.perf_counter() - t1
    return {
        "text": text,
        "audio_s": round(info.duration, 2),
        "infer_s": round(infer_s, 2),
        "load_s": round(load_s, 2),
    }
