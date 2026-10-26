"""Push-to-talk Hinglish STT comparison.

Run:  ./run.sh   then open http://localhost:8000

Set RUNPOD_STT_URL, RUNPOD_API_KEY and STT_TOKEN (run.sh reads them from
~/.audiotool.env) to add a column for the live/ server on RunPod Serverless.
"""
import io
import os
import threading
import time
from pathlib import Path

import httpx
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from faster_whisper import WhisperModel, decode_audio

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"

# id -> (label, local dir, description). A model shows as "not downloaded"
# until its directory contains model.bin.
MODELS = {
    "swift": ("Oriserve Swift", "swift-ct2", "Whisper base · int8 · 78 MB"),
    "apex": ("Oriserve Apex", "apex-ct2", "Whisper large-v3-turbo · int8 · 781 MB"),
    "prime": ("Oriserve Prime", "prime-ct2", "Whisper large-v3 · int8 · 1.5 GB"),
}

# The deployed live/ server, called over HTTP instead of loaded locally.
REMOTE_URL = os.environ.get("RUNPOD_STT_URL", "").rstrip("/")
REMOTE = ("RunPod GPU · Apex", "Whisper large-v3-turbo · int8_float16 · Serverless")

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
    models = [
        {"id": k, "label": v[0], "desc": v[2], "available": _available(k)}
        for k, v in MODELS.items()
    ]
    if REMOTE_URL:
        models.append({"id": "remote", "label": REMOTE[0], "desc": REMOTE[1], "available": True})
    return models


def _transcribe_remote(data: bytes) -> dict:
    # Decode locally and send 16 kHz PCM: smaller than the browser's webm, and
    # the server needn't understand whatever container the browser picked.
    pcm = (decode_audio(io.BytesIO(data), 16000) * 32767).astype(np.int16).tobytes()
    headers = {"Authorization": f"Bearer {os.environ.get('RUNPOD_API_KEY', '')}",
               "X-STT-Token": os.environ.get("STT_TOKEN", "")}
    t0 = time.perf_counter()
    # A scaled-to-zero endpoint answers "no workers available" until a worker
    # boots, so retry for up to ~2 minutes.
    for attempt in range(12):
        try:
            r = httpx.post(f"{REMOTE_URL}/transcribe?format=pcm16", content=pcm,
                           headers=headers, timeout=60)
        except httpx.TimeoutException:
            continue
        if r.status_code < 500:
            break
        time.sleep(10)
    else:
        raise HTTPException(504, "RunPod endpoint did not respond; is a worker able to start?")
    if r.status_code != 200:
        raise HTTPException(r.status_code, f"RunPod: {r.text[:200]}")
    j = r.json()
    rtt = time.perf_counter() - t0
    return {"text": j["text"], "audio_s": j["audio_s"], "infer_s": j["latency_s"],
            "load_s": 0, "rtt_s": round(rtt, 2)}


@app.post("/transcribe/{model_id}")
def transcribe(model_id: str, audio: UploadFile = File(...)):
    if model_id == "remote" and REMOTE_URL:
        return _transcribe_remote(audio.file.read())
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
