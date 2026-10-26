"""Live speech-to-text over WebSocket.

Clients stream raw 16 kHz mono 16-bit PCM to /ws. The server cuts the stream
into utterances (on silence, or when the client sends {"type": "end"}) and
replies with partial and final text. Protocol details are in README.md.

Run:  uvicorn server:app --host 0.0.0.0 --port 8000
"""
import asyncio
import collections
import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from faster_whisper import WhisperModel

ROOT = Path(__file__).parent
SAMPLE_RATE = 16000
FRAME = 480  # 30 ms at 16 kHz; the unit the endpointing works in

MODEL_DIR = os.environ.get("STT_MODEL_DIR", str(ROOT.parent / "models" / "prime-ct2"))
DEVICE = os.environ.get("STT_DEVICE", "cuda")
COMPUTE_TYPE = os.environ.get("STT_COMPUTE_TYPE", "int8_float16" if DEVICE == "cuda" else "int8")
# How many utterances the GPU transcribes at once (CTranslate2 worker threads).
WORKERS = int(os.environ.get("STT_WORKERS", "2"))
# Shared secret clients pass as ?token=...; empty disables auth (local testing only).
TOKEN = os.environ.get("STT_TOKEN", "")
# Oriserve models emit Roman-script Hinglish when decoded as "en"; "hi" gives Devanagari.
LANGUAGE = os.environ.get("STT_LANGUAGE", "en")
SPEECH_DBFS = float(os.environ.get("STT_SPEECH_DBFS", "-45"))  # louder than this counts as speech
SILENCE_MS = int(os.environ.get("STT_SILENCE_MS", "700"))  # this much quiet ends an utterance
MIN_SPEECH_MS = int(os.environ.get("STT_MIN_SPEECH_MS", "250"))  # shorter blips are dropped
MAX_UTTERANCE_S = float(os.environ.get("STT_MAX_UTTERANCE_S", "25"))  # Whisper's window is 30 s
PARTIAL_INTERVAL_S = float(os.environ.get("STT_PARTIAL_INTERVAL_S", "1.0"))  # 0 disables partials
PREROLL_MS = 300  # audio kept from before speech starts so the first syllable isn't clipped

log = logging.getLogger("stt")
model: WhisperModel | None = None
gpu_slots: asyncio.Semaphore | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model, gpu_slots
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not TOKEN:
        log.warning("STT_TOKEN is not set: anyone who can reach this port can use it")
    t0 = time.perf_counter()
    model = WhisperModel(MODEL_DIR, device=DEVICE, compute_type=COMPUTE_TYPE, num_workers=WORKERS)
    gpu_slots = asyncio.Semaphore(WORKERS)
    log.info("loaded %s on %s/%s in %.1fs", MODEL_DIR, DEVICE, COMPUTE_TYPE, time.perf_counter() - t0)
    yield


app = FastAPI(lifespan=lifespan)


def transcribe(pcm: np.ndarray, language: str, beam_size: int) -> str:
    audio = pcm.astype(np.float32) / 32768.0
    segments, _ = model.transcribe(
        audio, language=language, task="transcribe", beam_size=beam_size,
        temperature=0.0, condition_on_previous_text=False, without_timestamps=True,
    )
    return " ".join(s.text.strip() for s in segments).strip()


class Session:
    """One client connection: endpointing state plus an ordered queue of finals."""

    def __init__(self, ws: WebSocket, language: str, auto_end: bool):
        self.ws = ws
        self.language = language
        self.auto_end = auto_end
        self.send_lock = asyncio.Lock()
        self.finals: asyncio.Queue = asyncio.Queue()
        self.leftover = b""
        self.preroll = collections.deque(maxlen=PREROLL_MS * SAMPLE_RATE // 1000 // FRAME)
        self.utt_id = 0
        self.partial_task: asyncio.Task | None = None
        self._reset()

    def _reset(self):
        self.frames: list[np.ndarray] = []
        self.started = False
        self.speech_samples = 0
        self.silence_samples = 0
        self.last_partial = 0.0

    async def send(self, msg: dict):
        async with self.send_lock:
            await self.ws.send_text(json.dumps(msg, ensure_ascii=False))

    def feed(self, data: bytes):
        data = self.leftover + data
        usable = len(data) - len(data) % (FRAME * 2)
        self.leftover = data[usable:]
        pcm = np.frombuffer(data[:usable], dtype=np.int16)
        for i in range(0, len(pcm), FRAME):
            self._frame(pcm[i:i + FRAME])
        self._maybe_partial()

    def _frame(self, frame: np.ndarray):
        rms = np.sqrt(np.mean(frame.astype(np.float32) ** 2)) + 1e-9
        loud = 20 * np.log10(rms / 32768.0) > SPEECH_DBFS
        if not self.started:
            if not loud:
                self.preroll.append(frame)
                return
            self.started = True
            self.frames.extend(self.preroll)
            self.preroll.clear()
        self.frames.append(frame)
        if loud:
            self.speech_samples += len(frame)
            self.silence_samples = 0
        else:
            self.silence_samples += len(frame)
        if self.auto_end and self.silence_samples >= SILENCE_MS * SAMPLE_RATE // 1000:
            self.end_utterance()
        elif len(self.frames) * FRAME >= MAX_UTTERANCE_S * SAMPLE_RATE:
            self.end_utterance()

    def end_utterance(self):
        if self.started and self.speech_samples >= MIN_SPEECH_MS * SAMPLE_RATE // 1000:
            self.finals.put_nowait((self.utt_id, np.concatenate(self.frames)))
            self.utt_id += 1
        self._reset()

    def _maybe_partial(self):
        # Partials are best-effort: skipped while the GPU is busy so finals never wait on them.
        now = time.monotonic()
        if (not PARTIAL_INTERVAL_S or not self.started or gpu_slots.locked()
                or (self.partial_task and not self.partial_task.done())
                or now - self.last_partial < PARTIAL_INTERVAL_S
                or self.speech_samples < SAMPLE_RATE):
            return
        self.last_partial = now
        self.partial_task = asyncio.create_task(self._partial(self.utt_id, np.concatenate(self.frames)))

    async def _partial(self, utt_id: int, pcm: np.ndarray):
        async with gpu_slots:
            text = await asyncio.to_thread(transcribe, pcm, self.language, 1)
        if utt_id == self.utt_id and self.started and text:  # utterance still open
            await self.send({"type": "partial", "id": utt_id, "text": text})

    async def run_finals(self):
        while True:
            item = await self.finals.get()
            if item is None:  # marker queued by flush()
                await self.send({"type": "done"})
                continue
            utt_id, pcm = item
            t0 = time.perf_counter()
            async with gpu_slots:
                text = await asyncio.to_thread(transcribe, pcm, self.language, 5)
            await self.send({
                "type": "final", "id": utt_id, "text": text,
                "audio_s": round(len(pcm) / SAMPLE_RATE, 2),
                "latency_s": round(time.perf_counter() - t0, 2),
            })

    def flush(self):
        self.end_utterance()
        self.finals.put_nowait(None)


@app.get("/")
def index():
    return FileResponse(ROOT / "live.html")


@app.get("/health")
def health():
    return {"model": Path(MODEL_DIR).name, "device": DEVICE, "workers": WORKERS}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    q = ws.query_params
    if TOKEN and not secrets.compare_digest(q.get("token", ""), TOKEN):
        await ws.close(code=1008, reason="bad token")
        return
    await ws.accept()
    session = Session(ws, q.get("lang", LANGUAGE), auto_end=q.get("auto_end", "1") != "0")
    finals = asyncio.create_task(session.run_finals())
    await session.send({"type": "ready", "sample_rate": SAMPLE_RATE, "format": "s16le", "channels": 1})
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes"):
                session.feed(msg["bytes"])
            elif msg.get("text"):
                try:
                    kind = json.loads(msg["text"]).get("type")
                except (ValueError, AttributeError):
                    kind = None
                if kind == "end":
                    session.flush()
                elif kind == "reset":
                    session._reset()
                else:
                    await session.send({"type": "error", "message": f"unknown message: {msg['text'][:100]}"})
    except WebSocketDisconnect:
        pass
    finally:
        finals.cancel()
        if session.partial_task:
            session.partial_task.cancel()
