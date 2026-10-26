"""Stream audio to the live STT server and print what comes back.

  python client.py ws://HOST:PORT/ws --token SECRET clip.wav   # any file ffmpeg/PyAV can read
  python client.py ws://HOST:PORT/ws --token SECRET --mic      # needs: pip install sounddevice

Files are sent at real-time speed so the server sees them the way it would see
a live call; pass --fast to send as quickly as possible.
"""
import argparse
import asyncio
import json
import sys
import urllib.parse

import av
import numpy as np
import websockets

SAMPLE_RATE = 16000
CHUNK = 1600  # 100 ms per WebSocket message


def file_chunks(path: str):
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    buf = b""
    with av.open(path) as f:
        for frame in f.decode(audio=0):
            for out in resampler.resample(frame):
                buf += out.to_ndarray().tobytes()
                while len(buf) >= CHUNK * 2:
                    yield buf[:CHUNK * 2]
                    buf = buf[CHUNK * 2:]
        for out in resampler.resample(None):
            buf += out.to_ndarray().tobytes()
    if buf:
        yield buf


async def send_file(ws, path: str, fast: bool):
    for chunk in file_chunks(path):
        await ws.send(chunk)
        if not fast:
            await asyncio.sleep(CHUNK / SAMPLE_RATE)
    # A second of silence lets the server's endpointing close the last utterance
    # naturally; "end" then flushes anything still open and triggers "done".
    await ws.send(np.zeros(SAMPLE_RATE, dtype=np.int16).tobytes())
    await ws.send(json.dumps({"type": "end"}))


async def send_mic(ws):
    import sounddevice as sd

    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    stream = sd.RawInputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=CHUNK,
                               callback=lambda data, *_: loop.call_soon_threadsafe(q.put_nowait, bytes(data)))
    print("Listening… Ctrl+C to stop", file=sys.stderr)
    with stream:
        while True:
            await ws.send(await q.get())


async def receive(ws, until_done: bool):
    async for raw in ws:
        msg = json.loads(raw)
        if msg["type"] == "partial":
            print(f"\r\033[K… {msg['text']}", end="", flush=True)
        elif msg["type"] == "final":
            print(f"\r\033[K[{msg['id']}] {msg['text']}  ({msg['audio_s']}s audio, {msg['latency_s']}s)")
        elif msg["type"] == "done" and until_done:
            return
        elif msg["type"] == "error":
            print(f"\nserver error: {msg['message']}", file=sys.stderr)


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("url", help="e.g. ws://1.2.3.4:40123/ws")
    p.add_argument("file", nargs="?", help="audio file to stream")
    p.add_argument("--mic", action="store_true", help="stream the default microphone")
    p.add_argument("--token", default="")
    p.add_argument("--lang", default="en")
    p.add_argument("--fast", action="store_true", help="send the file faster than real time")
    a = p.parse_args()
    if bool(a.file) == a.mic:
        p.error("give an audio file or --mic")

    query = urllib.parse.urlencode({"token": a.token, "lang": a.lang})
    async with websockets.connect(f"{a.url}?{query}", max_size=None) as ws:
        ready = json.loads(await ws.recv())
        assert ready["type"] == "ready", ready
        sender = send_mic(ws) if a.mic else send_file(ws, a.file, a.fast)
        sender_task = asyncio.create_task(sender)
        await receive(ws, until_done=not a.mic)
        sender_task.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
