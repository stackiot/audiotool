# Live STT server

Clients stream audio over a WebSocket. The server splits the audio into utterances wherever the speaker pauses, transcribes them with the Apex model (Whisper large-v3-turbo, Hinglish fine-tune) on the GPU, and sends the text back.

```
client ──PCM audio──▶  /ws  ──▶ endpointing ──▶ faster-whisper (GPU)
       ◀──JSON text──        ◀── partial / final
```

## Protocol

Connect to `ws://HOST:PORT/ws?token=SECRET`.

Optional query parameters:
- `lang`: default `en`, which gives Roman-script Hinglish. `hi` gives Devanagari.
- `auto_end`: default `1`. Set it to `0` for push-to-talk clients that send `end` themselves.

**Client → server**

| Frame | Meaning |
|---|---|
| binary | Raw PCM, **16 kHz, mono, signed 16-bit little-endian**. Any chunk size works; 20–100 ms is typical. |
| `{"type":"end"}` | Finish the current utterance now (for example on PTT release). The server replies `done` once everything sent so far has been transcribed. |
| `{"type":"reset"}` | Throw away the current utterance. |

**Server → client**

| Message | Meaning |
|---|---|
| `{"type":"ready","sample_rate":16000,...}` | Sent on connect. Start sending audio after this arrives. |
| `{"type":"partial","id":0,"text":"..."}` | Rough text for the utterance still in progress. It may change. |
| `{"type":"final","id":0,"text":"...","audio_s":2.8,"latency_s":0.4}` | Finished text for utterance `id`. Finals always arrive in order. |
| `{"type":"done"}` | Reply to `end`: everything has been transcribed. |
| `{"type":"error","message":"..."}` | The client sent something the server doesn't understand. |

A connection with a wrong token is rejected with HTTP 403.

8 kHz telephony audio (Asterisk/RoIP G.711) has to be resampled to 16 kHz before sending.

## Run locally (CPU)

```sh
cd live
STT_DEVICE=cpu STT_TOKEN=test ../.venv/bin/uvicorn server:app --port 8000
../.venv/bin/python client.py ws://localhost:8000/ws --token test some.wav
```

Open http://localhost:8000 to use the browser client with your microphone.

## Deploy on a RunPod GPU Pod

The RunPod HTTP proxy closes connections after 100 seconds, so live WebSockets have to go through a **TCP port** instead.

1. GitHub Actions (`.github/workflows/docker.yml`) builds the image whenever `live/` changes on `main` and pushes it to `ghcr.io/stackiot/audiotool`. It's tagged `latest` and with the short commit hash. You can also start a build by hand from the repo's Actions tab.

   To build it yourself instead, run this from the repo root. The build downloads and converts the official Apex weights itself, so it needs no local model files, but it needs about 4 GB of RAM and 10 GB of free disk.
   ```sh
   docker build --platform linux/amd64 -f live/Dockerfile -t ghcr.io/stackiot/audiotool:latest .
   ```
2. Create a Pod:
   - GPU: any 16 GB card is enough.
   - Image: `ghcr.io/stackiot/audiotool:latest`. Pin a commit tag such as `:sha-0800fa0` if you want a fixed version.
   - If the package is private, add **Container Registry Auth** in RunPod Settings. Use registry `ghcr.io`, your GitHub username, and a GitHub token with the `read:packages` scope.
   - **Expose TCP Ports:** `8000`
   - Environment variable: `STT_TOKEN=<long random string>`. You can generate one with `openssl rand -hex 24`.
3. Under **Connect → Direct TCP Ports**, note the public `IP:PORT`, then run:
   ```sh
   python client.py ws://IP:PORT/ws --token <token> some.wav
   ```
4. **Stop the Pod when you're not using it.** You're billed while it runs.

The TCP port is plain `ws://`, not TLS, so the token and the audio travel unencrypted. That's fine for testing. For production, put TLS in front of the server (for example Caddy or a Cloudflare Tunnel) and connect with `wss://`.

Browsers only allow microphone access on HTTPS or localhost. To use the browser client against the Pod, forward the port over SSH and open http://localhost:8000:
```sh
ssh -L 8000:localhost:8000 root@IP -p <ssh port> -i ~/.ssh/id_ed25519
```

## Deploy on RunPod Serverless (pay only while transcribing)

A Serverless worker can't hold a live WebSocket, so this route uses `POST /transcribe`, **one utterance per request**. Your client decides where each utterance ends: on PTT release, or with its own silence detection. The image is the same one the Pod uses.

1. In RunPod, go to **Serverless → New Endpoint → Import from Docker registry**:
   - Image: `ghcr.io/stackiot/audiotool:latest`
   - Endpoint type: **Load balancer** (requests go straight to the worker's HTTP port, with no job queue)
   - GPU: 16 GB tier
   - **Expose HTTP Ports:** `8000`
   - Environment variables: `PORT=8000`, `STT_TOKEN=<long random string>`. The health check uses `/ping`, which is RunPod's default.
   - Workers: Min `0` (scales to zero, nothing billed while idle), Max as needed. Idle timeout of about 60 s or more keeps a warm worker between bursts of traffic.
2. Send each utterance:
   ```sh
   curl -X POST "https://<ENDPOINT_ID>.api.runpod.ai/transcribe" \
     -H "Authorization: Bearer <RUNPOD_API_KEY>" \
     -H "X-STT-Token: <token>" \
     --data-binary @utterance.wav
   # → {"text":"Hello, kal office mein meeting hai.","audio_s":2.8,"latency_s":0.4}
   ```
   For raw 16 kHz mono s16le PCM, add `?format=pcm16`. Audio can be at most 25 s long; split anything longer into utterances.

**Cold starts:** when the endpoint has scaled to zero, the first request waits for a worker to start and load the model, which can take tens of seconds. While no worker is ready, the request can fail with "no workers available", so retry 3 times, 5–10 s apart. To avoid cold starts for live traffic, set Min workers to `1`, but you then pay for that worker around the clock.

## Settings (environment variables)

| Variable | Default | |
|---|---|---|
| `STT_TOKEN` | *(empty: no auth)* | Shared secret that clients pass as `?token=`. |
| `STT_MODEL_DIR` | `../models/apex-ct2` | Any faster-whisper/CT2 model directory. |
| `STT_DEVICE` / `STT_COMPUTE_TYPE` | `cuda` / `int8_float16` | Use `cpu` / `int8` without a GPU. |
| `STT_WORKERS` | `2` | Number of utterances transcribed on the GPU at the same time. |
| `STT_SILENCE_MS` | `700` | Length of pause that ends an utterance. |
| `STT_SPEECH_DBFS` | `-45` | Loudness threshold for speech. Raise it toward `-35` for noisy radio channels. |
| `STT_MAX_UTTERANCE_S` | `25` | Utterances longer than this are cut. Whisper's limit is 30 s. |
| `STT_PARTIAL_INTERVAL_S` | `1.0` | How often partials are sent. `0` turns them off and saves GPU. |
