# Live STT server with the Apex model baked in. Build from the repo root:
#   docker build --platform linux/amd64 -f live/Dockerfile -t <registry-user>/hinglish-stt:1 .
# The model is converted from Oriserve's official weights during the build, so
# nothing outside this repo is needed. The convert stage needs ~4 GB of RAM and ~10 GB of disk.

# --- Stage 1: convert Oriserve/Whisper-Hindi2Hinglish-Apex to CTranslate2 ---
FROM python:3.12-slim AS convert
RUN pip install --no-cache-dir ctranslate2==4.8.3 transformers==5.20.1 \
    torch==2.9.1 --extra-index-url https://download.pytorch.org/whl/cpu
# Pinned to a known revision so rebuilds produce the same model.
ARG MODEL_REVISION=f3214eed20b4e4d4144e739982d911f87b9cb223
RUN ct2-transformers-converter --model Oriserve/Whisper-Hindi2Hinglish-Apex \
      --revision "$MODEL_REVISION" --low_cpu_mem_usage --quantization int8_float16 \
      --copy_files tokenizer.json preprocessor_config.json --output_dir /models/apex-ct2 \
    && rm -rf /root/.cache/huggingface

# --- Stage 2: CUDA runtime with the server ---
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY live/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY --from=convert /models/apex-ct2 /models/apex-ct2
COPY live/server.py live/live.html /app/

ENV STT_MODEL_DIR=/models/apex-ct2 STT_DEVICE=cuda
WORKDIR /app
EXPOSE 8000
# RunPod Serverless load-balancing endpoints tell the worker its port via $PORT.
CMD ["sh", "-c", "exec uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
