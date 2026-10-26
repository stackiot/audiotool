# Live STT server with the Prime model baked in. Build from the repo root:
#   docker build --platform linux/amd64 -f live/Dockerfile -t <registry-user>/hinglish-stt:1 .
# The model is converted from Oriserve's official weights during the build, so
# nothing outside this repo is needed. The convert stage needs ~8 GB of RAM and ~15 GB of disk.

# --- Stage 1: convert Oriserve/Whisper-Hindi2Hinglish-Prime to CTranslate2 ---
FROM python:3.12-slim AS convert
RUN pip install --no-cache-dir ctranslate2==4.8.3 transformers==5.20.1 \
    torch==2.9.1 --extra-index-url https://download.pytorch.org/whl/cpu
# Pinned to a known revision so rebuilds produce the same model.
ARG MODEL_REVISION=4edf7b8291f950e3ac4f32bdacb38c46800e77dc
RUN ct2-transformers-converter --model Oriserve/Whisper-Hindi2Hinglish-Prime \
      --revision "$MODEL_REVISION" --low_cpu_mem_usage --quantization int8_float16 \
      --copy_files tokenizer.json preprocessor_config.json --output_dir /models/prime-ct2 \
    && rm -rf /root/.cache/huggingface

# --- Stage 2: CUDA runtime with the server ---
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY live/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY --from=convert /models/prime-ct2 /models/prime-ct2
COPY live/server.py live/live.html /app/

ENV STT_MODEL_DIR=/models/prime-ct2 STT_DEVICE=cuda
WORKDIR /app
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
