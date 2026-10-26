# Live STT server with the Prime model baked in. Build from the repo root:
#   docker build --platform linux/amd64 -f live/Dockerfile -t <dockerhub-user>/hinglish-stt:1 .
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY live/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY models/prime-ct2 /models/prime-ct2
COPY live/server.py live/live.html /app/

ENV STT_MODEL_DIR=/models/prime-ct2 STT_DEVICE=cuda
WORKDIR /app
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
