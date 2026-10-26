#!/bin/sh
# Start the Hinglish STT comparison server, then open http://localhost:8000
cd "$(dirname "$0")"
# RunPod endpoint settings for the GPU column (RUNPOD_STT_URL, RUNPOD_API_KEY, STT_TOKEN).
[ -f ~/.audiotool.env ] && { set -a; . ~/.audiotool.env; set +a; }
echo "Open http://localhost:8000  (Ctrl+C to stop)"
exec .venv/bin/uvicorn app:app --port 8000
