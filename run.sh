#!/bin/sh
# Start the Hinglish STT comparison server, then open http://localhost:8000
cd "$(dirname "$0")"
echo "Open http://localhost:8000  (Ctrl+C to stop)"
exec .venv/bin/uvicorn app:app --port 8000
