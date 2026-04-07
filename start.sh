#!/bin/bash

# Start FastAPI backend
uvicorn server.app:app --host 0.0.0.0 --port 8000 &

# Start Gradio frontend
python ui.py