#!/bin/bash

echo "Starting chatbot RAG FastAPI service..."

uvicorn src.main:app --host 0.0.0.0 --port 8000
