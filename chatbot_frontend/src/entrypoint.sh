#!/bin/bash

echo "Starting medical chatbot frontend..."

exec streamlit run main.py \
  --server.address=0.0.0.0 \
  --server.port=8501
