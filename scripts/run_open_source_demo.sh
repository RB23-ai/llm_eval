#!/usr/bin/env bash
# Run the full eval suite in real open-source mode: Qwen (via Ollama) for
# generation and judging, sentence-transformers for embeddings, Chroma as
# the vector store. Requires Ollama installed and running locally
# (https://ollama.com) -- this script pulls the model for you if missing.
#
# This script cannot be run inside a sandboxed CI/demo environment without
# internet access to ollama.com and the model registry; run it on your own
# machine to get real (non-mock) numbers for reports/eval_report.md.
set -euo pipefail

MODEL="${OLLAMA_MODEL:-qwen2.5:1.5b}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama is not installed. Install it from https://ollama.com and re-run this script." >&2
  exit 1
fi

if ! ollama list | grep -q "${MODEL%%:*}"; then
  echo "Pulling ${MODEL} (one-time download)..."
  ollama pull "${MODEL}"
fi

if ! pgrep -x "ollama" >/dev/null 2>&1; then
  echo "Starting local Ollama server..."
  ollama serve >/tmp/ollama.log 2>&1 &
  sleep 3
fi

pip install -q sentence-transformers chromadb

export LLM_PROVIDER=ollama
export JUDGE_PROVIDER=ollama
export EMBEDDING_PROVIDER=hf_sentence_transformers
export VECTOR_STORE=chroma
export OLLAMA_MODEL="${MODEL}"

echo "Running full suite with real open-source models (LLM_PROVIDER=ollama, model=${MODEL})..."
python3 run_evals.py
