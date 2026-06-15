#!/usr/bin/env bash
# Setup LM Studio headless daemon (llmster) for SENTINEL Track B evaluation.
set -euo pipefail

echo "Installing LM Studio headless daemon (llmster)..."
curl -fsSL https://lmstudio.ai/install.sh | bash

export PATH="$HOME/.lmstudio/bin:$PATH"
lms bootstrap 2>/dev/null || true

echo "Starting LM Studio daemon..."
lms daemon up || lms server start

echo "Downloading a small chat model (Llama 3.1 8B GGUF)..."
lms get mlx-community/Meta-Llama-3.1-8B-Instruct-4bit || \
  lms get lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF

echo "Loading model and starting server on http://127.0.0.1:1234 ..."
lms load --gpu=max || lms load
lms server start

echo "Verify with: curl http://127.0.0.1:1234/v1/models"
echo "Run benchmark: python3 benchmark/run_endtoend.py --config config.lmstudio.yaml"
