#!/usr/bin/env bash
# Download the model set used by the AsymSpec paper.
# Usage: bash scripts/download_models.sh [--cross-family]

set -euo pipefail

if ! command -v hf >/dev/null 2>&1; then
  echo "The Hugging Face CLI is required: pip install huggingface_hub" >&2
  exit 1
fi

models=(
  "Qwen/Qwen3-32B"
  "Qwen/Qwen3-4B"
  "Qwen/Qwen3-1.7B"
  "Qwen/Qwen3-0.6B"
  "Qwen/Qwen3-VL-2B-Instruct"
  "Qwen/Qwen2.5-14B-Instruct"
  "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
)

if [[ "${1:-}" == "--cross-family" ]]; then
  models+=(
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.3-70B-Instruct"
  )
fi

for model in "${models[@]}"; do
  echo "==> ${model}"
  hf download "${model}"
done

echo "Model downloads complete. Set HF_HOME to relocate the cache."
