#!/usr/bin/env bash
# Download public datasets used by the AsymSpec experiments.
# GAIA and Llama-family assets require accepting their upstream terms first.

set -euo pipefail

if ! command -v hf >/dev/null 2>&1; then
  echo "The Hugging Face CLI is required: pip install huggingface_hub" >&2
  exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_dir="${ASYMSPEC_DATA_DIR:-${repo_root}/data}"
mkdir -p "${data_dir}/longbench/raw"

echo "==> LongBench"
hf download THUDM/LongBench data.zip --repo-type dataset \
  --local-dir "${data_dir}/longbench/source"
unzip -j -o "${data_dir}/longbench/source/data.zip" \
  data/hotpotqa.jsonl data/2wikimqa.jsonl data/musique.jsonl \
  -d "${data_dir}/longbench/raw"

echo "==> MultiChallenge"
hf download ScaleAI/MultiChallenge --repo-type dataset \
  --local-dir "${data_dir}/multi-challenge"

echo "==> API-Bank"
hf download liminghao1630/API-Bank --repo-type dataset \
  --local-dir "${data_dir}/DAMO-ConvAI/api-bank"

echo "==> MathVista and SimpleQA"
hf download AI4Math/MathVista --repo-type dataset
hf download basicv8vc/SimpleQA --repo-type dataset

echo "==> GAIA (gated)"
hf download gaia-benchmark/GAIA --repo-type dataset \
  --revision 682dd723ee1e1697e00360edccf2366dc8418dd9

echo "Dataset downloads complete under ${data_dir}."
