#!/usr/bin/env bash
# Download public datasets used by the AsymSpec experiments.
# GAIA and Llama-family assets require accepting their upstream terms first.

set -euo pipefail

for command_name in hf curl git; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Required command not found: ${command_name}" >&2
    exit 1
  fi
done

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
mc_revision="5ccefcca6a39020d66c1383c4e6a809cb07afa33"
mkdir -p "${data_dir}/multi-challenge/data"
curl --fail --location --retry 3 \
  "https://raw.githubusercontent.com/ekwinox117/multi-challenge/${mc_revision}/data/benchmark_questions.jsonl" \
  --output "${data_dir}/multi-challenge/data/benchmark_questions.jsonl"

echo "==> API-Bank"
apibank_revision="483554eae102996f5ec1f4feab4e78ef29c2a394"
apibank_target="${data_dir}/DAMO-ConvAI/api-bank"
if [[ ! -d "${apibank_target}/apis" || ! -d "${apibank_target}/lv1-lv2-samples" ]]; then
  tmp_dir="$(mktemp -d)"
  trap 'rm -rf "${tmp_dir}"' EXIT
  git -C "${tmp_dir}" init --quiet
  git -C "${tmp_dir}" remote add origin \
    https://github.com/AlibabaResearch/DAMO-ConvAI.git
  git -C "${tmp_dir}" sparse-checkout init --cone
  git -C "${tmp_dir}" sparse-checkout set api-bank
  git -C "${tmp_dir}" fetch --quiet --depth 1 origin "${apibank_revision}"
  git -C "${tmp_dir}" checkout --quiet --detach FETCH_HEAD
  mkdir -p "${apibank_target}"
  cp -R "${tmp_dir}/api-bank/." "${apibank_target}/"
fi

echo "==> MathVista and SimpleQA"
hf download AI4Math/MathVista --repo-type dataset
hf download basicv8vc/SimpleQA --repo-type dataset

echo "==> GAIA (gated)"
hf download gaia-benchmark/GAIA --repo-type dataset \
  --revision 682dd723ee1e1697e00360edccf2366dc8418dd9

echo "Dataset downloads complete under ${data_dir}."
