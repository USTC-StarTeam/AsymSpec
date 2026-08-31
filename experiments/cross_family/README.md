# Cross-family AsymSpec

This directory contains the heterogeneous-tokenizer extension used for the
LongBench portability experiments. It supports the two evaluated directions:

- Llama-3.2-3B drafter to Qwen3-32B verifier;
- Qwen3-4B drafter to Llama-3.3-70B verifier.

The extension aligns string-identical tokens in the two vocabularies. Draft
tokens and fused emissions are restricted to this exact shared-token map;
unmapped verifier tokens receive no delta contribution. The map also pairs the
families' turn-end and text-end special tokens explicitly.

## Build a vocabulary map

Llama drafter to Qwen verifier:

```bash
python experiments/cross_family/build_hetero_map.py \
  --output artifacts/hetero_llama3b_qwen32b.pt
```

Qwen drafter to Llama verifier:

```bash
python experiments/cross_family/build_hetero_map_qwen2llama.py \
  --output artifacts/hetero_qwen4b_llama70b.pt
```

Each artifact stores the exact drafter-to-verifier and verifier-to-drafter
maps, the verifier emission mask, aligned index tensors for delta fusion, and
model/vocabulary metadata.

## Deploy and run

Deploy the standard vLLM integration first, then enable the cross-vocabulary
extension:

```bash
python scripts/deploy_specsteer.py --apply
python experiments/cross_family/apply_hetero_patch.py
```

Run the Llama-to-Qwen configuration:

```bash
python experiments/cross_family/bench_lb_crossfamily.py \
  --mode specsteer --hetero \
  --drafter_path meta-llama/Llama-3.2-3B-Instruct \
  --verifier_path Qwen/Qwen3-32B \
  --hetero_map artifacts/hetero_llama3b_qwen32b.pt \
  --slm 4B --K 2 --beta 1.0 --gamma 0.5 --asym_method jsd \
  --cell llama_to_qwen --out outputs/cross_family/llama_to_qwen.json \
  --responses outputs/cross_family/llama_to_qwen.responses.jsonl
```

For Qwen-to-Llama, select the reverse map, set the corresponding model paths,
and choose the tensor-parallel degree required by the verifier deployment.

The mapping is tokenizer-pair-specific. Build a new artifact before using
other model families or tokenizers, and inspect the reported exact-token
coverage before interpreting results.
