# AsymSpec vLLM integration

This directory contains the implementation used by the AsymSpec experiments.
The internal `specsteer` module name predates the final paper title and is
retained to avoid a risky mass rename of the vLLM integration surface.

## Versions

- `v0.10.mm/`: current implementation for vLLM 0.19.0, including multimodal
  drafter support. This is the default deployed version.
- `v0.10/`: text-only implementation retained as a smaller reference.

The multimodal version is a strict superset of the text-only path. Use
`scripts/deploy_specsteer.py`; do not copy files manually.

```bash
python scripts/deploy_specsteer.py --check
python scripts/deploy_specsteer.py --apply
```

The helper locates the active vLLM installation dynamically, saves original
files under `.backups/`, and supports `--revert`.

## Per-step computation

AsymSpec evaluates:

1. the drafter on the full context to propose tokens and obtain `aug_logits`;
2. the same drafter on the compressed context to obtain `base_logits`;
3. the verifier on the compressed context to obtain `target_logits`.

The compressed-context drafter pass maintains a separate KV cache, reducing
its incremental cost to O(K). The sampler then applies CDA and delta fusion:

```text
gamma_eff = gamma * exp(-JSD(p_aug || p_base))
delta     = log p_aug - log p_base
fallback  = argmax(log p_target + beta * delta)
```

See [`IMPLEMENTATION.md`](IMPLEMENTATION.md) for the KV-cache implementation
and equivalence argument.

## Safety checks

The implementation fails fast when the compressed-context drafter logits are
missing or shape-incompatible. A silent `base_logits = aug_logits` fallback
would make the context delta zero and invalidate the method.

The patches modify installed vLLM source files. Always run inside a dedicated
environment and keep the generated `.backups/` directory until validation is
complete.
