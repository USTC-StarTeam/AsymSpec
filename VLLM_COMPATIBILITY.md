# vLLM compatibility assessment

The released integration currently supports **vLLM 0.19.0 only**. It replaces
several complete upstream Python files, so applying it to another release
without rebasing those changes can silently remove upstream fixes or fail at
runtime. `scripts/deploy_specsteer.py` therefore checks the installed version
before modifying it.

## Static migration assessment

The following comparison is source-level only; it is not a runtime support
claim.

- **vLLM 0.19.1 is the nearest migration candidate.** Relative to 0.19.0,
  `gpu_model_runner.py` is identical, `eagle.py` adds one model name, and
  `config/speculative.py` adds two model names. Rebasing the AsymSpec changes
  is mechanically small, but still requires the full GPU smoke suite before
  changing the compatibility pin.
- **vLLM 0.20.0 and later require a real port.** The speculative proposer was
  split out of `eagle.py` into `llm_base_proposer.py`, while the runner and
  speculative-configuration interfaces also changed. The current full-file
  patches must not be copied over these versions.

Upstream references:

- [v0.19.0...v0.19.1](https://github.com/vllm-project/vllm/compare/v0.19.0...v0.19.1)
- [v0.19.1...v0.20.0](https://github.com/vllm-project/vllm/compare/v0.19.1...v0.20.0)

## Required validation for a new pin

1. Rebase only the AsymSpec-specific hunks onto the target vLLM sources.
2. Run the text-only and multimodal deployment checks from a clean install.
3. Smoke-test Floor, Ceiling, standard SD, and AsymSpec at the paper defaults.
4. Confirm identical greedy outputs on a fixed sample and inspect acceptance
   and delta-fusion diagnostics.
5. Re-run throughput profiling before reporting compatibility.
