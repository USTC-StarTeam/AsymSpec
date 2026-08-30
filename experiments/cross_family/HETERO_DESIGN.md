# USD-AsymSpec: heterogeneous-vocabulary cross-family in vLLM 0.19 + SpecSteer

Goal: answer qJdq W-a/Q-b ("heterogeneous drafter or verifier pair (different family
and vocabulary) ... mismatched tokenizers") with a number directly comparable to the
paper: **verifier = Qwen3-32B (the paper's verifier, so Floor 45.0 / Ceiling 65.5
carry over unchanged), drafter = Llama-3.2-3B**, LongBench 3 subsets, jsd K=2 β=1.0
γ=0.5, TP=1 single GPU, bs=1 eager, scored by scripts/eval_lb.py.

Vocab facts (measured): Qwen3 model V_B=151936 (tokenizer 151669), Llama-3.2 model
V_A=128256. String-match intersection 109,566 = 85.4% of A, 72.2% of B; **94.7%
frequency-weighted coverage on real text**. Both byte-level BPE with 'Ġ' space sign
(no normalization needed); all 256 byte tokens shared → any string expressible in
the intersection.

## Staging
- **M1 (this build)**: all emitted tokens restricted to intersection ∪ {Qwen eos
  specials} → strict 1:1 A↔B stream, zero length bookkeeping. Bonus token dropped
  (not emitted) when outside the allow-set (~5% of all-accept steps; costs a little
  speed, never correctness).
- **M2 (optional, +2-3d)**: unrestricted verifier emission + text-roundtrip resync
  (variable-k drafter appends). Only if M1 lands early.

## Map artifact

Build the artifact with an explicit output path, for example:

```bash
python experiments/cross_family/build_hetero_map.py \
  --output artifacts/hetero_llama3b_qwen32b.pt
```
- `a2b` int64 [V_A=128256]: Llama id → Qwen id, −1 if unmapped
- `b2a` int64 [V_B=151936]: Qwen id → Llama id, −1 if unmapped
- `allow_b` bool [V_B]: image of a2b ∪ {151645 im_end, 151643 endoftext}
- special pairs: 128009 eot↔151645 im_end; 128001 end_of_text↔151643 endoftext;
  128008 eom→151645 (one-way)
- meta: names, sizes, coverage

## Engine edit points (rebuttal_env copy ONLY; .orig backups; live env untouched)
All hetero code behind `ASYMSPEC_HETERO_VOCAB=1` + `ASYMSPEC_HETERO_MAP=<pt path>`;
flag off ⇒ byte-identical to current behavior.

- **E0 loader** (specsteer_model.py `SpecSteerProposer.__init__`): if env set, load
  map to device buffers `self._h_a2b/_h_b2a/_h_allow_b/_h_suppress_a` (suppress_a =
  a2b<0). Log coverage.
- **E1 suffix translate** (top of `propose()`, before `_maybe_substitute_aug_tokens`,
  anchor ~line 2780): `target_token_ids = b2a[target_token_ids].clamp(min=0)` clone
  for the DRAFTER-side stream only. Prefill positions are garbage after this but are
  fully overwritten by aug substitution (assert every request has aug ids under
  hetero). Decode positions are guaranteed mapped because emission is allow_b-masked
  (E6) — assert no −1 at decode positions.
  NOTE: base mirror shares this translated stream at decode; at prefill base uses E4.
- **E2 drafting mask** (`_greedy_sample` override, anchor ~2305): capture logits
  (full A-space, unmasked — δ and JSD want the raw distribution), then
  `masked = logits.masked_fill(_h_suppress_a, -inf)`, return `masked.argmax(-1)`.
- **E3 draft id return** (`SpecSteerProposer.propose` return, anchor ~2925):
  `return a2b[draft_token_ids]` (assert ≥0; guaranteed by E2).
- **E4 base prefill**: base currently mirrors the target's MAIN prompt ids
  (`_base_mirror_forward(target_token_ids ...)` — B-space). Under hetero the base
  prefix must be `specsteer_base_prompt_ids` (A-encoded compressed context, NEW
  extra_arg) with its own length/positions/slots (block table has room: aug ≫ base).
  Find the prefill-time call site via `_base_prefilled` bookkeeping; dedicated
  one-time forward like the aug-prefill. Decode steps: base consumes E1-translated
  suffix at positions continuing from len_A(base).
- **E5 Gate-A bonus** (`_compute_aug_first_bonus`): its internal drafter argmax must
  apply `_h_suppress_a`; its returned bonus replaces `next_token_ids` (B-space) ⇒
  map `a2b[bonus]`.
- **E6 sampler** (specsteer_sampler.py `specsteer_greedy_sample`): hetero branch
  (env flag; map path loaded lazily once, cached module-global on device):
  - relax `assert target.shape == aug.shape == base.shape` → same num_tokens only
  - `d_B = draft_token_ids`; `d_A = b2a[d_B]` (assert ≥0)
  - `p_llm = softmax(t).gather(d_B)`; `p_base = softmax(b).gather(d_A)`
  - δ_A = a_log − b_log (also raw_aug/scd variants in A/B spaces:
    ours: A; raw_aug: A; scd (t−b) is cross-space — hetero scd unsupported, assert)
  - fused: `f = t_log.clone(); f[:, a2b_valid_dst] += β·δ_A[:, a2b_valid_src]`
    (precomputed index pair tensors); `f.masked_fill_(~allow_b, -inf)`;
    `fused_argmax = f.argmax(-1)`
  - JSD/CMA gates: computed from a_log/b_log (A-space) — unchanged code path
  - bonus: `bonus_ok = allow_b[bonus_token_ids]`; where !ok → PLACEHOLDER (drop
    bonus emission, never emit unmappable)
- **E7 runner glue**: any same-shape asserts between target/drafter logits in the
  runner's `_specsteer_sample` collection path — relax under flag.
- **E8 bench** (bench_lb_crossfamily.py): `--hetero` flag ⇒
  - verifier prompt: existing path (Qwen tokenizer + Qwen template, enable_thinking=False)
  - aug prompt: Llama tokenizer + Llama chat template (same CoT text template)
  - NEW `specsteer_base_prompt_ids`: compressed/main CONTENT through Llama tok+template
  - env ASYMSPEC_HETERO_VOCAB=1, ASYMSPEC_HETERO_MAP=<pt>
  - responses decoded with the VERIFIER tokenizer (stream is B-ids) — existing path

## Config gate
`verify_equal_vocab_size_if_draft_model` only fires for method=="draft_model";
specsteer bypasses it already — NO EDIT NEEDED (verified in rebuttal_env copy).

## Test ladder
1. unit: synthetic tensors through E6 hetero branch (CPU) — accept/reject/fused/bonus-drop
2. smoke n=5 (TP=1 single GPU, Qwen3-32B + Llama-3.2-3B): loads, generates, no −1
   assertions, decoded text coherent, answers extractable
3. anchor: re-run SAME-family Qwen3-32B+Qwen3-4B jsd K2 n=600 in THIS env → expect
   ≈59.7 F1 (engine integrity; hetero diff then attributable to hetero alone)
4. hetero n=600 → F1 + recovery vs Floor 45.0 / Ceiling 65.5
Steps 3&4 run in parallel on two free GPUs (TP=1 each) once Llama×Llama cells finish.

## Status log (append-only)
- [x] design; vocab overlap measured (85.4%/72.2%, 94.7% freq)
- [x] map artifact built (109,569 exact pairs + total b2a_sur; .pt at models/)
- [x] E0-E4+E6 engine edits applied (apply_hetero_patch.py; .orig_hetero backups;
      flag-off import verified byte-safe)
- [x] E8 bench edits (B1-B6 in bench_lb_crossfamily.py)
- [x] unit test PASS (test_hetero_sampler.py: accept/reject/δ-scatter/mask/bonus-none
      on real triton kernels; NOTE cu_num_draft_tokens has NO leading zero)
- [x] no-GPU dry run PASS (prompts kept 2/2, HETERO env set, config gate silent)
- [x] smoke n=5 PASS (GPU6): EXIT=0, AR=0.454, coherent Qwen prose, clean eos.
      Fixed en route: (1) _os UnboundLocal in E0 -> use module-level os (was
      breaking ALL specsteer starts incl. flag-off); (2) _use_fast_base_fwd
      default True -> forced PV path under hetero; (3) bench warmup lacked
      extra_args -> E4a assert (warmup now mirrors real requests); (4) PV
      one-shot hidden-diag uses target-frame index -> skipped under hetero.
      OPEN ITEM (dissected via K=3 probe): last-position drafts always reject —
      K=2 per_pos [0.91, 0.00], K=3 [0.855, 0.814, 0.000]. Anomaly tracks K-1
      (the LAST draft), not absolute position: within-step feedback is healthy
      (pos1 accepts 81% at K=3). Same-vocab anchor K=2 shows [0.90, 0.82] —
      hetero-only. Speed-only (MAL 2.67/4 at K=3; fused emissions fluent, F1
      unaffected). Candidate causes: last-row p_llm/p_base gather alignment
      under bonus=none, or drafter's terminal eot proposals mapping to im_end
      (p_llm≈0 mid-answer). Instrumented run needed to close; deprioritized.
- [ ] anchor repro (59.7) — run_hetero.sh anchor <gpu>
- [ ] hetero full n=600 — run_hetero.sh full <gpu>


## REVERSE PAIR (2026-07-10): Qwen3-4B drafter -> Llama-3.3-70B verifier, TP=2
- reverse map: build_hetero_map_qwen2llama.py -> hetero_qwen4b_llama70b.pt
  (A=Qwen V_A=151936, B=Llama V_B=128256; same 109,566 intersection; specials reversed)
- run: run_hetero_rev.sh <smoke|full> <gpu_pair>; --slm 4B (drafter=Qwen3-4B),
  --verifier_path Llama-70B, --hetero_map <reverse pt>, TP=2
- BUGS FOUND & FIXED (only surfaced in reverse config):
  1. E5 Gate-A bonus: _compute_aug_first_bonus argmax was raw DRAFTER-vocab id
     (Qwen 151645 <|im_end|>) written straight into next_token_ids -> IndexKernel
     OOB against Llama verifier width 128256. FIX: hetero branch masks to
     intersection + a2b-maps the bonus argmax (mirrors E2+E3). Forward config
     dodged it only because Llama drafter ids all fit inside Qwen's larger vocab.
  2. Sampler hetero branch: added CPU-side bounds guards (dst_b/src_a/draft/d_a)
     for clear errors instead of async CUDA asserts.
  3. TP=2 first validated here (70B verifier forces it); PV base path works.
- smoke n=5 PASS: EXIT 0, AR 0.312, fluent Llama output (repetition tendency like
  the same-family weak-drafter case; not a bug).

## FAST-PATH DRAFT MAPPING BUG (2026-07-11) — reverse-pair crash root cause
Symptom: reverse full (Qwen4B drafter -> Llama70B verifier) crashed at sampler
bounds guard `draft id range [151645,151645] outside target width 128256`
(151645 = Qwen <|im_end|>), while n=5 smoke passed.
Root cause: `_merged_aug_prefill_and_kdecode` (fast incremental draft path, taken
every step where next_token_ids is not None = the DOMINANT path) argmaxes drafts
in RAW drafter vocab (~line 1923) and its caller early-returns merged_drafts at
~line 2796, BYPASSING the E3 a2b map that only the slow super().propose() path
applies (~line 3084). So fast-path drafts reach the sampler unmapped. Forward
(Llama3B->Qwen32B) never crashed because Llama ids (<128256) fit inside Qwen's
larger width; reverse crashes on any Qwen id >128256 (e.g. im_end).
Fix (3 edits, hetero-guarded, deployed copy):
 - E2 mask at fast-path bonus argmax (~1793) and draft argmax (~1923):
   masked_fill(_h_suppress_a, -inf) before argmax; drafts stay DRAFTER-space to
   feed the next decode. RAW (unmasked) logits still appended to
   _draft_logits_per_pos (delta/JSD need the full drafter vocab).
 - E3 map at the fast-path early return (~2796): _h_a2b[merged_drafts], mirroring
   the slow-path E3, same min>=0 assert.
Validation: reverse smoke EXIT 0 (was crash), AR 0.31->0.39. FORWARD smoke AR
0.454->0.460 (UNCHANGED) -> forward 47.1 is NOT affected: accuracy is verifier +
delta-fusion driven, and the bug only touched draft acceptance id-matching, which
only overflowed in the reverse direction. Last-pos per_pos[K-1]=0 anomaly persists
(separate speed-only OPEN ITEM).
TODO(hygiene): fold these 3 edits into apply_hetero_patch.py so a re-deploy keeps
them (currently applied only to the deployed copy).
