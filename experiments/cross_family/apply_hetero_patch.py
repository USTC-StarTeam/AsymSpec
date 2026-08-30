#!/usr/bin/env python3
"""Apply AsymSpec heterogeneous-vocabulary edits (E0-E4, E6) to the active
vLLM 0.19 installation. Idempotent: skips if marker present. Backs up
pristine files to *.orig_hetero. All behavior behind ASYMSPEC_HETERO_VOCAB=1.
Design doc: ``experiments/cross_family/HETERO_DESIGN.md``.
"""
import shutil, sys, py_compile
from pathlib import Path
import vllm

V = Path(vllm.__file__).resolve().parent
MODEL = str(V / "v1" / "spec_decode" / "specsteer_model.py")
SAMPLER = str(V / "v1" / "sample" / "specsteer_sampler.py")
MARK = "HETERO (USD-AsymSpec)"


def patch(path, edits):
    src = open(path).read()
    if MARK in src:
        print(f"[skip] {path} already patched")
        return
    shutil.copy(path, path + ".orig_hetero")
    for name, old, new in edits:
        assert old in src, f"anchor NOT FOUND for {name} in {path}"
        assert src.count(old) == 1, f"anchor NOT UNIQUE for {name} in {path}"
        src = src.replace(old, new, 1)
        print(f"[ok] {name}")
    open(path, "w").write(src)
    py_compile.compile(path, doraise=True)
    print(f"[compiled] {path}")


# ---------------- specsteer_model.py ----------------
E0_OLD = """        # SpecSteer path — dual_forward base was an early experiment and is
        # always skipped here.
        self._pathb_skip_dual_base: bool = True"""
E0_NEW = """        # SpecSteer path — dual_forward base was an early experiment and is
        # always skipped here.
        self._pathb_skip_dual_base: bool = True
        # ---- HETERO (USD-AsymSpec): cross-vocabulary drafter/verifier ----
        # ASYMSPEC_HETERO_VOCAB=1 + ASYMSPEC_HETERO_MAP=<pt>. a2b: exact
        # draft->target (drafts intersection-masked in _greedy_sample so the
        # exact map always hits). b2a_sur: TOTAL target->draft surrogate
        # (exact on the intersection; first re-encoded token elsewhere) for
        # translating committed suffixes. Flag off => byte-identical behavior.
        self._hetero = os.environ.get("ASYMSPEC_HETERO_VOCAB", "0") == "1"
        if self._hetero:
            _mp = os.environ["ASYMSPEC_HETERO_MAP"]
            _hm = torch.load(_mp, map_location="cpu")
            self._h_a2b = _hm["a2b"].to(self.device)
            self._h_b2a_sur = _hm["b2a_sur"].to(self.device)
            self._h_b2a_sur_cpu = _hm["b2a_sur"]
            self._h_suppress_a = (_hm["a2b"] < 0).to(self.device)
            self._h_base_off: dict[str, int] = {}
            logger.info(
                "SpecSteer HETERO on: map=%s exact_pairs=%d V_A=%d V_B=%d",
                _mp, int((_hm["a2b"] >= 0).sum()),
                _hm["meta"]["V_A"], _hm["meta"]["V_B"])"""

E1_OLD = """        # Phase 1 + Phase 2 Option Z: drafter in gid=1 with aug-space when"""
E1_NEW = """        # HETERO E1: the engine streams committed tokens in TARGET vocab;
        # everything below this wrapper feeds drafter/base only. Translate to
        # drafter vocab via the total surrogate map. Prefill positions become
        # garbage here but are fully overwritten by the aug substitution (all
        # hetero requests carry specsteer_aug_prompt_ids).
        if getattr(self, "_hetero", False) and target_token_ids is not None:
            target_token_ids = self._h_b2a_sur[target_token_ids.long()].to(
                target_token_ids.dtype)
            kwargs["target_token_ids"] = target_token_ids
            if args:
                args = (target_token_ids,) + args[1:]
            if next_token_ids is not None:
                next_token_ids = self._h_b2a_sur[next_token_ids.long()].to(
                    next_token_ids.dtype)
                kwargs["next_token_ids"] = next_token_ids
                if len(args) > 3:
                    args = args[:3] + (next_token_ids,) + args[4:]

        # Phase 1 + Phase 2 Option Z: drafter in gid=1 with aug-space when"""

E2_OLD = """        logits = self.model.compute_logits(hidden_states)
        self._draft_logits_per_pos.append(logits.detach())
        self._base_logits_per_pos.append(logits.detach())  # placeholder
        return logits.argmax(dim=-1)"""
E2_NEW = """        logits = self.model.compute_logits(hidden_states)
        self._draft_logits_per_pos.append(logits.detach())
        self._base_logits_per_pos.append(logits.detach())  # placeholder
        if getattr(self, "_hetero", False):
            # HETERO E2: drafts must map 1:1 to target vocab -> mask
            # non-intersection ids BEFORE argmax. Captured logits above stay
            # full A-space (delta and JSD want the raw distribution).
            return logits.masked_fill(
                self._h_suppress_a.unsqueeze(0), float("-inf")).argmax(dim=-1)
        return logits.argmax(dim=-1)"""

E3_OLD = """        self._prof_report_if_due()
        return draft_token_ids"""
E3_NEW = """        self._prof_report_if_due()
        if getattr(self, "_hetero", False):
            # HETERO E3: engine/verifier consume TARGET-vocab ids. Drafts are
            # intersection-masked (E2) so the exact map always hits.
            _mapped = self._h_a2b[draft_token_ids.long()]
            assert int(_mapped.min()) >= 0, "hetero draft outside intersection"
            return _mapped.to(draft_token_ids.dtype)
        return draft_token_ids"""

E4_OLD = """            if req_id_i not in self._base_prefilled:
                start_pos_i = 0
                input_tokens_i = (
                    list(token_ids_i) + nt_list + drafts_per_req[i]
                )
                self._base_prefilled.add(req_id_i)
            else:
                # Incremental: writes [active-1 .. active+K]. Prior PV must
                # have populated [0..active-2].
                start_pos_i = active_i - 1
                input_tokens_i = (
                    [int(token_ids_i[-1])] + nt_list + drafts_per_req[i]
                )"""
E4_NEW = """            if req_id_i not in self._base_prefilled:
                start_pos_i = 0
                if getattr(self, "_hetero", False):
                    # HETERO E4a: base prefix = drafter-tokenizer encoding of
                    # the compressed context (specsteer_base_prompt_ids), NOT
                    # the verifier-vocab prompt ids; committed suffix beyond
                    # the prompt is translated 1:1 (total surrogate map).
                    _req = self.runner.requests.get(req_id_i)
                    _bi = None
                    if (_req is not None and _req.sampling_params is not None
                            and _req.sampling_params.extra_args is not None):
                        _bi = _req.sampling_params.extra_args.get(
                            "specsteer_base_prompt_ids")
                    assert _bi, "hetero requires specsteer_base_prompt_ids"
                    _npt = _req.num_prompt_tokens
                    _sur = self._h_b2a_sur_cpu
                    _gen = [int(_sur[int(t)]) for t in token_ids_i[_npt:]]
                    input_tokens_i = (
                        list(_bi) + _gen + nt_list + drafts_per_req[i]
                    )
                    self._h_base_off[req_id_i] = len(_bi) - _npt
                else:
                    input_tokens_i = (
                        list(token_ids_i) + nt_list + drafts_per_req[i]
                    )
                self._base_prefilled.add(req_id_i)
            else:
                # Incremental: writes [active-1 .. active+K]. Prior PV must
                # have populated [0..active-2].
                start_pos_i = active_i - 1
                if getattr(self, "_hetero", False):
                    # HETERO E4b: base runs at a constant per-request position
                    # offset (prefix length diff; suffix is 1:1).
                    start_pos_i += self._h_base_off.get(req_id_i, 0)
                    input_tokens_i = (
                        [int(self._h_b2a_sur_cpu[int(token_ids_i[-1])])]
                        + nt_list + drafts_per_req[i]
                    )
                else:
                    input_tokens_i = (
                        [int(token_ids_i[-1])] + nt_list + drafts_per_req[i]
                    )"""

E4F_OLD = """    def _base_forward_fast(
        self,"""
E4F_NEW = """    def _base_forward_fast(
        self,"""
# fast path guard is inserted via its docstring end instead (see below).

E4G_OLD = """    def _base_mirror_forward(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        common_attn_metadata,
    ) -> torch.Tensor | None:"""
E4G_NEW = """    def _base_mirror_forward(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        common_attn_metadata,
    ) -> torch.Tensor | None:
        assert not getattr(self, "_hetero", False), (
            "HETERO (USD-AsymSpec): dual/mirror base path unsupported; "
            "PathB PV is the only hetero base path")"""

model_edits = [
    ("E0 loader", E0_OLD, E0_NEW),
    ("E1 suffix translate", E1_OLD, E1_NEW),
    ("E2 drafting mask", E2_OLD, E2_NEW),
    ("E3 draft id map", E3_OLD, E3_NEW),
    ("E4 base prefill/incremental", E4_OLD, E4_NEW),
    ("E4g mirror guard", E4G_OLD, E4G_NEW),
]

# ---------------- specsteer_sampler.py ----------------
S1_OLD = """    assert draft_token_ids.ndim == 1
    assert target_logits.ndim == aug_logits.ndim == base_logits.ndim == 2
    assert target_logits.shape == aug_logits.shape == base_logits.shape"""
S1_NEW = """    assert draft_token_ids.ndim == 1
    assert target_logits.ndim == aug_logits.ndim == base_logits.ndim == 2
    # HETERO (USD-AsymSpec): drafter logits live in the DRAFTER vocab; only
    # the row counts must agree with the target.
    _hetero = os.environ.get("ASYMSPEC_HETERO_VOCAB", "0") == "1"
    if _hetero:
        assert aug_logits.shape == base_logits.shape
        assert target_logits.shape[0] == aug_logits.shape[0]
    else:
        assert target_logits.shape == aug_logits.shape == base_logits.shape"""

S2_OLD = """    _delta_src = os.environ.get("ASYMSPEC_DELTA_SRC", "ours")
    if _delta_src == "ours":
        delta = a_log - b_log
    elif _delta_src == "raw_aug":
        delta = a_log
    elif _delta_src == "scd":
        delta = t_log - b_log
    else:
        raise ValueError(
            f"ASYMSPEC_DELTA_SRC={_delta_src!r} not in {{ours, raw_aug, scd}}")
    delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
    fused = t_log + beta * delta
    fused_argmax = fused.argmax(dim=-1)   # [num_tokens] — used as reject token by all methods

    # Per-position scalar probabilities at the drafted token.
    idx = draft_token_ids.to(torch.int64).unsqueeze(-1)
    p_llm  = t_log.exp().gather(-1, idx).squeeze(-1)
    p_base = b_log.exp().gather(-1, idx).squeeze(-1)"""
S2_NEW = """    _delta_src = os.environ.get("ASYMSPEC_DELTA_SRC", "ours")
    if _hetero:
        # HETERO (USD-AsymSpec): delta lives in the drafter vocab; scatter it
        # into the target vocab over the exact-map pairs (unmapped target ids
        # keep their own logit, i.e. delta=0 — additive signal, NOT -inf).
        # Emission (fused argmax) is masked to allow_b: intersection image +
        # verifier eos specials, so every committed token maps 1:1 back.
        _m = _hetero_map(device)
        if _delta_src == "ours":
            delta_a = a_log - b_log
        elif _delta_src == "raw_aug":
            delta_a = a_log
        else:
            raise ValueError(
                "hetero supports ASYMSPEC_DELTA_SRC in {ours, raw_aug}; "
                f"got {_delta_src!r} (scd is cross-vocab)")
        delta_a = torch.nan_to_num(delta_a, nan=0.0, posinf=0.0, neginf=0.0)
        fused = t_log.clone()
        fused[:, _m["dst_b"]] += beta * delta_a[:, _m["src_a"]]
        fused.masked_fill_(~_m["allow_b"].unsqueeze(0), float("-inf"))
        fused_argmax = fused.argmax(dim=-1)
        idx = draft_token_ids.to(torch.int64).unsqueeze(-1)      # B-space
        d_a = _m["b2a"][draft_token_ids.to(torch.int64)]
        assert int(d_a.min()) >= 0, "hetero draft id not in intersection"
        p_llm  = t_log.exp().gather(-1, idx).squeeze(-1)
        p_base = b_log.exp().gather(-1, d_a.unsqueeze(-1)).squeeze(-1)
    else:
        if _delta_src == "ours":
            delta = a_log - b_log
        elif _delta_src == "raw_aug":
            delta = a_log
        elif _delta_src == "scd":
            delta = t_log - b_log
        else:
            raise ValueError(
                f"ASYMSPEC_DELTA_SRC={_delta_src!r} not in {{ours, raw_aug, scd}}")
        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
        fused = t_log + beta * delta
        fused_argmax = fused.argmax(dim=-1)   # [num_tokens] — used as reject token by all methods

        # Per-position scalar probabilities at the drafted token.
        idx = draft_token_ids.to(torch.int64).unsqueeze(-1)
        p_llm  = t_log.exp().gather(-1, idx).squeeze(-1)
        p_base = b_log.exp().gather(-1, idx).squeeze(-1)"""

S0_OLD = """def specsteer_greedy_sample("""
S0_NEW = """_HETERO_CACHE: dict = {}


def _hetero_map(device):
    \"\"\"HETERO (USD-AsymSpec): lazily load the vocab map onto `device`.\"\"\"
    key = str(device)
    if key not in _HETERO_CACHE:
        _m = torch.load(os.environ["ASYMSPEC_HETERO_MAP"], map_location="cpu")
        _HETERO_CACHE[key] = {
            k: _m[k].to(device) for k in ("b2a", "allow_b", "src_a", "dst_b")
        }
    return _HETERO_CACHE[key]


def specsteer_greedy_sample("""

sampler_edits = [
    ("S0 map loader", S0_OLD, S0_NEW),
    ("S1 shape assert relax", S1_OLD, S1_NEW),
    ("S2 hetero delta/fused/gather", S2_OLD, S2_NEW),
]

patch(MODEL, model_edits)
patch(SAMPLER, sampler_edits)
print("ALL PATCHES APPLIED")

# NOTE(2026-07-10): reverse-pair (Qwen drafter->Llama verifier) also needs
# E5 Gate-A bonus map + sampler bounds guards + fast-base bypass; these were
# applied directly to the copy env (see specsteer_model.py _compute_aug_first_bonus
# hetero branch and specsteer_sampler.py hetero gather guards).
