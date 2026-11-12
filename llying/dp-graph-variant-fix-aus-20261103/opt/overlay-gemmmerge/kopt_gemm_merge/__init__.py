"""KOPT_GEMM_MERGE=1: in GLM-5.2's decode / verify CUDA graphs, compute the DSA indexer's wk (hidden -> 128) and
weights_proj (hidden -> 32) in the same bf16 GEMM as the attention's fused_qkv_a (hidden -> 2624), and with
KOPT_GEMM_MERGE_QB=1 also the indexer's wq_b (q_lora -> 4096) in the same GEMM as q_b_proj (q_lora -> 16384).
Microbench (KERNEL-PROGRESS-20261109 section 15, 11-12, tuned tables): rows 64 / 88 / 96, fused_qkv_a + wk +
weights_proj 20.4 / 24.0 / 26.1 us separate -> 12.8 / 15.6 / 15.5 us merged; q_b_proj + wq_b 23.7 / 27.3 / 28.2 ->
22.0 / 23.3 / 22.9 us, before the copy the indexer needs (its Hadamard rotation views wq_b's output as [-1, 128]).

Scope: the target model's indexer layers (not skip_topk, not NextN), inside CUDA-graph capture of decode / verify
batches with at least KOPT_GEMM_MERGE_MIN_ROWS rows (default 49; up to 48 rows the KF fused indexer computes these
projections itself in one GEMV). Everything else runs the original path. After weight loading each layer's separate
weights become row views of one merged tensor (no extra memory), and the merged GEMM goes through the layer's own
linear method. The indexer's projections use their precomputed columns only when called with the same input tensor
(data pointer and shape) in the same layer forward; otherwise they compute as before.

Env: KOPT_GEMM_MERGE_QB=0|1 (default 0), KOPT_GEMM_MERGE_MIN_ROWS (default 49), KOPT_GEMM_MERGE_LOG=1.
"""
import importlib.abc
import os
import sys

_QB = os.environ.get("KOPT_GEMM_MERGE_QB", "0") == "1"
_MIN_ROWS = int(os.environ.get("KOPT_GEMM_MERGE_MIN_ROWS", "49"))
_LOG = os.environ.get("KOPT_GEMM_MERGE_LOG", "1") == "1"
_warned = set()


def log(msg):
    if _LOG:
        print(f"[kopt-gemm-merge] {msg}", file=sys.stderr, flush=True)


def _warn_once(key, msg):
    if key not in _warned:
        _warned.add(key)
        log(msg)


def _src(x):
    return (x.data_ptr(), tuple(x.shape))


class _Merged:
    """One bf16 weight [sum(sizes), K] whose row blocks are the separate modules' weights."""

    def __init__(self, owner, weight, sizes):
        import torch

        self.owner = owner
        self.sizes = sizes
        self.holder = torch.nn.Module()
        self.holder.weight = weight

    def __call__(self, x):
        return self.owner.quant_method.apply(self.holder, x, None)


def _merge_rows(modules):
    """Concatenate the modules' weights along rows and make each module's weight a view of the result."""
    import torch

    ws = [getattr(m, "weight", None) for m in modules]
    k = ws[0].shape[-1] if ws[0] is not None else None
    for m, w in zip(modules, ws):
        if (w is None or type(w.data) is not torch.Tensor or w.dtype != torch.bfloat16 or w.dim() != 2
                or w.shape[1] != k or not w.is_contiguous() or getattr(m, "bias", None) is not None
                or type(getattr(m, "quant_method", None)).__name__ != "UnquantizedLinearMethod"):
            return None
    merged = torch.cat([w.data for w in ws], 0)
    sizes, off = [], 0
    for m in modules:
        n = m.weight.shape[0]
        m.weight.data = merged[off:off + n]
        sizes.append(n)
        off += n
    return _Merged(modules[0], merged, sizes)


def _take_from_stash(mod, stash, key):
    """mod(x) returns stash[key] when stash[key] was computed from this very x, else runs as before."""
    import torch

    orig = mod.forward

    def forward(x, *args, **kwargs):
        hit = stash.pop(key, None)
        if hit is not None:
            src, out = hit
            if isinstance(x, torch.Tensor) and _src(x) == src:
                return out, None
            _warn_once(key, f"{key}: input differs from the merged GEMM's, computing it separately")
        return orig(x, *args, **kwargs)

    mod.forward = forward


def _armed(forward_batch, x):
    import torch
    from sglang.srt.model_executor.runner import get_is_capture_mode

    mode = forward_batch.forward_mode
    return (isinstance(x, torch.Tensor) and x.dim() == 2 and x.shape[0] >= _MIN_ROWS and get_is_capture_mode()
            and (mode.is_decode() or mode.is_target_verify()))


def _build(model):
    import torch

    n_qkva = n_qb = 0
    for layer in getattr(getattr(model, "model", None), "layers", []):
        attn = getattr(layer, "self_attn", None)
        idx = getattr(attn, "indexer", None)
        if idx is None or getattr(attn, "skip_topk", False) or getattr(attn, "is_nextn", False):
            continue
        idx._kopt_stash = {}
        fq, wk, wp = (getattr(attn, "fused_qkv_a_proj_with_mqa", None), getattr(idx, "wk", None),
                      getattr(idx, "weights_proj", None))
        if fq is not None and wk is not None and wp is not None:
            m = _merge_rows([fq, wk, wp])
            if m is not None:
                attn._kopt_qkva = m
                _take_from_stash(wk, idx._kopt_stash, "wk")
                _take_from_stash(wp, idx._kopt_stash, "weights_proj")
                n_qkva += 1
        qb, wq = getattr(attn, "q_b_proj", None), getattr(idx, "wq_b", None)
        if _QB and qb is not None and wq is not None:
            m = _merge_rows([qb, wq])
            if m is not None:
                attn._kopt_qb = m
                _take_from_stash(wq, idx._kopt_stash, "wq_b")
                n_qb += 1
    torch.cuda.empty_cache()
    log(f"merged fused_qkv_a + wk + weights_proj in {n_qkva} layers, q_b_proj + wq_b in {n_qb} layers "
        f"(min rows {_MIN_ROWS})")


def _patch_deepseek(mod):
    attn_cls, model_cls = mod.DeepseekV2AttentionMLA, mod.DeepseekV2ForCausalLM
    orig_prepare, orig_qb, orig_post = attn_cls.prepare_qkv_latent, attn_cls.q_b_proj_forward, model_cls.post_load_weights

    def prepare_qkv_latent(self, hidden_states, forward_batch):
        m = getattr(self, "_kopt_qkva", None)
        self._kopt_armed = m is not None and _armed(forward_batch, hidden_states)
        if m is None:
            return orig_prepare(self, hidden_states, forward_batch)
        stash = self.indexer._kopt_stash
        stash.clear()
        if not self._kopt_armed:
            return orig_prepare(self, hidden_states, forward_batch)
        out = m(hidden_states)
        n0, nk, _ = m.sizes
        src = _src(hidden_states)
        stash["wk"] = (src, out[:, n0:n0 + nk])
        stash["weights_proj"] = (src, out[:, n0 + nk:])
        return out[:, :n0]

    def q_b_proj_forward(self, q_lora):
        m = getattr(self, "_kopt_qb", None)
        if m is None or not getattr(self, "_kopt_armed", False) or q_lora.shape[0] < _MIN_ROWS:
            return orig_qb(self, q_lora)
        out = m(q_lora)
        n0 = m.sizes[0]
        self.indexer._kopt_stash["wq_b"] = (_src(q_lora), out[:, n0:].contiguous())
        return out[:, :n0].view(-1, self.num_local_heads, self.qk_head_dim)

    def post_load_weights(self, is_nextn=False, weight_names=None):
        orig_post(self, is_nextn=is_nextn, weight_names=weight_names)
        if not is_nextn:
            _build(self)

    attn_cls.prepare_qkv_latent = prepare_qkv_latent
    attn_cls.q_b_proj_forward = q_b_proj_forward
    model_cls.post_load_weights = post_load_weights
    log("patched DeepseekV2AttentionMLA.prepare_qkv_latent / q_b_proj_forward, DeepseekV2ForCausalLM.post_load_weights")


_TARGETS = {"sglang.srt.models.deepseek_v2": [_patch_deepseek]}


class _Loader(importlib.abc.Loader):
    def __init__(self, inner, fns):
        self.inner, self.fns = inner, fns

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        for fn in self.fns:
            fn(module)


class _Finder(importlib.abc.MetaPathFinder):
    busy = False

    def find_spec(self, name, path, target=None):
        fns = _TARGETS.get(name)
        if not fns or self.busy:
            return None
        self.busy = True
        try:
            for f in sys.meta_path:
                if f is self:
                    continue
                spec = getattr(f, "find_spec", lambda *a: None)(name, path, target)
                if spec is not None:
                    spec.loader = _Loader(spec.loader, fns)
                    return spec
        finally:
            self.busy = False
        return None


def install():
    # Another finder of this kind (kopt_megatp's) asks every other finder but itself, so two of them recurse into each
    # other; when kopt_megatp's hook is installed, register with it instead.
    km = sys.modules.get("kopt_megatp")
    targets = getattr(km, "_TARGETS", None)
    what = f"q_b_proj + wq_b {'on' if _QB else 'off'}, min rows {_MIN_ROWS}"
    if isinstance(targets, dict) and any(type(f).__module__ == "kopt_megatp" for f in sys.meta_path):
        for name, fns in _TARGETS.items():
            targets[name] = list(targets.get(name, [])) + list(fns)
        log(f"installed through kopt_megatp's import hook ({what})")
    elif not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
        log(f"installed ({what})")
