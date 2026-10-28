"""Drop the KernelForge GLM-5.2 decode kernels into a running sglang (v0.5.21, 11bfd868) without editing it.

Loaded by ../sitecustomize.py in every python process of the container. Installs a post-import hook that
patches four call sites once their modules are imported, each behind its own switch:

  INFERA_KERNEL_SPARSE=1   sglang.kernels.ops.attention.dsa.triton_sparse_mla.triton_sparse_mla_fwd
  INFERA_KERNEL_INDEXER=1  sglang.srt.layers.attention.dsa.dsa_indexer.aiter_paged_mqa_logits
                      sglang.srt.layers.attention.dsa.dsa_topk_backend.DSATopKBackend.topk_transform
  INFERA_KERNEL_MOE=1      sglang.srt.layers.moe.moe_runner.aiter.AiterRunnerCore.run

Every patch gates on the shapes the candidate was validated for and otherwise calls the original. The first
eligible call that is not under graph capture (sglang runs two eager warm-ups per graph batch size before
capturing) also runs the original and logs the comparison. All decisions are logged with prefix [infera-kernels].
INFERA_KERNEL_CHECK=0 disables the comparisons.
"""
import importlib.abc
import importlib.util
import os
import sys

_ON = {k: os.environ.get(f"INFERA_KERNEL_{k}", "0") == "1" for k in ("SPARSE", "INDEXER", "MOE", "DCP_SPARSE", "DCP_INDEXER")}
_CHECK = os.environ.get("INFERA_KERNEL_CHECK", "1") == "1"
_HERE = os.path.dirname(os.path.abspath(__file__))
_stats = {}
FORCE_ORIG = False      # tests: route every call to the original (kernel-level A/B in one process)
_disabled = set()       # kernels whose KF path raised once: original from then on


def _guard(kind, fn, fallback):
    """Run the KF path; on any exception log it once, disable this kernel and run the original."""
    try:
        return fn()
    except Exception as e:
        _disabled.add(kind)
        log(f"KF {kind} RAISED {type(e).__name__}: {e} -> disabled, using the original from now on")
        return fallback()


def log(msg):
    print(f"[infera-kernels] pid={os.getpid()} {msg}", file=sys.stderr, flush=True)


def _count(key, limit=3):
    """True for the first `limit` occurrences of key (keeps the logs short)."""
    n = _stats.get(key, 0) + 1
    _stats[key] = n
    return n <= limit


def _capturing():
    import torch
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _load(name):
    spec = importlib.util.spec_from_file_location(f"infera_kernels.{name}", os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _snr(ref, x):
    import torch
    ref, x = ref.float(), x.float()
    n = torch.mean((x - ref) ** 2).item()
    s = torch.mean(ref ** 2).item()
    return 100.0 if n == 0 else 10 * __import__("math").log10(max(s, 1e-30) / n)


# ----------------------------------------------------------------------------------------- forward mode
# The sparse-MLA and indexer candidates were validated only on TARGET_VERIFY rows (T = 6 * bs, one query
# token per row, row-expanded page tables). Their entry points do not see the forward mode, so the callers
# one level up record it here: DeepseekSparseAttnBackend.forward_{extend,decode} (sparse MLA) and
# Indexer._get_topk_paged (indexer; also reached by DECODE and DRAFT_EXTEND_V2, which must stay on the
# original -- 2026-10-23 r4-idx-real: a non-verify rows=82 call into the KF top-k faulted the GPU).
import threading

_mode = threading.local()


def _verify(kind):
    return getattr(_mode, kind, False)


def _wrap_mode(cls, name, kind):
    orig = getattr(cls, name)

    def wrapper(self, *args, **kwargs):
        fb = kwargs.get("forward_batch")
        if fb is None:
            fb = next((a for a in args if hasattr(a, "forward_mode")), None)
        prev = getattr(_mode, kind, False)
        setattr(_mode, kind, bool(fb is not None and fb.forward_mode.is_target_verify()))
        try:
            return orig(self, *args, **kwargs)
        finally:
            setattr(_mode, kind, prev)

    wrapper.__wrapped__ = orig
    setattr(cls, name, wrapper)


def _patch_sparse_mode(mod):
    for name in ("forward_extend", "forward_decode"):
        _wrap_mode(mod.DeepseekSparseAttnBackend, name, "sparse")
    log("patched DeepseekSparseAttnBackend.forward_{extend,decode} (mode)")


def _patch_indexer_mode(mod):
    _wrap_mode(mod.Indexer, "_get_topk_paged", "indexer")
    log("patched Indexer._get_topk_paged (mode)")


# ---------------------------------------------------------------------------------------------- sparse MLA
SPARSE_MAX_ROWS = int(os.environ.get("INFERA_KERNEL_SPARSE_MAX_ROWS", "512"))   # decode / verify rows only


def _patch_sparse(mod):
    orig = mod.triton_sparse_mla_fwd
    cand = _load("kf_sparse_mla").triton_sparse_mla_fwd
    checked = []

    def triton_sparse_mla_fwd(q_nope, q_rope, kv, indices, sm_scale, d_v=512, topk_length=None,
                              max_topk_length=None):
        rows, heads = q_nope.shape[0], q_nope.shape[1]
        ok = not FORCE_ORIG and "sparse" not in _disabled and _verify("sparse") and rows <= SPARSE_MAX_ROWS and heads == 64 and d_v == 512 and topk_length is None
        if _count(f"sparse:{ok}:{rows}", 1):
            log(f"sparse_mla rows={rows} heads={heads} kv={tuple(kv.shape)} dtype={q_nope.dtype} -> "
                f"{'KF' if ok else 'orig'}{' (capture)' if _capturing() else ''}")
        if not ok:
            return orig(q_nope, q_rope, kv, indices, sm_scale, d_v, topk_length, max_topk_length)
        out = _guard("sparse", lambda: cand(q_nope, q_rope, kv, indices, sm_scale, d_v, topk_length, max_topk_length),
                     lambda: orig(q_nope, q_rope, kv, indices, sm_scale, d_v, topk_length, max_topk_length))
        if _CHECK and not checked and not _capturing():
            checked.append(1)
            try:
                ref = orig(q_nope, q_rope, kv, indices, sm_scale, d_v, topk_length, max_topk_length)
                log(f"CHECK sparse_mla rows={rows}: snr(KF vs orig)={_snr(ref, out):.2f} dB "
                    f"maxabs={(ref.float() - out.float()).abs().max().item():.4g}")
            except Exception as e:  # a failed comparison must never take the server down
                log(f"CHECK failed: {type(e).__name__}: {e}")
        return out

    mod.triton_sparse_mla_fwd = triton_sparse_mla_fwd
    log("patched triton_sparse_mla_fwd")


# ------------------------------------------------------------------------------------------------- indexer
INDEXER_MAX_ROWS = 128   # kf_indexer._RMAX
_kf_indexer = None


def _kfi():
    global _kf_indexer
    if _kf_indexer is None:
        _kf_indexer = _load("kf_indexer")
    return _kf_indexer


def _patch_indexer_logits(mod):
    orig = mod.aiter_paged_mqa_logits
    if orig is None:
        log("dsa_indexer.aiter_paged_mqa_logits is None (no aiter); not patched")
        return
    checked = []

    def aiter_paged_mqa_logits(q_fp8, kv_cache_fp8, weights, seq_lens, block_tables, max_seq_len, *,
                               preshuffle, kv_block_size):
        rows = q_fp8.shape[0]
        ok = (not FORCE_ORIG and "logits" not in _disabled and _verify("indexer") and preshuffle and kv_block_size == 64 and rows <= INDEXER_MAX_ROWS
              and q_fp8.dim() == 3 and block_tables.shape[0] == rows and seq_lens.numel() == rows)
        if _count(f"logits:{ok}:{rows}", 1):
            log(f"indexer_logits rows={rows} q={tuple(q_fp8.shape)} max_seq_len={max_seq_len} -> "
                f"{'KF' if ok else 'orig'}{' (capture)' if _capturing() else ''}")
        if not ok:
            return orig(q_fp8, kv_cache_fp8, weights, seq_lens, block_tables, max_seq_len,
                        preshuffle=preshuffle, kv_block_size=kv_block_size)
        out = _guard("logits",
                     lambda: _kfi().aiter_paged_mqa_logits(q_fp8, kv_cache_fp8, weights, seq_lens, block_tables,
                                                           max_seq_len, preshuffle=preshuffle,
                                                           kv_block_size=kv_block_size),
                     lambda: orig(q_fp8, kv_cache_fp8, weights, seq_lens, block_tables, max_seq_len,
                                  preshuffle=preshuffle, kv_block_size=kv_block_size))
        if _CHECK and not checked and not _capturing():
            checked.append(1)
            try:
                import torch
                ref = orig(q_fp8, kv_cache_fp8, weights, seq_lens, block_tables, max_seq_len,
                           preshuffle=preshuffle, kv_block_size=kv_block_size)
                # compare only the valid [0, L) prefix of every row (the rest is never read by top-k)
                L = seq_lens.reshape(-1).long()
                col = torch.arange(ref.shape[1], device=ref.device)[None, :]
                m = col < L[:, None]
                d = (ref[m] - out[m]).abs()
                log(f"CHECK indexer_logits rows={rows}: valid={int(m.sum())} maxabs={d.max().item() if d.numel() else 0:.4g} "
                    f"exact={bool(torch.equal(ref[m], out[m]))}")
            except Exception as e:  # a failed comparison must never take the server down
                log(f"CHECK failed: {type(e).__name__}: {e}")
        return out

    mod.aiter_paged_mqa_logits = aiter_paged_mqa_logits
    log("patched dsa_indexer.aiter_paged_mqa_logits")


def _patch_indexer_topk(mod):
    cls = mod.DSATopKBackend
    orig = cls.topk_transform
    envs = mod.envs
    PAGED = mod.TopkTransformMethod.PAGED
    checked = []

    def topk_transform(self, logits, lengths, topk, topk_transform_method, attn_metadata,
                       cu_seqlens_q_topk=None, topk_indices_offset=None, row_starts=None,
                       batch_idx_list=None, force_unfused_topk=False):
        rows = logits.shape[0]
        ok = (not FORCE_ORIG and "topk" not in _disabled and envs.SGLANG_DSA_FUSE_TOPK.get() and not force_unfused_topk and self.is_sgl_kernel()
              and not self.should_use_topk_v2() and topk_transform_method == PAGED
              and row_starts is None and batch_idx_list is None and topk == 2048
              and rows <= INDEXER_MAX_ROWS and lengths.shape[0] == rows
              and logits.dtype.is_floating_point and logits.stride(1) == 1
              and getattr(attn_metadata, "page_table_1", None) is not None
              and attn_metadata.page_table_1.shape[0] == rows and _verify("indexer"))
        if _count(f"topk:{ok}:{rows}", 1):
            log(f"indexer_topk rows={rows} method={topk_transform_method} v2={self.should_use_topk_v2()} -> "
                f"{'KF' if ok else 'orig'}{' (capture)' if _capturing() else ''}")
        args = (logits, lengths, topk, topk_transform_method, attn_metadata, cu_seqlens_q_topk,
                topk_indices_offset, row_starts, batch_idx_list, force_unfused_topk)
        if not ok:
            return orig(self, *args)
        pt1 = attn_metadata.page_table_1
        out = _guard("topk", lambda: _kfi().topk_transform_fused(logits, lengths, pt1, cu_seqlens_q_topk, topk),
                     lambda: orig(self, *args))
        if _CHECK and not checked and not _capturing():
            checked.append(1)
            try:
                import torch
                ref = orig(self, *args)
                same = sum(int(torch.equal(torch.sort(ref[r]).values, torch.sort(out[r]).values)) for r in range(rows))
                log(f"CHECK indexer_topk rows={rows}: identical index sets in {same}/{rows} rows")
            except Exception as e:  # a failed comparison must never take the server down
                log(f"CHECK failed: {type(e).__name__}: {e}")
        return out

    cls.topk_transform = topk_transform
    log("patched DSATopKBackend.topk_transform")


# ----------------------------------------------------------------------------------------------------- MoE
MOE_M_LO, MOE_M_HI = 97, 455     # kf_moe validated M range (decode cfg key 97..512; HIP sort M*9 <= 4096)
_kf_moe = None


def _patch_moe(mod):
    cls = mod.AiterRunnerCore
    orig = cls.run
    QT = mod.AiterQuantType
    checked = []

    def run(self, runner_input, quant_info, running_state, hooks=None):
        h = runner_input.hidden_states
        M = h.shape[0]
        kw = dict(quant_info.fused_moe_kwargs or {})
        gate = kw.pop("gate_mode", None)
        try:
            from aiter.ops.flydsl.moe_common import GateMode
            gate_ok = gate in (None, GateMode.SEPARATED.value)
        except Exception:
            gate_ok = False
        ok = (not FORCE_ORIG and "moe" not in _disabled and MOE_M_LO <= M <= MOE_M_HI and h.shape[1] == 6144 and h.dtype.itemsize == 2
              and runner_input.quant_type == QT.PER_1X32 and quant_info.expert_mask is None
              and runner_input.a1_scale is None and quant_info.a13_scale is None and quant_info.a2_scale is None
              and quant_info.b13 is None and quant_info.b2 is None and not quant_info.doweight_stage1
              and quant_info.hidden_pad == 0 and quant_info.intermediate_pad == 0 and not kw and gate_ok
              and runner_input.num_local_tokens is None and runner_input.output_dtype in (None, h.dtype)
              and not self.config.no_combine and self.config.activation == "silu"
              and quant_info.swiglu_limit <= 0 and mod._MOE_SORTING_DISPATCH_POLICY == 2
              and quant_info.w13_weight.shape[0] == 257 and runner_input.topk_ids.shape[1] == 9
              # sglang 0.5.22 fields (absent on v0.5.21): MORI output arena, fp8-activation (A8W4) MoE
              and getattr(runner_input, "output", None) is None
              and not getattr(quant_info, "fp8_activations", False))
        if _count(f"moe:{ok}:{M}", 1):
            log(f"moe M={M} E={quant_info.w13_weight.shape[0]} topk={runner_input.topk_ids.shape[1]} "
                f"qt={runner_input.quant_type} mask={quant_info.expert_mask is not None} kw={sorted((quant_info.fused_moe_kwargs or {}).keys())} "
                f"act={self.config.activation} -> {'KF' if ok else 'orig'}{' (capture)' if _capturing() else ''}")
        if not ok:
            return orig(self, runner_input, quant_info, running_state, hooks)
        def kf():
            global _kf_moe
            if _kf_moe is None:
                _kf_moe = _load("kf_moe")
            return _kf_moe.glm52_moe_mxfp4(h, quant_info.w13_weight, quant_info.w2_weight, quant_info.w13_scale,
                                           quant_info.w2_scale, runner_input.topk_weights, runner_input.topk_ids)
        out = _guard("moe", kf, lambda: None)
        if out is None:
            return orig(self, runner_input, quant_info, running_state, hooks)
        if _CHECK and not checked and not _capturing():
            checked.append(1)
            try:
                ref = orig(self, runner_input, quant_info, running_state, hooks).hidden_states
                log(f"CHECK moe M={M}: snr(KF vs orig)={_snr(ref, out):.2f} dB")
            except Exception as e:  # a failed comparison must never take the server down
                log(f"CHECK failed: {type(e).__name__}: {e}")
        return mod.AiterRunnerOutput(hidden_states=out)

    cls.run = run
    log("patched AiterRunnerCore.run")



# ------------------------------------------------------------------------------------------------- DCP
# DCP4 decode (TP4 EP4 DP1 --dcp-size 4, sglang 93034eba91): KF candidates from tasks/dcp_sparse and tasks/dcp_indexer.
#   INFERA_KERNEL_DCP_SPARSE=1   triton_sparse_mla_decode.triton_sparse_mla_decode_splitk (DCP path: return_lse + lengths)
#   INFERA_KERNEL_DCP_INDEXER=1  Indexer._dcp_split_decode_topk: KF logits over the owned pages + KF local top-2048/pack,
#                           the real all_gather over the DCP group, KF merge + slot transform.
def _patch_dcp_sparse(mod):
    orig = mod.triton_sparse_mla_decode_splitk
    cand = _load("kf_dcp_sparse").triton_sparse_mla_decode_splitk
    checked = []

    def triton_sparse_mla_decode_splitk(q_nope, q_rope, kv, indices, sm_scale, d_v=512, kv_splits=None,
                                        workspace=None, return_lse=False, lengths=None):
        rows, heads = q_nope.shape[0], q_nope.shape[1]
        ok = (not FORCE_ORIG and "dcp_sparse" not in _disabled and return_lse and lengths is not None
              and heads == 64 and d_v == 512 and kv_splits is None and indices.shape[-1] == 2048)
        if _count(f"dcp_sparse:{ok}:{rows}", 1):
            log(f"dcp_sparse rows={rows} heads={heads} lse={return_lse} lengths={lengths is not None} -> "
                f"{'KF' if ok else 'orig'}{' (capture)' if _capturing() else ''}")
        args = dict(q_nope=q_nope, q_rope=q_rope, kv=kv, indices=indices, sm_scale=sm_scale, d_v=d_v,
                    kv_splits=kv_splits, workspace=workspace, return_lse=return_lse, lengths=lengths)
        if not ok:
            return orig(**args)
        # the KF launcher keeps its own grow-only split-K buffers (layout may differ from the backend's workspace)
        out = _guard("dcp_sparse", lambda: cand(**dict(args, workspace=None)), lambda: orig(**args))
        if _CHECK and not checked and not _capturing():
            checked.append(1)
            try:
                ro, rl = orig(**args)
                o, l = out
                log(f"CHECK dcp_sparse rows={rows}: snr(KF vs orig)={_snr(ro, o):.2f} dB "
                    f"lse maxabs={(rl - l).abs().max().item():.4g}")
            except Exception as e:  # a failed comparison must never take the server down
                log(f"CHECK failed: {type(e).__name__}: {e}")
        return out

    mod.triton_sparse_mla_decode_splitk = triton_sparse_mla_decode_splitk
    log("patched triton_sparse_mla_decode_splitk")


_kf_dcpi = None


def _patch_dcp_indexer(mod):
    cls = mod.Indexer
    orig = cls._dcp_split_decode_topk
    envs = mod.envs
    checked = []

    def _dcp_split_decode_topk(self, q_fp8, kv_cache_fp8, weights, lens, block_tables, page_size, metadata, layer_id):
        global _kf_dcpi
        rows = q_fp8.shape[0]
        group = mod.get_parallel().dcp_group
        ok = (not FORCE_ORIG and "dcp_indexer" not in _disabled and _verify("indexer") and page_size == 64
              and group.world_size == 4 and self.index_topk == 2048 and q_fp8.dim() == 3
              and envs.SGLANG_DSA_FUSE_TOPK.get() and not metadata.force_unfused_topk
              and envs.SGLANG_DSA_DCP_SPLIT_INDEXER_TRITON.get() and lens.dtype.itemsize == 4)
        if _count(f"dcp_indexer:{ok}:{rows}", 1):
            log(f"dcp_indexer rows={rows} kv={kv_cache_fp8.dtype}{tuple(kv_cache_fp8.shape)} -> "
                f"{'KF' if ok else 'orig'}{' (capture)' if _capturing() else ''}")
        a = (q_fp8, kv_cache_fp8, weights, lens, block_tables, page_size, metadata, layer_id)
        if not ok:
            return orig(self, *a)

        def kf():
            global _kf_dcpi
            if _kf_dcpi is None:
                _kf_dcpi = _load("kf_dcp_indexer")
            from sglang.srt.layers.attention.dsa.dsa_dcp_split_indexer import OWNED_INPUTS_CACHE, owned_inputs
            w, r = group.world_size, group.rank_in_group
            tabs, lns = owned_inputs(OWNED_INPUTS_CACHE, metadata=metadata.attn_metadata, layer_id=layer_id,
                                     block_tables=block_tables, lens=lens, page=page_size, w=w, r=r)
            lns = lns.to(torch.int32) if lns.dtype != torch.int32 else lns
            logits = _kf_dcpi._dcp_logits(q_fp8, kv_cache_fp8, weights, lns, tabs, tabs.shape[1] * page_size)
            pack = _kf_dcpi.local_topk_pack(logits, lns, 2048, page_size, w, r)
            g = group.all_gather(pack, dim=0)                       # [2 * w, rows, 2048], rank order
            return _kf_dcpi.merge2_candidates_triton(g[0:2].contiguous(), g[2:], w, 2048, block_tables, page_size)

        import torch
        out = _guard("dcp_indexer", kf, lambda: orig(self, *a))
        if _CHECK and not checked and not _capturing():
            checked.append(1)
            try:
                ref = orig(self, *a)
                same = sum(int(torch.equal(torch.sort(ref[i]).values, torch.sort(out[i]).values)) for i in range(rows))
                log(f"CHECK dcp_indexer rows={rows}: identical slot sets in {same}/{rows} rows")
            except Exception as e:  # a failed comparison must never take the server down
                log(f"CHECK failed: {type(e).__name__}: {e}")
        return out

    cls._dcp_split_decode_topk = _dcp_split_decode_topk
    log("patched Indexer._dcp_split_decode_topk")

# ------------------------------------------------------------------------------------------ import hook
_TARGETS = {}
if _ON["SPARSE"]:
    _TARGETS["sglang.kernels.ops.attention.dsa.triton_sparse_mla"] = [_patch_sparse]
    _TARGETS["sglang.srt.layers.attention.dsa_backend"] = [_patch_sparse_mode]
if _ON["INDEXER"]:
    _TARGETS["sglang.srt.layers.attention.dsa.dsa_indexer"] = [_patch_indexer_logits, _patch_indexer_mode]
    _TARGETS["sglang.srt.layers.attention.dsa.dsa_topk_backend"] = [_patch_indexer_topk]
if _ON["DCP_SPARSE"]:
    _TARGETS["sglang.kernels.ops.attention.dsa.triton_sparse_mla_decode"] = [_patch_dcp_sparse]
if _ON["DCP_INDEXER"]:
    _TARGETS.setdefault("sglang.srt.layers.attention.dsa.dsa_indexer", []).extend([_patch_dcp_indexer, _patch_indexer_mode]
        if not _ON["INDEXER"] else [_patch_dcp_indexer])
if _ON["MOE"]:
    _TARGETS["sglang.srt.layers.moe.moe_runner.aiter"] = [_patch_moe]


class _Loader(importlib.abc.Loader):
    def __init__(self, inner, fns):
        self.inner, self.fns = inner, fns

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        for fn in self.fns:
            try:
                fn(module)
            except Exception as e:   # never take the server down; the original stays in place
                log(f"PATCH FAILED {module.__name__}.{fn.__name__}: {type(e).__name__}: {e}")


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        fns = _TARGETS.get(name)
        if not fns:
            return None
        for f in sys.meta_path:
            if f is self:
                continue
            spec = getattr(f, "find_spec", lambda *a: None)(name, path, target)
            if spec is not None:
                spec.loader = _Loader(spec.loader, fns)
                return spec
        return None


def install():
    if _TARGETS and not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
        log(f"installed: {sorted(k for k, v in _ON.items() if v)}")
