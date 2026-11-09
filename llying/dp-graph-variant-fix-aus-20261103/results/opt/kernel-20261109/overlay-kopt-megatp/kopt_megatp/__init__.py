"""KOPT_MEGATP=1: run the TP8 decode MoE layers of GLM-5.2 (DeepseekV2MoE under attention DP, TP MoE) as aiter's
MegaMoeTP: one persistent kernel for the all-gather of this rank's tokens, the MoE (prologue, GEMM1, act, GEMM2) and
the reduce-scatter back to this rank's tokens (KERNEL-PROGRESS-20261109 section 14; aiter op test: ~204 us per layer
at 704 global tokens vs ~250 us for the production all-gather + fused_moe + reduce-scatter).

Only inside CUDA-graph capture (every rank replays the same graph with the same padded local token count, which is
MegaMoeTP's collective contract), for decode / target-verify batches of the target model's MoE layers; everything else
runs the original path. For such a layer:
  * the FFN input boundary keeps its residual update + norm but its dp_gather returns this rank's rows;
  * the router and top-k (with the fused shared expert, routed scaling folded into the weights) run on those rows;
  * MegaMoeTP (bf16 comm by default) returns this rank's rows of the reduced output;
  * the FFN exit records that output as complete (no reduce-scatter, no deferred sum).
One MegaMoeTP instance serves all layers (set_weights before each call).

Env: KOPT_MEGATP_COMM=bf16|fp8, KOPT_MEGATP_MAX_LOCAL (default 128), KOPT_MEGATP_LOG=1 (one line per new shape).
"""
import importlib.abc
import os
import sys

_COMM = os.environ.get("KOPT_MEGATP_COMM", "bf16")
_MAX_LOCAL = int(os.environ.get("KOPT_MEGATP_MAX_LOCAL", "128"))
_LOG = os.environ.get("KOPT_MEGATP_LOG", "1") == "1"


class _State:
    armed = False       # inside an eligible layer's forward
    gather_off = False  # inside that layer's FFN input boundary
    mega = None
    prepared = set()
    seen = set()


def log(msg):
    if _LOG:
        print(f"[kopt-megatp] {msg}", file=sys.stderr, flush=True)


def _eligible(layer, forward_batch):
    from sglang.srt.model_executor.runner import get_is_capture_mode
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

    mlp = getattr(layer, "mlp", None)
    if not isinstance(mlp, DeepseekV2MoE) or getattr(mlp, "is_nextn", False) or not get_is_capture_mode():
        return False
    mode = forward_batch.forward_mode
    if not (mode.is_decode() or mode.is_target_verify()):
        return False
    m = forward_batch.input_ids.shape[0]
    return 0 < m <= _MAX_LOCAL


def _tp_group():
    try:
        from sglang.srt.runtime_context import get_parallel

        return get_parallel().tp_group.device_group
    except Exception:
        try:
            from sglang.srt.distributed import get_tp_group

            return get_tp_group().device_group
        except Exception:
            return None


def _get_mega(moe, m, topk):
    import torch
    import torch.distributed as dist
    from aiter import ActivationType
    from aiter.ops.flydsl.mega_moe_tp import MegaMoeTP, MegaMoeTPConfig

    e = moe.experts
    w1 = e.w13_weight.view(torch.float4_e2m1fn_x2)
    w2 = e.w2_weight.view(torch.float4_e2m1fn_x2)
    if _State.mega is None:
        group = _tp_group()
        rank, world = dist.get_rank(group), dist.get_world_size(group)
        E, two_inter, _ = w1.shape
        cfg = MegaMoeTPConfig(rank=rank, world_size=world, model_dim=w1.shape[2] * 2, inter_dim=two_inter // 2,
                              experts=E, topk=topk, max_local_tokens=_MAX_LOCAL, activation=ActivationType.Silu,
                              comm_mode="ag_rs", comm_dtype=_COMM)
        _State.mega = MegaMoeTP(cfg, w1=w1, w1_scale=e.w13_weight_scale, w2=w2, w2_scale=e.w2_weight_scale,
                                group=group)
        log(f"MegaMoeTP rank {rank}/{world} H {cfg.model_dim} inter {cfg.inter_dim} E {E} topk {topk} "
            f"max_local {_MAX_LOCAL} comm {_COMM} w13 {tuple(w1.shape)} s13 {tuple(e.w13_weight_scale.shape)} "
            f"w2 {tuple(w2.shape)} s2 {tuple(e.w2_weight_scale.shape)}")
    else:
        _State.mega.set_weights(w1=w1, w1_scale=e.w13_weight_scale, w2=w2, w2_scale=e.w2_weight_scale)
    if m not in _State.prepared and not torch.cuda.is_current_stream_capturing():
        _State.mega.prepare([m])
        _State.prepared.add(m)
    return _State.mega


def _mega_moe_forward(moe, hidden_states, forward_batch, mod):
    import torch

    m = forward_batch.input_ids.shape[0]
    if hidden_states.shape[0] != m:
        raise RuntimeError(f"kopt-megatp: FFN input has {hidden_states.shape[0]} rows, expected the {m} local rows "
                           "(the dp_gather bypass did not take effect)")
    router_logits, partials = mod._hip_moe.forward_gate(moe, hidden_states, None, fused_gate=True)
    kw = {"router_logits_partials": partials} if partials is not None else {}
    topk = moe.topk(hidden_states, router_logits, num_token_non_padded=None, expert_location_dispatch_info=None, **kw)
    tw, ids = topk.topk_weights, topk.topk_ids
    mega = _get_mega(moe, m, ids.shape[1])
    if m not in _State.seen:
        _State.seen.add(m)
        log(f"layer {moe.layer_id} m {m} x {tuple(hidden_states.shape)} {hidden_states.dtype} topk {tuple(ids.shape)} "
            f"{ids.dtype} w {tw.dtype} capturing {torch.cuda.is_current_stream_capturing()}")
    out = torch.empty_like(hidden_states)
    return mega(hidden_states, tw.float(), ids.to(torch.int32), out=out)


class _MegaExit:
    """FFN exit for an output that is already this rank's reduced rows: the write-back into the residual the
    production exit runs after its reduce-scatter, then the record (exit.py FfnExit.finish / ExitPolicy._complete_now)."""

    def __init__(self, boundary, forward_batch, update_context=None):
        from sglang.srt.layers.layer_boundary.residual import bind_update
        from sglang.srt.layers.layer_boundary.residual.batch import stream_of

        self.policy = boundary.plan.output
        self.steps = self.policy.plan.path_for(forward_batch)
        self.update = bind_update(self.steps.output.update, forward_batch, update_context)
        self.stream = stream_of(forward_batch)
        self.defer_moe_finalize = False

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return None

    def finish(self, hidden_states):
        residual = self.stream.residual
        if residual is not None and (self.update.applied_at_exit or self.steps.writes_at_handoff):
            hidden_states = self.update.update(hidden_states, residual)
            residual = None
        return self.policy._record_output(hidden_states, residual, self.stream, self.update, None)


def _patch_deepseek(mod):
    layer_cls, moe_cls = mod.DeepseekV2DecoderLayer, mod.DeepseekV2MoE
    orig_layer, orig_moe = layer_cls.forward, moe_cls.forward

    def layer_forward(self, positions, hidden_states, forward_batch, *args, **kwargs):
        if not _eligible(self, forward_batch):
            return orig_layer(self, positions, hidden_states, forward_batch, *args, **kwargs)
        _State.armed = True
        try:
            return orig_layer(self, positions, hidden_states, forward_batch, *args, **kwargs)
        finally:
            _State.armed = False

    def moe_forward(self, hidden_states, forward_batch=None, *args, **kwargs):
        if not _State.armed:
            return orig_moe(self, hidden_states, forward_batch, *args, **kwargs)
        return _mega_moe_forward(self, hidden_states, forward_batch, mod)

    layer_cls.forward, moe_cls.forward = layer_forward, moe_forward
    log("patched DeepseekV2DecoderLayer.forward, DeepseekV2MoE.forward")


def _patch_prepare(mod):
    orig = mod.dp_gather

    def dp_gather(hidden_states, *args, **kwargs):
        if _State.gather_off:
            return hidden_states
        return orig(hidden_states, *args, **kwargs)

    mod.dp_gather = dp_gather
    log("patched layer_boundary.prepare.dp_gather")


def _patch_stage(mod):
    cls = mod.StageBoundary
    orig_exit, orig_prepare = cls.exit, cls.prepare
    ffn = mod.StageKind.FFN

    def exit(self, forward_batch, *, update_context=None):
        if _State.armed and self.kind is ffn:
            return _MegaExit(self, forward_batch, update_context)
        return orig_exit(self, forward_batch, update_context=update_context)

    def prepare(self, hidden_states, forward_batch, *, cache=None, **call):
        if not (_State.armed and self.kind is ffn):
            return orig_prepare(self, hidden_states, forward_batch, cache=cache, **call)
        _State.gather_off = True
        try:
            return orig_prepare(self, hidden_states, forward_batch, cache=cache, **call)
        finally:
            _State.gather_off = False

    cls.exit, cls.prepare = exit, prepare
    log("patched StageBoundary.exit / prepare")


_TARGETS = {
    "sglang.srt.models.deepseek_v2": [_patch_deepseek],
    "sglang.srt.layers.layer_boundary.prepare": [_patch_prepare],
    "sglang.srt.layers.layer_boundary.stage": [_patch_stage],
}


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
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
        log(f"installed (comm {_COMM}, max_local {_MAX_LOCAL})")
