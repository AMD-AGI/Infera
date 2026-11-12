###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Trace-driven frontend: any PyTorch model to the same ``Op`` list.

The model is compiled with a custom ``torch.compile`` backend on meta tensors,
so nothing is allocated and no GPU is needed. Dynamo hands the backend the FX
graph the model really runs; each node is then executed under a dispatch mode
that records the aten ops it lowers to, which is where the FLOPs come from --
PyTorch's own ``flop_registry`` formulas, not a reimplementation.

Two node types are not left to the dispatcher. Scaled-dot-product attention
decomposes there into two batched GEMMs and a softmax over a materialised
score matrix, which is not what a flash kernel does; it is priced as one fused
attention op from its operand shapes, halved when the call is causal. And
view-only nodes (reshape, transpose, unbind) move no bytes and are dropped.

Every matrix multiply and every convolution a node dispatches is a GEMM op
carrying the ``(M, N, K, batch)`` problem it runs as, so the GEMM model --
Origami when attached -- prices a convolution exactly as it prices a linear
layer. A convolution is lowered to its implicit GEMM (``conv_as_gemm``), the
form MIOpen's implicit-GEMM solvers run; its bytes are its real tensors, not
an im2col buffer that is never written. Any other node is one memory-bound
pass over its inputs and outputs.

The trace is one rank of a ``tp = 1`` model. Under tensor parallelism the
GEMMs and attention shard ``tp`` ways and the norms and residuals do not,
which is how ``scale_for_tp`` divides them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from .ops import ATTENTION, GEMM, MEMORY, Op

_GEMM_ATEN = {"mm", "addmm", "bmm", "baddbmm", "_scaled_mm"}
_CONV_ATEN = {"convolution", "_convolution", "cudnn_convolution", "miopen_convolution"}
# Filters or taps per group below which a convolution is not run as a GEMM.
_MIN_TILE = 16
_VIEW_ATEN = {
    "t", "view", "_unsafe_view", "reshape", "transpose", "permute", "unbind", "expand",
    "slice", "select", "unsqueeze", "squeeze", "alias", "detach", "split", "split_with_sizes",
    "chunk", "as_strided", "lift_fresh", "_reshape_alias", "unflatten", "flatten", "sym_size",
    "sym_numel", "sym_stride", "dim", "size", "stride", "is_contiguous", "lift_fresh_copy",
}  # fmt: skip


def _dtype_name(dtype: Any) -> str:
    import torch

    return {
        torch.bfloat16: "bf16",
        torch.float16: "fp16",
        torch.float32: "fp32",
        getattr(torch, "float8_e4m3fn", None): "fp8",
        getattr(torch, "float8_e4m3fnuz", None): "fp8",
        getattr(torch, "float8_e5m2", None): "fp8",
        getattr(torch, "float8_e5m2fnuz", None): "fp8",
    }.get(dtype, "bf16")


def _tensors(obj: Any) -> list[Any]:
    import torch
    from torch.utils._pytree import tree_flatten

    return [t for t in tree_flatten(obj)[0] if isinstance(t, torch.Tensor)]


def _nbytes(obj: Any) -> float:
    return float(sum(t.numel() * t.element_size() for t in _tensors(obj)))


@dataclass
class TraceResult:
    ops: list[Op] = field(default_factory=list)
    # Activation tensors entering the graph, so the caller can tell how many
    # tokens the trace was taken at.
    input_shapes: list[tuple[int, ...]] = field(default_factory=list)
    fx_nodes: int = 0

    def flops(self, kind: str | None = None) -> float:
        return sum(o.flops * o.count for o in self.ops if kind is None or o.kind == kind)


def _prod(xs: Any) -> int:
    p = 1
    for x in xs:
        p *= int(x)
    return p


def conv_as_gemm(
    x_shape: tuple[int, ...],
    w_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    groups: int = 1,
    transposed: bool = False,
) -> tuple[int, int, int, int]:
    """The implicit GEMM a convolution runs as, ``(M, N, K, batch)``.

    A forward convolution is, per group, ``M = N x prod(out spatial)`` output
    pixels, ``N = C_out / groups`` filters and ``K = C_in / groups x prod(kernel)``
    taps; the groups are a batch of identical GEMMs. A transposed convolution
    is the scatter form: every input pixel times every filter tap, ``M = N x
    prod(in spatial)``, ``N = C_out / groups x prod(kernel)``, ``K = C_in /
    groups``. Either way ``2 x M x N x K x batch`` is PyTorch's FLOP count.
    """
    g = max(1, int(groups))
    kernel = _prod(w_shape[2:])
    if transposed:
        return (x_shape[0] * _prod(x_shape[2:]), w_shape[1] * kernel, w_shape[0] // g, g)
    return (out_shape[0] * _prod(out_shape[2:]), w_shape[0] // g, w_shape[1] * kernel, g)


def _lowered_op(name: str, args: tuple, out: Any, flops: float) -> Op:
    """One matrix-multiply or convolution call, with the GEMM it runs as."""
    if name in _CONV_ATEN:
        x, w = args[0], args[1]
        transposed = bool(args[6]) if name == "convolution" and len(args) > 6 else False
        if transposed:
            groups = int(args[8]) if len(args) > 8 else 1
        else:
            # Every convolution entry point agrees on the shapes, not on where
            # ``groups`` sits among the arguments.
            groups = max(1, int(x.shape[1]) // max(1, int(w.shape[1])))
        m, n, k, b = conv_as_gemm(
            tuple(x.shape), tuple(w.shape), tuple(out.shape), groups, transposed
        )
        if not flops:
            flops = 2.0 * m * n * k * b
        # A depthwise or near-depthwise convolution has too few filters or taps
        # per group to fill a matrix tile; it runs as a direct kernel, so it is
        # left on the roofline over the bytes it really moves.
        gemm = (m, n, k, b) if min(n, k) >= _MIN_TILE else None
        operands = [x, w] + ([args[2]] if len(args) > 2 and args[2] is not None else [])
    else:
        if name == "_scaled_mm":
            a, w = args[0], args[1]
        else:
            mats = [t for t in _tensors(args) if t.dim() >= 2]
            a, w = mats[-2], mats[-1]
        b = _prod(a.shape[:-2]) if name in ("bmm", "baddbmm") else 1
        m, k, n = int(a.shape[-2]), int(a.shape[-1]), int(w.shape[-1])
        if a.dim() > 2 and name not in ("bmm", "baddbmm"):
            m *= _prod(a.shape[:-2])
        gemm = (m, n, k, b)
        operands = _tensors(args)
    dtype = _dtype_name(args[0].dtype) if hasattr(args[0], "dtype") else "bf16"
    return Op(name, GEMM, flops, _nbytes(operands), _nbytes(out), dtype, gemm=gemm)


def _attention_op(name: str, args: tuple, kwargs: dict, out: Any) -> Op:
    q, k, v = args[0], args[1], args[2]
    causal = bool(kwargs.get("is_causal", args[5] if len(args) > 5 else False))
    *batch, sq, d = q.shape
    skv, dv = k.shape[-2], v.shape[-1]
    bh = 1
    for b in batch:
        bh *= int(b)
    flops = 2.0 * bh * sq * skv * (d + dv)
    if causal:
        flops /= 2.0
    return Op(
        name,
        ATTENTION,
        flops=flops,
        bytes_read=_nbytes((q, k, v)),
        bytes_written=_nbytes(out),
        dtype=_dtype_name(q.dtype),
    )


def trace_model(model: Any, *example_inputs: Any, **example_kwargs: Any) -> TraceResult:
    """Ops of one forward of ``model`` at the shapes of the example inputs.

    Pass meta tensors (``torch.device("meta")`` for the model and inputs) to
    trace a full-size model without memory.
    """
    import torch
    import torch.fx
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils.flop_counter import flop_registry

    result = TraceResult(input_shapes=[tuple(t.shape) for t in _tensors(example_inputs)])
    sdpa = torch.nn.functional.scaled_dot_product_attention

    class _Recorder(TorchDispatchMode):
        def __init__(self) -> None:
            super().__init__()
            self.names: list[str] = []
            self.gemms: list[Op] = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            out = func(*args, **kwargs)
            packet = func._overloadpacket
            name = packet.__name__
            self.names.append(name)
            if name in _GEMM_ATEN or name in _CONV_ATEN:
                flops = 0.0
                if packet in flop_registry:
                    flops = float(flop_registry[packet](*args, **kwargs, out_val=out))
                self.gemms.append(_lowered_op(name, args, out, flops))
            return out

    class _Interp(torch.fx.Interpreter):
        def run_node(self, n: torch.fx.Node) -> Any:
            if n.op not in ("call_function", "call_method", "call_module"):
                return super().run_node(n)
            if n.op == "call_function" and n.target is sdpa:
                args, kwargs = self.fetch_args_kwargs_from_env(n)
                out = super().run_node(n)
                result.ops.append(_attention_op(n.name, args, kwargs, out))
                return out
            args, kwargs = self.fetch_args_kwargs_from_env(n)
            rec = _Recorder()
            with rec:
                out = super().run_node(n)
            if rec.gemms:
                for i, op in enumerate(rec.gemms):
                    op.name = n.name if i == 0 else f"{n.name}.{i}"
                    result.ops.append(op)
                return out
            if not [x for x in rec.names if x not in _VIEW_ATEN]:
                return out
            read = _nbytes(_tensors((args, kwargs)))
            result.ops.append(Op(n.name, MEMORY, 0.0, read, _nbytes(out)))
            return out

    def backend(gm: torch.fx.GraphModule, inputs: list[Any]):
        result.fx_nodes = len(gm.graph.nodes)
        with torch.no_grad():
            _Interp(gm).run(*inputs)
        return gm.forward

    torch._dynamo.reset()
    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    with torch.no_grad():
        compiled(*example_inputs, **example_kwargs)
    torch._dynamo.reset()
    return result


def scale_for_tp(ops: list[Op], tp: int) -> list[Op]:
    """One TP rank's share of a ``tp = 1`` trace."""
    if tp <= 1:
        return list(ops)
    out = []
    for op in ops:
        if op.kind in (GEMM, ATTENTION):
            out.append(
                replace(
                    op,
                    flops=op.flops / tp,
                    bytes_read=op.bytes_read / tp,
                    bytes_written=op.bytes_written / tp,
                )
            )
        else:
            out.append(op)
    return out
