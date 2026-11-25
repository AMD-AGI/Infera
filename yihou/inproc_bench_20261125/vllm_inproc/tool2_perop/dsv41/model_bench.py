###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################
# Provenance: adapted from AKLR `aklr/benchmarks/models/{glm52,deepseekv4}/model_bench.py` @ 33689332
# (research/aklr_src/) and the SGLang DSv4.1 port (sglang_inproc/tool2_perop/dsv41/model_bench.py), for vLLM
# 0.34.0rc1.dev897+g9c1d45c21 (image infera-vllm-dsv41:0007cd8087f1-nightly-9c1d45c2), model files
# vllm/models/deepseek_v41/amd/{model,rocm}.py + deepseek_v41/attention.py + deepseek_v4/amd/model.py.
# Batches are built exactly the way vLLM's own `vllm bench forward` builds them (vllm/benchmarks/forward/
# {shapes,batch,kv_init,capture}.py, shipped in the image): a SchedulerOutput without the scheduler, run
# through the real GPUModelRunner.execute_model (input prep, attention metadata, DP sync); the bench takes
# over the eager model call (model.forward) and does all of tool 2's work inside it, under the
# ForwardContext the runner built. PARITY_tool2_vllm.md lists every adaptation and why.

"""DeepSeek-V4.1 (vLLM) parts of tool 2: engine setup, batches, layer kinds, op tree, recording and replay
(module hooks + call-site wrappers), cold-weight rotation, memory."""


import contextlib
import importlib
import os
import socket
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from vllm_bench import (  # noqa: E402
    GRAPH_REPS,
    BalancedRouting,
    _partitions,
    agreed,
    ctx_now,
    ctx_scope,
    snapshot,
    table,
    world_rank,
)

SUPPORTED_ARCHS = ("DeepseekV41ForCausalLM", "DeepseekV41LLMForCausalLM")
MODEL_TAG_DEFAULT = "dsv41"


def model_tag(model_path: str) -> str:
    name = Path(str(model_path)).name.lower()
    return "dsv41_flash" if "flash" in name else ("dsv41_pro" if "pro" in name else MODEL_TAG_DEFAULT)


# ----------------------------------------------------------------------------------------------
# CLI and engine.


def make_parser():
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser(description="tool 2 (per-op micro-bench), vLLM DSv4.1")
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(enable_prefix_caching=False)
    parser.add_argument("--ISL", type=int, default=4096, help="context per request before the step (decode) / prompt (prefill)")
    parser.add_argument("--reps", type=int, default=50, help="graph replays per op")
    parser.add_argument("--repeat", type=int, default=1, help="timing passes per op, fastest one wins")
    parser.add_argument("--full-step", type=int, default=0, metavar="N",
                        help="time N(+2) whole steps under vLLM's own cuda graphs (the preset's compilation config) and exit")
    parser.add_argument("--out", help="markdown report path (default: auto-named in --result-dir)")
    parser.add_argument("--result-dir", default=None, help="directory for report / trace / export (default: cwd)")
    parser.add_argument("--kv-fill", choices=["random", "zeros"], default="random",
                        help="random: the case's KV blocks filled with bytes < 0x7C (AKLR); zeros: vLLM bench forward's init")
    parser.add_argument("--routing", choices=["balanced", "native"], default="balanced",
                        help="balanced: AKLR seeded draw (seed 1234) at the router; native: keep the router (diagnostic)")
    parser.add_argument("--isl-file", default=None,
                        help="per-request context list for each DP rank (one int per line); overrides --ISL per request")
    parser.add_argument("--kv-len-jitter", type=float, default=0.0,
                        help="vLLM bench forward's per-request context jitter (+-fraction); 0 = every request at --ISL (AKLR)")
    return parser


def setup_env() -> None:
    """vLLM bench forward's environment: in-process engine core, torchrun world or a 1-rank fake one."""
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    if "RANK" in os.environ:
        return
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    os.environ.update(RANK="0", LOCAL_RANK="0", WORLD_SIZE="1", MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))


DUMMY_WARNING = (
    "--load-format dummy. Operators whose cost depends on values are mistimed (AKLR GLM-5.2: the DSA indexer"
    " read 24.1% of the step at ISL 100k on dummy weights vs 9.5% on real ones). Do not act on these numbers."
)


def dummy_banner(load_format) -> str:
    return f"\n> **WARNING: {DUMMY_WARNING}**\n" if str(load_format) == "dummy" else ""


def build_engine_args(args, n_reqs: int, n_tokens: int, cuda_graph: bool = False):
    """EngineArgs for the bench. Forced (AKLR forces --disable-cuda-graph for the same reason): compilation
    mode NONE and cudagraph NONE unless `cuda_graph`, so module hooks fire and the runner's eager branch calls
    `model(**inputs)`; external_launcher (torchrun, one process per GPU); no prefix caching; `auto` KV -> fp8."""
    from vllm.config.compilation import CompilationMode, CUDAGraphMode
    from vllm.engine.arg_utils import EngineArgs

    if str(args.load_format) == "dummy":
        print(f"[BENCH] WARNING: {DUMMY_WARNING}", flush=True)
    ea = EngineArgs.from_cli_args(args)
    ea.distributed_executor_backend = "external_launcher"
    ea.enable_prefix_caching = False
    if not cuda_graph:
        ea.compilation_config.mode = CompilationMode.NONE
        ea.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    if ea.kv_cache_dtype == "auto":
        ea.kv_cache_dtype = "fp8_e4m3"
    ea.max_num_seqs = max(ea.max_num_seqs or 0, n_reqs)
    ea.max_num_batched_tokens = max(ea.max_num_batched_tokens or 0, n_tokens)
    return ea


def setup(engine_args, tag: str = "BENCH"):
    """Build the engine in this process; returns (llm, runner, rank_print)."""
    from vllm import LLM

    llm = LLM.from_engine_args(engine_args)
    runner = llm.llm_engine.model_executor.driver_worker.worker.model_runner
    validate_model(runner)
    if world_rank() != 0:
        return llm, runner, lambda *a, **k: None

    def rank_print(msg=""):
        print("\n".join(f"[{tag}] {ln}" for ln in str(msg).split("\n")), flush=True)

    return llm, runner, rank_print


def validate_model(runner) -> None:
    name = type(runner.model).__name__
    if name not in SUPPORTED_ARCHS:
        raise SystemExit(f"tool2 dsv41 supports {SUPPORTED_ARCHS}, got {name}")


def lm(runner):
    """The text model (DeepseekV41LLMForCausalLM) under the VL wrapper."""
    return getattr(runner.model, "language_model", runner.model)


def parallel(runner) -> dict:
    pc = runner.parallel_config
    return {"tp": pc.tensor_parallel_size, "dp": pc.data_parallel_size, "ep": bool(pc.enable_expert_parallel),
            "dp_rank": getattr(runner, "dp_rank", 0)}


def moe_spans_ranks(runner) -> bool:
    """MoE collectives across ranks (vLLM without EP still TP-shards experts over DP x TP)."""
    p = parallel(runner)
    return p["tp"] * p["dp"] > 1


def parallel_line(runner) -> str:
    p = parallel(runner)
    moe_tp = p["tp"] * p["dp"]
    moe = (f"EP {moe_tp}" if p["ep"] else f"experts TP-sharded {moe_tp} ways over DP x TP (naive DP dispatch/combine)")
    return f"TP {p['tp']}, DP {p['dp']} (attention per DP rank), MoE {moe}"


def parallel_tags(runner) -> list[str]:
    p = parallel(runner)
    return [f"tp{p['tp']}"] + ([f"dp{p['dp']}"] if p["dp"] > 1 else []) + (["ep"] if p["ep"] else [])


def spec_config(runner):
    return runner.vllm_config.speculative_config


def num_spec(runner) -> int:
    return int(getattr(runner, "num_speculative_steps", 0) or 0) if spec_config(runner) is not None else 0


def accept_len(runner):
    """Synthetic acceptance length the server runs under (rejection_sample_method=synthetic), else None."""
    sc = spec_config(runner)
    if sc is None:
        return None
    if str(getattr(sc, "rejection_sample_method", "")) != "synthetic":
        return None
    v = getattr(sc, "synthetic_acceptance_length", None)
    if v:
        return float(v)
    # vLLM resolves the length into per-position unconditional rates and clears it (config/speculative.py):
    # expected tokens per verify step = 1 (bonus) + sum of the rates.
    rates = getattr(sc, "synthetic_acceptance_rates", None)
    return 1.0 + float(sum(rates)) if rates else None


def spec_note(runner) -> str:
    sc = spec_config(runner)
    return (f"{sc.method}, {num_spec(runner)} draft tokens; verify rows of 1+{num_spec(runner)} query tokens."
            " The per-op rows are the target verify forward; the drafter is one `draft model` row.")


def weights_line(runner) -> str:
    qc = getattr(runner.vllm_config, "quant_config", None)
    return f"`{runner.load_config.load_format}` load, quant `{type(qc).__name__ if qc else 'none'}`"


# ----------------------------------------------------------------------------------------------
# Batches (vLLM bench forward's builder) and the forward takeover.


def sampling_params(runner):
    from vllm.sampling_params import SamplingParams

    d = runner.model_config.get_diff_sampling_param()
    return SamplingParams.from_optional(**d) if d else SamplingParams()


def read_isl_file(path: str) -> list[int]:
    """Per-request contexts, one integer per line ('#' comments), e.g. vllm-tool3's isl_shapeA_per_dp_rank.txt."""
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(int(line))
    if not out:
        raise ValueError(f"{path}: no contexts")
    return out


def decode_case(name: str, n_reqs: int, isl: int, drafts: int, lens: list[int] | None = None):
    """`n_reqs` verify (or decode) rows: 1+drafts query tokens on `isl` context, or on lens[i % len(lens)]
    (the same per-rank list on every DP rank) when a per-request context list is given."""
    from vllm.benchmarks.forward.shapes import BenchCase, BenchRequest

    q = 1 + drafts
    ctx = [lens[i % len(lens)] for i in range(n_reqs)] if lens else [isl] * n_reqs
    return BenchCase(name, tuple(BenchRequest(q, c + q, drafts) for c in ctx))


def prefill_case(name: str, n_reqs: int, prefix: int, extend: int, chunked: bool):
    from vllm.benchmarks.forward.shapes import BenchCase, BenchRequest

    return BenchCase(name, tuple([BenchRequest(extend, prefix + extend, 0, chunked=chunked and prefix > 0)] * n_reqs))


def lookahead(runner, drafter: bool) -> int:
    return num_spec(runner) + 1 if drafter and spec_config(runner) is not None else 0


class Capacity(Exception):
    """The case does not fit; reported as [BLOCKED_CAPACITY], never reduced."""


def build(runner, case, seed: int, jitter: float = 0.0, drafter: bool = True):
    """(BenchBatch, case) with disjoint KV blocks; Capacity when blocks / model length do not fit."""
    import numpy as np
    from vllm.benchmarks.forward.batch import build_batch
    from vllm.benchmarks.forward.shapes import jitter_kv_lens

    if jitter:
        case = jitter_kv_lens(case, jitter, runner.model_config.max_model_len, seed)
    la = lookahead(runner, drafter)
    longest = max(r.kv_len for r in case.requests) + la
    if longest > runner.model_config.max_model_len:
        raise Capacity(f"context {longest:,} (incl. draft lookahead {la}) > --max-model-len {runner.model_config.max_model_len:,}")
    kv = runner.kv_cache_config
    try:
        batch = build_batch(case, kv.kv_cache_groups, kv.num_blocks, runner.model_config.get_vocab_size(),
                            np.random.default_rng([seed, runner.dp_rank]),
                            runner.parallel_config.decode_context_parallel_size, la, sampling_params(runner))
    except ValueError as e:
        raise Capacity(str(e)) from e
    return batch, case


def kv_views(runner, batch):
    """Every KV tensor slice of the case's blocks: [(tensor_view, element_size)]."""
    from vllm.benchmarks.forward.kv_init import _block_dim, _group_tensors

    context = runner.vllm_config.compilation_config.static_forward_context
    caches = {name: getattr(layer, "kv_cache", None) for name, layer in context.items()}
    n = runner.kv_cache_config.num_blocks
    out = []
    for groups in runner.attn_groups:
        for group in groups:
            ranges = batch.group_block_ranges[group.kv_cache_group_id]
            for t in _group_tensors(caches, group, n):
                dim, factor = _block_dim(t, n)
                for begin, end in ranges:
                    out.append(t.narrow(dim, begin * factor, (end - begin) * factor))
    return out


@torch.inference_mode()
def fill_kv(runner, batch, how: str) -> dict:
    """AKLR `randomize_kv` on the case's blocks only (no prefill forward): 1-byte buffers get bytes < 0x7C (no
    fp8 Inf/NaN; the MXFP4 indexer payload and UE8M0 scales included), wider ones uniform(-1, 1); `zeros` is
    vLLM bench forward's init. Returns bytes filled per dtype."""
    filled: dict[str, int] = {}
    seen = set()
    for v in kv_views(runner, batch):
        key = (v.data_ptr(), v.numel())
        if key in seen or not v.numel():
            continue
        seen.add(key)
        if how == "zeros":
            v.zero_()
        elif v.element_size() == 1:
            v.view(torch.uint8).random_(0, 0x7C)
        elif v.is_floating_point():
            v.uniform_(-1, 1)
        else:
            v.zero_()
        k = str(v.dtype).replace("torch.", "")
        filled[k] = filled.get(k, 0) + v.numel() * v.element_size()
    return filled


@dataclass
class Forward:
    """The runner's eager model call, taken over: its inputs, the ForwardContext and InputBatch it built."""

    inputs: dict
    ctx: Any
    input_batch: Any
    forward: Callable  # the model's own forward (unpatched)
    runner: Any

    def run_model(self):
        with ctx_scope(self.ctx):
            return self.forward(**self.inputs)

    def logits_rows(self, hidden):
        h = hidden[0] if isinstance(hidden, tuple) else hidden
        return h[self.input_batch.logits_indices]

    def step(self):
        """Target forward + logits: the per-step target work (vLLM computes logits outside model.forward,
        in sample_tokens; AKLR's SGLang forward includes the LogitsProcessor, so both are the reference)."""
        out = self.run_model()
        with ctx_scope(self.ctx):
            self.runner.model.compute_logits(self.logits_rows(out))
        return out


def run_inside_forward(runner, batch, body: Callable[[Forward], Any]):
    """Run `batch` through runner.execute_model; `body(Forward)` runs in place of the eager model call (all
    ranks reach it in lockstep after the runner's DP sync). Removes the batch's requests afterwards."""
    from vllm.benchmarks.forward.capture import write_input_ids

    model = runner.model
    state: dict = {}
    prepare_inputs = runner.prepare_inputs
    own_forward = model.forward

    def prep(*a, **k):
        state["ib"] = prepare_inputs(*a, **k)
        return state["ib"]

    def fwd(**inputs):
        write_input_ids(batch, state["ib"], inputs.get("input_ids"))
        f = Forward(inputs, ctx_now(), state["ib"], own_forward, runner)
        state["result"] = body(f)
        return f.run_model()

    runner.prepare_inputs = prep
    model.forward = fwd
    try:
        runner.execute_model(batch.scheduler_output)
    finally:
        del runner.prepare_inputs, model.forward
        runner.execute_model_state = None
        runner.execute_model(batch.finish())
    return state.get("result")


# ----------------------------------------------------------------------------------------------
# Layer kinds.


def layer_kind(layer, i: int, late_start: int | None) -> str:
    """DSv4.1 layers differ by compression ratio, routing (hash vs gate), owning a compressor (kv source) /
    indexer, carrying an Engram, and sitting past the decoder SWA bounded-replay cut (prefill runs those on
    the tail rows only). Same naming as the SGLang port, so the two reports line up."""
    attn, ffn = layer.attn, layer.ffn
    kind = "moe" if hasattr(ffn, "experts") else "dense"
    if getattr(getattr(ffn, "gate", None), "tid2eid", None) is not None:
        kind += "+hash"
    kind += f"+c{int(getattr(attn, 'compress_ratio', 0))}"
    if getattr(attn, "compressor", None) is not None:
        kind += "+kvsrc"
    if getattr(attn, "indexer", None) is not None:
        kind += "+indexer"
    if getattr(layer, "engram", None) is not None:
        kind += "+engram"
    if late_start is not None and i >= late_start:
        kind += "+late"
    return kind


def _ids(ids: list[int]) -> str:
    out, run = [], [ids[0]]
    for x in ids[1:]:
        if x == run[-1] + 1:
            run.append(x)
        else:
            out.append(run)
            run = [x]
    out.append(run)
    return ",".join(f"{r[0]}" if len(r) == 1 else f"{r[0]}-{r[-1]}" for r in out)


class Layers:
    def __init__(self, runner):
        model = lm(runner).model
        self.model = model
        self.layers = model.layers
        self.start, self.end = model.start_layer, model.end_layer
        self.n_all = self.end - self.start
        self.late_start = model.decoder_replay_start if getattr(model, "decoder_replay_layers", None) is not None else None
        self.kinds: dict[str, list[int]] = {}
        for i in range(self.start, self.end):
            self.kinds.setdefault(layer_kind(self.layers[i], i, self.late_start), []).append(i)

    @property
    def counts(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.kinds.items()}

    def describe(self) -> str:
        return f"{self.n_all} layers -- " + ", ".join(f"{k} {len(v)} ({_ids(v)})" for k, v in self.kinds.items())

    def moe_pairs(self):
        return moe_pairs(self.layers[i] for i in range(self.start, self.end))


def moe_pairs(layers) -> list:
    return [(layer.ffn, layer.ffn.gate) for layer in layers if hasattr(getattr(layer, "ffn", None), "gate")]


def install_routing(pairs, routing: str):
    if routing == "native" or not pairs:
        return None
    return BalancedRouting(pairs)


# ----------------------------------------------------------------------------------------------
# The op tree. Rows keep the SGLang port's names where the operator is the same (attn.wq_b, attn.wo_b,
# mlp.experts, ...); vLLM's attribute paths are mapped below. Depth 3 under mlp.experts: vLLM's MoE runner
# calls the gate and the shared expert inside its own forward.

TREE = [
    ("layer", 0),
    ("engram", 1),
    ("input_layernorm", 1),
    ("mhc_attn", 1),
    ("mhc_post", 1),
    ("attn", 1),
    ("attn.wqkv_a", 2),
    ("attn.compressor_kv", 2),
    ("attn.indexer_w", 2),
    ("attn.qkv_norm", 2),
    ("attn.wq_b", 2),
    ("attn.kv_insert", 2),
    ("attn.compressor", 2),
    ("attn.compressor_insert", 2),
    ("attn.indexer", 2),
    ("attn.indexer_topk", 2),
    ("attn.mla", 2),
    ("attn.wo_a", 2),
    ("attn.wo_b", 2),
    ("attn.<rest>", 2),
    ("post_attention_layernorm", 1),
    ("mhc_ffn", 1),
    ("mlp", 1),
    ("mlp.experts", 2),
    ("mlp.experts.gate", 3),
    ("mlp.experts.shared_experts", 3),
    ("mlp.experts.<rest>", 3),
    ("mlp.<rest>", 2),
    ("layer.<rest>", 1),
]
_DEPTH = dict(TREE)
PARENTS = ("layer", "attn", "mlp", "mlp.experts")


def _children(parent: str) -> list[str]:
    d = _DEPTH[parent]
    pre = "" if parent == "layer" else parent + "."
    return [n for n, dd in TREE if dd == d + 1 and n.startswith(pre) and not n.endswith("<rest>")]


RESIDUALS = {p: _children(p) for p in PARENTS}
LEAVES = [n for n, _ in TREE if n not in RESIDUALS]
# Per-kind rows issued from the model loop, outside the layer's forward (summed x layer count, no parent).
OUTSIDE_LAYER = ("engram_lookup",)
GLOBAL_OPS = ("embed_tokens", "engram_hasher", "engram_gather", "hc_post", "hc_head", "norm", "logits")
CYCLED = {"layer": "", "attn": "attn", "mlp": "ffn"}  # parent rows replayed over every layer of their kind

RESIDUAL_CONTENTS = {
    "attn.<rest>": "attn-out alloc, stream fork/join, glue between the call sites",
    "mlp.experts.<rest>": "top-k routing, DP dispatch all-gather, routed MXFP4 experts, combine reduce-scatter",
    "mlp.<rest>": "reshape (no kernel expected)",
    "layer.<rest>": "glue between the seams (no kernel expected)",
    "engram_lookup": "Engram host-table rows (cpu_offload) fetched from the model loop",
    "mhc_attn": "mHC attention seam (post fold + pre + Sinkhorn + attn_norm, Mega-MHC)",
    "mhc_ffn": "mHC FFN seam (post fold + pre + Sinkhorn + ffn_norm, Mega-MHC)",
    "hc_head": "final hc collapse",
}


@dataclass
class Op:
    """One timed row: a module (forward pre-hook) or a call site (owner.attr wrapped), optionally matched
    on its arguments when one module/function serves several rows; `wmod` is the module whose `.weight`
    the cold-weight ring rotates."""

    name: str
    module: torch.nn.Module | None = None
    site: tuple | None = None  # (owner, attr)
    match: Callable | None = None  # (args, kwargs) -> bool
    wmod: torch.nn.Module | None = None
    layer_index: int | None = None


def _get(obj, path: str):
    for part in path.split("."):
        obj = getattr(obj, part, None) if obj is not None else None
    return obj


AMD_MODEL = "vllm.models.deepseek_v41.amd.model"
AMD_ROCM = "vllm.models.deepseek_v41.amd.rocm"


def layer_ops(layer, prefix: str, i: int | None = None) -> list[Op]:
    """Every row of one decoder layer, keyed `<prefix>|<row>`."""
    attn = layer.attn
    ops: list[Op] = []

    def mod(row, path, wpath=None):
        m = _get(layer, path) if path else layer
        if isinstance(m, torch.nn.Module):
            ops.append(Op(f"{prefix}|{row}", module=m, wmod=_get(layer, wpath) if wpath else None, layer_index=i))

    def meth(row, owner_path, attr, wpath=None):
        owner = _get(layer, owner_path)
        if owner is not None and hasattr(owner, attr):
            ops.append(Op(f"{prefix}|{row}", site=(owner, attr), wmod=_get(layer, wpath) if wpath else None, layer_index=i))

    mod("layer", "")
    mod("engram", "engram")
    mod("input_layernorm", "attn_norm")
    mod("post_attention_layernorm", "ffn_norm")
    mod("mhc_post", "mhc_post")
    # The two seams share one module (mhc_pre_delayed) and, on the MXFP8 path, one function: matched on the
    # mix weights passed in (attention seam: hc_attn_fn or its broadcast fold; FFN seam: hc_ffn_fn).
    pre = getattr(layer, "mhc_pre_delayed", None)
    attn_fns = [w for w in (getattr(layer, "hc_attn_fn", None), getattr(layer, "hc_attn_fn_broadcast", None)) if w is not None]
    ffn_fn = getattr(layer, "hc_ffn_fn", None)
    for row, fns in (("mhc_attn", attn_fns), ("mhc_ffn", [ffn_fn])):
        if pre is None:
            continue
        mod_match = (lambda a, kw, _f=fns: any((a[1] if len(a) > 1 else kw.get("fn")) is f for f in _f))
        fn_match = (lambda a, kw, _p=pre, _f=fns: len(a) > 2 and a[0] is _p and any(a[2] is f for f in _f))
        ops.append(Op(f"{prefix}|{row}", module=pre, match=mod_match, layer_index=i))
        ops.append(Op(f"{prefix}|{row}", site=(AMD_MODEL, "mhc_pre_delayed_mxfp8"), match=fn_match, layer_index=i))
    mod("attn", "attn")
    meth("attn.wqkv_a", "attn", "_fused_wqa_wkv_gemm", "attn.fused_wqa_wkv")
    mod("attn.compressor_kv", "attn.compressor.fused_wkv_wgate", "attn.compressor.fused_wkv_wgate")
    mod("attn.indexer_w", "attn.indexer.weights_proj", "attn.indexer.weights_proj")
    meth("attn.qkv_norm", "attn", "_split_qkv_and_norm")
    meth("attn.wq_b", "attn", "_wq_b_proj", "attn.wq_b")
    meth("attn.kv_insert", "attn", "_fused_qnorm_rope_kv_insert")
    mod("attn.compressor", "attn.compressor")
    meth("attn.compressor_insert", "attn.compressor", "insert_cache")
    mod("attn.indexer", "attn.indexer")
    mod("attn.indexer_topk", "attn.indexer.indexer_op")
    meth("attn.mla", "attn", "forward_mqa")
    wo_a = getattr(attn, "wo_a", None)
    if wo_a is not None:
        ops.append(Op(f"{prefix}|attn.wo_a", site=(AMD_ROCM, "rocm_mxfp8_wo_a_bmm"),
                      match=lambda a, kw, _w=wo_a: len(a) > 2 and a[2] is _w, layer_index=i))
        ops.append(Op(f"{prefix}|attn.wo_a", site=(AMD_ROCM, "rocm_inv_rope_einsum"),
                      match=lambda a, kw, _w=wo_a: len(a) > 6 and a[6] is _w, layer_index=i))
    meth("attn.wo_b", "attn", "_wo_b_after_wo_a", "attn.wo_b")
    mod("mlp", "ffn")
    mod("mlp.experts", "ffn.experts")
    mod("mlp.experts.gate", "ffn.gate", "ffn.gate")
    mod("mlp.experts.shared_experts", "ffn.shared_experts")
    if getattr(layer, "engram", None) is not None:
        meth("engram_lookup", "engram", "prepare_embeddings")
    return ops


def global_ops(runner) -> list[Op]:
    text = lm(runner)
    m = text.model
    ops = []
    for row, obj in (("embed_tokens", m.embed_tokens), ("engram_hasher", getattr(m, "engram_hash", None)),
                     ("hc_post", m.mhc_post), ("norm", m.norm), ("logits", text.logits_processor)):
        if isinstance(obj, torch.nn.Module):
            ops.append(Op(row, module=obj))
    ops.append(Op("engram_gather", site=(AMD_MODEL, "gather_engram_hashes")))
    # The hc_collapse op is wrapped on the torch.ops.vllm namespace. A TorchDispatchMode (AKLR_EXPORT's op
    # recorder) resolves the overload through that same attribute, so the row is dropped under export.
    if os.environ.get("AKLR_EXPORT", "0") in ("", "0", "false", "False"):
        ops.append(Op("hc_head", site=("torch.ops.vllm", "hc_collapse_triton")))
    return ops


def collect_ops(layers: Layers, runner) -> list[Op]:
    """Every row to time: `kind|row` for one representative layer per kind, plus globals."""
    ops = []
    for kind, ids in layers.kinds.items():
        ops += layer_ops(layers.layers[ids[0]], kind, ids[0])
    return ops + global_ops(runner)


def op_names(ops: list[Op]) -> list[str]:
    return list(dict.fromkeys(o.name for o in ops))


# ----------------------------------------------------------------------------------------------
# Recording and replay.


class Recorded(NamedTuple):
    args: tuple
    kwargs: dict
    ctx: Any  # (ForwardContext, fields) at the call
    fn: Callable | None  # the original callable for a call site; None = module __call__
    op: Op


ROW_STACK: list[str] = []


def rows_now() -> list[str]:
    return list(ROW_STACK)


def _resolve_owner(owner):
    if isinstance(owner, str):
        if owner == "torch.ops.vllm":
            return torch.ops.vllm
        return importlib.import_module(owner)
    return owner


@contextlib.contextmanager
def record_inputs(ops: list[Op], into: dict, calls: dict | None = None):
    """First-call inputs (snapshotted before in-place mutation) + the ForwardContext of every row, from one
    real forward; `calls` counts every call per row (globals called several times per step, e.g. the model
    mhc_post for the DSpark aux hidden states, are multiplied by it)."""
    memo: dict = {}
    calls = {} if calls is None else calls
    handles, restores = [], []

    def keep(op: Op, args, kwargs, fn=None):
        calls[op.name] = calls.get(op.name, 0) + 1
        if op.name not in into:
            memo["__call__"] = memo.get("__call__", 0) + 1
            into[op.name] = Recorded(snapshot(args, memo), snapshot(kwargs, memo), ctx_now(), fn, op)

    try:
        for op in ops:
            if op.module is None:
                continue

            def pre(_m, args, kwargs, _op=op):
                if _op.match is not None and not _op.match(args, kwargs):
                    return
                if ROW_STACK and ROW_STACK[-1] == _op.name:  # the module called from its own matched call site
                    return
                keep(_op, args, kwargs)
                ROW_STACK.append(_op.name)
                _m._bench_pushed = getattr(_m, "_bench_pushed", 0) + 1

            def post(_m, _args, _kwargs, _out, _op=op):
                if getattr(_m, "_bench_pushed", 0) and ROW_STACK and ROW_STACK[-1] == _op.name:
                    ROW_STACK.pop()
                    _m._bench_pushed -= 1

            handles.append(op.module.register_forward_pre_hook(pre, with_kwargs=True))
            handles.append(op.module.register_forward_hook(post, with_kwargs=True, always_call=True))
        groups: dict = {}
        for op in ops:
            if op.site is not None:
                owner = _resolve_owner(op.site[0]) if isinstance(op.site[0], str) else op.site[0]
                groups.setdefault((id(owner), op.site[1]), (owner, op.site[1], []))[2].append(op)
        for owner, attr, group in groups.values():
            try:
                orig = getattr(owner, attr)
            except AttributeError:
                continue
            had = attr in getattr(owner, "__dict__", {})

            def wrapped(*a, _orig=orig, _group=group, **kw):
                hit = next((o for o in _group if o.match is None or o.match(a, kw)), None)
                if hit is None:
                    return _orig(*a, **kw)
                keep(hit, a, kw, fn=_orig)
                ROW_STACK.append(hit.name)
                try:
                    return _orig(*a, **kw)
                finally:
                    ROW_STACK.pop()

            try:
                setattr(owner, attr, wrapped)
            except (AttributeError, TypeError):
                continue
            restores.append((owner, attr, orig, had))
        yield into
    finally:
        for h in handles:
            h.remove()
        for owner, attr, orig, had in reversed(restores):
            if had or not isinstance(owner, torch.nn.Module):
                setattr(owner, attr, orig)
            else:
                owner.__dict__.pop(attr, None)
        ROW_STACK.clear()


def make_call(entry: Recorded, target=None) -> Callable:
    """Replay a row on its recorded arguments inside its recorded ForwardContext. `target` replaces the module
    (parent rows cycled over the real layers of the kind)."""
    fn = entry.fn if entry.fn is not None else (target if target is not None else entry.op.module)
    if entry.fn is not None and target is not None:
        fn = target
    args, kwargs, ctx = entry.args, entry.kwargs, entry.ctx

    def call():
        with ctx_scope(ctx):
            return fn(*args, **kwargs)

    return call


def cycled_calls(entry: Recorded, layers: Layers, kind: str, row: str) -> list[Callable] | None:
    """Parent rows (layer/attn/mlp) replayed over every real layer of their kind, so weights are not
    cache-hot from one representative (AKLR deepseekv4)."""
    if row not in CYCLED or entry.fn is not None or kind not in layers.kinds:
        return None
    attr = CYCLED[row]
    return [make_call(entry, _get(layers.layers[i], attr) if attr else layers.layers[i]) for i in layers.kinds[kind]]


# MI355X's last-level cache.
MALL_BYTES = 256 << 20
# Twice the cache, so a slot is evicted before its next read.
ROTATION_BYTES = 2 * MALL_BYTES
MAX_ROTATION_SLOTS = 256


def rotation_plan(nbytes: int) -> int:
    """Ring slots for a weight of ``nbytes``; 0 = no rotation (too big for the MALL to matter, or too small)."""
    slots = min(ROTATION_BYTES // max(nbytes, 1), MAX_ROTATION_SLOTS)
    if nbytes >= MALL_BYTES or slots < 2 or slots * nbytes <= MALL_BYTES:
        return 0
    return slots


@contextlib.contextmanager
def rotated_calls(entry: Recorded):
    """Yield ``(fns, inner)`` reading a different copy of the row's 2-D weight each call (AKLR). For a call
    site the rotated weight is the module it reads (`Op.wmod`, e.g. attn.fused_wqa_wkv behind
    `_fused_wqa_wkv_gemm`); the call site reads `module.weight` at call time, so the swap takes."""
    call = make_call(entry)
    wmod = entry.op.wmod if entry.op.wmod is not None else entry.op.module
    w = getattr(wmod, "weight", None) if isinstance(wmod, torch.nn.Module) else None
    slots = rotation_plan(w.numel() * w.element_size()) if isinstance(w, torch.Tensor) and w.dim() == 2 else 0
    ring = None
    if slots:
        try:
            ring = torch.empty((slots, *w.shape), dtype=w.dtype, device=w.device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            ring = None
    # Agreed across ranks: a rank that fell back alone would issue mismatched collectives and wedge the world.
    if not agreed(ring is not None):
        if ring is not None:
            del ring
            torch.cuda.empty_cache()
        yield [call], GRAPH_REPS
        return
    ring.copy_(w)
    params = [torch.nn.Parameter(ring[i], requires_grad=False) for i in range(slots)]

    def rotate_to(slot):
        def fn():
            wmod.weight = slot
            return call()

        return fn

    try:
        # The assignment runs during capture, so each launch records its own slot's pointer.
        yield [rotate_to(p) for p in params], max(GRAPH_REPS, slots)
    finally:
        wmod.weight = w
        del params, ring
        torch.cuda.empty_cache()


@contextlib.contextmanager
def child_scopes(children: list[Op]):
    """Kineto child ranges for a parent's sample: modules via kernel_breakdown.child_ranges, call sites by
    wrapping them in record_function(CHILD) (a function has no module hook)."""
    import kernel_breakdown
    from torch.profiler import record_function

    mods = [o.module for o in children if o.module is not None and o.match is None]
    restores = []
    seen = set()
    for o in children:
        if o.site is None or (id(o.site[0]), o.site[1]) in seen:
            continue
        seen.add((id(o.site[0]), o.site[1]))
        owner = _resolve_owner(o.site[0]) if isinstance(o.site[0], str) else o.site[0]
        orig = getattr(owner, o.site[1], None)
        if orig is None:
            continue
        had = o.site[1] in getattr(owner, "__dict__", {})

        def wrapped(*a, _o=orig, **kw):
            with record_function(kernel_breakdown.CHILD):
                return _o(*a, **kw)

        try:
            setattr(owner, o.site[1], wrapped)
        except (AttributeError, TypeError):
            continue
        restores.append((owner, o.site[1], orig, had))
    try:
        with kernel_breakdown.child_ranges(mods):
            yield
    finally:
        for owner, attr, orig, had in reversed(restores):
            if had or not isinstance(owner, torch.nn.Module):
                setattr(owner, attr, orig)
            else:
                owner.__dict__.pop(attr, None)


@contextlib.contextmanager
def engram_inline(runner):
    """Per-op phase: Engram host lookups run inline on the current stream (prefetch stream and its event
    detached), so a standalone replay neither forks an unjoined side stream inside capture nor waits on an
    event recorded inside another graph. The server overlaps the lookup on its prefetch stream; the
    `engram_lookup` row is the lookup's own cost."""
    saved = []
    for layer in Layers(runner).layers:
        e = getattr(layer, "engram", None)
        if e is not None and hasattr(e, "_prefetch_stream"):
            saved.append((e, e._prefetch_stream, getattr(e, "_prefetch_done", None)))
            e._prefetch_stream, e._prefetch_done = None, None
    try:
        yield len(saved)
    finally:
        for e, st, ev in saved:
            e._prefetch_stream, e._prefetch_done = st, ev


def children_ops(ops_by_name: dict, label: str, times: dict) -> list[Op]:
    kind, _, row = label.rpartition("|")
    out = []
    for child in RESIDUALS.get(row, ()):
        key = f"{kind}|{child}" if kind else child
        if key in times:
            out += ops_by_name.get(key, [])
    return out


# ----------------------------------------------------------------------------------------------
# Roll-up.


def residual_owner(name: str) -> str:
    kind, _, op = name.rpartition("|")
    if not kind or kind == "draft" or op == "layer" or op in OUTSIDE_LAYER:
        return ""
    parent = op.rpartition(".")[0] or "layer"
    if parent not in RESIDUALS:
        return ""
    return f"{kind}|{parent}.<rest>"


def add_residuals(times: dict, counts: dict) -> None:
    """Give every parent a `<rest>` row: itself minus its measurable children."""
    for kind in counts:
        for parent, kids in RESIDUALS.items():
            if f"{kind}|{parent}" in times:
                got = sum(times[f"{kind}|{k}"] for k in kids if f"{kind}|{k}" in times)
                times[f"{kind}|{parent}.<rest>"] = times[f"{kind}|{parent}"] - got


def leaf_totals(times: dict, counts: dict, gcalls: dict | None = None) -> dict:
    """ms per forward for the rows that partition a step without overlapping; globals x their calls/step."""
    gcalls = gcalls or {}
    totals = {}
    for name in list(LEAVES) + list(OUTSIDE_LAYER):
        ms = sum(
            times[f"{k}|{name}"] * n
            for k, n in counts.items()
            if f"{k}|{name}" in times and (name in OUTSIDE_LAYER or _partitions(times, k, name))
        )
        if ms:
            totals[name] = ms
    totals.update({n: times[n] * gcalls.get(n, 1) for n in GLOBAL_OPS if n in times})
    return totals


def kernel_totals(profiles: dict, counts: dict, times: dict, gcalls: dict | None = None) -> dict:
    """Roll representative kernel samples up with the same multiplicities as the graph rows."""
    import kernel_breakdown

    gcalls = gcalls or {}
    totals = {}
    for label, kernels in profiles.items():
        kind, _, name = label.rpartition("|")
        if label not in times or (kind and kind not in counts):
            continue
        if kind and name not in OUTSIDE_LAYER and not _partitions(times, kind, name):
            continue
        row = name + ".<rest>" if name in RESIDUALS else name
        dest = totals.setdefault(row, {})
        mult = counts[kind] if kind else gcalls.get(name, 1)
        for kernel, (ms, n, frames) in kernels.items():
            kernel_breakdown.add_sample(dest, kernel, ms * mult, n * mult, frames)
    return totals


def skipped_note(skipped: dict) -> str:
    if not skipped:
        return ""
    rows = []
    for name, err in sorted(skipped.items()):
        kind, _, op = name.rpartition("|")
        owner = residual_owner(name)
        if kind == "draft":
            where = "the `draft model` row"
        elif op in RESIDUALS:
            where = "its parent's `<rest>`; its children leave the roll-up too"
        elif owner:
            where = f"`{owner.split('|', 1)[1]}`"
        else:
            where = "nowhere -- the roll-up is short by it"
        rows.append((kind or "--", f"`{op}`", where, err.replace("|", "\\|")))
    return table(rows, ("layer kind", "op", "its ms now sits in", "why it could not be replayed")) + (
        "\n\nThese ops ran in the recording forward, so their cost is real and is in the parent row; what is"
        " missing is the row that would separate it out. Read the named `<rest>` row as an upper bound."
    )


def coverage_note(names: list[str], saved: dict) -> str:
    missing = [n for n in names if n not in saved]
    if not missing:
        return ""
    rows = []
    for name in missing:
        kind, _, op = name.rpartition("|")
        owner = residual_owner(name)
        rows.append((kind or "--", op, owner.split("|", 1)[1] if owner else "unattributed (no parent residual)"))
    return table(rows, ("layer kind", "hook / call site not called", "possible containing row")) + (
        "\n\nHooked but never called during the recording forward: this layer kind does not run that op, or a"
        " fused kernel does the work without going through the module (vLLM DSv4.1 on gfx950: the Mega-MHC seam"
        " applies attn_norm / ffn_norm itself; the ROCm attention reads the preshuffled fused_wqa_wkv / wo_b"
        " weights directly, timed through their call sites instead)."
    )


def tree_name(name: str, depth: int, contents: dict | None = None) -> str:
    c = (RESIDUAL_CONTENTS if contents is None else contents).get(name)
    return f"`{'  ' * depth}{name}{' :: ' + c if c else ''}`"


# ----------------------------------------------------------------------------------------------
# Memory.

GB = 1 << 30


def memory_section(runner, batch=None, filled: dict | None = None, drafter=None) -> str:
    """Weights, KV tensors per cache group and the case's share of the block pool, off the real buffers."""
    context = runner.vllm_config.compilation_config.static_forward_context
    kvc = runner.kv_cache_config
    rows, total_kv = [], 0
    for gi, group in enumerate(kvc.kv_cache_groups):
        seen, nbytes = set(), 0
        for name in group.layer_names:
            cache = getattr(context.get(name), "kv_cache", None)
            for t in (cache if isinstance(cache, (list, tuple)) else [cache]):
                if isinstance(t, torch.Tensor) and t.numel() and t.data_ptr() not in seen:
                    seen.add(t.data_ptr())
                    nbytes += t.untyped_storage().nbytes()
        total_kv += nbytes
        specs = list(group.kv_cache_specs.values())
        spec = specs[0] if specs else None
        rows.append((f"group {gi} `{type(spec).__name__}`", len(group.layer_names), group.block_size,
                     str(getattr(spec, "dtype", "")).replace("torch.", ""), nbytes / GB))
    models = [runner.model] + ([drafter] if drafter is not None else [])
    dev = host = 0
    for m in models:
        for t in (*m.parameters(), *m.buffers()):
            b = t.numel() * t.element_size()
            if t.is_cuda:
                dev += b
            else:
                host += b
    hbm = torch.cuda.mem_get_info()[1]
    rows += [("**kv total** (storage, groups may alias)", "", "", "", total_kv / GB), ("weights on device", "", "", "", dev / GB),
             ("weights on host (Engram cpu_offload table)", "", "", "", host / GB), ("**HBM**", "", "", "", hbm / GB)]
    notes = [f"`--gpu-memory-utilization {runner.cache_config.gpu_memory_utilization}`; KV block pool"
             f" {kvc.num_blocks:,} blocks (block ids shared by all groups)"]
    if batch is not None:
        notes.append(f"this case allocates {batch.num_blocks:,} block ids, {batch.num_blocks / max(kvc.num_blocks, 1) * 100:.1f}% of the pool")
    if filled:
        notes.append("KV bytes filled for the case: " + ", ".join(f"{k} {v / GB:.2f} GB" for k, v in filled.items()))
    return table(rows, ("component", "layers", "block", "dtype", "GB"), fmt=".2f") + "\n\n" + "\n".join(f"- {n}" for n in notes)
