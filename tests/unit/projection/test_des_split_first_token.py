###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""What a split-served request pays between submission and its first token.

Four pieces, each checked against a stub cost model so the numbers follow from
it rather than from the projector's physics:

* a decode replica's clock is its own: a handoff delivered while it is busy
  waits for it, and does not move it;
* the first token is stamped when a decode replica admits the request, so a
  request waiting for a decode slot waits before its first token;
* host latency measured with each pool's anchor is paid once per server;
* SGLang's prefill delayer holds a lone attention-DP prefill for up to its
  pass limit, and only while there are decodes to run instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from infera.projection.core.projection.inference_projection import des as des_mod
from infera.projection.core.projection.inference_projection import pareto as P
from infera.projection.core.projection.inference_projection.performance import (
    InferencePerformanceProjector,
)

_BASE_MS = 2.0
_PER_SEQ_MS = 0.05
_PER_PREFILL_TOKEN_MS = 0.002


@dataclass
class _Req:
    input_seq_len: int = 1024
    output_seq_len: int = 256
    max_concurrency: int = 8
    max_num_batched_tokens: int = 8192
    chunked_prefill_size: int = 0
    speculative_num_tokens: int = 0
    speculative_acceptance_rate: float = 0.0
    tokenize_overhead_us: float = 0.0
    detokenize_overhead_us: float = 0.0
    serving_engine: str = ""

    def resolved_max_concurrency(self) -> int:
        return self.max_concurrency

    def resolved_max_context_len(self) -> int:
        return self.input_seq_len + self.output_seq_len

    def resolved_prefix_cache_hit_rate(self) -> float:
        return 0.0


@dataclass
class _Disagg:
    enabled: bool = True
    prefill_replicas: int = 1
    decode_replicas: int = 1
    prefill_max_running: int | None = None
    decode_max_running: int | None = None


@dataclass
class _MP:
    tensor_model_parallel_size: int = 4
    pipeline_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    attention_data_parallel_size: int = 1


@dataclass
class _PoolCfg:
    model_parallel_config: _MP = field(default_factory=_MP)


@dataclass
class _Cfg:
    request_config: _Req
    disaggregation_config: _Disagg = field(default_factory=_Disagg)


class _Pool:
    def __init__(self, scale: float = 1.0, host_ms: float = 0.0):
        self.scale = scale
        self.cfg = _PoolCfg()
        self.host_ms = host_ms

    def decode_step_latency_ms(self, batch, ctx, q_len):
        return self.scale * (_BASE_MS + _PER_SEQ_MS * batch)

    def mixed_step_latency_ms(self, num_decode, prefill_tokens, ctx, prefill_kv, q_len):
        return self.scale * (
            _BASE_MS + _PER_SEQ_MS * num_decode + _PER_PREFILL_TOKEN_MS * prefill_tokens
        )

    def measured_host_ms(self, prompt_len):
        return self.host_ms


class _Projector:
    def __init__(self, prefill: _Pool, decode: _Pool, handoff_ms: float = 0.5):
        self.prefill = prefill
        self.decode = decode
        self.handoff_ms = handoff_ms

    def pool_projectors(self):
        return self.prefill, self.decode

    def kv_handoff_ms(self, decode_proj, context_len):
        return self.handoff_ms


def _split(
    concurrency=16,
    *,
    decode_scale=1.0,
    prefill_host=0.0,
    decode_host=0.0,
    decode_max_running=None,
    input_len=1024,
    output_len=256,
    range_ratio=1.0,
    handoff_ms=0.5,
    prefill_replicas=1,
):
    cfg = _Cfg(
        _Req(max_concurrency=concurrency, input_seq_len=input_len, output_seq_len=output_len),
        _Disagg(prefill_replicas=prefill_replicas, decode_max_running=decode_max_running),
    )
    proj = _Projector(
        _Pool(host_ms=prefill_host), _Pool(decode_scale, host_ms=decode_host), handoff_ms
    )
    return des_mod.simulate_disaggregated(
        cfg,
        proj,
        rate_per_s=0.0,
        num_requests=concurrency * 8,
        warmup_frac=0.0,
        closed_loop_clients=concurrency,
        range_ratio=range_ratio,
    )


def test_a_busy_decode_replica_is_not_moved_by_a_handoff():
    """TPOT is the decode step at the batch decode ran, not more.

    Handoffs are delivered against the furthest-ahead station's clock. Moving
    a busy replica's clock up to each one -- here a KV transfer far longer than
    a decode step -- skipped the decode steps its running requests still had
    to take, and charged the skipped time to their TPOT.
    """
    res = _split(16, input_len=8192, range_ratio=0.5, handoff_ms=80.0)
    step = _BASE_MS + _PER_SEQ_MS * res.packing["avg_decode_reqs"]
    assert res.tpot["mean"] < 1.1 * step, (res.tpot, step)


def test_a_request_waiting_for_a_decode_slot_waits_before_its_first_token(monkeypatch):
    """The router streams from the decode server, so the wait lands in TTFT."""
    monkeypatch.setenv("INFERASIM_DES_SPLIT_TTFT", "prefill")
    at_prefill = _split(16, decode_max_running=4)
    monkeypatch.delenv("INFERASIM_DES_SPLIT_TTFT")
    at_admit = _split(16, decode_max_running=4)
    assert at_admit.ttft["mean"] > 2.0 * at_prefill.ttft["mean"]
    # Neither changes when tokens are produced, only when the first is stamped.
    assert abs(at_admit.makespan_ms - at_prefill.makespan_ms) < 1e-6 * at_prefill.makespan_ms


def test_each_server_charges_its_own_measured_host_latency(monkeypatch):
    base = _split(1)
    both = _split(1, prefill_host=40.0, decode_host=60.0)
    assert abs(both.ttft["mean"] - base.ttft["mean"] - 100.0) < 1.0
    monkeypatch.setenv("INFERASIM_SPLIT_HOST", "prefill")
    one = _split(1, prefill_host=40.0, decode_host=60.0)
    assert abs(one.ttft["mean"] - base.ttft["mean"] - 40.0) < 1.0


def test_the_anchor_host_fit_is_read_and_grows_with_the_prompt():
    p = InferencePerformanceProjector.__new__(InferencePerformanceProjector)
    p._read_prefill_host({"host_fit": {"fixed_ms": 45.0, "ms_per_token": 0.005}})
    assert abs(p.measured_host_ms(0) - 45.0) < 1e-9
    assert abs(p.measured_host_ms(8000) - 85.0) < 1e-9
    p._read_prefill_host({"ms_per_token": 0.02})
    assert p.measured_host_ms(8000) == 0.0


class _Kernel:
    def decode_step_latency_ms(self, batch, ctx, q_len):
        return _BASE_MS + _PER_SEQ_MS * batch

    def mixed_step_latency_ms(self, num_decode, prefill_tokens, ctx, prefill_kv, q_len, **kw):
        return _BASE_MS + _PER_SEQ_MS * num_decode + _PER_PREFILL_TOKEN_MS * prefill_tokens


@dataclass
class _ColoCfg:
    request_config: _Req


def _colocated(concurrency, monkeypatch, passes):
    monkeypatch.setenv("INFERASIM_DES_LOCKSTEP", "4")
    if passes:
        monkeypatch.setenv("INFERASIM_PREFILL_DELAY_PASSES", str(passes))
    else:
        monkeypatch.delenv("INFERASIM_PREFILL_DELAY_PASSES", raising=False)
    return des_mod.simulate_once(
        _ColoCfg(_Req(max_concurrency=concurrency, chunked_prefill_size=4096)),
        _Kernel(),
        rate_per_s=0.0,
        arrival_model="closed",
        num_requests=concurrency * 12,
        warmup_frac=0.0,
        closed_loop_clients=concurrency,
        range_ratio=0.5,
    )


def test_the_prefill_delayer_holds_a_lone_prefill(monkeypatch):
    """Up to its pass limit, so TTFT grows by at most that many decode steps."""
    off = _colocated(8, monkeypatch, 0)
    on = _colocated(8, monkeypatch, 30)
    step = _BASE_MS + 4 * _PER_SEQ_MS * 8
    assert on.ttft_arrival["mean"] > off.ttft_arrival["mean"] + 5 * step
    assert on.ttft_arrival["mean"] < off.ttft_arrival["mean"] + 31 * step


def test_the_prefill_delayer_never_holds_an_idle_rank(monkeypatch):
    """With nothing decoding there is no step for a held prefill to wait out."""
    off = _colocated(1, monkeypatch, 0)
    on = _colocated(1, monkeypatch, 30)
    assert abs(on.ttft_arrival["mean"] - off.ttft_arrival["mean"]) < 1e-6


def test_a_scenario_carries_the_inferencex_range_ratio():
    sc = P.Scenario.parse("8k/1k@0.8")
    assert (sc.isl, sc.osl, sc.range_ratio) == (8192, 1024, 0.8)
    assert P.Scenario.parse("8k/1k").range_ratio == 1.0
    kw = dict(
        trace_path=None,
        precision="fp8",
        engine="sglang",
        par=P.Parallelism(8),
        concurrency=16,
        gpu_arch="mi355x",
        hbm_gb=288.0,
        num_requests=160,
        anchor=None,
        checkpoint="X",
    )
    argv = P.point_argv(scenario=sc, **kw)
    assert argv[argv.index("--des-range-ratio") + 1] == "0.8"
    assert "--des-range-ratio" not in P.point_argv(scenario=P.Scenario.parse("8k/1k"), **kw)


def test_sglang_attention_dp_turns_the_delayer_on(monkeypatch):
    """The InferenceX SGLang attention-DP recipes pass --enable-prefill-delayer."""
    import os

    # project_point writes os.environ directly; restore it for the tests after.
    import infera.projection.cli as cli
    import infera.projection.core.projection.inference_projection as ip

    class _Parser:
        def parse_known_args(self, argv):
            return None, []

    def _launch(parsed, overrides):
        raise RuntimeError("not projected here")

    monkeypatch.setattr(cli, "build_parser", lambda: _Parser())
    monkeypatch.setattr(ip, "launch_projection_from_cli", _launch)
    saved = dict(os.environ)
    try:
        for engine, dp, want in (("sglang", 8, "30"), ("sglang", 1, None), ("atom", 8, None)):
            point = dict(
                model="m",
                checkpoint="c",
                engine=engine,
                precision="fp8",
                gpu_arch="mi355x",
                scenario="1k/1k",
                isl=1024,
                osl=1024,
                mode="simulate",
                kv_config="hbm",
                parallelism="x",
                tp=8,
                ep=1,
                dp_attn=dp,
                pp=1,
                gpus=8,
                concurrency=16,
            )
            P.project_point({"point": point, "preset": "x", "argv": ["inference", "--bogus"]})
            assert os.environ.get("INFERASIM_PREFILL_DELAY_PASSES") == want, (engine, dp)
    finally:
        os.environ.clear()
        os.environ.update(saved)
