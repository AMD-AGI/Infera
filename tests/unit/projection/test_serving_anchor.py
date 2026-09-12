###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""An anchor has to be measured on the engine it is going to predict.

Offline ``LLM()`` and a real server do not resolve the same kernels, so the
anchor path launches a server by default and records which kernels it got. The
launch is the only part that differs between engines: both expose ``/health``
and an OpenAI-compatible completions route, so readiness and load generation
are shared.
"""

from __future__ import annotations

from argparse import Namespace

import pytest

from infera.projection.core.projection.inference_projection import benchmark_serving
from infera.projection.core.projection.inference_projection.benchmark_serving import (
    _engine_argv,
    _fit_prefill_curve,
    prefill_probe_lengths,
    resolved_kernels,
)

VLLM_LOG = """
Overriding with ROCM_AITER_FA out of potential backends: ['ROCM_AITER_FA']
Using 'AITER_MXFP4_BF16' Mxfp4 MoE backend
"""

SERVER_LOG = """
Overriding with ROCM_AITER_UNIFIED_ATTN out of potential backends: ['TRITON_ATTN']
Using 'TRITON' Mxfp4 MoE backend
"""


def spec(**over) -> Namespace:
    base = dict(
        model="openai/gpt-oss-120b",
        serving_backend="vllm",
        max_model_len=8192,
        enable_expert_parallel=False,
        enforce_eager=False,
        quantization=None,
        kv_cache_dtype=None,
        server_args="",
    )
    base.update(over)
    return Namespace(**base)


def test_the_two_entrypoints_do_not_run_the_same_kernels():
    """The finding the serving default exists to fix, pinned as a test."""
    offline = resolved_kernels(VLLM_LOG)
    served = resolved_kernels(SERVER_LOG)
    assert offline["resolved_attention_backend"] == "ROCM_AITER_FA"
    assert offline["resolved_moe_backend"] == "AITER_MXFP4_BF16"
    assert served["resolved_attention_backend"] == "ROCM_AITER_UNIFIED_ATTN"
    assert served["resolved_moe_backend"] == "TRITON"
    assert offline != served


@pytest.mark.parametrize("backend, expected", [
    ("vllm", ["--load-format", "dummy"]),
    ("sglang", ["--load-format", "dummy"]),
    # ATOM names the fill, not the loader, and rejects the vLLM spelling hard
    # enough that the server dies during argv parsing. The fill is "empty"
    # because zero-filling a packed-MXFP4 checkpoint has no torch kernel.
    ("atom", ["--load_dummy", "empty"]),
])
def test_random_weights_are_requested_the_way_each_engine_spells_it(
        backend, expected):
    argv = _engine_argv(spec(serving_backend=backend, load_format="dummy"),
                        port=8000, tp=4)
    assert "--load-format" not in argv or backend != "atom"
    for i in range(len(argv) - 1):
        if argv[i] == expected[0]:
            assert argv[i + 1] == expected[1]
            break
    else:
        raise AssertionError(f"{expected[0]} absent from {argv}")


def probe_spec(**over) -> Namespace:
    base = dict(input_len=8192, prefill_anchor_short=1024,
                prefill_anchor_points=0, prefill_anchor_validate=False)
    base.update(over)
    return Namespace(**base)


def test_two_probe_points_are_the_default_and_bracket_the_anchor_length():
    """The long point is the anchor's own length, so the rate is interpolated."""
    assert prefill_probe_lengths(probe_spec()) == [1024, 8192]


def test_asking_for_more_points_spaces_them_evenly_to_the_anchor_length():
    """Evenly, not geometrically: the quadratic term needs leverage at the long
    end, and a geometric ladder puts almost nothing there."""
    lengths = prefill_probe_lengths(probe_spec(prefill_anchor_points=5))
    assert lengths == [1024, 2816, 4608, 6400, 8192]
    gaps = {b - a for a, b in zip(lengths, lengths[1:])}
    assert gaps == {1792}


def test_a_ladder_too_tight_to_resolve_falls_back_rather_than_measuring_noise():
    """Adjacent points closer than the noise floor measure the noise floor."""
    lengths = prefill_probe_lengths(
        probe_spec(input_len=1024, prefill_anchor_short=512,
                   prefill_anchor_points=6))
    assert lengths == [512, 1024]


def test_the_curve_fit_recovers_coefficients_it_was_given():
    """Fixed cost, per-token and per-token^2 are separable when the probe has
    enough points; this is the whole reason for taking more than two."""
    fixed, a, b = 180.0, 0.012, 1.5e-6
    pts = [(n, fixed + a * n + b * n * n)
           for n in (1024, 2816, 4608, 6400, 8192)]
    fit = _fit_prefill_curve(pts)
    assert fit["fixed_ms"] == pytest.approx(fixed, abs=1e-3)
    assert fit["ms_per_token"] == pytest.approx(a, rel=1e-6)
    assert fit["ms_per_token_sq"] == pytest.approx(b, rel=1e-6)
    assert fit["r2"] == pytest.approx(1.0)


def test_three_points_are_refused_because_a_parabola_through_them_is_exact():
    """With no residual left there is nothing to judge the fit by, and noise in
    any one point can put the intercept anywhere."""
    pts = [(n, 180.0 + 0.012 * n) for n in (1024, 4608, 8192)]
    assert _fit_prefill_curve(pts) is None


def test_a_two_point_chord_hides_curvature_in_its_intercept():
    """Why the extra probes are worth their runtime. The chord through two
    points of a curved prefill reports an intercept that is nowhere near the
    real fixed cost, and that intercept is what gets carried across tensor
    parallelism as though it did not shard."""
    fixed, a, b = 146.4, 0.0558, -1.74e-6
    def truth(n):
        return fixed + a * n + b * n * n
    lo, hi = 4096, 8192
    chord_rate = (truth(hi) - truth(lo)) / (hi - lo)
    chord_fixed = truth(lo) - chord_rate * lo
    assert chord_fixed > fixed * 1.3
    pts = [(n, truth(n)) for n in (1024, 2816, 4608, 6400, 8192)]
    assert _fit_prefill_curve(pts)["fixed_ms"] == pytest.approx(fixed, abs=1e-3)


def test_kernels_are_absent_rather_than_wrong_for_another_engine():
    """A log that never names a vLLM backend must not report one."""
    assert resolved_kernels("Loading weights...\nServer started") == {
        "resolved_attention_backend": None,
        "resolved_moe_backend": None,
    }


@pytest.mark.parametrize(
    "backend, model_flag, tp_flag, http_flag",
    [
        ("vllm", None, "--tensor-parallel-size", "--port"),
        ("sglang", "--model-path", "--tp", "--port"),
        ("atom", "--model", "--tensor-parallel-size", "--server-port"),
    ],
)
def test_each_engine_is_launched_the_way_it_expects(backend, model_flag, tp_flag, http_flag):
    argv = _engine_argv(spec(serving_backend=backend), port=8123, tp=4)
    if model_flag is None:
        assert argv[0] == "openai/gpt-oss-120b"  # vLLM takes the model positionally
    else:
        assert argv[argv.index(model_flag) + 1] == "openai/gpt-oss-120b"
    assert argv[argv.index(tp_flag) + 1] == "4"
    assert argv[argv.index(http_flag) + 1] == "8123"


def test_atom_keeps_its_rendezvous_port_off_the_http_port():
    """ATOM's --port is MASTER_PORT, not the API; sharing one number deadlocks."""
    argv = _engine_argv(spec(serving_backend="atom"), port=8123, tp=1)
    assert argv[argv.index("--server-port") + 1] == "8123"
    assert argv[argv.index("--port") + 1] != "8123"


def test_the_context_length_flag_follows_the_engine():
    """The same intent, spelled differently by each engine."""
    assert "--max-model-len" in _engine_argv(spec(), port=1, tp=1)
    assert "--context-length" in _engine_argv(spec(serving_backend="sglang"), port=1, tp=1)


def test_caller_server_args_always_win_by_coming_last():
    argv = _engine_argv(spec(server_args="--gpu-memory-utilization 0.85"), port=1, tp=1)
    assert argv[-2:] == ["--gpu-memory-utilization", "0.85"]


# --- what one launch brings back --------------------------------------------
# A served point costs a whole client run, so which batches a launch sweeps is
# the difference between an anchor that answers off its measured point and one
# that does not.


def _sweep_batches(monkeypatch, **over):
    """Run the serving anchor with the engine and load generator stubbed."""
    measured: list[int] = []

    class FakeEngine:
        async def start(self):
            return None

        async def stop(self):
            return None

    monkeypatch.setattr(benchmark_serving, "_build_engine", lambda *a, **k: FakeEngine())
    # The load generator is stubbed out below, so claim one exists: these tests
    # are about which batches get measured, not about what is installed.
    monkeypatch.setattr(benchmark_serving.shutil, "which", lambda _: "/usr/bin/vllm")
    monkeypatch.setattr(
        benchmark_serving,
        "_measure_concurrency",
        lambda port, batch, args, out_dir: measured.append(batch) or float(batch),
    )
    fields = dict(
        tp=8,
        pp=1,
        benchmark_gpus=4,
        batch=16,
        batches=None,
        concurrency=None,
        input_len=1024,
        output_len=128,
        env=[],
        quantization="mxfp4",
    )
    fields.update(over)
    return measured, benchmark_serving.run_serving_benchmark(spec(**fields))


def test_a_concurrency_anchor_sweeps_the_ladder_rather_than_one_batch(monkeypatch):
    """Asking for a concurrency must not collapse to the default batch.

    ``--concurrency`` is what benchmark mode requests, and honouring it only on
    the offline path left the served sweep measuring a single point at whatever
    ``--batch`` happened to default to -- neither the shape asked for, nor a
    curve. The projector then held that one point flat across every batch while
    wearing a measurement's credibility.
    """
    measured, artifact = _sweep_batches(monkeypatch, concurrency=32)

    assert len(measured) > 1
    assert measured == sorted(measured)
    # The ladder has to reach the concurrency, so any batch up to it buckets up
    # to something actually measured.
    assert max(measured) >= 32
    assert artifact["meta"]["decode_pad_to_capture"] is True
    assert artifact["meta"]["concurrency"] == 32
    # The anchor point is the bucket covering the concurrency, not --batch.
    assert artifact["meta"]["batch"] == 32


def test_explicit_batches_are_measured_as_given(monkeypatch):
    """The ladder is for --concurrency; a caller naming batches still gets them,
    and no claim that decode may be bucketed."""
    measured, artifact = _sweep_batches(monkeypatch, batches="8,32")
    assert measured == [8, 32]
    assert artifact["meta"]["decode_pad_to_capture"] is False


def test_benchmark_mode_can_choose_the_engine_that_can_load_the_model():
    """``--profiling-mode benchmark`` must not be pinned to one engine.

    The harness already launches vLLM, SGLang and ATOM through the platform's
    own adapters, but the projector built its command line without ever naming
    one, so benchmark mode always measured under vLLM. That is not a cosmetic
    default: a vLLM build that cannot load an architecture makes the model
    unmeasurable, while an SGLang build that can would have measured it. The
    two are also different measurements of the same config, which is why
    ``serving_backend`` is part of the anchor cache key rather than ignored.
    """
    from argparse import Namespace
    from types import SimpleNamespace
    from unittest import mock

    from infera.projection.core.projection.inference_projection import benchmark

    cfg = SimpleNamespace(
        model_parallel_config=SimpleNamespace(
            tensor_model_parallel_size=2,
            expert_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        ),
        request_config=SimpleNamespace(
            input_seq_len=1024, output_seq_len=128, max_concurrency=32, batch_size=32
        ),
    )

    def argv_for(**over):
        args = Namespace(
            bench_model="deepseek-ai/DeepSeek-V4-Flash",
            save_benchmark="/tmp/anchor.json",
            benchmark_gpus=None,
            **over,
        )
        seen = {}
        with (
            mock.patch.object(benchmark, "json") as js,
            mock.patch("builtins.open", mock.mock_open(read_data="{}")),
        ):
            js.load.return_value = {}
            with mock.patch(
                "infera.projection.core.projection.inference_projection.benchmark_vllm.main",
                side_effect=lambda argv: seen.setdefault("argv", argv),
            ):
                benchmark.spawn_inference_benchmark(args, cfg)
        return seen["argv"]

    sglang = argv_for(bench_serving_backend="sglang")
    assert "--serving-backend" in sglang
    assert sglang[sglang.index("--serving-backend") + 1] == "sglang"

    # Unset stays unset so the harness keeps its own default rather than
    # having one restated in two places.
    assert "--serving-backend" not in argv_for(bench_serving_backend=None)


def test_the_client_is_not_a_second_reason_a_model_cannot_be_measured(monkeypatch):
    """A load generator is needed, but vLLM's in particular is not.

    The models worth measuring under SGLang are the ones the local vLLM build
    cannot load, and such an image need not carry vLLM's client either. Falling
    back to the engine's own client keeps the missing package from deciding
    what is measurable.
    """
    monkeypatch.setattr(benchmark_serving.shutil, "which", lambda _: "/usr/bin/vllm")
    # Where vLLM exists it drives both engines, so anchors stay comparable.
    assert benchmark_serving.client_kind(spec(serving_backend="vllm")) == "vllm"
    assert benchmark_serving.client_kind(spec(serving_backend="sglang")) == "vllm"

    monkeypatch.setattr(benchmark_serving.shutil, "which", lambda _: None)
    assert benchmark_serving.client_kind(spec(serving_backend="sglang")) == "sglang"
    # An engine with no client of its own says so, rather than failing later
    # inside a subprocess that was never going to exist.
    with pytest.raises(RuntimeError, match="no load generator"):
        benchmark_serving.client_kind(spec(serving_backend="vllm"))


def test_each_client_is_read_the_way_it_writes(tmp_path):
    """The two clients report the same metrics in different file shapes."""
    one = tmp_path / "vllm.json"
    one.write_text('{"mean_tpot_ms": 12.5, "mean_ttft_ms": 300.0}')
    assert benchmark_serving._client_result(str(one), "vllm")["mean_tpot_ms"] == 12.5

    # JSON Lines, and appended to: the run just finished is the last line, not
    # the first.
    many = tmp_path / "sglang.jsonl"
    many.write_text('{"mean_tpot_ms": 99.0}\n{"mean_tpot_ms": 12.5}\n')
    assert benchmark_serving._client_result(str(many), "sglang")["mean_tpot_ms"] == 12.5

    with pytest.raises(RuntimeError, match="no result"):
        (tmp_path / "empty.jsonl").write_text("")
        benchmark_serving._client_result(str(tmp_path / "empty.jsonl"), "sglang")


def test_the_packed_probe_varies_how_many_sequences_share_a_step(monkeypatch):
    """The axis the length sweep cannot supply, and the guards around it.

    At concurrency 1 the step's token count and the sequence's attention
    context are the same number, so widening the step and lengthening the
    context are indistinguishable and the curve fit charges one for the other.
    This probe holds the length fixed and varies how many arrive together,
    which is the only way to separate them.
    """
    calls = []

    def fake_client(port, args, out_dir, tag, *, batch, input_len, output_len,
                    num_prompts):
        calls.append((tag, batch, input_len, num_prompts))
        # 80 ms floor plus 20 us per token in the step: a clean packed curve.
        return {"p99_ttft_ms": 80.0 + 0.020 * batch * input_len}

    monkeypatch.setattr(benchmark_serving, "_run_client", fake_client)

    args = Namespace(prefill_packed_points=4, max_num_batched_tokens=16384)
    got = benchmark_serving.packed_prefill_probe(0, args, "/tmp", 1024)

    assert got["seq_len"] == 1024
    # Powers of two while the wave still fits one step; 16 x 1024 is the last.
    assert [p["seqs"] for p in got["points"]] == [1, 2, 4, 8]
    assert got["ms_per_token"] == pytest.approx(0.020)
    # The floor the difference throws away is recovered as the intercept, which
    # is the check that the slope really is marginal cost and nothing else.
    assert got["implied_fixed_ms"] == pytest.approx(80.0)
    # One discarded warmup at the widest wave, then each count repeated so its
    # p99 is a median over waves rather than one draw from the tail.
    assert calls[0][0] == "packed_warmup"
    assert len(calls) == 1 + 4 * benchmark_serving._PACKED_PROBE_REPEATS
    # Every measured wave is exactly as wide as the point it measures; refilling
    # the loop would put steady-state queueing into the same number.
    for tag, batch, input_len, num_prompts in calls[1:]:
        assert batch == num_prompts and input_len == 1024


def test_a_wider_step_cannot_be_cheaper(monkeypatch):
    """Non-monotonic points are discarded, not fitted through.

    Least squares reports a tidy positive rate through points that are not a
    curve at all. Three vLLM harvests came back with the widest wave *faster*
    than the one before it and still fitted a clean slope, which would have
    been used as a measurement of step cost.
    """
    ladder = {1024: 114.7, 2048: 481.6, 4096: 532.9, 8192: 334.8}

    monkeypatch.setattr(
        benchmark_serving, "_run_client",
        lambda port, args, out_dir, tag, *, batch, input_len, output_len,
        num_prompts: {"p99_ttft_ms": ladder[batch * input_len]})

    args = Namespace(prefill_packed_points=4, max_num_batched_tokens=16384)
    assert benchmark_serving.packed_prefill_probe(0, args, "/tmp", 1024) is None


def test_the_packed_probe_declines_rather_than_measuring_nothing(monkeypatch):
    """Three ways it can have nothing to say, each said rather than guessed."""
    monkeypatch.setattr(benchmark_serving, "_run_client",
                        lambda *a, **k: {"p99_ttft_ms": 100.0})

    # Off by request.
    assert benchmark_serving.packed_prefill_probe(
        0, Namespace(prefill_packed_points=0, max_num_batched_tokens=16384),
        "/tmp", 1024) is None

    # A prompt as long as the budget cannot be packed with anything, so there
    # is no packing axis here to measure.
    assert benchmark_serving.packed_prefill_probe(
        0, Namespace(prefill_packed_points=4, max_num_batched_tokens=16384),
        "/tmp", 16384) is None

    # A flat or falling wave is not a prefill curve; returning None leaves the
    # projector on the single-sequence curve and warning about it, rather than
    # anchoring on noise.
    assert benchmark_serving.packed_prefill_probe(
        0, Namespace(prefill_packed_points=4, max_num_batched_tokens=16384),
        "/tmp", 1024) is None
