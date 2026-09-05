"""Measurement driver for the sglang sampler vocab-softmax forge-loop task.

forge-loop treats this file as a black box invoked as ``python driver.py <args>``
and reads only stdout. Three modes (same contract as the shipped examples):

  * Correctness  ``python driver.py`` -> the complete scored suite; prints
    ``case_snr: <case> <db>`` / ``case_allclose: <case> <bool>`` per case and
    then the aggregate ``SNR: <db> dB`` / ``allclose: <bool>``.
  * Benchmark    ``python driver.py --bench-mode --warmup N --iters N`` ->
    ``wall_ms: <t>`` samples plus one ``case_ms: <case> <median>`` per case.
  * Profiling    ``python driver.py --profile-run`` -> runs only the target on
    the representative case, no reference, no timing.

The driver is the correctness ORACLE and the perf MEASURER. forge never edits it.
It imports the kernel under optimization by its stable public name
``sampler_softmax`` from ``sampler_softmax_kernel.py``.

Cases are the traced production shape and its neighbours in batch:
``vocab = 151936`` is Qwen3-0.6B's vocabulary; ``batch`` is the number of
running decode requests. ``B8`` is the shape actually measured in the trace
(55.59 us with ATen); B1 and B32 bracket it so a change cannot win on one batch
size by wrecking another.
"""

from __future__ import annotations

import argparse
import math
import sys

import torch

from graph_harness import cuda_graph_bench
from sampler_softmax_kernel import sampler_softmax

_VOCAB = 151936
# (batch, vocab). B8 is the traced shape and the profiling representative.
_CASES = (
    (1, _VOCAB),
    (8, _VOCAB),
    (32, _VOCAB),
)
_PROFILE_CASE = (8, _VOCAB)

_SEED = 0


def _case_id(batch: int, vocab: int) -> str:
    return f"B{batch}_V{vocab}"


def _make_logits(batch: int, vocab: int, device: str) -> torch.Tensor:
    """Build logits with a realistic scale.

    Real sampler logits are temperature-scaled and have a long tail; randn*4
    gives a peaked-but-not-degenerate distribution, which keeps the
    max-subtraction meaningful (a kernel that skips it overflows exp()).
    """
    torch.manual_seed(_SEED)
    return torch.randn(batch, vocab, device=device, dtype=torch.float32) * 4.0


def _reference(logits: torch.Tensor) -> torch.Tensor:
    """Float64 reference so the oracle is independent of the fp32 kernel."""
    return torch.softmax(logits.double(), dim=-1).float()


def _snr_db(reference: torch.Tensor, test: torch.Tensor) -> float:
    reference = reference.double()
    test = test.double()
    noise = test - reference
    signal_power = torch.mean(reference * reference).item()
    noise_power = torch.mean(noise * noise).item()
    if noise_power <= 0.0:
        return 100.0
    if signal_power <= 0.0:
        return 0.0
    return 10.0 * math.log10(signal_power / noise_power)


def _run_correctness_suite(device: str) -> int:
    snr_values = []
    allclose_values = []
    for batch, vocab in _CASES:
        x = _make_logits(batch, vocab, device)
        out = torch.empty_like(x)
        sampler_softmax(x, out)
        ref = _reference(x)
        snr = _snr_db(ref, out)
        # Probabilities must also be a distribution: rows sum to 1.
        rows_ok = torch.allclose(
            out.double().sum(dim=-1),
            torch.ones(batch, device=device, dtype=torch.float64),
            atol=1e-4,
            rtol=0.0,
        )
        passed = bool(torch.allclose(out, ref, atol=1e-6, rtol=1e-3) and rows_ok)
        snr_values.append(snr)
        allclose_values.append(passed)
        print(f"case_snr: {_case_id(batch, vocab)} {snr:.2f}")
        print(f"case_allclose: {_case_id(batch, vocab)} {passed}")
    print(f"SNR: {min(snr_values):.2f} dB")
    print(f"allclose: {all(allclose_values)}")
    return 0


def _run_bench_case(batch: int, vocab: int, warmup: int, iters: int, device: str) -> int:
    x = _make_logits(batch, vocab, device)
    out = torch.empty_like(x)
    ref = _reference(x)

    def step() -> None:
        sampler_softmax(x, out)

    result = cuda_graph_bench(
        step,
        warmup=warmup,
        iters=iters,
        dirty=lambda: out.zero_(),
        verify=lambda: torch.allclose(out, ref, atol=1e-6, rtol=1e-3),
    )

    print(f"# bench mode: {result['mode']}")
    for t in result["times_ms"]:
        print(f"wall_ms: {t:.6f}")
    times = sorted(result["times_ms"])
    print(f"case_ms: {_case_id(batch, vocab)} {times[len(times) // 2]:.6f}")
    return 0


def _run_bench_suite(warmup: int, iters: int, device: str) -> int:
    for batch, vocab in _CASES:
        rc = _run_bench_case(batch, vocab, warmup, iters, device)
        if rc != 0:
            return rc
    return 0


def _run_profile(device: str) -> int:
    batch, vocab = _PROFILE_CASE
    x = _make_logits(batch, vocab, device)
    out = torch.empty_like(x)
    for _ in range(3):
        sampler_softmax(x, out)
    torch.cuda.synchronize()
    for _ in range(3):
        sampler_softmax(x, out)
    torch.cuda.synchronize()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="sglang sampler softmax driver")
    parser.add_argument("--bench-mode", action="store_true")
    parser.add_argument("--profile-run", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    args, _unknown = parser.parse_known_args()

    if not torch.cuda.is_available():
        print("error: no GPU available (torch.cuda.is_available() is False)")
        return 1

    device = "cuda"
    if args.profile_run:
        return _run_profile(device)
    if args.bench_mode:
        return _run_bench_suite(args.warmup, args.iters, device)
    return _run_correctness_suite(device)


if __name__ == "__main__":
    sys.exit(main())
