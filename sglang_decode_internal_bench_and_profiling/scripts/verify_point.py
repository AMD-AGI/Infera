#!/usr/bin/env python3
"""Gate, not formatter: assert a TP8 / DP-attention point really is the configuration we asked for.

Exits non-zero on any mismatch. Mirrors the packup's collect_noep_sweep_yihou.py checks; the
expected EP size is a parameter (8 for the expert-parallel sweep, 1 for the EP-off sweep) while
tp=8, dp=8 and dpa=True are fixed for this workspace.
"""
import argparse
import json
import pathlib
import sys

EXPECTED_ACCEPT = 3.6134393063583814


def check(point: pathlib.Path, expect_ep: int = 8, expect_input_len: int = 70000,
          expect_output_len: int = 10000, expect_heterogeneous: bool = False,
          expect_accept: float = EXPECTED_ACCEPT) -> tuple[str, dict]:
    problems = []
    result_path = point / "result_yihou.json"
    status_path = point / "launch_status.json"
    if not result_path.is_file():
        return "missing", {"problems": ["result_yihou.json absent"]}
    result = json.loads(result_path.read_text())

    if not result.get("complete"):
        problems.append("complete is not true")
    concurrency = result.get("batch_size")
    expected_tokens = (concurrency or 0) * result.get("output_len", 0)
    if result.get("useful_output_tokens") != expected_tokens:
        problems.append(
            f"useful_output_tokens {result.get('useful_output_tokens')} != C*OSL {expected_tokens}"
        )
    if result.get("output_len") != expect_output_len:
        problems.append(f"output_len {result.get('output_len')} != {expect_output_len}")
    # A ragged run reports input_len as None on purpose (no single number is "the" ISL), so the
    # uniform gate below would fail it with a message about 70000 that explains nothing. Branch
    # explicitly and say which mode was expected.
    uniform = result.get("input_len_uniform")  # absent in results predating heterogeneous ISL
    input_lens = result.get("input_lens")
    if expect_heterogeneous:
        if uniform is not False:
            problems.append(f"expected a heterogeneous batch but input_len_uniform is {uniform}")
        if input_lens is not None and concurrency and len(input_lens) != concurrency:
            problems.append(f"input_lens has {len(input_lens)} entries, expected {concurrency} (global batch)")
    elif result.get("input_len") != expect_input_len:
        hint = " (this run is heterogeneous; pass --expect-heterogeneous)" if uniform is False else ""
        problems.append(f"input_len {result.get('input_len')} != {expect_input_len}{hint}")
    topology = (
        result.get("ep_size"),
        result.get("tp_size"),
        result.get("dp_size"),
        result.get("enable_dp_attention"),
    )
    if topology != (expect_ep, 8, 8, True):
        problems.append(f"topology {topology} != (ep={expect_ep}, tp=8, dp=8, dpa=True)")
    if result.get("moe_a2a_backend") != "none":
        problems.append(f"moe_a2a_backend {result.get('moe_a2a_backend')} != none")
    if result.get("local_batch_size") != (concurrency or 0) // 8:
        problems.append("local_batch_size is not C/8")
    # The default is the published 70000/10000 point. It is deterministic given the parameters but
    # NOT given only --accept-length: it also depends on how many iterations the run takes, so any
    # other OSL has its own value. Parameterised rather than hardcoded so the gate is usable at a
    # configuration other than the one it was born for.
    if abs(result.get("realized_accept_length", 0) - expect_accept) > 1e-9:
        problems.append(
            f"realized_accept_length {result.get('realized_accept_length')} != {expect_accept}"
        )

    if not status_path.is_file():
        problems.append("launch_status.json absent (unconfirmed driver)")
    else:
        status = json.loads(status_path.read_text())
        if status.get("exit_code") != 0:
            problems.append(f"driver exit_code {status.get('exit_code')} != 0")

    summary = {
        "concurrency": concurrency,
        "input_len": result.get("input_len"),
        "input_len_uniform": uniform,
        "input_len_min": min(input_lens) if input_lens else None,
        "input_len_max": max(input_lens) if input_lens else None,
        "input_len_distinct": len(set(input_lens)) if input_lens else None,
        "ep_size": result.get("ep_size"),
        "local_batch_size": result.get("local_batch_size"),
        "tpot_ms": result.get("effective_token_latency_ms_per_user"),
        "output_tokens_per_second": result.get("output_tokens_per_second"),
        "output_tokens_per_second_per_gpu": result.get("output_tokens_per_second_per_gpu"),
        "verify_iterations": result.get("verify_iterations"),
        "realized_accept_length": result.get("realized_accept_length"),
        "decode_seconds": (result.get("phase_seconds") or {}).get("decode"),
        "load_pool_capture_seconds": (result.get("phase_seconds") or {}).get("load_pool_capture"),
        "reserved_tokens": (result.get("memory") or {}).get("reserved_tokens"),
        "max_memory_allocated_bytes": result.get("max_memory_allocated_bytes"),
        "pinned_sglang": result.get("pinned_sglang"),
        "model_path": result.get("model_path"),
        "problems": problems,
    }
    return ("pass" if not problems else "fail"), summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("points", nargs="+", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--expect-ep", type=int, default=8, help="expected ep_size (8 or 1)")
    parser.add_argument("--expect-input-len", type=int, default=70000,
                        help="expected uniform ISL; ignored under --expect-heterogeneous")
    parser.add_argument("--expect-output-len", type=int, default=10000)
    parser.add_argument("--expect-accept", type=float, default=EXPECTED_ACCEPT,
                        help="expected realized_accept_length; the default is the published "
                             "70000/10000 point and does not transfer to another output length")
    parser.add_argument("--expect-heterogeneous", action="store_true",
                        help="expect a ragged batch: assert input_len_uniform is False and that "
                             "input_lens covers the global batch, instead of pinning one ISL")
    args = parser.parse_args()

    rows, failed = [], False
    for point in args.points:
        verdict, summary = check(point, expect_ep=args.expect_ep,
                                 expect_input_len=args.expect_input_len,
                                 expect_output_len=args.expect_output_len,
                                 expect_heterogeneous=args.expect_heterogeneous,
                                 expect_accept=args.expect_accept)
        failed |= verdict != "pass"
        rows.append({"point": str(point), "verdict": verdict, **summary})

    text = json.dumps(rows, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
