#!/usr/bin/env python3
"""Gate, not formatter: assert a point really is the configuration we asked for.

Exits non-zero on any mismatch. vLLM port of the GLM-5.2 verify_point.py. Same checks (complete, exact
useful_output_tokens == C*OSL, OSL, ISL or --expect-heterogeneous, topology, local_batch == C/dp,
realized_accept_length == the deterministic constant of the run's parameters, driver exit code 0).
Adapted: topology is vLLM's (tp, dp, ep) and parameterized (default: the llying decode leg DP2 TP1, MoE
tensor-parallel); `moe_a2a_backend` (SGLang) does not exist in vLLM, the bench records
`enable_expert_parallel` instead. Added: graph replay counters for target and draft == verify_iterations on
every rank unless --allow-eager (the fast-path detector the KB handoff names); realized acceptance within
--accept-tolerance of the configured target (vLLM draws acceptance per request, so the realized mean is a
sample of the target, deterministic for a given seed/ISL/OSL/batch but not equal to it); profiled runs fail.
"""
import argparse
import json
import pathlib
import sys

# Deterministic given (accept 3.51 rates, ISL vector, OSL, batch, seed): record it from the first clean run of a
# configuration and pass it with --expect-accept; there is no transferable constant.
EXPECTED_ACCEPT = None


def check(point: pathlib.Path, expect_tp: int = 1, expect_dp: int = 2, expect_ep: int = 1,
          expect_input_len: int = 173000, expect_output_len: int = 1024, expect_heterogeneous: bool = False,
          expect_accept=EXPECTED_ACCEPT, target_accept: float = 3.51, accept_tolerance: float = 0.02,
          allow_eager: bool = False) -> tuple[str, dict]:
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
        problems.append(f"useful_output_tokens {result.get('useful_output_tokens')} != C*OSL {expected_tokens}")
    if result.get("output_len") != expect_output_len:
        problems.append(f"output_len {result.get('output_len')} != {expect_output_len}")
    uniform = result.get("input_len_uniform")
    input_lens = result.get("input_lens")
    if expect_heterogeneous:
        if uniform is not False:
            problems.append(f"expected a heterogeneous batch but input_len_uniform is {uniform}")
        if input_lens is not None and concurrency and len(input_lens) != concurrency:
            problems.append(f"input_lens has {len(input_lens)} entries, expected {concurrency} (global batch)")
    elif result.get("input_len") != expect_input_len:
        hint = " (this run is heterogeneous; pass --expect-heterogeneous)" if uniform is False else ""
        problems.append(f"input_len {result.get('input_len')} != {expect_input_len}{hint}")
    topology = (result.get("tp_size"), result.get("dp_size"), result.get("ep_size"))
    if topology != (expect_tp, expect_dp, expect_ep):
        problems.append(f"topology (tp, dp, ep) {topology} != ({expect_tp}, {expect_dp}, {expect_ep})")
    if result.get("enable_expert_parallel") != (expect_ep > 1):
        problems.append(f"enable_expert_parallel {result.get('enable_expert_parallel')} != {expect_ep > 1}")
    if result.get("local_batch_size") != (concurrency or 0) // expect_dp:
        problems.append(f"local_batch_size is not C/{expect_dp}")
    realized = result.get("realized_accept_length", 0)
    if expect_accept is not None and abs(realized - expect_accept) > 1e-9:
        problems.append(f"realized_accept_length {realized} != {expect_accept}")
    if target_accept is not None and abs(realized - target_accept) > accept_tolerance:
        problems.append(f"realized_accept_length {realized} not within {accept_tolerance} of target {target_accept}")
    iterations = result.get("verify_iterations")
    by_rank = result.get("graph_execution_counts_by_rank") or {}
    if not allow_eager:
        if not by_rank:
            problems.append("graph_execution_counts_by_rank absent")
        for rank, counts in sorted(by_rank.items()):
            for name in ("target", "draft"):
                if counts.get(name) != iterations:
                    problems.append(f"rank {rank} {name} graph replays {counts.get(name)} != iterations {iterations}")
        if result.get("target_graph_iterations") != iterations:
            problems.append(f"target_graph_iterations {result.get('target_graph_iterations')} != {iterations}")
    if result.get("profiled"):
        problems.append("profiled run: not a performance measurement (is_performance_measurement=false)")

    if not status_path.is_file():
        problems.append("launch_status.json absent (unconfirmed driver)")
    else:
        status = json.loads(status_path.read_text())
        if status.get("exit_code") != 0:
            problems.append(f"driver exit_code {status.get('exit_code')} != 0")

    phase = result.get("phase_seconds") or {}
    memory = result.get("memory") or {}
    summary = {
        "concurrency": concurrency,
        "input_len": result.get("input_len"),
        "input_len_uniform": uniform,
        "input_len_min": min(input_lens) if input_lens else None,
        "input_len_max": max(input_lens) if input_lens else None,
        "input_len_mean": sum(input_lens) / len(input_lens) if input_lens else None,
        "input_len_distinct": len(set(input_lens)) if input_lens else None,
        "dp_size": result.get("dp_size"),
        "ep_size": result.get("ep_size"),
        "local_batch_size": result.get("local_batch_size"),
        "tpot_ms": result.get("effective_token_latency_ms_per_user"),
        "step_ms_p50": (result.get("step_seconds_p50") or 0) * 1000 or None,
        "output_tokens_per_second": result.get("output_tokens_per_second"),
        "output_tokens_per_second_per_gpu": result.get("output_tokens_per_second_per_gpu"),
        "verify_iterations": iterations,
        "realized_accept_length": realized,
        "graph_execution_counts": result.get("graph_execution_counts"),
        "decode_seconds": phase.get("decode"),
        "load_pool_capture_seconds": phase.get("load_pool_capture"),
        "reserved_tokens": memory.get("reserved_tokens_per_request_max"),
        "kv_block_utilization": memory.get("kv_block_utilization"),
        "max_memory_allocated_bytes": result.get("max_memory_allocated_bytes"),
        "pinned_vllm": result.get("pinned_vllm"),
        "model_path": result.get("model_path"),
        "problems": problems,
    }
    return ("pass" if not problems else "fail"), summary


def add_expectation_args(parser):
    parser.add_argument("--expect-tp", type=int, default=1)
    parser.add_argument("--expect-dp", type=int, default=2)
    parser.add_argument("--expect-ep", type=int, default=1, help="1, or DP*TP with expert parallelism")
    parser.add_argument("--expect-input-len", type=int, default=173000,
                        help="expected uniform ISL; ignored under --expect-heterogeneous")
    parser.add_argument("--expect-output-len", type=int, default=1024)
    parser.add_argument("--expect-accept", type=float, default=EXPECTED_ACCEPT,
                        help="exact realized_accept_length of this configuration (from its first clean run); omit to skip")
    parser.add_argument("--target-accept", type=float, default=3.51, help="configured synthetic acceptance length")
    parser.add_argument("--accept-tolerance", type=float, default=0.02)
    parser.add_argument("--expect-heterogeneous", action="store_true",
                        help="expect a ragged batch: assert input_len_uniform is False and input_lens covers the global batch")
    parser.add_argument("--allow-eager", action="store_true", help="do not gate graph replay counts (eager runs)")


def expectations(args):
    return dict(expect_tp=args.expect_tp, expect_dp=args.expect_dp, expect_ep=args.expect_ep,
                expect_input_len=args.expect_input_len, expect_output_len=args.expect_output_len,
                expect_heterogeneous=args.expect_heterogeneous, expect_accept=args.expect_accept,
                target_accept=args.target_accept, accept_tolerance=args.accept_tolerance,
                allow_eager=args.allow_eager)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("points", nargs="+", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    add_expectation_args(parser)
    args = parser.parse_args()
    rows, failed = [], False
    for point in args.points:
        verdict, summary = check(point, **expectations(args))
        failed |= verdict != "pass"
        rows.append({"point": str(point), "verdict": verdict, **summary})
    text = json.dumps(rows, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
