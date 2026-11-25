#!/usr/bin/env python3
"""Gate, not formatter: assert a point really is the configuration we asked for.

Exits non-zero on any mismatch. Port of the GLM-5.2 verify_point.py. Same checks (complete, exact
useful_output_tokens == C*OSL, OSL, ISL or --expect-heterogeneous, topology, moe_a2a_backend,
local_batch == C/dp, realized_accept_length == the deterministic constant, driver exit code 0).
Adapted: the topology gate is parameterized (default: the #5034 shape TP4 / DPA4 / EP4) instead of
fixed TP8/DP8, and the default ISL/OSL/accept are this port's reference point. Added (the KB handoff
names it as the fast-path detector, the GLM script did not gate it): graph execute counts for target
and draft == verify_iterations on every rank unless --allow-eager.
"""
import argparse
import json
import pathlib
import sys

# Deterministic given (accept 3.51, OSL 1024, seed 1234, match-expected): measured on the first clean full run of
# this port (tp4_ep4_dpa_on_c64_yihou, 288 iterations) and reproduced bit for bit by t3_h1_hetero and
# tp4_noep_dpa_on_c64_yihou (VALIDATION_tool3.md). Any other OSL / accept / seed has its own value
# (OSL 4096: 3.520618556701031 over 1164 iterations); pass --expect-accept for those.
EXPECTED_ACCEPT = 3.5625


def check(point: pathlib.Path, expect_tp: int = 4, expect_dp: int = 4, expect_ep: int = 4, expect_dpa: bool = True,
          expect_input_len: int = 8192, expect_output_len: int = 1024, expect_heterogeneous: bool = False,
          expect_accept=EXPECTED_ACCEPT, allow_eager: bool = False) -> tuple[str, dict]:
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
    topology = (result.get("ep_size"), result.get("tp_size"), result.get("dp_size"), result.get("enable_dp_attention"))
    if topology != (expect_ep, expect_tp, expect_dp, expect_dpa):
        problems.append(f"topology {topology} != (ep={expect_ep}, tp={expect_tp}, dp={expect_dp}, dpa={expect_dpa})")
    if result.get("moe_a2a_backend") != "none":
        problems.append(f"moe_a2a_backend {result.get('moe_a2a_backend')} != none")
    if result.get("local_batch_size") != (concurrency or 0) // expect_dp:
        problems.append(f"local_batch_size is not C/{expect_dp}")
    if expect_accept is not None and abs(result.get("realized_accept_length", 0) - expect_accept) > 1e-9:
        problems.append(f"realized_accept_length {result.get('realized_accept_length')} != {expect_accept}")
    iterations = result.get("verify_iterations")
    by_rank = result.get("graph_execution_counts_by_rank") or {}
    if not allow_eager:
        for rank, counts in sorted(by_rank.items()):
            for name in ("target", "draft"):
                if counts.get(name) != iterations:
                    problems.append(f"rank {rank} {name} graph executes {counts.get(name)} != iterations {iterations}")
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
        "verify_iterations": iterations,
        "realized_accept_length": result.get("realized_accept_length"),
        "graph_execution_counts": result.get("graph_execution_counts"),
        "decode_seconds": phase.get("decode"),
        "load_pool_capture_seconds": phase.get("load_pool_capture"),
        "reserved_tokens": (result.get("memory") or {}).get("reserved_tokens"),
        "max_memory_allocated_bytes": result.get("max_memory_allocated_bytes"),
        "pinned_sglang": result.get("pinned_sglang"),
        "model_path": result.get("model_path"),
        "problems": problems,
    }
    return ("pass" if not problems else "fail"), summary


def add_expectation_args(parser):
    parser.add_argument("--expect-tp", type=int, default=4)
    parser.add_argument("--expect-dp", type=int, default=4, help="attention-DP size (1 with DPA off)")
    parser.add_argument("--expect-ep", type=int, default=4)
    parser.add_argument("--expect-no-dpa", action="store_true", help="expect enable_dp_attention False")
    parser.add_argument("--expect-input-len", type=int, default=8192,
                        help="expected uniform ISL; ignored under --expect-heterogeneous")
    parser.add_argument("--expect-output-len", type=int, default=1024)
    parser.add_argument("--expect-accept", type=float, default=EXPECTED_ACCEPT,
                        help="expected realized_accept_length (deterministic per accept/OSL/seed); omit to skip")
    parser.add_argument("--expect-heterogeneous", action="store_true",
                        help="expect a ragged batch: assert input_len_uniform is False and input_lens covers the global batch")
    parser.add_argument("--allow-eager", action="store_true", help="do not gate graph execute counts (eager runs)")


def expectations(args):
    return dict(expect_tp=args.expect_tp, expect_dp=args.expect_dp, expect_ep=args.expect_ep,
                expect_dpa=not args.expect_no_dpa, expect_input_len=args.expect_input_len,
                expect_output_len=args.expect_output_len, expect_heterogeneous=args.expect_heterogeneous,
                expect_accept=args.expect_accept, allow_eager=args.allow_eager)


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
