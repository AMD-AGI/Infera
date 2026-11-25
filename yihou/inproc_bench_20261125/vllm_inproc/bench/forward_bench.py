"""vLLM forward bench wrapper: vLLM's own `vllm bench forward` (vllm/benchmarks/forward, shipped in the image)
with kit conventions: --result-dir, case presets, provenance-friendly result.json.

What upstream does (code read of the pinned image, vllm 0.34.0rc1.dev897+g9c1d45c21): builds exact batches
WITHOUT the scheduler, runs them through the real GPUModelRunner.execute_model (input prep, attention
metadata, DP sync), captures target forward + sampler (+ the configured MTP/DSpark drafter, assuming every
draft is accepted) as a one-shot CUDA graph and times replays. KV contents are zeros; MoE routing is real
for real weights (or balanced / zipf fixtures). CPU step overhead is NOT measured (see engine_step_bench.py).

Single GPU: python3 forward_bench.py --result-dir R --model M --case decode=32*1@8192
Multi-GPU (TP/DP/EP): TORCHRUN_NPROC=N via scripts/run_point.sh (torchrun, one process per GPU).
--case-file F : one case per line (`NAME=REQS`, '#' comments), e.g. presets/dsv41_agentx_cases.txt
"""

import json
import os
import sys
from pathlib import Path


def read_case_file(path):
    cases = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            cases.append(line)
    return cases


def main(argv=None):
    from vllm.benchmarks.forward import main as fwd
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    p = FlexibleArgumentParser(description=__doc__)
    p.add_argument("--result-dir", required=True)
    p.add_argument("--case-file", action="append", default=[])
    p.add_argument("--profile", action="store_true", help="trace a few replays per case into result-dir/profile")
    fwd.add_cli_args(p)
    a = p.parse_args(argv)
    out = Path(a.result_dir)
    out.mkdir(parents=True, exist_ok=True)
    for f in a.case_file:
        a.case += read_case_file(f)
    a.output_json = str(out / "forward.json")
    if a.profile:
        a.profile_dir = str(out / "profile")
    fwd.main(a)
    rank = int(os.environ.get("RANK", "0"))
    if rank != 0:
        return 0
    rows = json.loads((out / "forward.json").read_text())
    world = int(os.environ.get("WORLD_SIZE", "1"))
    gpus_per_replica = (a.tensor_parallel_size or 1) * (a.pipeline_parallel_size or 1)  # one DP replica runs one batch
    cases = []
    for r in rows:
        rank0 = r["ranks"][0]
        reqs = r["requests"]
        n_out = sum(1 + q["num_draft_tokens"] if q["query_len"] == 1 + q["num_draft_tokens"] and q["kv_len"] > q["query_len"]
                    else 0 for q in reqs)  # decode/verify rows emit 1 + accepted drafts (all accepted in this bench)
        med_s = r["median_us"] / 1e6
        cases.append({
            "case": r["case"], "num_requests": len(reqs), "num_tokens": rank0["num_tokens"],
            "median_us": r["median_us"], "sample_spread_pct": rank0["latency_us"].get("sample_spread_pct"),
            "graph_vs_eager_rel_rms": rank0.get("graph_vs_eager", {}).get("relative_rms"),
            "forward_tok_s_per_replica": rank0["num_tokens"] / med_s,
            "forward_tok_s_per_gpu": rank0["num_tokens"] / med_s / gpus_per_replica,
            "decode_out_tok_s_per_replica_all_accepted": (n_out / med_s) if n_out else None,
            "decode_out_tok_s_per_gpu_all_accepted": (n_out / med_s / gpus_per_replica) if n_out else None,
            "routing": rank0.get("routing"),
        })
    res = {"engine": "vllm", "tool": "forward_bench", "complete": len(cases) == len(a.case) and len(cases) > 0,
           "world_size": world, "gpus_per_replica": gpus_per_replica, "model": a.model, "load_format": a.load_format, "cases": cases}
    (out / "result.json").write_text(json.dumps(res, indent=2))
    for c in cases:
        print("[forward_bench] " + json.dumps(c), flush=True)
    return 0 if res["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
