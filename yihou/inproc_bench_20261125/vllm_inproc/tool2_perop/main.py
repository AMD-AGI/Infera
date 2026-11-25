###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################
# Provenance: adapted from AKLR `aklr/benchmarks/models/main.py` @ 33689332 (research/aklr_src/main.py).
# vLLM adaptation: the in-process engine (LLM over external_launcher, one process per GPU under torchrun,
# the preset's own cuda-graph config) runs real text through the real scheduler: a real chunked prefill,
# then 256 decode steps. vLLM offers no teacher forcing through its engine API, so the decode tokens are the
# model's own greedy tokens with ignore_eos (same shapes per step as AKLR's forced ones; token values differ).
# TTFT = wall of a max_tokens=1 generate; step = (wall(257 tokens) - wall(1 token)) / 256. --text-file reads a
# local UTF-8 text (sha256 logged), as the SGLang port does for offline nodes.

"""e2e stage on real text: ``torchrun ... main.py dsv41 <decode|prefill> [EngineArgs] --text-file F --result-json P``."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

DECODE_STEPS = 256


def main():
    here = Path(__file__).resolve().parent
    models = sorted(p.parent.name for p in here.glob("*/model_bench.py"))
    head = argparse.ArgumentParser(add_help=False)
    head.add_argument("model", choices=models)
    sys.path.insert(0, str(here / head.parse_known_args()[0].model))
    sys.path.insert(0, str(here))
    import model_bench as mb
    import vllm_bench as vb

    parser = mb.make_parser()
    parser.add_argument("model_dir", choices=models, metavar="model")
    parser.add_argument("phase", choices=["decode", "prefill"])
    parser.add_argument("--CONC", type=int, default=1, help="decode: concurrent requests, global over DP ranks")
    parser.add_argument("--text-file", required=True, help="local UTF-8 text (e.g. one PG-19 book)")
    parser.add_argument("--result-json", required=True)
    args = parser.parse_args()
    if str(args.load_format) == "dummy":
        raise SystemExit("e2e performance requires real weights")
    if args.speculative_config:
        raise SystemExit("e2e does not run speculative decoding's verify shape")
    mb.setup_env()
    dp = args.data_parallel_size or 1
    n = max(1, args.CONC // dp) if args.phase == "decode" else 1
    llm, runner, rank_print = mb.setup(mb.build_engine_args(args, n, args.max_num_batched_tokens or 16384, cuda_graph=True))
    from vllm import SamplingParams, TokensPrompt

    raw = Path(args.text_file).read_bytes()
    rank_print(f"text {args.text_file} sha256 {hashlib.sha256(raw).hexdigest()}")
    tokens = llm.get_tokenizer().encode(raw.decode("utf-8"), add_special_tokens=False)
    if args.ISL > len(tokens):
        raise SystemExit(f"--ISL {args.ISL:,} exceeds the {len(tokens):,}-token text")
    rank = vb.world_rank()
    prompts = [TokensPrompt(prompt_token_ids=(tokens + tokens)[s: s + args.ISL])
               for s in ((i + rank * n) * len(tokens) // (n * dp) for i in range(n))]

    def run(max_tokens):
        sp = SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True)
        vb.barrier()
        t = time.perf_counter()
        llm.generate(prompts, sp, use_tqdm=False)
        return (time.perf_counter() - t) * 1000.0

    run(2)  # warm-up (JIT, graphs)
    ttft = run(1)
    total = run(DECODE_STEPS + 1)
    step = (total - ttft) / DECODE_STEPS
    rank_print(f"e2e: TTFT {ttft:.1f} ms ({n} request(s)/rank), decode step {step:.3f} ms")
    if rank == 0:
        res = ({"tokens_per_s": args.CONC * 1000 / step, "latency_ms": step} if args.phase == "decode"
               else {"tokens_per_s": args.ISL * 1000 / ttft, "latency_ms": ttft})
        Path(args.result_json).write_text(json.dumps(res))


if __name__ == "__main__":
    main()
