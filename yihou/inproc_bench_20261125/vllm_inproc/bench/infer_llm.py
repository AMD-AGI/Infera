"""vLLM quick inference through the in-process offline engine (vllm.LLM): no API server, no frontend process.

Every engine flag (EngineArgs) works, speculative decoding included. Two modes, same as the SGLang kit:
  generate   : a few prompts (optional chat template) -> completions printed (smoke / sanity)
  throughput : N random-token requests, ISL / OSL, ignore_eos, all at once -> offline output tok/s

  python3 infer_llm.py --model /models/M [engine flags...] generate --prompt "1+1=" --chat
  python3 infer_llm.py --model /models/M [engine flags...] throughput --num 64 --isl 4096 --osl 256 --result-dir R
"""

import json
import sys
import time
from pathlib import Path


def main(argv=None):
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    p = FlexibleArgumentParser(description=__doc__)
    EngineArgs.add_cli_args(p)
    sub = p.add_subparsers(dest="mode", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--prompt", action="append", default=[])
    g.add_argument("--prompt-file")
    g.add_argument("--chat", action="store_true")
    g.add_argument("--max-new-tokens", type=int, default=128)
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--result-dir")
    t = sub.add_parser("throughput")
    t.add_argument("--num", type=int, default=64)
    t.add_argument("--isl", type=int, default=1024)
    t.add_argument("--osl", type=int, default=128)
    t.add_argument("--data-seed", type=int, default=1234)
    t.add_argument("--result-dir", required=True)
    a = p.parse_args(argv)
    ea = EngineArgs.from_cli_args(a)
    t0 = time.perf_counter()
    llm = LLM.from_engine_args(ea)
    startup_s = time.perf_counter() - t0
    if a.mode == "generate":
        prompts = list(a.prompt)
        if a.prompt_file:
            prompts += [l.rstrip("\n") for l in open(a.prompt_file) if l.strip()]
        prompts = prompts or ["The capital of France is"]
        sp = SamplingParams(temperature=a.temperature, max_tokens=a.max_new_tokens)
        t = time.perf_counter()
        if a.chat:
            outs = llm.chat([[{"role": "user", "content": x}] for x in prompts], sp)
        else:
            outs = llm.generate(prompts, sp)
        wall = time.perf_counter() - t
        rows = []
        for x, o in zip(prompts, outs):
            print("=" * 30, "\n[prompt]", x[-300:], "\n[output]", o.outputs[0].text, flush=True)
            rows.append({"prompt": x, "text": o.outputs[0].text, "num_tokens": len(o.outputs[0].token_ids)})
        res = {"mode": "generate", "startup_s": startup_s, "wall_s": wall, "outputs": rows,
                   "complete": len(rows) == len(prompts) and all(r["text"].strip() for r in rows)}
    else:
        import random
        rng = random.Random(a.data_seed)
        prompts = [{"prompt_token_ids": [rng.randrange(0, 10000) for _ in range(a.isl)]} for _ in range(a.num)]
        sp = SamplingParams(temperature=0.0, max_tokens=a.osl, ignore_eos=True)
        t = time.perf_counter()
        outs = llm.generate(prompts, sp)
        wall = time.perf_counter() - t
        out_tok = sum(len(o.outputs[0].token_ids) for o in outs)
        ngpu = (ea.tensor_parallel_size or 1) * (ea.pipeline_parallel_size or 1) * (ea.data_parallel_size or 1)
        res = {"mode": "throughput", "num": a.num, "isl": a.isl, "osl": a.osl, "startup_s": startup_s, "wall_s": wall,
               "completion_tokens": out_tok, "complete": out_tok == a.num * a.osl, "num_gpus": ngpu,
               "output_tok_s": out_tok / wall, "output_tok_s_per_gpu": out_tok / wall / ngpu,
               "total_tok_s": (out_tok + a.num * a.isl) / wall}
        print("[infer_llm] " + json.dumps(res), flush=True)
    if getattr(a, "result_dir", None):
        Path(a.result_dir).mkdir(parents=True, exist_ok=True)
        (Path(a.result_dir) / "result.json").write_text(json.dumps(res, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
