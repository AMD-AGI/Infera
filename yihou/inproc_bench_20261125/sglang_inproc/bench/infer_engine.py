"""SGLang quick inference through the in-process offline Engine (sglang.Engine): no HTTP server, no router.

The Engine is the real serving engine (tokenizer manager + scheduler + detokenizer processes started in-process),
so every server flag works here, speculative decoding and DP attention included. Two modes:

  generate   : run a few prompts (text, optional chat template), print the completions -> smoke / sanity check
  throughput : N requests of random token ids with ISL / OSL and ignore_eos, all submitted at once ->
               offline tok/s of the full engine (scheduler + model + sampling; no network, no client)

Usage (inside the container):
  python3 infer_engine.py --model-path /models/M --tp-size 4 [server flags...] generate --prompt "1+1=" --chat
  python3 infer_engine.py --model-path /models/M [server flags...] throughput --num 64 --isl 4096 --osl 256 --result-dir R
"""

import argparse
import json
import sys
import time
from pathlib import Path


def build_parser():
    from sglang.srt.server_args import ServerArgs
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ServerArgs.add_cli_args(p)
    sub = p.add_subparsers(dest="mode", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--prompt", action="append", default=[])
    g.add_argument("--prompt-file")
    g.add_argument("--chat", action="store_true", help="apply the tokenizer chat template")
    g.add_argument("--max-new-tokens", type=int, default=128)
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--result-dir")
    t = sub.add_parser("throughput")
    t.add_argument("--num", type=int, default=64)
    t.add_argument("--isl", type=int, default=1024)
    t.add_argument("--osl", type=int, default=128)
    t.add_argument("--data-seed", type=int, default=1234)
    t.add_argument("--result-dir", required=True)
    return p


def main(argv=None):
    import sglang as sgl
    from sglang.srt.server_args import ServerArgs

    ns = build_parser().parse_args(argv)
    server_args = ServerArgs.from_cli_args(ns)
    t0 = time.perf_counter()
    engine = sgl.Engine(server_args=server_args)
    startup_s = time.perf_counter() - t0
    try:
        if ns.mode == "generate":
            prompts = list(ns.prompt)
            if ns.prompt_file:
                prompts += [l.rstrip("\n") for l in open(ns.prompt_file) if l.strip()]
            prompts = prompts or ["The capital of France is"]
            if ns.chat:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(server_args.tokenizer_path or server_args.model_path,
                                                    trust_remote_code=server_args.trust_remote_code)
                prompts = [tok.apply_chat_template([{"role": "user", "content": x}], tokenize=False,
                                                   add_generation_prompt=True) for x in prompts]
            sp = {"temperature": ns.temperature, "max_new_tokens": ns.max_new_tokens}
            t = time.perf_counter()
            outs = engine.generate(prompt=prompts, sampling_params=sp)
            wall = time.perf_counter() - t
            rows = []
            for x, o in zip(prompts, outs):
                print("=" * 30, "\n[prompt]", x[-300:], "\n[output]", o["text"], flush=True)
                rows.append({"prompt": x, "text": o["text"], "meta": o.get("meta_info", {})})
            res = {"mode": "generate", "startup_s": startup_s, "wall_s": wall, "outputs": rows,
                   "complete": len(rows) == len(prompts) and all(r["text"].strip() for r in rows)}
        else:
            import random
            rng = random.Random(ns.data_seed)
            ids = [[rng.randrange(0, 10000) for _ in range(ns.isl)] for _ in range(ns.num)]
            sp = {"temperature": 0.0, "max_new_tokens": ns.osl, "ignore_eos": True}
            t = time.perf_counter()
            outs = engine.generate(input_ids=ids, sampling_params=sp)
            wall = time.perf_counter() - t
            out_tok = sum(o["meta_info"]["completion_tokens"] for o in outs)
            in_tok = sum(o["meta_info"]["prompt_tokens"] for o in outs)
            res = {"mode": "throughput", "num": ns.num, "isl": ns.isl, "osl": ns.osl, "startup_s": startup_s,
                   "wall_s": wall, "prompt_tokens": in_tok, "completion_tokens": out_tok,
                   "complete": out_tok == ns.num * ns.osl,
                   "output_tok_s": out_tok / wall, "total_tok_s": (in_tok + out_tok) / wall,
                   "num_gpus": server_args.tp_size * max(1, server_args.pp_size)}
            res["output_tok_s_per_gpu"] = res["output_tok_s"] / res["num_gpus"]
            print("[infer_engine] " + json.dumps({k: v for k, v in res.items() if k != "outputs"}), flush=True)
        if getattr(ns, "result_dir", None):
            Path(ns.result_dir).mkdir(parents=True, exist_ok=True)
            (Path(ns.result_dir) / "result.json").write_text(json.dumps(res, indent=2, default=str))
    finally:
        engine.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
