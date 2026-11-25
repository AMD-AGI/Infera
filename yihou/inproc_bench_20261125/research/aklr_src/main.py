###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""The e2e stage's run on real text, the same for every model:
``python main.py <model> <phase> [the model's flags] --result-json PATH``."""

import argparse
import hashlib
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from sglang_bench import (
    bench_eager,
    forward_batch_of,
    forward_ctx,
    local_conc,
    prefill_conc,
    prepare_decode,
)

# Teacher-forced steps timed after prefill.
DECODE_STEPS = 256


@dataclass(frozen=True)
class HubText:
    """One text of a parquet file in an HF dataset repo, pinned by revision and hash;
    read from the HF cache like the weights, so CI's offline cache must hold it."""

    repo: str
    file: str
    revision: str
    # the row's ``url`` column, which picks the text out of the file
    url: str
    sha256: str

    def read(self) -> str:
        import pyarrow.parquet
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(self.repo, self.file, repo_type="dataset", revision=self.revision)
        rows = pyarrow.parquet.read_table(path, columns=["url", "text"]).to_pylist()
        text = next(row["text"] for row in rows if row["url"] == self.url)
        if hashlib.sha256(text.encode()).hexdigest() != self.sha256:
            raise ValueError(f"{path}: {self.url} is not the pinned text")
        return text


TEXTS = {
    # One PG-19 test book (PG-19 is Apache-2.0, the book public domain), about 340k tokens.
    "pg19-3608": HubText(
        repo="emozilla/pg19-test",
        file="data/test-00000-of-00001-29a571947c0b5ccc.parquet",
        revision="c5e39bf32e33f9111323aa68d7d9000d22722035",
        url="http://www.gutenberg.org/ebooks/3608",
        sha256="51978ae38dd137023fb6bf13b6400dcf2bf02ab5556a23e811eeb61cb720e4ba",
    ),
}


def work(server_args, port_args, args, gpu_id, tp_rank):
    """Time one model on real text: a real chunked prefill, then teacher-forced
    decode steps. Rank 0 writes ``--result-json`` for ``--phase``."""
    # Imported here, not at module level: a spawned rank re-imports this file without
    # running __main__, though it inherits the sys.path that finds the model's directory.
    import model_bench
    from sglang.benchmark.one_batch import prepare_synthetic_inputs_for_latency_test
    from transformers import AutoTokenizer

    # setup returns (model_runner, [spec,] rank_print), by model
    mr, *_, rank_print = model_bench.setup(server_args, port_args, gpu_id, tp_rank)
    tokenizer = AutoTokenizer.from_pretrained(
        server_args.tokenizer_path or server_args.model_path,
        trust_remote_code=server_args.trust_remote_code,
    )
    tokens = tokenizer.encode(TEXTS[args.text].read(), add_special_tokens=False)
    length = args.ISL + DECODE_STEPS
    n_reqs = local_conc(args, mr)
    if length > len(tokens):
        raise ValueError(f"--ISL {args.ISL:,} exceeds the {len(tokens):,}-token text")
    if length * n_reqs > mr.max_total_num_tokens:
        raise ValueError(
            f"{n_reqs} requests of {length:,} tokens exceed the KV pool's"
            f" {mr.max_total_num_tokens:,}; raise --mem-fraction-static"
        )
    # Each request reads the text from its own offset, wrapping round.
    ids = [
        (tokens + tokens)[start : start + length]
        for start in (i * len(tokens) // n_reqs for i in range(n_reqs))
    ]
    reqs = prepare_synthetic_inputs_for_latency_test(
        n_reqs, args.ISL, custom_inputs=[x[: args.ISL] for x in ids]
    )

    with forward_ctx(mr):
        rank_print(f"e2e: prefill {args.ISL:,} tokens of real text")
        # The scheduler admits chunk_tokens new tokens per forward across all requests.
        page = getattr(mr, "page_size", 1) or 1
        per_req = model_bench.chunk_tokens(args, mr) // len(reqs)
        budget = max(page, per_req // page * page)
        ttft_ms = 0.0
        for prefix in range(0, args.ISL, budget):
            end = min(prefix + budget, args.ISL)
            # Keep each request's row: SGLang reads the next chunk's prefix from it.
            for req in reqs:
                req.prefix_len = prefix
                req.extend_end = end
            batch = model_bench.make_extend_batch(reqs, mr, reset=prefix == 0)
            fb = forward_batch_of(batch, mr)
            # Rerunning a chunk rewrites the same KV slots, so the warm-up passes are harmless.
            ttft_ms += bench_eager(lambda fb=fb: mr.forward(fb), reps=1)

        rank_print(f"e2e: {DECODE_STEPS} teacher-forced decode steps")
        step_ms = []
        for step in range(DECODE_STEPS):
            forced = [x[args.ISL + step] for x in ids]
            prepare_decode(batch, mr, torch.tensor(forced, device=mr.device))
            fb = forward_batch_of(batch, mr)
            torch.cuda.synchronize()
            start = time.perf_counter()
            mr.forward(fb)
            torch.cuda.synchronize()
            step_ms.append((time.perf_counter() - start) * 1000.0)

    if tp_rank != 0:
        return
    out = Path(args.result_json)
    step = statistics.median(step_ms)
    rank_print(f"e2e: TTFT {ttft_ms:.1f} ms, median step {step:.3f} ms")
    if args.phase == "decode":
        result = {"tokens_per_s": args.CONC * 1000 / step, "latency_ms": step}
    else:
        result = {"tokens_per_s": args.ISL * 1000 / ttft_ms, "latency_ms": ttft_ms}
    out.write_text(json.dumps(result))


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    models = sorted(path.parent.name for path in here.glob("*/model_bench.py"))
    # The model picks the parser, so it is read first.
    head = argparse.ArgumentParser(add_help=False)
    head.add_argument("model", choices=models)
    sys.path.insert(0, str(here / head.parse_known_args()[0].model))
    import model_bench

    parser = model_bench.make_parser()
    parser.add_argument("model", choices=models, help="a directory beside this file")
    parser.add_argument("phase", choices=["decode", "prefill"], help="the throughput reported")
    parser.add_argument("--CONC", type=int, default=1, help="decode: concurrent requests, global")
    parser.add_argument("--text", choices=TEXTS, default="pg19-3608")
    parser.add_argument(
        "--result-json",
        metavar="PATH",
        required=True,
        help='write {"tokens_per_s", "latency_ms"} to PATH',
    )
    args = parser.parse_args()
    # Dummy weights distort data-dependent timings; MTP uses a different verify shape.
    if args.load_format == "dummy":
        raise SystemExit("e2e performance requires real weights")
    if args.speculative_algorithm is not None:
        raise SystemExit("e2e does not run speculative decoding's verify shape")
    if args.phase == "prefill":
        args.CONC = prefill_conc(args)
    model_bench.run(work, model_bench.build_server_args(args, cuda_graph=True), args)
