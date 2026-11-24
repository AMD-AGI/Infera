#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Correctness probes against an OpenAI-compatible endpoint (the infera router).

    accuracy_probe.py gsm8k  URL MODEL OUT_JSONL [--n 200] [--concurrency 32]
    accuracy_probe.py needle URL MODEL OUT_JSONL --ctx-tokens 131072 [--n 32]

gsm8k   zero-shot GSM8K test questions (openai/gsm8k, fetched from the Hugging Face hub into
        HF_HOME, or --gsm8k-file); correct when the answer's last number equals the target.
needle  ~ctx-tokens of random filler per request (no shared prefix) holding four
        "record <name>: access code <code>" facts at random depths; asks for one code. Long
        prompts are where the sparse indexer selects a top-k and where a P/D KV transfer that
        drops or misplaces blocks shows up; GSM8K's ~150-token prompts exercise neither.

Both count what a score alone hides: errors (HTTP / timeouts), runaway (finish_reason=length,
a model looping) and garbled (template tags emitted as text, i.e. broken decoding). The last
line of output is a machine-readable summary.

Run with REAL verification (DSpark block rejection): synthetic acceptance commits draft tokens
without checking them and would score near zero.
"""

import argparse
import json
import random
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

TAGS = ("<think>", "</think>", "<｜tool▁calls▁begin｜>", "<｜tool▁call▁begin｜>", "<｜end▁of▁sentence｜>")
WORDS = ("river mountain window garden morning evening silver golden quiet bright market harbor "
         "station village forest meadow lantern bridge tower valley ocean island desert canyon "
         "teacher farmer sailor painter doctor student baker driver writer singer pilot miner "
         "walked carried painted followed noticed counted gathered opened closed watched built "
         "slowly gently quickly carefully often rarely always never softly loudly early late "
         "red blue green yellow old young tall short warm cold heavy light empty full narrow wide "
         "table chair basket letter candle mirror ladder blanket kettle compass ribbon pebble").split()
NAMES = ("Avery Blake Carter Dana Ellis Finley Gray Harper Indigo Jordan Kendall Logan Morgan "
         "Noel Oakley Parker Quinn Reese Sawyer Taylor").split()


def last_number(text):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text or "")
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def chat(url, model, prompt, max_tokens, timeout):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.load(resp)
    choice, usage = out["choices"][0], out.get("usage", {})
    msg = choice["message"]
    return {"finish_reason": choice.get("finish_reason"), "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
            "content": msg.get("content") or "", "seconds": round(time.time() - t0, 1)}


def gsm8k_jobs(a):
    if a.gsm8k_file:
        rows = [json.loads(line) for line in open(a.gsm8k_file) if line.strip()]
    else:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        path = hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset")
        rows = pq.read_table(path).to_pylist()
    jobs = []
    for i, row in enumerate(rows[: a.n]):
        target = row["answer"].split("####")[-1].strip().replace(",", "")
        prompt = (f"{row['question']}\n\nSolve the problem step by step and end your answer with "
                  "'#### <number>'.")
        jobs.append((i, "gsm8k", prompt, target))
    return jobs


def needle_jobs(a):
    n_words = int(a.ctx_tokens / 1.3)
    jobs = []
    for i in range(a.n):
        rng = random.Random(a.seed * 100003 + i)
        sents, total = [], 0
        while total < n_words:
            k = rng.randint(8, 16)
            s = " ".join(rng.choice(WORDS) for _ in range(k))
            sents.append(s[0].upper() + s[1:] + ".")
            total += k
        names = rng.sample(NAMES, 4)
        codes = [f"{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.randint(2, 9)}-{rng.randint(1000, 9999)}"
                 for _ in names]
        for name, code in zip(names, codes):
            sents.insert(rng.randint(0, len(sents)), f"Record {name}: the access code is {code}.")
        k = rng.randrange(4)
        prompt = " ".join(sents) + f"\n\nWhat is the access code of record {names[k]}? Answer with the code only."
        jobs.append((i, "needle", prompt, codes[k]))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=("gsm8k", "needle"))
    ap.add_argument("url")
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--ctx-tokens", type=int, default=16384)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gsm8k-file", default=None, help="jsonl with question/answer (default: the HF hub)")
    a = ap.parse_args()
    a.n = a.n or (200 if a.task == "gsm8k" else 32)
    jobs = gsm8k_jobs(a) if a.task == "gsm8k" else needle_jobs(a)

    def one(job):
        idx, kind, prompt, target = job
        try:
            r = chat(a.url, a.model, prompt, a.max_tokens, a.timeout)
        except Exception as e:  # noqa: BLE001 -- a failed request is a result here
            r = {"error": repr(e), "finish_reason": None, "prompt_tokens": None, "completion_tokens": None,
                 "reasoning": "", "content": ""}
        ok = (target in r["content"]) if kind == "needle" else (last_number(r["content"]) == target)
        r.update(idx=idx, kind=kind, target=target, correct=ok,
                 garbled=any(t in r["reasoning"] + r["content"] for t in TAGS))
        return r

    rows = []
    with open(a.out, "w") as f, ThreadPoolExecutor(a.concurrency) as pool:
        for fut in as_completed([pool.submit(one, job) for job in jobs]):
            r = fut.result()
            rows.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
    n = len(rows)
    pt = sorted(r["prompt_tokens"] or 0 for r in rows)
    ct = sorted(r["completion_tokens"] or 0 for r in rows)
    label = a.task if a.task == "gsm8k" else f"needle@{a.ctx_tokens // 1024}k"
    print(f"{label}: n={n} correct={sum(r['correct'] for r in rows)} accuracy={sum(r['correct'] for r in rows) / n:.3f} "
          f"errors={sum('error' in r for r in rows)} runaway={sum(r['finish_reason'] == 'length' for r in rows)} "
          f"garbled={sum(r['garbled'] for r in rows)} prompt_tokens_p50={pt[n // 2]} completion_tokens_p50={ct[n // 2]} "
          f"completion_tokens_max={ct[-1]}")


if __name__ == "__main__":
    main()
