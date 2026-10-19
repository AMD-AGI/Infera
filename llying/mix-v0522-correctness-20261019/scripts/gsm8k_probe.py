#!/usr/bin/env python3
"""Replay the GSM8K prompts of an lm-eval samples file straight to an engine and report
per request what lm-eval cannot see: finish_reason, completion tokens, reasoning length.

A runaway (finish_reason == "length" at max_tokens) is the defect's signature: the garbled
text sits in reasoning_content, so lm-eval only sees an empty answer.

Usage: gsm8k_probe.py SAMPLES_JSONL ENGINE_URL MODEL OUT_JSONL [--limit 64] [--concurrency 64]
"""

import argparse
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def last_number(text):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text or "")
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def run_one(url, model, messages, max_tokens):
    body = json.dumps({"model": model, "messages": messages, "temperature": 0, "top_p": 1,
                       "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        out = json.load(resp)
    choice = out["choices"][0]
    msg = choice["message"]
    return {"finish_reason": choice.get("finish_reason"),
            "completion_tokens": out.get("usage", {}).get("completion_tokens"),
            "reasoning": msg.get("reasoning_content") or "", "content": msg.get("content") or "",
            "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples")
    ap.add_argument("url")
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=64)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=16384)
    a = ap.parse_args()

    docs = [json.loads(l) for l in open(a.samples) if l.strip()]
    docs = [d for d in docs if d["filter"] == "strict-match"][: a.limit]
    jobs = [(d["doc_id"], json.loads(d["arguments"]["gen_args_0"]["arg_0"][0]),
             d["target"].split("####")[-1].strip().replace(",", "")) for d in docs]

    with ThreadPoolExecutor(a.concurrency) as pool:
        futs = [pool.submit(run_one, a.url, a.model, m, a.max_tokens) for _, m, _ in jobs]
        rows = []
        for (doc_id, _, target), fut in zip(jobs, futs):
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001 -- a failed request is a result here
                r = {"error": repr(e), "finish_reason": None, "completion_tokens": None,
                     "reasoning": "", "content": ""}
            r.update(doc_id=doc_id, target=target, correct=last_number(r["content"]) == target)
            rows.append(r)

    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n = len(rows)
    runaway = sum(r["finish_reason"] == "length" for r in rows)
    errors = sum("error" in r for r in rows)
    correct = sum(r["correct"] for r in rows)
    toks = sorted(r["completion_tokens"] or 0 for r in rows)
    print(f"n={n} correct={correct / n:.3f} runaway={runaway} errors={errors} "
          f"completion_tokens p50={toks[n // 2]} max={toks[-1]}")


if __name__ == "__main__":
    main()
