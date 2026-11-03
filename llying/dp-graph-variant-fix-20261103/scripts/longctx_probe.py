#!/usr/bin/env python3
"""Long-context correctness probe: every request is long enough (default ~16k tokens) for the DSA
indexer to select a sparse top-k (it does so above 2048 tokens), unlike GSM8K's ~1k-token prompts.

Each request gets its own random filler text (no shared prefix) and one of two tasks:
  needle  -- four "record <name>: access code <code>" facts at random depths; asks for one code,
             correct when the code appears verbatim in the answer;
  gsm8k   -- a GSM8K question after the filler; correct when the last number equals the target.
garbled / runaway are counted as in gsm8k_probe.py (template tags as text; max_tokens reached).

Usage: longctx_probe.py SAMPLES_JSONL ENGINE_URL MODEL OUT_JSONL [--n 128] [--concurrency 32]
                        [--ctx-tokens 16000] [--max-tokens 8192] [--seed 0]
"""

import argparse
import json
import random
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

TAGS = ("<think>", "</think>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>",
        "<tool_call>", "</tool_call>")
WORDS = ("river mountain window garden morning evening silver golden quiet bright market harbor "
         "station village forest meadow lantern bridge tower valley ocean island desert canyon "
         "teacher farmer sailor painter doctor student baker driver writer singer pilot miner "
         "walked carried painted followed noticed counted gathered opened closed watched built "
         "slowly gently quickly carefully often rarely always never softly loudly early late "
         "red blue green yellow old young tall short warm cold heavy light empty full narrow wide "
         "table chair basket letter candle mirror ladder blanket kettle compass ribbon pebble").split()
NAMES = ("Avery Blake Carter Dana Ellis Finley Gray Harper Indigo Jordan Kendall Logan Morgan "
         "Noel Oakley Parker Quinn Reese Sawyer Taylor").split()


def filler(rng, n_words):
    out, total = [], 0
    while total < n_words:
        k = rng.randint(8, 16)
        s = " ".join(rng.choice(WORDS) for _ in range(k))
        out.append(s[0].upper() + s[1:] + ".")
        total += k
    return out


def needle_prompt(rng, n_words):
    sents = filler(rng, n_words)
    names = rng.sample(NAMES, 4)
    codes = [f"{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.randint(2, 9)}-{rng.randint(1000, 9999)}"
             for _ in names]
    for name, code in zip(names, codes):
        sents.insert(rng.randint(0, len(sents)), f"Record {name}: the access code is {code}.")
    k = rng.randrange(4)
    question = (f"\n\nWhat is the access code of record {names[k]}? "
                "Answer with the code only.")
    return " ".join(sents) + question, codes[k]


def gsm8k_prompt(rng, n_words, question, target):
    text = " ".join(filler(rng, n_words))
    return (text + "\n\nIgnore the text above. Solve the following problem and end your answer with "
            f"'#### <number>'.\nQuestion: {question}\nAnswer:"), target


def last_number(text):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text or "")
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def run_one(url, model, prompt, max_tokens):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0, "top_p": 1, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as resp:
        out = json.load(resp)
    choice, usage = out["choices"][0], out.get("usage", {})
    return {"finish_reason": choice.get("finish_reason"), "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning": choice["message"].get("reasoning_content") or "",
            "content": choice["message"].get("content") or "", "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples")
    ap.add_argument("url")
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--ctx-tokens", type=int, default=16000)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    docs = [json.loads(l) for l in open(a.samples) if l.strip()]
    docs = [d for d in docs if d["filter"] == "strict-match"]
    n_words = int(a.ctx_tokens / 1.3)
    jobs = []
    for i in range(a.n):
        rng = random.Random(a.seed * 100003 + i)
        if i % 2 == 0:
            prompt, target = needle_prompt(rng, n_words)
            jobs.append((i, "needle", prompt, target))
        else:
            d = docs[(i // 2) % len(docs)]
            prompt, target = gsm8k_prompt(rng, n_words, d["doc"]["question"],
                                          d["target"].split("####")[-1].strip().replace(",", ""))
            jobs.append((i, "gsm8k", prompt, target))

    def one(job):
        idx, kind, prompt, target = job
        try:
            r = run_one(a.url, a.model, prompt, a.max_tokens)
        except Exception as e:  # noqa: BLE001 -- a failed request is a result here
            r = {"error": repr(e), "finish_reason": None, "prompt_tokens": None,
                 "completion_tokens": None, "reasoning": "", "content": ""}
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

    for kind in ("needle", "gsm8k"):
        rs = [r for r in rows if r["kind"] == kind]
        pt = sorted(r["prompt_tokens"] or 0 for r in rs)
        print(f"{kind}: n={len(rs)} correct={sum(r['correct'] for r in rs)} "
              f"garbled={sum(r['garbled'] for r in rs)} "
              f"runaway={sum(r['finish_reason'] == 'length' for r in rs)} "
              f"errors={sum('error' in r for r in rs)} prompt_tokens p50={pt[len(pt) // 2]}")


if __name__ == "__main__":
    main()
