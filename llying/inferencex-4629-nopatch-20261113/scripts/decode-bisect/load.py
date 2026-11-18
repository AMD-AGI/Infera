#!/usr/bin/env python3
"""Fixed long-context decode load for one server, then steady-state step time and a torch profile.

Sends N requests of random token ids (same ids for every variant: seeded by request index),
waits until prefill is over, records the server's "Decode batch" lines for --steady seconds,
then asks the server for a --profile-steps step torch profile. Writes summary.json.
"""
import argparse
import json
import os
import random
import re
import statistics as st
import threading
import time
from datetime import datetime

import requests

DECODE = re.compile(
    r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (DP\d+) TP\d+[^\]]*\] Decode batch, #running-req: (\d+), "
    r"#token: (\d+), token usage: ([\d.]+), accept len: ([\d.]+).*?gen throughput \(token/s\): ([\d.]+), "
    r"#queue-req: (\d+)"
)
PREFILL = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (DP\d+)[^\]]*\] Prefill batch")


def ts(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp()


def log(msg):
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} [load] {msg}", flush=True)


class LogTail:
    def __init__(self, path):
        self.f = open(path, errors="ignore")
        self.last_prefill = None
        self.decode = []

    def poll(self):
        for line in self.f:
            m = DECODE.search(line)
            if m:
                t, dp, run, tok, use, acc, gen, q = m.groups()
                self.decode.append(
                    dict(t=ts(t), dp=dp, run=int(run), tok=int(tok), use=float(use), acc=float(acc),
                         gen=float(gen), queue=int(q))
                )
                continue
            m = PREFILL.search(line)
            if m:
                self.last_prefill = ts(m.group(1))


def summarize(rows):
    rows = [r for r in rows if r["gen"] >= 1 and r["run"] >= 1]
    for r in rows:
        r["step_ms"] = r["run"] * r["acc"] / r["gen"] * 1000
    out = {"samples": len(rows)}
    if rows:
        out["step_ms_median"] = st.median(r["step_ms"] for r in rows)
        out["running_median"] = st.median(r["run"] for r in rows)
        out["kv_tokens_median"] = st.median(r["tok"] for r in rows)
        out["accept_len_median"] = st.median(r["acc"] for r in rows)
        out["per_dp"] = {}
        for dp in sorted({r["dp"] for r in rows}):
            xs = [r for r in rows if r["dp"] == dp]
            out["per_dp"][dp] = {
                "samples": len(xs),
                "step_ms_median": st.median(r["step_ms"] for r in xs),
                "running_median": st.median(r["run"] for r in xs),
                "kv_tokens_median": st.median(r["tok"] for r in xs),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--input-len", type=int, default=170000)
    ap.add_argument("--max-new", type=int, default=30000)
    ap.add_argument("--quiet-prefill-s", type=int, default=60)
    ap.add_argument("--prefill-timeout-s", type=int, default=3000)
    ap.add_argument("--steady", type=int, default=180)
    ap.add_argument("--profile-steps", type=int, default=8)
    a = ap.parse_args()

    base = f"http://127.0.0.1:{a.port}"
    vocab = json.load(open("/model/config.json"))["vocab_size"]
    lo, hi = 1000, vocab - 2000
    tail = LogTail(a.log)
    tail.f.seek(0, os.SEEK_END)

    def send(i):
        rng = random.Random(1000 + i)
        ids = [rng.randrange(lo, hi) for _ in range(a.input_len)]
        payload = {"input_ids": ids,
                   "sampling_params": {"max_new_tokens": a.max_new, "ignore_eos": True, "temperature": 0.0}}
        try:
            requests.post(f"{base}/generate", json=payload, timeout=None)
        except Exception:
            pass

    t_send = time.time()
    for i in range(a.n):
        threading.Thread(target=send, args=(i,), daemon=True).start()
    log(f"sent {a.n} requests x {a.input_len} tokens, max_new {a.max_new}")

    while True:
        time.sleep(10)
        tail.poll()
        now = time.time()
        if tail.last_prefill and now - tail.last_prefill >= a.quiet_prefill_s and tail.decode:
            break
        if now - t_send > a.prefill_timeout_s:
            log("prefill phase timed out")
            break
    latest = {}
    for r in tail.decode:
        latest[r["dp"]] = r
    log(f"prefill over after {time.time() - t_send:.0f}s; latest per DP: "
        + ", ".join(f"{dp} run={r['run']} q={r['queue']} tok={r['tok']}" for dp, r in sorted(latest.items())))

    t0 = time.time()
    n0 = len(tail.decode)
    time.sleep(a.steady)
    tail.poll()
    steady = [r for r in tail.decode[n0:] if r["t"] >= t0 - 1]
    summary = {"input_len": a.input_len, "n": a.n, "max_new": a.max_new, "prefill_s": t0 - t_send,
               "steady_s": a.steady, "steady": summarize(steady)}
    log("steady: " + json.dumps({k: v for k, v in summary["steady"].items() if k != "per_dp"}))

    prof_dir = os.path.join(a.out, "profile")
    os.makedirs(prof_dir, exist_ok=True)
    try:
        r = requests.post(f"{base}/start_profile",
                          json={"output_dir": prof_dir, "num_steps": a.profile_steps}, timeout=900)
        log(f"start_profile -> {r.status_code} {r.text[:200]}")
    except Exception as e:
        log(f"start_profile failed: {e}")
    last, stable = -1, 0
    for _ in range(90):
        time.sleep(10)
        size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(prof_dir) for f in fs)
        stable = stable + 1 if size == last and size > 0 else 0
        last = size
        if stable >= 3:
            break
    summary["profile_bytes"] = last
    tail.poll()
    summary["all_decode"] = summarize(list(tail.decode))
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    log(f"profile bytes {last}; summary written")


if __name__ == "__main__":
    main()
