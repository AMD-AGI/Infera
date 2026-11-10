#!/usr/bin/env python3
"""Fixed-concurrency decode load for a decode trace without AgentX: every stream sends one chat request after another
(random ~PROMPT_TOKENS prompt, ignore_eos, MAX_TOKENS out), so the decode keeps ~CONC requests running. The kit's
simulated acceptance (SIM_ACCEPT_LEN) makes the MTP accept length independent of the text. Stages raise the
concurrency in steps (streams beyond the current stage's CONC wait); the process exits after the last stage.
AgentX needs ~40 min (cold prefill of 222 long contexts + cache warm-up) before its decode batch is representative,
longer than the 30 min a job is exempt from preemption on AUS; this load gets there within a minute or two.

Usage: synth_decode_load.py ROUTER_URL MODEL --stages 96:150,160:150,192:150 [--prompt-tokens 2048] [--max-tokens 4096]
"""

import argparse
import json
import random
import threading
import time
import urllib.request

LETTERS = "abcdefghijklmnopqrstuvwxyz"


def prompt(rng, n_tokens):
    return " ".join("".join(rng.choice(LETTERS) for _ in range(rng.randint(3, 7))) for _ in range(n_tokens // 2))


def stream(i, url, model, a, active, stop, stats, lock):
    rng = random.Random(i)
    while not stop.is_set():
        if i >= active[0]:
            time.sleep(1)
            continue
        body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt(rng, a.prompt_tokens)}],
                           "max_tokens": a.max_tokens, "ignore_eos": True, "temperature": 0}).encode()
        req = urllib.request.Request(url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=900) as resp:
                n = json.load(resp).get("usage", {}).get("completion_tokens", 0)
            with lock:
                stats["done"] += 1
                stats["tokens"] += n
        except Exception as e:  # noqa: BLE001
            with lock:
                stats["errors"] += 1
                stats["last_error"] = f"{type(e).__name__}: {str(e)[:120]}"
            time.sleep(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("model")
    ap.add_argument("--stages", required=True, help="CONC:SECONDS,...")
    ap.add_argument("--prompt-tokens", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=4096)
    a = ap.parse_args()
    stages = [tuple(int(v) for v in s.split(":")) for s in a.stages.split(",")]
    active, stop, lock = [0], threading.Event(), threading.Lock()
    stats = {"done": 0, "tokens": 0, "errors": 0, "last_error": ""}
    threads = [threading.Thread(target=stream, args=(i, a.url, a.model, a, active, stop, stats, lock), daemon=True)
               for i in range(max(c for c, _ in stages))]
    for t in threads:
        t.start()
    for conc, secs in stages:
        active[0] = conc
        print(f"{time.strftime('%H:%M:%S', time.gmtime())} stage conc={conc} for {secs}s", flush=True)
        t_end = time.time() + secs
        while time.time() < t_end:
            time.sleep(min(30, max(0, t_end - time.time())))
            with lock:
                print(f"{time.strftime('%H:%M:%S', time.gmtime())}   conc={conc} done={stats['done']} "
                      f"tokens={stats['tokens']} errors={stats['errors']} {stats['last_error']}", flush=True)
    stop.set()


if __name__ == "__main__":
    main()
