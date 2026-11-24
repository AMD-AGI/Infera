"""Compile prefill's on-demand kernels before a scored run: concurrent max_tokens=1 requests through the router whose
prompts span 1k-64k words, so prefill steps of many token counts (each a separate FlyDSL / Triton MoE and attention
shape) run once. Prompts are random words, so they share no prefix with each other or with the benchmark.

usage: jit_warm.py ROUTER_URL MODEL [ROUNDS=3]
Prints one line per round: requests ok / sent and the round's wall time.
"""
import concurrent.futures as cf
import json
import random
import sys
import time
import urllib.request

LENGTHS = [1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152, 65536]


def one(url, model, n_words, seed):
    rng = random.Random(seed)
    words = " ".join(f"w{rng.randrange(1 << 20):x}" for _ in range(n_words))
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": words}], "max_tokens": 1}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def main():
    url = sys.argv[1].rstrip("/") + "/v1/chat/completions"
    model = sys.argv[2]
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    for r in range(rounds):
        t = time.time()
        jobs = [(n, r * 1000 + i) for i, n in enumerate(LENGTHS)]
        with cf.ThreadPoolExecutor(len(jobs)) as ex:
            ok = sum(ex.map(lambda j: one(url, model, *j), jobs))
        print(f"round {r}: {ok}/{len(jobs)} ok in {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
