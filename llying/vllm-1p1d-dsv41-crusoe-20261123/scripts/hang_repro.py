"""Reproduce the DP2 lockstep hang without AgentX: idle, then wake both legs with new requests, repeatedly.

Every hang so far came on the first step after both ranks of a leg had idled for minutes, with one rank running a
real batch (often a request whose KV just arrived) and the other a dummy batch. A cycle here is:
  idle IDLE_S seconds -> one long request alone -> a burst of BURST requests of mixed lengths -> wait for all.
Prompts are random words (no shared prefix), max_tokens=MAX_TOKENS so decode admits each request and runs a few
steps. The loop stops when a request fails at the transport level and the engine logs show a fatal error, or when
the time budget ends.

usage: hang_repro.py ROUTER_URL MODEL LOG_DIR [BUDGET_S=14400] [IDLE_S=150] [BURST=48] [MAX_TOKENS=32]
Prints one line per cycle; the last line is "RESULT cycles=N hangs=H ...".
"""
import concurrent.futures as cf
import json
import os
import random
import sys
import time
import urllib.request

FATAL = ("engine core reported fatal failure", "LLM became permanently unhealthy")


def request(url, model, n_words, seed, max_tokens):
    rng = random.Random(seed)
    words = " ".join(f"w{rng.randrange(1 << 20):x}" for _ in range(n_words))
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": words}],
                       "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            r.read()
            return r.status == 200, time.time() - t
    except Exception:  # noqa: BLE001
        return False, time.time() - t


def engine_died(log_dir):
    died = []
    for leg in ("prefill", "decode"):
        try:
            with open(os.path.join(log_dir, f"{leg}.log"), errors="replace") as f:
                if any(m in f.read() for m in FATAL):
                    died.append(leg)
        except OSError:
            pass
    return died


def main():
    url = sys.argv[1].rstrip("/") + "/v1/chat/completions"
    model, log_dir = sys.argv[2], sys.argv[3]
    budget = float(sys.argv[4]) if len(sys.argv) > 4 else 14400
    idle = float(sys.argv[5]) if len(sys.argv) > 5 else 150
    burst = int(sys.argv[6]) if len(sys.argv) > 6 else 48
    max_tokens = int(sys.argv[7]) if len(sys.argv) > 7 else 32
    t0, cycle = time.time(), 0
    while time.time() - t0 + idle + 120 < budget:
        cycle += 1
        time.sleep(idle)
        tc = time.time()
        ok1, dt1 = request(url, model, 7500, cycle * 7919, max_tokens)
        lengths = [random.Random(cycle * 31 + i).choice((1024, 4096, 8192, 16384, 32768, 65536)) for i in range(burst)]
        with cf.ThreadPoolExecutor(burst) as ex:
            res = list(ex.map(lambda a: request(url, model, a[1], cycle * 1000 + a[0], max_tokens),
                              enumerate(lengths)))
        ok = sum(r[0] for r in res)
        died = engine_died(log_dir)
        print(f"{time.strftime('%H:%M:%S', time.gmtime())} cycle {cycle}: single ok={ok1} {dt1:.1f}s; "
              f"burst {ok}/{burst} ok, slowest {max(r[1] for r in res):.0f}s; cycle {time.time() - tc:.0f}s"
              f"{'; ENGINE DIED: ' + ' '.join(died) if died else ''}", flush=True)
        if died:
            print(f"RESULT cycles={cycle} hangs=1 died={'+'.join(died)} elapsed={time.time() - t0:.0f}s", flush=True)
            return
    print(f"RESULT cycles={cycle} hangs=0 elapsed={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
