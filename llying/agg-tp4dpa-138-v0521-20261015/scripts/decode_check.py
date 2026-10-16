#!/usr/bin/env python3
"""Decode correctness under prefill load: retrieval probes answered by generated tokens (not just the
first token) while other requests keep every DP rank prefilling, so decode/verify steps run next to
prefill steps (the heterogeneous steps SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION changes).

  background: FILL_THREADS threads send fresh ~FILL_TOKENS-token prompts (max_tokens 1) until the
              probes finish;
  probes:     PROBES prompts of ~PROBE_TOKENS tokens, each stating a pass code at the start of a long
              random text and asking for it at the end; thinking off, greedy, max_tokens 16. The answer
              must contain the code (MTP simulation forces acceptance lengths, so this checks content,
              not bit-equality).

Usage: decode_check.py ROUTER_URL MODEL OUT_JSON
Env: PROBES=32 PROBE_TOKENS=8000 FILL_THREADS=24 FILL_TOKENS=60000
Exit 0 if every probe answers correctly and none fails, else 1.
"""

import concurrent.futures
import json
import os
import random
import sys
import threading
import time
import urllib.request

ROUTER, MODEL, OUT = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]
PROBES = int(os.environ.get("PROBES", 32))
PROBE_TOKENS = int(os.environ.get("PROBE_TOKENS", 8000))
FILL_THREADS = int(os.environ.get("FILL_THREADS", 24))
FILL_TOKENS = int(os.environ.get("FILL_TOKENS", 60000))
WORDS = ("river stone paper window garden silver orange planet market winter engine signal camera "
         "forest pocket ladder candle bridge harbor rocket violin desert castle mirror button tunnel "
         "anchor feather jungle marble needle saddle temple velvet walnut yellow zipper").split()
CODES = ("MAGENTA", "TUNDRA", "QUASAR", "PELICAN", "OBSIDIAN", "ZEPHYR", "LANTERN", "GLACIER",
         "HARPOON", "NEBULA", "SAFFRON", "TORNADO")


def text(seed: int, tokens: int) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(int(tokens / 1.15)))


def ask(prompt: str, session: str, max_tokens: int) -> dict:
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    request = urllib.request.Request(ROUTER + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "X-Dynamo-Session-ID": session})
    start = time.time()
    reply = json.load(urllib.request.urlopen(request, timeout=1800))
    return {"text": reply["choices"][0]["message"].get("content") or "", "seconds": round(time.time() - start, 2),
            "usage": reply.get("usage")}


def main() -> int:
    stop = threading.Event()
    fill_stats = {"sent": 0, "errors": 0}
    salt = int(time.time())

    def filler(k: int) -> None:
        i = 0
        while not stop.is_set():
            try:
                ask(text(salt * 1000 + k * 100000 + i, FILL_TOKENS), f"fill-{salt}-{k}-{i}", 1)
                fill_stats["sent"] += 1
            except Exception:  # noqa: BLE001
                fill_stats["errors"] += 1
            i += 1

    threads = [threading.Thread(target=filler, args=(k,), daemon=True) for k in range(FILL_THREADS)]
    for t in threads:
        t.start()
    time.sleep(20)  # let the prefill load build up on every rank

    def probe(i: int) -> dict:
        code = random.Random(salt + i).choice(CODES)
        prompt = (f"Remember this: the pass code of vault {i} is {code}.\n\n" + text(salt + 7000 + i, PROBE_TOKENS)
                  + f"\n\nWhat is the pass code of vault {i}? Answer with the code only.")
        try:
            r = ask(prompt, f"probe-{salt}-{i}", 16)
            return {"i": i, "code": code, "ok": code in r["text"].upper(), **r}
        except Exception as exc:  # noqa: BLE001
            return {"i": i, "code": code, "ok": False, "error": str(exc)[:200]}

    with concurrent.futures.ThreadPoolExecutor(PROBES) as pool:
        results = list(pool.map(probe, range(PROBES)))
    stop.set()
    passed = sum(r["ok"] for r in results)
    errors = sum("error" in r for r in results)
    out = {"verdict": "pass" if passed == PROBES else "fail", "passed": passed, "probes": PROBES,
           "probe_errors": errors, "fill": fill_stats, "results": results}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "results"}))
    for r in results:
        if not r["ok"]:
            print("FAILED probe", r["i"], r["code"], repr(r.get("text", r.get("error")))[:120])
    return 0 if passed == PROBES else 1


if __name__ == "__main__":
    sys.exit(main())
