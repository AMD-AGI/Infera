#!/usr/bin/env python3
"""DO NOT USE AS-IS — sending a request straight to a disaggregated leg CRASHES it.

  AssertionError: req.bootstrap_room should not be None. Do not send requests directly to
  prefill or decode instances; send to the router instead.

The DP controller (follow_bootstrap_room_scheduler) asserts on a missing bootstrap_room and
SIGQUITs the whole leg. Kept only as a record of that gotcha. To inspect prefill-side numerics,
go through the router, or launch the same weights as a NON-disaggregated server.

Original intent: is the PREFILL leg itself computing correct activations, or is the corruption
introduced by / after the KV transfer?

A prefill-role sglang server refuses normal generation, but it will score a prompt: we ask for
1 token with logprobs. If prefill's own forward pass on a long novel shape is healthy, the
top logprob token should be sane and the prompt logprobs finite (no NaN/-inf storm).

Compares a long novel shape against a short control on the SAME leg.

usage: prefill_logprob_test.py PREFILL_URL MODEL "len1,len2,..."
"""
import json, math, sys, time, urllib.request

BASE, MODEL = sys.argv[1], sys.argv[2]
LENS = [int(x) for x in sys.argv[3].split(",")]

FILLER = ("Record {i}: the maintenance crew inspected corridor {a} and logged "
          "routine status code {b} with no anomalies reported that shift.")
NEEDLE = ("Record SECRET-B: the calibration constant for the orbital gyroscope is "
          "exactly 82931.")
TOK_PER_LINE = 28.1


def build(tokens, salt):
    n = max(4, int(tokens / TOK_PER_LINE))
    lines = [FILLER.format(i=i + salt, a=(i * 7 + salt) % 400,
                           b=1000 + (i * 13 + salt * 31) % 8000) for i in range(n)]
    lines.insert(n // 2, NEEDLE)
    return ("Below is a long maintenance log.\n\n<log>\n" + "\n".join(lines)
            + "\n</log>\n\nThe calibration constant for the orbital gyroscope is")


for k, L in enumerate(LENS):
    body = json.dumps({"model": MODEL, "prompt": build(L, (k + 1) * 5077),
                       "max_tokens": 1, "temperature": 0,
                       "logprobs": 5, "echo": False}).encode()
    req = urllib.request.Request(f"{BASE}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=900))
    except Exception as e:
        print(f"req~{L}: ERROR {e}", flush=True)
        continue
    ch = d["choices"][0]
    lp = ch.get("logprobs") or {}
    toks = lp.get("tokens")
    vals = lp.get("token_logprobs") or []
    finite = [v for v in vals if v is not None and math.isfinite(v)]
    print(f"req~{L} {time.time()-t0:.1f}s text={ch.get('text')!r} "
          f"top_tokens={toks} logprobs={vals} finite={len(finite)}/{len(vals)}", flush=True)
