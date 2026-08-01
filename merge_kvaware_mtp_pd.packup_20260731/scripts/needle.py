#!/usr/bin/env python3
"""Needle-in-a-haystack across prefill chunk boundaries.

The mooncake early-send race corrupts every prefill chunk EXCEPT the last: the
transfer worker RDMA-reads pages while the forward that writes them is still
running. It never raises. The signature is a needle that is retrieved correctly
when it sits in the final chunk and comes back mangled when it sits in an earlier
one -- classically the first digits followed by repeated `</think>`.

So the test only means anything when the prompt is genuinely longer than
`--chunked-prefill-size`, and only when the needle is placed at several depths:
a needle in the last chunk passes even on unpatched code.

usage: needle.py BASE MODEL PROMPT_TOKENS [DEPTHS_CSV] [OUT_JSON]
"""
import json
import os
import re
import sys
import urllib.request

BASE = sys.argv[1]
MODEL = sys.argv[2]
PROMPT_TOKENS = int(sys.argv[3]) if len(sys.argv) > 3 else 24000
DEPTHS = [float(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "0,0.25,0.5,0.75,1.0").split(",")]
OUT = sys.argv[5] if len(sys.argv) > 5 else "/tmp/needle.json"

TOK_PER_LINE = 28.1
FILLER = ("Record {i}: the maintenance crew inspected corridor {a} and logged "
          "routine status code {b} with no anomalies reported that shift.")


def build(depth: float, secret: int) -> str:
    n = max(8, int(PROMPT_TOKENS / TOK_PER_LINE))
    lines = [FILLER.format(i=i, a=(i * 7) % 400, b=1000 + (i * 13) % 8000) for i in range(n)]
    at = min(n, max(0, int(n * depth)))
    lines.insert(at, f"Record SECRET: the calibration constant for the orbital "
                     f"gyroscope is exactly {secret}.")
    return ("Below is a long maintenance log. Read it carefully, then answer the "
            "question at the end using only information from the log.\n\n<log>\n"
            + "\n".join(lines) +
            "\n</log>\n\nQuestion: What is the calibration constant for the orbital "
            "gyroscope? State the number.")


MAX_TOKENS = int(sys.argv[6]) if len(sys.argv) > 6 else 1024


# GLM-5.2's own generation_config.json: temperature 1.0, top_p 0.95. Use it.
#
# The earlier `temperature: 0` here was a probe defect, and a costly one. Greedy
# decoding sends a reasoning model into repetition on a long prompt, and EAGLE/MTP
# AMPLIFIES it: the draft model predicts a loop perfectly, so `accept len` pins at
# its maximum (4.00 with --speculative-num-draft-tokens 4) and the loop runs to
# max_tokens. The result reads exactly like KV corruption -- needle missing,
# `</think>` repeating hundreds of times, finish=length -- while the KV path is
# fine. Measured: MTP on gave 3/5 at temp 0 and 5/5 at these settings, with the
# same image, same prefill leg, same prompt.
#
# top_k is not in generation_config; 40 is the value GLM's own serving docs use.
TEMPERATURE = float(os.environ.get("NEEDLE_TEMPERATURE", "1.0"))
TOP_P = float(os.environ.get("NEEDLE_TOP_P", "0.95"))
TOP_K = int(os.environ.get("NEEDLE_TOP_K", "40"))


def ask(prompt: str) -> tuple[str, int, int, str]:
    body = json.dumps({"model": MODEL,
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": MAX_TOKENS,
                       "temperature": TEMPERATURE,
                       "top_p": TOP_P,
                       "top_k": TOP_K}).encode()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    ch = d["choices"][0]
    usage = d.get("usage", {})
    # finish_reason separates a truncated probe from a real retrieval failure:
    # "length" means we cut the model off, not that it got the needle wrong.
    return (ch["message"]["content"], usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0), ch.get("finish_reason", "?"))


rows, ok = [], 0
for i, depth in enumerate(DEPTHS):
    secret = 10000 + i * 7919 % 89999
    prompt = build(depth, secret)
    try:
        txt, ptok, ctok, finish = ask(prompt)
    except Exception as e:
        print(f"[XX] depth={depth:<5} ERROR {e}")
        rows.append({"depth": depth, "secret": secret, "error": str(e)})
        continue
    hit = str(secret) in txt
    ok += hit
    # The failure mode is a truncated needle plus a repeating tag, so record both.
    tag_repeats = len(re.findall(r"</think>", txt))
    rows.append({"depth": depth, "secret": secret, "found": hit,
                 "prompt_tokens": ptok, "completion_tokens": ctok,
                 "finish_reason": finish, "think_tag_repeats": tag_repeats,
                 "text": txt[:600]})
    print(f"[{'OK' if hit else 'XX'}] depth={depth:<5} want={secret} "
          f"ptok={ptok} ctok={ctok} finish={finish} </think>x{tag_repeats} "
          f":: {txt[:100]!r}")

json.dump({"prompt_tokens_target": PROMPT_TOKENS, "rows": rows}, open(OUT, "w"), indent=1)
print(f"\n{ok}/{len(DEPTHS)} needles retrieved  -> {OUT}")
sys.exit(0 if ok == len(DEPTHS) else 1)
