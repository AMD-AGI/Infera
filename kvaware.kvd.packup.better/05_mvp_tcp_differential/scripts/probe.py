#!/usr/bin/env python3
"""Differential probe — captures the RAW output text, not just a verdict.

For this experiment a pass/fail count is useless. Both arms score 0/4; the
whole finding lives in WHAT the wrong answers look like. So this probe dumps
each completion verbatim (with repr(), so control characters and CJK survive
the trip through logs and diffs) and writes a machine-readable JSON alongside
so the two arms can be compared field by field.

Usage:
    python3 probe.py http://10.2.122.10:8100 qwen3 --arm A --json /tmp/r4.json

Read the output with three questions, in this order:

  1. Did a completion come back at all?      no  -> transport problem
  2. Is the content right?                   no  -> keep going
  3. Does the OTHER arm look the same way?   yes -> not caused by the feature

Question 3 is the only one that attributes anything, and it cannot be answered
from one run. Run both arms.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

CASES = [
    ("The capital of France is", "paris"),
    ("The capital of China is", "beijing"),
    ("2+2=", "4"),
    ("The largest planet in our solar system is", "jupiter"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", nargs="?", default="http://127.0.0.1:8100")
    ap.add_argument("model", nargs="?", default="qwen3")
    ap.add_argument("--arm", default="?", help="label for this run, e.g. A or B")
    ap.add_argument("--json", default=None, help="write structured results here")
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()

    print(f"# arm={args.arm} base={args.base} model={args.model} temperature=0")
    print()

    records = []
    ok = wrong = err = 0
    kvcache_failures = 0

    for prompt, want in CASES:
        body = json.dumps(
            {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.max_tokens,
                "temperature": 0,
            }
        ).encode()
        req = urllib.request.Request(
            f"{args.base}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        rec = {"prompt": prompt, "want": want}
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.load(r)
            txt = d["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:400]
            except Exception:
                pass
            err += 1
            is_kv = "kvcache" in detail.lower()
            kvcache_failures += is_kv
            rec.update(status="ERROR", http=e.code, detail=detail, kvcache=bool(is_kv))
            print(f"[ERROR] {prompt!r}")
            print(f"        HTTP {e.code}: {detail!r}")
            records.append(rec)
            continue
        except Exception as e:
            err += 1
            rec.update(status="ERROR", detail=f"{type(e).__name__}: {e}")
            print(f"[ERROR] {prompt!r} -> {type(e).__name__}: {e}")
            records.append(rec)
            continue

        hit = want.lower() in txt.lower()
        status = "OK" if hit else "WRONG"
        ok += hit
        wrong += not hit
        # repr() on purpose: newlines, control chars and CJK must survive being
        # pasted into a results file and diffed against the other arm.
        rec.update(status=status, text=txt, first_16=txt[:16])
        print(f"[{status:5s}] {prompt!r}")
        print(f"        -> {txt!r}")
        records.append(rec)

    n = len(CASES)
    print()
    print(f"arm={args.arm}: {ok}/{n} correct, {wrong} wrong-content, {err} errored")

    # The opening tokens are the single most diagnostic field: if two arms with
    # different features open with the SAME token, whatever produced it is not
    # the feature.
    firsts = [r.get("first_16") for r in records if r.get("first_16") is not None]
    if firsts:
        print(f"arm={args.arm}: first 16 chars of each completion: {firsts!r}")

    if kvcache_failures:
        print(
            f"\n>>> {kvcache_failures}/{n} were KV-transfer errors. On a same-host PD\n"
            ">>> pair that is the mooncake cross-rail loopback limitation, and it\n"
            ">>> means this arm never got far enough to say anything about content."
        )

    if wrong and not ok:
        print(
            "\n>>> Completions came back and every one is wrong. This is NOT yet\n"
            ">>> attributable to anything. Run the other arm and compare — if it\n"
            ">>> is equally wrong, the cause is in the shared substrate, not in\n"
            ">>> the switch you flipped. Do NOT try to isolate this by probing a\n"
            ">>> PD leg directly; a PD leg only serves through the pair."
        )

    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "arm": args.arm,
                    "base": args.base,
                    "model": args.model,
                    "ok": ok,
                    "wrong": wrong,
                    "errored": err,
                    "kvcache_failures": kvcache_failures,
                    "records": records,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"\nwrote {args.json}")

    return 0 if ok >= 3 else 1


if __name__ == "__main__":
    sys.exit(main())
