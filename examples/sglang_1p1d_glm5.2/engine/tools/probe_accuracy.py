#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
"""Decide whether a benchmark score from this deployment would mean anything, before paying
for one.

A GSM8K run against this stack costs tens of minutes and can come back at 0.00 for reasons
that have nothing to do with accuracy -- an empty ``content`` field, a router serving with
one leg registered. It can equally come back at a plausible 0.85 while prefix caching hands
one request another request's KV. Neither is visible in the score.

So this sends ~30 requests and reports the four things that separate an interpretable score
from a decorative one:

    reachability   router healthy and BOTH legs registered
    answerability  does ``content`` arrive, or does reasoning eat the whole budget
    stability      the same question, asked repeatedly, must yield the same ANSWER
    isolation      a shared prefix must not change the answer to the question behind it

WHY STABILITY IS CHECKED ON THE ANSWER AND NOT ON THE BYTES. This deployment is not
byte-deterministic at ``temperature=0``, and that is expected rather than broken. Measured
here, ten serial repeats of one GSM8K question all returned 72 while their completion
lengths ranged over 214-289 tokens: MTP verification, dp8 attention and batch-dependent
kernel selection each perturb the numerics without changing the argmax that matters. A
byte-for-byte check would fail on every healthy run, which is worse than no check at all.

Isolation is the failure mode nothing else in this kit can see. ``bench.sh`` builds every
prompt independently, so a radix-cache or kvd lookup returning the wrong block is never
exercised there.

Usage:
    python3 probe_accuracy.py --url http://<ip>:<port> --model <served-name> [options]

Exit status is 0 when every HARD check passes, 1 otherwise, so it can gate a script. Soft
findings (unstable completion lengths, some empty content when the budget is genuinely
tight) are reported loudly but do NOT fail.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# Questions whose answer is not in dispute. The point is NOT to measure ability -- a model
# that gets these wrong is not a weaker model, it is a broken deployment. Keeping them
# trivial is what makes a failure unambiguous; a hard question coming back wrong would leave
# "the model is not that good" as a live explanation.
QUESTIONS = [
    (
        "natalia",
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many "
        "clips in May. How many clips did Natalia sell altogether in April and May?",
        "72",
    ),
    ("multiply", "What is 17 multiplied by 23? Give the number.", "391"),
    (
        "speed",
        "A train travels 60 miles in 1.5 hours. What is its average speed in miles per hour?",
        "40",
    ),
]

# Long enough to span many KV blocks (block size is 64 here), repetitive enough that the
# radix cache will certainly retain it, and semantically inert so that any change in the
# answer is attributable to the cache rather than to the content.
SHARED_PREFIX = (
    "You are a careful assistant. "
    + "Background fact number 7 is irrelevant to any question. " * 300
)

RED, GRN, YEL, NC = "\033[0;31m", "\033[0;32m", "\033[0;33m", "\033[0m"


def mark(ok: bool) -> str:
    return f"{GRN}ok{NC}" if ok else f"{RED}FAIL{NC}"


class Findings:
    """Hard findings fail the probe; soft ones are observations about the deployment."""

    def __init__(self) -> None:
        self.hard: list[str] = []
        self.soft: list[str] = []

    def fail(self, msg: str) -> None:
        self.hard.append(msg)

    def note(self, msg: str) -> None:
        self.soft.append(msg)


def post(url: str, model: str, prompt: str, args) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=args.timeout) as fh:
        d = json.load(fh)
    choice = d["choices"][0]
    msg, usage = choice["message"], (d.get("usage") or {})
    # Both fields are None rather than "" when the model produces neither, and `or ""` is the
    # difference between a report and a TypeError.
    return {
        "content": msg.get("content") or "",
        "reasoning": msg.get("reasoning_content") or "",
        "finish": choice.get("finish_reason"),
        "ctok": usage.get("completion_tokens"),
        "ptok": usage.get("prompt_tokens"),
    }


def last_number(text: str) -> str | None:
    """The scorer SGLang's own gsm8k eval uses: the last number in the text."""
    found = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return found[-1].rstrip(".") if found else None


def answer_of(resp: dict) -> str | None:
    """Read the answer from ``content``, falling back to ``reasoning_content``.

    The fallback is not leniency. With --reasoning-parser glm45 the chain of thought is a
    separate field, and a reply that ran out of budget mid-thought has the answer there and
    nothing in `content`. Refusing to look would report a budget problem as a wrong answer.
    """
    return last_number(resp["content"]) or last_number(resp["reasoning"])


def worst_ngram_repeat(text: str, n: int = 8) -> tuple[int, str]:
    """Count of the most-repeated n-gram, i.e. how degenerate the output looks.

    Degeneration is what README note 5 looks like from the client side: the DSA-on-ROCm env
    block not taking effect produces repeated tokens rather than an error. An 8-gram is long
    enough that ordinary prose does not repeat one.
    """
    words = text.split()
    if len(words) < n:
        return 0, ""
    grams = collections.Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    gram, count = grams.most_common(1)[0]
    return count, " ".join(gram)


def check_workers(url: str, f: Findings) -> bool:
    """Both legs registered. A router with one leg still answers and still produces a score,
    but that score describes a different deployment and nothing downstream would say so."""
    print("===== 1. router and workers =====")
    try:
        with urllib.request.urlopen(url + "/v1/workers", timeout=30) as fh:
            workers = json.load(fh).get("workers", [])
    except Exception as exc:  # noqa: BLE001
        f.fail(f"could not read {url}/v1/workers: {exc}")
        print(f"  {RED}unreachable: {exc}{NC}")
        return False
    roles = collections.Counter(w.get("disagg_mode") for w in workers)
    for role in ("prefill", "decode"):
        n = roles.get(role, 0)
        print(f"  {role:<8} workers registered: {n}")
        if n < 1:
            f.fail(f"no {role} worker registered")
    print(f"  model names: {sorted({w.get('model_name') for w in workers})}")
    return True


def check_answers(ask, args, f: Findings) -> None:
    print("\n===== 2. answers: correct, and the same every time =====")
    print(
        f"  (temperature={args.temperature} max_tokens={args.max_tokens} "
        f"repeat={args.repeat} per question)"
    )
    empty = capped = total = 0
    for key, prompt, expect in QUESTIONS:
        answers, ctoks, worst_rep, worst_gram = [], [], 0, ""
        for _ in range(args.repeat):
            try:
                r = ask(prompt)
            except Exception as exc:  # noqa: BLE001
                f.fail(f"request failed on '{key}': {exc}")
                print(f"  {key:<10} {RED}request failed: {exc}{NC}")
                break
            total += 1
            answers.append(answer_of(r))
            ctoks.append(r["ctok"])
            empty += not r["content"].strip()
            capped += r["finish"] == "length"
            rep, gram = worst_ngram_repeat(r["reasoning"] + " " + r["content"])
            if rep > worst_rep:
                worst_rep, worst_gram = rep, gram
        if not answers:
            continue

        dist = collections.Counter(answers)
        correct = all(a == expect for a in answers)
        print(
            f"  {key:<10} expect={expect:<5} got={dict(dist)!s:<28} "
            f"ctok={min(ctoks)}-{max(ctoks):<6} rep8={worst_rep:<4} "
            f"{mark(correct and len(dist) == 1)}"
        )
        if not correct:
            f.fail(f"'{key}' answered {dict(dist)}, expected {expect}")
        elif len(dist) > 1:
            f.fail(f"'{key}' gave {len(dist)} different answers: {dict(dist)}")
        # Soft: a model asked for a table or a list can legitimately repeat a stem, and this
        # probe should not fail on style.
        if worst_rep >= 4:
            f.note(f"'{key}' repeated an 8-gram {worst_rep}x: {worst_gram[:60]!r}")
        # Soft by design: byte-instability at temperature 0 is this deployment's normal state
        # (see the module docstring). Recorded so a later run can see whether it changed.
        if len(set(ctoks)) > 1:
            f.note(
                f"'{key}' completion length varied {min(ctoks)}-{max(ctoks)} tokens "
                f"at temperature {args.temperature}"
            )

    if not total:
        return
    pct_empty, pct_capped = 100.0 * empty / total, 100.0 * capped / total
    print(
        f"  empty content: {empty}/{total} ({pct_empty:.0f}%)   "
        f"finish_reason=length: {capped}/{total} ({pct_capped:.0f}%)"
    )
    # Empty content plus finish_reason=length is reasoning eating the whole budget, which
    # scores 0.00 on every eval while the deployment is healthy. Empty content while stopping
    # normally is a different bug. Distinguished here so the fix is obvious.
    if pct_empty <= 20.0:
        return
    if pct_capped > 20.0:
        f.fail(
            f"{pct_empty:.0f}% of replies had empty content and {pct_capped:.0f}% hit the "
            f"token cap — raise --max-tokens above {args.max_tokens} before reading a score"
        )
    else:
        f.fail(
            f"{pct_empty:.0f}% of replies had empty content while stopping normally — "
            f"the reasoning parser or chat template is wrong"
        )


def check_batch(ask, args, f: Findings) -> None:
    """Serial repeats share little with a real eval, which sends dozens of requests at once.
    Batch composition changes kernel selection, DP rank assignment and MTP verification, so a
    batched answer disagreeing with the serial one points at batching, not at the model."""
    print(f"\n===== 3. same question, {args.concurrency} at once =====")
    key, prompt, expect = QUESTIONS[0]
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            batch = list(pool.map(lambda _: ask(prompt), range(args.concurrency)))
    except Exception as exc:  # noqa: BLE001
        f.fail(f"batched request failed: {exc}")
        print(f"  {RED}failed: {exc}{NC}")
        return
    dist = collections.Counter(answer_of(r) for r in batch)
    ok = len(dist) == 1 and next(iter(dist)) == expect
    print(f"  {key:<10} expect={expect:<5} got={dict(dist)}  {mark(ok)}")
    if not ok:
        f.fail(
            f"under concurrency {args.concurrency}, '{key}' gave {dict(dist)}, "
            f"expected all {expect}"
        )


def check_prefix(ask, f: Findings) -> None:
    print("\n===== 4. shared prefix must not change the answer =====")
    print("  (the failure this catches is invisible to bench.sh — no shared prefix there)")
    for key, prompt, expect in QUESTIONS:
        try:
            alone, behind = ask(prompt), ask(SHARED_PREFIX + "\n\n" + prompt)
        except Exception as exc:  # noqa: BLE001
            f.fail(f"prefix check failed on '{key}': {exc}")
            print(f"  {key:<10} {RED}request failed: {exc}{NC}")
            continue
        a1, a2 = answer_of(alone), answer_of(behind)
        print(
            f"  {key:<10} standalone={str(a1):<6} behind-prefix={str(a2):<6} "
            f"ptok {alone['ptok']}->{behind['ptok']}  {mark(a1 == a2 == expect)}"
        )
        if not a1 == a2 == expect:
            f.fail(
                f"'{key}' answered {a1} standalone but {a2} behind a shared prefix "
                f"(expected {expect}) — suspect radix cache / kvd / kv-aware routing"
            )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--url", required=True, help="router base url, e.g. http://10.0.0.1:8100")
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--repeat", type=int, default=4, help="repeats per question")
    ap.add_argument("--concurrency", type=int, default=8, help="requests in the batched check")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    f = Findings()
    if not check_workers(args.url, f):
        # Everything below goes through that router; there is nothing to learn by asking
        # thirty more times.
        print(f"\n{RED}PROBE FAILED{NC} — router unreachable")
        return 1

    ask = lambda prompt: post(args.url, args.model, prompt, args)
    check_answers(ask, args, f)
    check_batch(ask, args, f)
    check_prefix(ask, f)

    print("\n===== verdict =====")
    for s in f.soft:
        print(f"  {YEL}note{NC}  {s}")
    if not f.soft:
        print("  (no soft findings)")
    for h in f.hard:
        print(f"  {RED}FAIL{NC}  {h}")
    if f.hard:
        print(f"\n{RED}PROBE FAILED{NC} — a score measured now would not be interpretable")
        return 1
    print(f"\n{GRN}PROBE PASSED{NC} — scores from this deployment are worth measuring")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
