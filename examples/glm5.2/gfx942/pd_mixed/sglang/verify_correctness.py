#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Correctness checks for GLM-5.2 on gfx942. Two checks, two failure modes.

`needle` — long-context retrieval. GLM-5.2's `index_topk` is 2048, so for any shorter
prompt the sparse indexer's top-k selects the whole prompt and the model behaves like dense
attention. A short smoke test therefore says nothing about the DSA path — and when that path
is wrong the prose still reads fine while a fact buried in the context goes missing. Hence
retrieval prompts well past 2048 tokens.

A wrong top-k is deterministic, so one pass finds it. Logits corruption on a cold prefill is
not: the one behind note 1 in the README hit ~8% of cold prefills and never reproduced once
the prefix cache was warm. These prompts are seeded, which makes a rerun byte-identical and
therefore served warm — so pass `--flush-cache` to keep every prefill cold, and `--repeat`
for the sample size, because calling such a bug fixed takes on the order of a hundred clean
cold requests.

`idle` — output corruption on the first request after the engine's run queue drains. This
was measured on vLLM with this model: after the queue emptied one request came back as
token-level garbage with nothing in any server log, and the identical request right after it
was fine. Here it is a regression guard. It is worth keeping because a back-to-back
benchmark passes it cleanly while a code-agent workload, whose requests are separated by
thinking time, sits exactly on it. The three conditions separate "idleness triggers it" from
"it happens at random".

  ./verify_correctness.py                  # both checks, 12 cases
  ./verify_correctness.py --checks idle     # 3 conditions, ~1 min of deliberate sleeping
  ./verify_correctness.py --checks needle   # 9 cases
  ./verify_correctness.py --checks needle --lengths 65536 --depths 50  # one long case
  ./verify_correctness.py --checks needle --flush-cache --repeat 12    # 108 cold prefills
"""

from __future__ import annotations

import argparse
import random
import sys
import time

import requests

ALL_CHECKS = ("idle", "needle")

# Bland prose with no digits, so the needle is the only number in the prompt and
# retrieving it is unambiguous evidence of a read.
FILLER = [
    "The archive room stayed cool even in summer, and the shelves were always full.",
    "Maintenance crews swept the corridors before the building opened each morning.",
    "A narrow window near the stairwell let in a thin band of afternoon light.",
    "The catalogue was reorganised twice, first by subject and later by author.",
    "Visitors often remarked that the reading desks were unusually comfortable.",
    "Rain collected in the courtyard and drained slowly through the old channels.",
    "The east wing housed periodicals never transferred to microfilm.",
    "Librarians kept a spare key behind the counter for the basement door.",
    "Every autumn the heating pipes made a low knocking sound for a week or so.",
    "The reference desk was staffed by rotation, and the schedule changed monthly.",
]


def build_prompt(length: int, depth: int) -> tuple[str, int]:
    """Seeded by (length, depth) so a rerun sends a byte-identical prompt."""
    rng = random.Random(length * 1000 + depth)
    code = rng.randint(1_000_000, 9_999_999)
    needle = f"Important: the secret access code for Ravenna is {code}. Remember this number."
    body = [rng.choice(FILLER) for _ in range(max(4, length // 16))]  # ~16 tokens/sentence
    body.insert(min(len(body) * depth // 100, len(body)), needle)
    return " ".join(body) + (
        "\n\nBased on the text above, what is the secret access code for Ravenna? "
        "Reply with just the number."
    ), code


class Server:
    def __init__(self, base_url: str, timeout: float):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.sess = requests.Session()
        self.model = self.sess.get(f"{self.base}/v1/models", timeout=30).json()["data"][0]["id"]

    def chat(self, prompt: str, max_tokens: int) -> dict:
        r = self.sess.post(
            f"{self.base}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def flush_cache(self) -> None:
        # /flush_cache answers 200 but skips the flush while requests are in flight, so a
        # caller that needs cold prefills must also check cached_tokens per response.
        self.sess.post(f"{self.base}/flush_cache", timeout=60).raise_for_status()


def report(ok: bool, text: str) -> None:
    print(f"  [{' ok ' if ok else 'FAIL'}] {text}")


def check_idle(srv: Server, args: argparse.Namespace) -> tuple[int, int]:
    """First request after the run queue drains — see the module docstring."""
    probe = "What is 17 * 23? Reply with just the number."
    n = args.idle_trials
    answered = lambda d: "391" in (d["choices"][0]["message"].get("content") or "")  # noqa: E731

    srv.chat(probe, 32)  # warm the engine, then send without pausing
    hot = sum(answered(srv.chat(probe, 32)) for _ in range(n))

    cold = 0
    for _ in range(n):
        time.sleep(args.idle_seconds)
        cold += answered(srv.chat(probe, 32))

    warmed = 0
    for _ in range(n):
        time.sleep(args.idle_seconds)
        srv.chat("hi", 8)  # discarded warm-up request, the known workaround
        warmed += answered(srv.chat(probe, 32))

    conditions = [
        (hot, "back-to-back (baseline)"),
        (cold, f"first request after {args.idle_seconds:g}s idle"),
        (warmed, f"after {args.idle_seconds:g}s idle + a warm-up request"),
    ]
    for good, label in conditions:
        report(good == n, f"{label:<48} {good}/{n} sane")
    return sum(good != n for good, _ in conditions), len(conditions)


def check_needle(srv: Server, args: argparse.Namespace) -> tuple[int, int]:
    """Long-context retrieval — see the module docstring."""
    cases = [
        (n, d)
        for n in (int(x) for x in args.lengths.split(",") if x)
        for d in (int(x) for x in args.depths.split(",") if x)
    ]
    failures = sparse = total = warm = 0
    for round_no in range(1, args.repeat + 1):
        if args.repeat > 1:
            print(f"  round {round_no}/{args.repeat}")
        for length, depth in cases:
            prompt, code = build_prompt(length, depth)
            if args.flush_cache:
                srv.flush_cache()
            d = srv.chat(prompt, 64)
            got = (d["choices"][0]["message"].get("content") or "").strip()
            ptok = d["usage"]["prompt_tokens"]
            warm += ((d["usage"].get("prompt_tokens_details") or {}).get("cached_tokens") or 0) > 0
            ok = str(code) in got.replace(",", "").replace(" ", "")
            # Below index_topk the top-k selects everything, so the run is dense-equivalent
            # and proves nothing about the sparse path.
            regime = "sparse" if ptok > args.index_topk else "dense-eq"
            sparse += regime == "sparse"
            failures += not ok
            total += 1
            report(
                ok,
                f"~{length // 1024}k tok depth {depth:>2}%  prompt_tok={ptok:<6} {regime}"
                + ("" if ok else f"  got={got[:40]!r} want={code}"),
            )

    if not sparse:
        print(
            f"  warning: no prompt exceeded index_topk={args.index_topk}, so the sparse "
            "path was never exercised — raise --lengths."
        )
    if args.flush_cache and warm:
        print(
            f"  warning: {warm}/{total} requests reused cached prompt tokens, so those "
            "prefills were not cold and do not count towards a cold-prefill sample."
        )
    return failures, total


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--checks", default=",".join(ALL_CHECKS), help=f"any of {ALL_CHECKS}")
    ap.add_argument(
        "--lengths", default="4096,16384,65536", help="needle context lengths, comma separated"
    )
    ap.add_argument("--depths", default="10,50,90", help="needle depth in the context, percent")
    ap.add_argument("--index-topk", type=int, default=2048, help="the model's index_topk")
    ap.add_argument("--repeat", type=int, default=1, help="run the needle case list N times")
    ap.add_argument(
        "--flush-cache",
        action="store_true",
        help="POST /flush_cache before every needle request, so every prefill is cold",
    )
    ap.add_argument("--idle-trials", type=int, default=3, help="trials per idle condition")
    ap.add_argument("--idle-seconds", type=float, default=10.0, help="how long to stay idle")
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()

    checks = [c for c in args.checks.split(",") if c]
    unknown = [c for c in checks if c not in ALL_CHECKS]
    if unknown:
        ap.error(f"unknown check(s) {unknown}, pick from {ALL_CHECKS}")

    srv = Server(args.base_url, args.timeout)
    print(f"server: {srv.base}\nmodel : {srv.model}")

    failures = total = 0
    for name in checks:
        print(f"\n{name}")
        f, t = {"idle": check_idle, "needle": check_needle}[name](srv, args)
        failures += f
        total += t

    print(f"\nresult: {total - failures}/{total} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
