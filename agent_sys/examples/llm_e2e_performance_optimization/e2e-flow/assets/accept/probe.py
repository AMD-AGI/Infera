#!/usr/bin/env python3
"""Is a score from this deployment interpretable? Gate the eval on the answer.

RUNS ON THE COMPUTE NODE. Standard library only.

Ported from `examples/sglang_1p1d_glm5.2/engine/tools/probe_accuracy.py`, which
runs before that kit's `lm_eval quick` for a reason worth restating: **every
failure it catches produces a number rather than an error**, and that number is
indistinguishable from a real regression. An eval that scores 0.42 because the
reasoning parser ate the answer looks exactly like an eval that scores 0.42
because a kernel is wrong.

One thing changed in the port, and it is the reason this is a copy rather than a
call. The original requires a prefill worker *and* a decode worker to be
registered, because the 1P1D kit deploys disaggregated. This package deploys MIX,
where `/v1/workers` holds exactly one worker with `disagg_mode: mixed`, and the
original's check fails against a perfectly healthy deployment.

The four properties, unchanged:

  reachable    the router answers and a worker is registered
  answerable   `content` arrives — not eaten by the reasoning budget
  stable       the same question gives the same ANSWER across repeats, and under
               concurrency. Not the same bytes: temperature 0 is not
               byte-reproducible here, so comparing bytes would fail on a
               healthy deployment
  isolated     a long shared prefix in front of the question does not change the
               answer. This is the one failure nothing else in the package can
               see: a radix-cache or kv-aware lookup returning the wrong block
               is never exercised by a benchmark that builds every prompt
               independently, and a kernel patch is exactly the kind of change
               that could cause it
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
import urllib.error
import urllib.request

QUESTIONS = {
    "natalia": (
        "Natalia sold clips to 48 of her friends in April, and then she sold half as "
        "many clips in May. How many clips did Natalia sell altogether in April and May? "
        "Give the final number.",
        "72",
    ),
    "multiply": ("Compute 17 * 23. Reply with only the number.", "391"),
    "speed": (
        "A car travels 120 kilometres in 3 hours. What is its average speed in "
        "kilometres per hour? Give the final number.",
        "40",
    ),
}

#: Long enough to span several router blocks and to be worth caching, and
#: irrelevant to every question, so any change in the answer is the cache
#: misbehaving rather than the model reading it.
SHARED_PREFIX = "You are a careful assistant. " + (
    "Background fact number 7 is irrelevant to any question. " * 300
)

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def last_number(text: str) -> str | None:
    found = _NUMBER.findall((text or "").replace(",", ""))
    return found[-1] if found else None


def ask(url: str, model: str, prompt: str, max_tokens: int, temperature: float,
        top_p: float, timeout: int) -> dict:
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read())
    choice = payload["choices"][0]
    message = choice["message"]
    content = message.get("content") or ""
    return {
        "content": content,
        "reasoning": message.get("reasoning_content") or "",
        # Last number in the answer, GSM8K-style, falling back to the reasoning
        # when the answer is empty -- which happens when the token budget ran out
        # before the model stopped thinking.
        "answer": last_number(content) or last_number(message.get("reasoning_content") or ""),
        "finish_reason": choice.get("finish_reason"),
        "usage": payload.get("usage") or {},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    failures: list[str] = []
    notes: list[str] = []
    report: dict = {"checks": {}}

    def call(prompt: str) -> dict:
        return ask(args.url, args.model, prompt, args.max_tokens, args.temperature,
                   args.top_p, args.timeout)

    # ---- 1. reachable --------------------------------------------------------
    print("===== 1. router and workers =====")
    try:
        with urllib.request.urlopen(f"{args.url}/v1/workers", timeout=15) as resp:
            workers = json.loads(resp.read())["workers"]
    except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
        print(f"  FAIL cannot read the worker list: {exc}")
        json.dump({"ok": False, "failures": [f"worker list unreadable: {exc}"]},
                  open(args.out, "w"), indent=2)
        return 1
    mixed = [w for w in workers if w.get("disagg_mode") == "mixed"]
    print(f"  workers registered: {len(workers)}, mixed: {len(mixed)}")
    report["checks"]["workers"] = {"total": len(workers), "mixed": len(mixed)}
    if not mixed:
        failures.append("no worker with disagg_mode 'mixed' is registered")

    # ---- 2. answerable, and the same answer every time -----------------------
    print(f"===== 2. answers: correct, and the same across {args.repeat} repeats =====")
    empty = length_capped = 0
    total = 0
    per_question = {}
    for key, (prompt, expected) in QUESTIONS.items():
        seen, ctoks = [], []
        for _ in range(args.repeat):
            try:
                got = call(prompt)
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                failures.append(f"{key}: request failed: {exc}")
                break
            total += 1
            seen.append(got["answer"])
            ctoks.append(got["usage"].get("completion_tokens"))
            if not got["content"].strip():
                empty += 1
                if got["finish_reason"] == "length":
                    length_capped += 1
        unique = sorted({s for s in seen if s is not None})
        ok = seen and all(s == expected for s in seen)
        per_question[key] = {"expected": expected, "answers": seen, "completion_tokens": ctoks,
                             "ok": bool(ok)}
        print(f"  {key:<10} expect={expected:<5} got={unique} ctok={ctoks} "
              f"{'ok' if ok else 'FAIL'}")
        if not ok:
            failures.append(
                f"{key}: expected {expected}, got {unique or 'nothing'}"
                + (" (answers differ between repeats)" if len(unique) > 1 else "")
            )
    report["checks"]["answers"] = per_question

    if total and empty / total > 0.2:
        if length_capped / max(empty, 1) > 0.2:
            failures.append(
                f"{empty}/{total} replies had empty content and most hit the token cap — "
                f"raise --max-tokens above {args.max_tokens}"
            )
        else:
            failures.append(
                f"{empty}/{total} replies had empty content without hitting the cap — "
                "the reasoning parser or the chat template is wrong for this model"
            )

    # ---- 3. the same question, all at once -----------------------------------
    print(f"===== 3. the same question, {args.concurrency} at once =====")
    batched = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for key, (prompt, expected) in QUESTIONS.items():
            futures = [pool.submit(call, prompt) for _ in range(args.concurrency)]
            answers = []
            for fut in futures:
                try:
                    answers.append(fut.result()["answer"])
                except Exception as exc:  # noqa: BLE001 - a probe reports
                    answers.append(f"error: {exc}")
            ok = all(a == expected for a in answers)
            batched[key] = {"expected": expected, "answers": answers, "ok": ok}
            print(f"  {key:<10} expect={expected:<5} got={sorted(set(answers))} "
                  f"{'ok' if ok else 'FAIL'}")
            if not ok:
                failures.append(f"{key}: wrong or inconsistent under concurrency {args.concurrency}")
    report["checks"]["batched"] = batched

    # ---- 4. a shared prefix must not change the answer -----------------------
    print("===== 4. a shared prefix must not change the answer =====")
    prefixed = {}
    for key, (prompt, expected) in QUESTIONS.items():
        try:
            alone = call(prompt)
            behind = call(SHARED_PREFIX + prompt)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            failures.append(f"{key}: prefix check failed to run: {exc}")
            continue
        ok = alone["answer"] == behind["answer"] == expected
        prefixed[key] = {
            "standalone": alone["answer"],
            "behind_prefix": behind["answer"],
            "prompt_tokens": [alone["usage"].get("prompt_tokens"),
                              behind["usage"].get("prompt_tokens")],
            "ok": ok,
            # The answer is the last number in the text, so a wrong answer and a
            # wrong extraction look the same from the outside -- a reply ending
            # "...which leaves 7 in the second month" scores 7 while saying 72
            # earlier. The text goes in the record on failure so the next reader
            # can tell the two apart instead of rerunning to find out.
            **({} if ok else {
                "standalone_text": alone["content"][-300:],
                "behind_prefix_text": behind["content"][-300:],
                "behind_prefix_finish": behind.get("finish_reason"),
            }),
        }
        print(f"  {key:<10} standalone={alone['answer']} behind-prefix={behind['answer']} "
              f"ptok {prefixed[key]['prompt_tokens']} {'ok' if ok else 'FAIL'}")
        if not ok:
            failures.append(
                f"{key}: the answer changed behind a shared prefix "
                f"({alone['answer']} -> {behind['answer']}). Prefix reuse is returning "
                "the wrong blocks."
            )
    report["checks"]["shared_prefix"] = prefixed

    print("===== verdict =====")
    for note in notes:
        print(f"  note  {note}")
    for failure in failures:
        print(f"  FAIL  {failure}")
    report["ok"] = not failures
    report["failures"] = failures
    report["notes"] = notes
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    if failures:
        print("PROBE FAILED — a score measured now would not be interpretable")
        return 1
    print("PROBE PASSED — scores from this deployment are worth measuring")
    return 0


if __name__ == "__main__":
    sys.exit(main())
