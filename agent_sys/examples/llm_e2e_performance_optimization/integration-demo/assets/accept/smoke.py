#!/usr/bin/env python3
"""The short end of "long and short text": four checks, each blind to the others.

RUNS ON THE COMPUTE NODE. Standard library only, plus `docker` on PATH.

The mission asks for "服务启动的简单验证：长短文本/needle". Split three ways,
because each direction fails in a way the other two cannot see:

  short prompt, short output   arithmetic. Catches a deployment that is serving
                               fluent nonsense — the failure mode that looks
                               green in every latency dashboard.
  short prompt, long output    512 tokens of continuous text. Catches
                               degeneration into repetition, which produces a
                               *score* on an eval rather than an error, and a
                               score is indistinguishable from a real regression.
  long prompt, short output    needle.py, separately.

Plus two structural checks that cost one HTTP call each: exactly one worker in
`mixed` mode (two means a stale etcd registration is silently halving every
measurement that follows), and an engine log with no fault lines.

The repetition check is borrowed from `probe_accuracy.py`, where it is a soft
finding. Here it is hard: this package's whole output is a comparison between two
arms, and an arm that degenerates produces numbers rather than errors.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

ARITHMETIC = "Compute 17 * 23. Reply with only the number."
ARITHMETIC_ANSWER = "391"
LONG_PROMPT = (
    "Explain how a modern GPU executes a matrix multiplication, from the host launch "
    "through to the result landing in memory. Write continuous prose, at least three "
    "hundred words, and do not repeat yourself."
)

FAULT_PATTERNS = ("memory access fault", "HIP error", "CUDA error", "Traceback")


#: Turn the reasoning pass off for the long-generation check only.
#:
#: **Measured twice, and this is the second fix.** The check asks whether the
#: deployment can produce long continuous text without degenerating. With
#: reasoning on it never gets to try: at max_tokens 512 the whole budget went to
#: thinking and `content` came back empty, and at 2048 it did it again — 1470
#: words of reasoning, zero words of answer, `finish_reason: length`. Raising the
#: cap further is a race against a model that will use whatever it is given.
#:
#: `--thinking-mode glm-45` in the evaluator sets `enable_thinking: True` through
#: this same kwarg, so the template takes it; passing False is the documented way
#: to get a direct answer. The eval and the needle keep reasoning on, because
#: there the reasoning is part of what is being measured.
NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}


def post(url: str, payload: dict, timeout: int) -> dict:
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read())
    out["_wall_s"] = round(time.time() - started, 2)
    return out


def get(url: str, path: str, timeout: int = 15):
    with urllib.request.urlopen(f"{url}{path}", timeout=timeout) as resp:
        return json.loads(resp.read())


def max_ngram_repeat(text: str, n: int = 8) -> int:
    words = text.split()
    if len(words) < n:
        return 0
    counts = collections.Counter(
        " ".join(words[i : i + n]) for i in range(len(words) - n + 1)
    )
    return max(counts.values())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--container", required=True)
    ap.add_argument("--out", required=True)
    # 2048, not 512. The engine runs with --reasoning-parser glm45, so the model
    # thinks before it answers and `content` stays empty until it stops.
    # Measured at 512: the whole budget went to reasoning, `content` came back
    # empty with finish_reason "length", and the check failed for a reason that
    # had nothing to do with degeneration.
    ap.add_argument("--long-tokens", type=int, default=2048)
    ap.add_argument("--max-repeat", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    checks = []

    def record(name: str, ok: bool, **detail):
        checks.append({"name": name, "ok": bool(ok), **detail})
        print(f"  {name:<16} {'ok' if ok else 'FAIL'}  {detail}")

    # ---- workers -------------------------------------------------------------
    try:
        workers = get(args.url, "/v1/workers")["workers"]
        modes = [w.get("disagg_mode") for w in workers]
        record(
            "workers",
            len(workers) == 1 and modes == ["mixed"],
            count=len(workers),
            disagg_modes=modes,
        )
    except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
        record("workers", False, error=str(exc))

    try:
        names = [m["id"] for m in get(args.url, "/v1/models")["data"]]
        record("models", args.model in names, served=names)
    except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
        record("models", False, error=str(exc))

    # ---- short prompt, short output -----------------------------------------
    try:
        answer = post(
            args.url,
            {
                "model": args.model,
                "messages": [{"role": "user", "content": ARITHMETIC}],
                "max_tokens": 512,
                "temperature": 0.0,
                "top_p": 1.0,
            },
            args.timeout,
        )
        content = (answer["choices"][0]["message"].get("content") or "").strip()
        record(
            "arithmetic",
            ARITHMETIC_ANSWER in content,
            expected=ARITHMETIC_ANSWER,
            got=content[:80],
            wall_s=answer["_wall_s"],
        )
    except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
        record("arithmetic", False, error=str(exc))

    # ---- short prompt, long output ------------------------------------------
    try:
        answer = post(
            args.url,
            {
                "model": args.model,
                "messages": [{"role": "user", "content": LONG_PROMPT}],
                "max_tokens": args.long_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
                **NO_THINKING,
            },
            args.timeout,
        )
        choice = answer["choices"][0]
        content = (choice["message"].get("content") or "").strip()
        reasoning = (choice["message"].get("reasoning_content") or "").strip()
        usage = answer.get("usage") or {}
        # If the deployment ignored the kwarg and reasoned anyway, judge the text
        # it did produce rather than reporting nothing. 1470 words of
        # non-repeating reasoning answers the question this check is asking —
        # "can it generate long coherent text" — even though it is not the text
        # that was asked for. The substitution is recorded, not hidden.
        judged, judged_from = (content, "content")
        if not content and reasoning:
            judged, judged_from = (reasoning, "reasoning_content")
        repeat = max_ngram_repeat(judged)
        words = len(judged.split())
        # Named, because an answer that is empty because the budget ran out and
        # one that is empty because the deployment is broken look identical from
        # the outside, and only the second is a finding.
        starved = not content and choice.get("finish_reason") == "length"
        record(
            "long_generation",
            bool(judged) and repeat <= args.max_repeat and words >= 50,
            completion_tokens=usage.get("completion_tokens"),
            judged_from=judged_from,
            words=words,
            answer_words=len(content.split()),
            reasoning_words=len(reasoning.split()),
            max_8gram_repeat=repeat,
            repeat_bar=args.max_repeat,
            finish_reason=choice.get("finish_reason"),
            starved=starved,
            diagnosis=(
                "reasoning was requested off and happened anyway; the degeneration check "
                "was applied to the reasoning text instead"
                if judged_from == "reasoning_content"
                else ""
            ),
            wall_s=answer["_wall_s"],
        )
    except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
        record("long_generation", False, error=str(exc))

    # ---- the engine's own account -------------------------------------------
    # Read here rather than trusted from the deployment record: this runs after
    # the requests above, so it also covers faults those requests caused.
    proc = subprocess.run(
        ["docker", "exec", args.container, "tail", "-c", "400000", "/tmp/glm53_mix.log"],
        capture_output=True,
        text=True,
    )
    log = proc.stdout
    hits = {p: len(re.findall(re.escape(p), log)) for p in FAULT_PATTERNS}
    record(
        "engine_log",
        proc.returncode == 0 and not any(hits.values()),
        bytes_scanned=len(log),
        fault_hits=hits,
    )

    report = {"checks": checks, "ok": all(c["ok"] for c in checks)}
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("SMOKE_OK" if report["ok"] else "SMOKE_FAIL")
    # Exit 0 regardless: this is evidence for `compare` to weigh, and taking the
    # measurement task down here would discard the eval and the replay that
    # follow it.
    return 0


if __name__ == "__main__":
    sys.exit(main())
