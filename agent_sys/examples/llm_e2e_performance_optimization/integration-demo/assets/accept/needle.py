#!/usr/bin/env python3
"""Bury a distinctive string in a multi-chunk prompt and ask for it back.

RUNS ON THE COMPUTE NODE, because the router binds the node's private address.
Standard library only: it runs under the node's system python, not a venv.

**Why this exists at all.** sglang prefills in `--chunked-prefill-size` pieces
(8192 here), and a stack that drops every piece but the last still answers a
short prompt perfectly and still answers a needle placed near the end. The
repository's own guidance for GLM-5.2 on gfx942 says it plainly: bury the needle
at head, middle *and* tail of a prompt several times one chunk long, because
"losing only the head reads as 'it works' if you happen to probe the tail".
Nothing in this repository shipped a script that does it, so this is written from
scratch.

Three things were measured before the defaults here were chosen, and each of them
changed something. See `../../temp/manual/FINDINGS.md` §3.

**The needle is lexical, and the filler must not resemble it.** A six-digit code
buried in filler sentences that are themselves numbered is not retrieved at head
depth in a 76 472-token prompt — three attempts returned the identical wrong
string, so it is deterministic model behaviour rather than a flaky kernel — while
the same position with `MAGENTA-QUASAR-17` in it is retrieved every time. A
numeric needle in numeric filler tests whether the model can tell two numbers
apart, which is not the question.

**Length is read back, not assumed.** `usage.prompt_tokens` comes from the
server, so this needs no tokenizer and cannot drift with one. Measured
12.0–12.7 tokens per filler sentence; the estimate is deliberately generous and
the result is asserted against the target.

**The result is not a stable property of the deployment, so there is almost
nothing here to gate on.** Nine measurements, and no pattern that survives them:

    length   needle    budget   head  middle  tail
    31 302   numeric      256    ok     ok     ok
    41 532   numeric      256    ok      -      -
    51 862   numeric      256   FAIL     -      -
    62 222   numeric      256    ok      -      -
    76 472   numeric      256   FAIL    ok     ok
    76 473   lexical      256    ok     ok     ok
   127 053   lexical      256   FAIL    ok     ok
    31 082   lexical     2048    ok    FAIL    ok
    79 492   lexical     2048    ok    FAIL   FAIL

Not monotonic in length. Sensitive to the needle's wording. And — the one that
settled the design — **sensitive to the generation budget in the wrong
direction**: raising `--max-tokens` from 256 to 2048 turned two passing depths
into failures, with the model reasoning its way to "I don't have any information
about a maintenance passphrase" rather than reading it off the prompt. A small
budget starves the answer and a large one talks it away, and there is no value
that is right for both.

So this is a **regression detector, not a capability gate**. What the caller does
with it is compare two arms; the only absolute floor asserted is that at least
one depth retrieves, which is what separates a working long-context path from a
stack that has dropped every prefill chunk but the last. `check_acceptance`
enforces that floor and nothing more, and `compare` looks for a depth the stock
arm retrieved and the patched arm did not.

Do not quote a number from here as GLM-5.3-Flash's long-context capability. It is
not one.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

FILLER = "Background fact number {n} is irrelevant to any question. "
NEEDLE = "The maintenance passphrase for vault {vault} is {code}. "
QUESTION = "What is the maintenance passphrase for vault {vault}? Reply with only the passphrase."

#: Lexically distant from the filler and from each other. Hyphenated words rather
#: than digits, for the reason in the module docstring.
CODES = ("MAGENTA-QUASAR-17", "OBSIDIAN-FALCON-42", "CINNABAR-LATTICE-93")

#: Measured, at the low end of 12.0–12.7 so the estimate overshoots rather than
#: undershoots. Undershooting would silently make this a shorter test than it
#: claims, which is the one failure mode a long-context probe must not have.
TOKENS_PER_SENTENCE = 12.0


def haystack(sentences: int, depth: float, vault: str, code: str) -> str:
    at = max(0, min(sentences - 1, int(sentences * depth)))
    parts = []
    for i in range(sentences):
        if i == at:
            parts.append(NEEDLE.format(vault=vault, code=code))
        parts.append(FILLER.format(n=i))
    return "".join(parts)


def ask(url: str, model: str, prompt: str, max_tokens: int, timeout: int) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
        }
    ).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}
    )
    started = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read())
    out["_wall_s"] = round(time.time() - started, 2)
    return out


def one_run(url: str, model: str, label: str, target_tokens: int, depths: list[float],
            max_tokens: int, timeout: int, min_ratio: float) -> dict:
    sentences = max(1, int(target_tokens / TOKENS_PER_SENTENCE))
    results = []
    for i, depth in enumerate(depths):
        vault = f"K{i}{int(depth * 100):02d}"
        code = CODES[i % len(CODES)]
        prompt = haystack(sentences, depth, vault, code) + QUESTION.format(vault=vault)
        record = {"depth": depth, "vault": vault, "expected": code}
        try:
            answer = ask(url, model, prompt, max_tokens, timeout)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            record.update(ok=False, error=str(exc), prompt_tokens=None)
            results.append(record)
            continue
        message = answer["choices"][0]["message"]
        content = (message.get("content") or "").strip()
        usage = answer.get("usage") or {}
        got = usage.get("prompt_tokens")
        finish = answer["choices"][0].get("finish_reason")
        # An empty answer that hit the token cap is a budget failure, not a
        # retrieval failure, and the two are worth telling apart: the engine runs
        # with --reasoning-parser glm45, so the model thinks before it answers and
        # `content` stays empty until it stops. Measured: at max_tokens=256 this
        # deployment spent the whole budget reasoning about a 31k-token haystack
        # and returned nothing, which reads exactly like a needle it could not
        # find. Hence the default of 2048 and this named diagnosis.
        starved = not content and finish == "length"
        record.update(
            ok=code in content,
            starved=starved,
            # The reasoning parser splits thinking out of the answer. A model that
            # found the needle while thinking and lost it in the reply has still
            # failed to answer, so only `content` decides -- but the distinction
            # is worth recording, because the two have different causes.
            in_reasoning=code in (message.get("reasoning_content") or ""),
            got=content[:120],
            prompt_tokens=got,
            completion_tokens=usage.get("completion_tokens"),
            finish_reason=finish,
            wall_s=answer["_wall_s"],
            long_enough=bool(got and got >= target_tokens * min_ratio),
        )
        results.append(record)
        print(
            f"  {label} depth={depth:<5} ptok={got} wall={record['wall_s']}s "
            f"expect={code} got={record['got'][:40]!r} "
            + ("STARVED (raise --max-tokens)" if starved else ("ok" if record["ok"] else "FAIL"))
        )
    retrieved = sum(1 for r in results if r.get("ok"))
    return {
        "label": label,
        "target_tokens": target_tokens,
        "sentences": sentences,
        "min_token_ratio": min_ratio,
        "depths": results,
        "retrieved": retrieved,
        "of": len(results),
        # Kept for the record, and deliberately not the thing anything gates on:
        # see the table in the module docstring.
        "passed": retrieved == len(results),
        "long_enough": all(r.get("long_enough") for r in results),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    # "gated" is the shorter run and "frontier" the longer one. Neither is a
    # pass/fail on its own any more -- see the table above -- but the two lengths
    # are still worth having, because a difference between the arms that appears
    # at one length and not the other is information.
    ap.add_argument("--gated-tokens", type=int, default=76000)
    # 0 turns the frontier run off, which is what a first bring-up wants.
    ap.add_argument("--frontier-tokens", type=int, default=127000)
    ap.add_argument("--depths", default="0.02,0.5,0.98")
    # 2048 for a passphrase of four words. The engine runs with
    # --reasoning-parser glm45, so the budget is spent thinking first and
    # `content` only appears once the model stops; measured at 256, two of three
    # depths returned nothing at all on a 31k-token haystack.
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--min-token-ratio", type=float, default=0.95)
    # The only absolute floor. One depth retrieving separates a working
    # long-context path from a stack that dropped every prefill chunk but the
    # last; requiring more than that is requiring something no configuration
    # measured here achieves reliably.
    ap.add_argument("--min-retrieved", type=int, default=1)
    args = ap.parse_args()

    depths = [float(x) for x in args.depths.split(",") if x.strip()]
    runs = [
        one_run(args.url, args.model, "gated", args.gated_tokens, depths,
                args.max_tokens, args.timeout, args.min_token_ratio)
    ]
    if args.frontier_tokens > 0:
        runs.append(
            one_run(args.url, args.model, "frontier", args.frontier_tokens, depths,
                    args.max_tokens, args.timeout, args.min_token_ratio)
        )

    gated = runs[0]
    report = {
        "tokens_per_sentence": TOKENS_PER_SENTENCE,
        "chunked_prefill_size": 8192,
        "max_tokens": args.max_tokens,
        "min_retrieved": args.min_retrieved,
        "runs": runs,
        # The floor, not a capability claim: the prompt reached the length it
        # says it did, and at least one depth came back. Everything else this
        # produces is for the arm-to-arm comparison.
        "ok": gated["retrieved"] >= args.min_retrieved and gated["long_enough"],
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(
        f"needle: gated {gated['retrieved']}/{gated['of']} depths retrieved at "
        f"~{gated['target_tokens']} tokens (floor {args.min_retrieved})"
    )
    print("NEEDLE_OK" if report["ok"] else "NEEDLE_FAIL")
    # Exit 0 either way. A needle result is evidence for `compare` to weigh, and
    # a non-zero exit here would take the whole measurement task down with it,
    # losing the eval and the replay that had already succeeded.
    return 0


if __name__ == "__main__":
    sys.exit(main())
