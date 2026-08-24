#!/usr/bin/env python3
"""Needle-in-a-haystack probe targeting the mooncake chunked-prefill KV race (#33970).

The defect corrupts every prefill chunk EXCEPT the last: non-final chunks are handed to
the mooncake transfer worker, which reads device memory outside the CUDA stream, while
the forward that writes those pages may still be running. So the probe is only
meaningful if the needle lands in a non-final chunk — a needle in the final chunk is
retrieved correctly even on a broken build, and would read as a false PASS.

Hence: place needles at several depths, tokenize with the real tokenizer so the chunk
index of each depth is known rather than assumed, and report per-depth so the
"corruption boundary lands on the chunk boundary" claim is checkable rather than
asserted.

Usage:
  python3 needle.py --url http://IP:8100 --model glm5.2-mxfp4 \
      --tokenizer /data/models/GLM-5.2-MXFP4 --chunk-size 131072 \
      --prompt-tokens 200000 --depths 0.1,0.2,...  --out result.json
"""

import argparse
import concurrent.futures as cf
import json
import random
import re
import sys
import urllib.error
import urllib.request

# Filler must be cheap to tokenize and carry no digits, so the only digits in the whole
# prompt are the needle's. That makes a wrong answer unambiguous rather than a
# near-miss against some other number in the haystack.
FILLER_WORDS = (
    "the quick brown fox jumps over a lazy dog while grass grows near the quiet river "
    "and clouds drift above the hills as wind moves through the tall pine trees "
).split()


def build_prompt(tok, total_tokens, depth, secret):
    """Return (prompt, needle_token_offset). The needle sits at `depth` of the body."""
    needle = (
        f" The special magic number for this document is {secret}. "
        f"Remember it. "
    )
    question = (
        "\n\nQuestion: What is the special magic number for this document? "
        "Reply with ONLY the digits, nothing else."
    )
    n_needle = len(tok.encode(needle, add_special_tokens=False))
    n_question = len(tok.encode(question, add_special_tokens=False))
    body_budget = total_tokens - n_needle - n_question
    if body_budget <= 0:
        raise SystemExit("total_tokens too small for the needle and question")

    rng = random.Random(1234)
    # Grow to ~4x the budget in words first, then trim by tokens: one pass of encode on a
    # 200k-token string is slow enough that a grow-and-check loop dominates the runtime.
    words = [rng.choice(FILLER_WORDS) for _ in range(int(body_budget * 1.2) + 64)]
    body_ids = tok.encode(" ".join(words), add_special_tokens=False)
    while len(body_ids) < body_budget:
        words += [rng.choice(FILLER_WORDS) for _ in range(body_budget)]
        body_ids = tok.encode(" ".join(words), add_special_tokens=False)
    body_ids = body_ids[:body_budget]

    cut = int(len(body_ids) * depth)
    head = tok.decode(body_ids[:cut])
    tail = tok.decode(body_ids[cut:])
    prompt = head + needle + tail + question
    needle_offset = len(tok.encode(head, add_special_tokens=False))
    return prompt, needle_offset


def ask(url, model, prompt, max_tokens, timeout):
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            # Greedy: a sampling-induced miss would be indistinguishable from a
            # corruption-induced one.
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
    ).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    msg = d["choices"][0]["message"]
    return (msg.get("content") or ""), (msg.get("reasoning_content") or ""), d.get("usage", {})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--model", default="glm5.2-mxfp4")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--prompt-tokens", type=int, default=200000)
    # The RESOLVED per-step prefill budget, not the value passed on the launch
    # command line. With DP-attention on, sglang divides the global budget by
    # dp_size -- 131072 at dp_size=4 resolves to 32768 -- and the chunk index of
    # every needle is computed from this. Getting it wrong mislabels which
    # needles are non-final, which is the entire control in this probe.
    # --resolved-chunk-from reads it off the leg's own server_args line so it
    # cannot drift; --chunk-size stays as a manual override.
    p.add_argument("--chunk-size", type=int, default=131072)
    p.add_argument(
        "--resolved-chunk-from",
        default="",
        help="path to the leg's log; parse chunked_prefill_size= off its server_args line",
    )
    p.add_argument("--depths", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--out", default="")
    a = p.parse_args()

    if a.resolved_chunk_from:
        # trap 7: read the value the engine actually resolved, never the launch arg.
        txt = open(a.resolved_chunk_from, errors="replace").read()
        hits = re.findall(r"chunked_prefill_size=(\d+)", txt)
        if not hits:
            raise SystemExit(
                f"no chunked_prefill_size= in {a.resolved_chunk_from}; "
                "the leg may not have logged server_args yet"
            )
        resolved = int(hits[-1])
        if resolved != a.chunk_size:
            print(
                f"NOTE: resolved chunked_prefill_size={resolved} differs from "
                f"--chunk-size={a.chunk_size} (DP-attention divides by dp_size). "
                f"Using {resolved}.",
                flush=True,
            )
        a.chunk_size = resolved

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    depths = [float(x) for x in a.depths.split(",")]
    rng = random.Random(20260819)

    cases = []
    for d in depths:
        secret = str(rng.randint(1_000_000, 9_999_999))
        prompt, off = build_prompt(tok, a.prompt_tokens, d, secret)
        chunk_idx = off // a.chunk_size
        n_chunks = (a.prompt_tokens + a.chunk_size - 1) // a.chunk_size
        cases.append(
            {
                "depth": d,
                "secret": secret,
                "prompt": prompt,
                "needle_token_offset": off,
                "needle_chunk": chunk_idx,
                "n_chunks_est": n_chunks,
                "is_final_chunk": chunk_idx >= n_chunks - 1,
            }
        )

    print(
        f"prompt_tokens={a.prompt_tokens} chunk_size={a.chunk_size} "
        f"-> ~{cases[0]['n_chunks_est']} chunks",
        flush=True,
    )

    def run(c):
        try:
            content, reasoning, usage = ask(
                a.url, a.model, c["prompt"], a.max_tokens, a.timeout
            )
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            return {**{k: v for k, v in c.items() if k != "prompt"},
                    "ok": False, "error": f"{type(e).__name__}: {e}"}
        # GLM-5.2 is a reasoning model: with thinking on it emits the answer in
        # `reasoning_content` and leaves `content` empty until it finishes, so
        # scoring `content` alone reports a miss for a needle the model plainly
        # retrieved. That failure mode is arm-independent -- it would depress
        # stock AND patched -- which is exactly what would make the A/B useless.
        # The filler carries no digits, so the secret appearing anywhere in the
        # response is unambiguous.
        digits = re.findall(r"\d+", f"{content}\n{reasoning}")
        ok = c["secret"] in digits
        return {
            **{k: v for k, v in c.items() if k != "prompt"},
            "ok": ok,
            "got": content.strip()[:200],
            "reasoning_head": reasoning.strip()[:120],
            "prompt_tokens": usage.get("prompt_tokens"),
        }

    with cf.ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        results = list(ex.map(run, cases))

    n_ok = sum(1 for r in results if r.get("ok"))
    print(f"\n{'depth':>6} {'chunk':>6} {'final':>6} {'ok':>4}  want / got", flush=True)
    for r in results:
        print(
            f"{r['depth']:>6.2f} {r['needle_chunk']:>6} "
            f"{str(r['is_final_chunk']):>6} {str(r.get('ok')):>4}  "
            f"{r['secret']} / {r.get('got', r.get('error', ''))!r}",
            flush=True,
        )
    print(f"\nSCORE: {n_ok}/{len(results)}", flush=True)

    # Non-final chunks are the ones the defect can corrupt; the final-chunk score is the
    # built-in control that separates "the race" from "the model just cannot do this".
    nf = [r for r in results if not r["is_final_chunk"]]
    fi = [r for r in results if r["is_final_chunk"]]
    if nf:
        print(f"  non-final-chunk needles: {sum(1 for r in nf if r.get('ok'))}/{len(nf)}")
    if fi:
        print(f"  final-chunk needles    : {sum(1 for r in fi if r.get('ok'))}/{len(fi)}")

    if a.out:
        with open(a.out, "w") as f:
            json.dump({"args": vars(a), "results": results, "score": [n_ok, len(results)]}, f, indent=2)
        print(f"  -> {a.out}")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
