#!/usr/bin/env python3
"""Step-2 workload: drive prefix reuse so kvd and kv-aware scoring actually work.

Step 1 proved kvaware+kvd are wired and don't break correctness, but kvd served
zero traffic (gets=0 sets=0): four short, prefix-disjoint prompts give the
offload path nothing to do, and the overlap scorer nothing to score.

This sends N sessions x M turns where every turn in a session repeats a long
shared system prefix. That is the shape kv-aware routing and an L3 cache are
built for:

  phase 1 (warm)  — each session's first turn is a MISS; the prefix gets stored.
  phase 2 (reuse) — same prefixes again; should hit the radix cache / kvd.

The prefix must exceed the hicache prefetch_threshold (64 tokens as infera
configures it, 256 by sglang default) or the prefetch is skipped outright:
  hiradix_cache.py:1603  `or prefetch_length < self.prefetch_threshold: return`

Also checks correctness on every response, so a cache hit that returns *wrong*
text can't pass silently — the whole point is correctness under reuse.
"""

import argparse
import json
import sys
import time
import urllib.request

# ~900 tokens of stable preamble. Long enough to clear any threshold, and
# deterministic so every session's prefix hashes identically.
#
# NOTE on the wording: an earlier version said "Answer strictly from the
# reference material below". GLM-5.2 obeyed it perfectly and refused every
# question ("the reference material provided is about system components...")
# -> 1/32, which measured my prompt, not the cache. The reference block is
# only here to be a long, stable, shared prefix; it must not gate the answer.
PREFIX = (
    "You are a helpful assistant. Answer the user's question directly using "
    "general knowledge. The reference material below is background context "
    "only -- ignore it unless the question is about it. Reply with just the "
    "answer, no explanation, no reasoning.\n\n=== BACKGROUND (ignore) ===\n"
) + "\n".join(
    f"Section {i}: The system component number {i} is responsible for "
    f"subsystem {i % 7}, operates in mode {i % 3}, and reports to "
    f"controller {i % 5}. Its nominal throughput is {1000 + i * 7} units "
    f"per second under standard load conditions."
    for i in range(120)
)

# (question, expected substring) — factual, temp=0, verifiable.
TURNS = [
    ("What is the capital of France? Answer with just the city name.", "paris"),
    ("What is 2+2? Answer with just the number.", "4"),
    ("What is the largest planet in our solar system? One word.", "jupiter"),
    ("What is the capital of Japan? Answer with just the city name.", "tokyo"),
]


def ask(base, model, session, question, timeout):
    """One request. The system message is identical across all sessions ->
    that is the shared prefix. A per-session marker follows it so sessions are
    distinguishable without breaking the common prefix."""
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": PREFIX},
            {"role": "user", "content": f"[session {session}] {question}"},
        ],
        # 128, not 32: GLM-5.2 emits a short reasoning preamble before the
        # answer, and 32 truncated mid-thought so the expected substring never
        # appeared. Matches what probe.py needs on this model.
        "max_tokens": 128,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        f"{base}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return d["choices"][0]["message"]["content"], time.time() - t0


def phase(name, base, model, sessions, timeout):
    print(f"\n--- {name} ---", flush=True)
    ok = total = 0
    lat = []
    for s in range(sessions):
        for question, want in TURNS:
            total += 1
            try:
                txt, dt = ask(base, model, s, question, timeout)
            except Exception as e:
                print(f"  [ERR] s{s} {question[:30]!r}: {e}", flush=True)
                continue
            lat.append(dt)
            hit = want.lower() in txt.lower()
            ok += hit
            if not hit:
                print(f"  [XX] s{s} want={want!r} got={txt[:90]!r}", flush=True)
    med = sorted(lat)[len(lat) // 2] if lat else float("nan")
    print(f"  {ok}/{total} correct | median latency {med:.2f}s", flush=True)
    return ok, total, med


def main():
    p = argparse.ArgumentParser()
    p.add_argument("base")
    p.add_argument("model", nargs="?", default="glm5.2-mxfp4")
    p.add_argument("--sessions", type=int, default=4)
    p.add_argument("--timeout", type=int, default=300)
    a = p.parse_args()

    print(f"prefix chars={len(PREFIX)} (~{len(PREFIX)//4} tokens est.) "
          f"sessions={a.sessions} turns/session={len(TURNS)}")

    ok1, tot1, med1 = phase("PHASE 1 (cold — populates cache)",
                            a.base, a.model, a.sessions, a.timeout)
    ok2, tot2, med2 = phase("PHASE 2 (reuse — same prefixes)",
                            a.base, a.model, a.sessions, a.timeout)

    ok, tot = ok1 + ok2, tot1 + tot2
    print(f"\nTOTAL {ok}/{tot} correct")
    print(f"median latency: phase1 {med1:.2f}s -> phase2 {med2:.2f}s")
    # Correctness is the gate. Latency is reported, not asserted: a warm radix
    # cache can serve the reuse phase without kvd ever being touched, so a
    # speedup here does NOT by itself prove kvd did anything. Read kvd's own
    # gets/sets counters for that.
    sys.exit(0 if ok == tot else 1)


if __name__ == "__main__":
    main()
