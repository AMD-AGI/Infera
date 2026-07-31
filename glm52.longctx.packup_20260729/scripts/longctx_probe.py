#!/usr/bin/env python3
"""Long-context correctness probe for GLM-5.2-MXFP4 (needle-in-a-haystack).

Builds a ~TARGET-token haystack of numbered filler lines, buries N distinct needles
at controlled depths, and asks the model to retrieve each one (temp=0).
Verifies the *real* prompt length via usage.prompt_tokens returned by the server.

usage: longctx_probe.py [BASE_URL] [MODEL] [TARGET_TOKENS] [OUT_JSON]
"""
import json, sys, time, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:30000"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "glm5.2-mxfp4"
TARGET = int(sys.argv[3]) if len(sys.argv) > 3 else 65536
OUT = sys.argv[4] if len(sys.argv) > 4 else "/tmp/longctx_result.json"

# Filler: neutral, non-repetitive-enough lines so the needle is not trivially unique by novelty.
FILLER = ("Record {i}: the maintenance crew inspected corridor {a} and logged "
          "routine status code {b} with no anomalies reported that shift.")

NEEDLES = [
    # (depth fraction, sentence to bury, question, expected substring)
    (0.05, "Record SECRET-A: the emergency access passphrase for vault seven is "
           "'crimson-lantern-4417'.",
     "What is the emergency access passphrase for vault seven? Answer with the passphrase only.",
     "crimson-lantern-4417"),
    (0.50, "Record SECRET-B: the calibration constant for the orbital gyroscope is "
           "exactly 82931.",
     "What is the calibration constant for the orbital gyroscope? Answer with the number only.",
     "82931"),
    (0.95, "Record SECRET-C: the backup generator was relocated to building "
           "'Kestrel-Nine' on March 14th.",
     "Which building was the backup generator relocated to? Answer with the building name only.",
     "kestrel-nine"),
]


def post(messages, max_tokens=64, timeout=1800):
    body = json.dumps({"model": MODEL, "messages": messages,
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return d, time.time() - t0


def build(n_lines, needle_specs):
    """needle_specs: list of (depth, text). Returns haystack string."""
    lines = [FILLER.format(i=i, a=(i * 7) % 400, b=1000 + (i * 13) % 8000)
             for i in range(n_lines)]
    for depth, text in sorted(needle_specs, key=lambda x: -x[0]):
        lines.insert(max(0, min(n_lines, int(n_lines * depth))), text)
    return "\n".join(lines)


def make_prompt(hay, question):
    return [{"role": "user", "content":
             "Below is a long maintenance log. Read it carefully, then answer the "
             "question at the end using only information from the log.\n\n"
             "<log>\n" + hay + "\n</log>\n\nQuestion: " + question}]


# ---- calibrate line count to hit TARGET prompt tokens ----
# measured 28.1 tok/line for this filler; undershoot deliberately so the calibration probe
# never exceeds the server's context window (that returns HTTP 400, not a clamp).
n = int(TARGET / 32)
calib = []
for attempt in range(4):
    hay = build(n, [(d, t) for d, t, _, _ in NEEDLES])
    d, _ = post(make_prompt(hay, "Reply with the single word OK."), max_tokens=4)
    pt = d["usage"]["prompt_tokens"]
    calib.append({"lines": n, "prompt_tokens": pt})
    print(f"[calib] lines={n} -> prompt_tokens={pt}", flush=True)
    if abs(pt - TARGET) <= TARGET * 0.02:
        break
    # 0.99 factor: always approach the target from below, never overshoot the ctx window
    n = max(16, int(n * TARGET / pt * 0.99))

results = {"base": BASE, "model": MODEL, "target_tokens": TARGET,
           "calibration": calib, "cases": []}
ok = 0
for depth, text, question, expect in NEEDLES:
    hay = build(n, [(depth, text)])          # one needle at a time = clean signal
    resp, dt = post(make_prompt(hay, question), max_tokens=96)
    txt = resp["choices"][0]["message"]["content"]
    pt = resp["usage"]["prompt_tokens"]
    ct = resp["usage"]["completion_tokens"]
    hit = expect.lower() in txt.lower()
    ok += hit
    print(f"[{'OK' if hit else 'XX'}] depth={depth:.0%} prompt_tokens={pt} "
          f"gen={ct} {dt:.1f}s -> {txt.strip()[:200]!r}", flush=True)
    results["cases"].append({"depth": depth, "prompt_tokens": pt, "completion_tokens": ct,
                             "latency_s": round(dt, 2), "expect": expect,
                             "output": txt, "pass": bool(hit)})

# ---- multi-needle: all three at once, one question each turn is overkill; ask for all ----
hay = build(n, [(d, t) for d, t, _, _ in NEEDLES])
q = ("List, one per line: (1) the vault seven passphrase, (2) the orbital gyroscope "
     "calibration constant, (3) the building the backup generator was relocated to.")
resp, dt = post(make_prompt(hay, q), max_tokens=160)
txt = resp["choices"][0]["message"]["content"]
pt = resp["usage"]["prompt_tokens"]
hits = [e for _, _, _, e in NEEDLES if e.lower() in txt.lower()]
print(f"[multi] prompt_tokens={pt} {dt:.1f}s hits={len(hits)}/3 -> {txt.strip()[:400]!r}",
      flush=True)
results["multi"] = {"prompt_tokens": pt, "latency_s": round(dt, 2),
                    "hits": len(hits), "output": txt}

results["single_needle_pass"] = f"{ok}/{len(NEEDLES)}"
with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nsingle-needle {ok}/{len(NEEDLES)}, multi-needle {len(hits)}/3 -> {OUT}")
sys.exit(0 if (ok == len(NEEDLES) and len(hits) == 3) else 1)
