#!/usr/bin/env python3
"""Concurrency stress that CAPTURES EVERY OUTPUT and classifies degeneration.

bench_serving only reports completion; a "digit loop" completes normally and is invisible to it.
This sends N prompts at a fixed concurrency through the router, stores every full output, and
classifies each one. Sampling follows the model's own generation_config (see below), NOT greedy.

Verdicts:
  DIGIT_LOOP        a short numeric-ish span repeats many times  <- the reported bug
  CORRUPT_REASONING token salad inside the reasoning             <- the FIXED KV race (want 0)
  TAIL_REPEAT       coherent + right answer, only the post-</think> tail loops
  WRONG             coherent but the expected answer is absent
  TRUNCATED         hit max_tokens mid-reasoning (probe artifact, not a bug)
  CLEAN             coherent + right answer

Prompts are ~ISL tokens of filler with an embedded arithmetic fact whose answer is a number, so
correctness is checkable and the natural answer is numeric (the mode we are hunting sits right
next to a legitimate numeric answer -- the classifier must not confuse them).

usage: stress_capture.py BASE MODEL CONC NPROMPTS [ISL] [OSL] [OUT_JSON] [SALT]
"""
import json, re, sys, time, threading, queue, urllib.request, collections

BASE   = sys.argv[1]
MODEL  = sys.argv[2]
CONC   = int(sys.argv[3])
N      = int(sys.argv[4])
ISL    = int(sys.argv[5]) if len(sys.argv) > 5 else 1024
OSL    = int(sys.argv[6]) if len(sys.argv) > 6 else 1024
OUT    = sys.argv[7] if len(sys.argv) > 7 else "/tmp/stress_capture.json"
SALT   = int(sys.argv[8]) if len(sys.argv) > 8 else 0

# From GLM-5.2's own generation_config.json. top_k is not in that file; 40 is the
# value GLM's serving docs use. Overridable for a deliberate A/B.
import os as _os
TEMPERATURE = float(_os.environ.get("STRESS_TEMPERATURE", "1.0"))
TOP_P       = float(_os.environ.get("STRESS_TOP_P", "0.95"))
TOP_K       = int(_os.environ.get("STRESS_TOP_K", "40"))

TOK_PER_LINE = 28.1
FILLER = ("Record {i}: the maintenance crew inspected corridor {a} and logged "
          "routine status code {b} with no anomalies reported that shift.")


def build(idx):
    """Deterministic per-idx prompt. Answer = a 5-digit number unique to idx."""
    secret = 10000 + (idx * 7919 + SALT * 131) % 89999
    n = max(6, int(ISL / TOK_PER_LINE))
    s = idx * 977 + SALT
    lines = [FILLER.format(i=i + s, a=(i * 7 + s) % 400,
                           b=1000 + (i * 13 + s * 31) % 8000) for i in range(n)]
    lines.insert(n // 2, f"Record SECRET-{idx}: the calibration constant for the orbital "
                         f"gyroscope is exactly {secret}.")
    p = ("Below is a long maintenance log. Read it carefully, then answer the question at the "
         "end using only information from the log.\n\n<log>\n" + "\n".join(lines) +
         "\n</log>\n\nQuestion: What is the calibration constant for the orbital gyroscope? "
         "Explain your reasoning briefly, then state the number.")
    return p, str(secret)


# ---------------- classifiers ----------------
def split_reasoning(t):
    i = t.find("</think>")
    return (t, "") if i < 0 else (t[:i], t[i:])


def digit_loop(t):
    """A short numeric-ish span repeated many times, anywhere in the output.

    Deliberately narrow so a legitimate numeric answer ('the number is 82931') never fires:
      (a) same digit-run repeated >=6 times with <=8 chars of junk between, or
      (b) one digit-run occupies a large share of a long output, or
      (c) a <=12-char span that is >=40% digits repeated >=8 times back to back.
    """
    if re.search(r"(\d{1,8})(?:[^\w\n]{0,8}\1){5,}", t):
        return True
    if re.search(r"([^\n]{1,12})\1{7,}", t) and sum(c.isdigit() for c in t) > 0.30 * max(1, len(t)):
        return True
    # same number literal repeated a lot AND dominating the output. The density test matters:
    # a legitimate long chain-of-thought quotes its answer ~12x over 1.5 KB and must NOT fire
    # (observed false positive: idx=460, 12x '52780' in 1484 clean chars).
    runs = re.findall(r"\d+", t)
    if len(t) > 300 and runs:
        lit, k = collections.Counter(runs).most_common(1)[0]
        if k >= 12 and len(lit) >= 2 and k * len(lit) > 0.25 * len(t):
            return True
    return False


def salad(t):
    # NOTE: CJK is deliberately NOT a salad signal. GLM-5.2 sometimes reasons in
    # Chinese on an English prompt and answers correctly -- a `>5 CJK chars` rule
    # flagged one such response as CORRUPT_REASONING in a conc=128 run where the
    # needle was retrieved, `finish=stop`, and the text was entirely coherent.
    # Corrupted output is caught by the token-boundary and repetition signals
    # below, which are script-independent.
    return (len(re.findall(r"\b[Tt]he\d", t)) > 1
            or len(re.findall(r"\d\s+the\b", t)) > 2
            # The documented chunk-boundary signature is a truncated needle with
            # the closing tag cycling: "2183</think>2183</think>218</think>...".
            # The word-repetition rule below cannot see it -- the digits differ
            # between tags, so no token is adjacent to a copy of itself -- and it
            # used to fall through to WRONG, which the BAD tally ignores. A
            # healthy response emits `</think>` exactly once.
            or t.count("</think>") >= 3
            or len(re.findall(r"(\b\w+\b)(?:\W+\1\b){4,}", t)) > 0)


def classify(txt, expect, finish):
    reasoning, tail = split_reasoning(txt)
    if digit_loop(txt):
        return "DIGIT_LOOP"
    found = expect in txt
    # A response carrying the right needle and stopping on its own is not corrupt,
    # whatever the heuristics below think of its prose. Checking this FIRST keeps a
    # stylistic signal from masquerading as a KV-corruption hit.
    if found and finish == "stop" and not digit_loop(txt):
        return "TAIL_REPEAT" if tail.count("</think>") > 1 else "CLEAN"
    # salad on the reasoning AND on the whole text: when the salad starts before the first
    # </think>, `reasoning` is a 2-word stub and the evidence all sits in the tail.
    if salad(reasoning) or salad(txt):
        return "CORRUPT_REASONING"
    if found and tail.count("</think>") > 1:
        return "TAIL_REPEAT"
    if found:
        return "CLEAN"
    if finish == "length":
        return "TRUNCATED"
    return "WRONG"


# ---------------- driver ----------------
# IDX="71,84,101" replays exactly those prompt indices (optionally REP times each) instead of
# range(N). Lets us re-run the failing subset at conc=1 -- prompt content is a pure function of
# idx+salt, so it is byte-identical to what failed under load.
import os
_idx = os.environ.get("IDX", "")
if _idx:
    REP = int(os.environ.get("REP", "1"))
    IDS = [int(x) for x in _idx.split(",") if x.strip() != ""] * REP
else:
    IDS = list(range(N))
N = len(IDS)

q, out, lock = queue.Queue(), [], threading.Lock()
for i in IDS:
    q.put(i)


def worker():
    while True:
        try:
            idx = q.get_nowait()
        except queue.Empty:
            return
        p, expect = build(idx)
        # GLM-5.2's own generation_config.json (temperature 1.0, top_p 0.95),
        # NOT greedy. Greedy decoding sends a reasoning model into repetition on
        # a long prompt, and EAGLE/MTP amplifies it -- the draft model predicts
        # the loop perfectly, `accept len` pins at its maximum, and the response
        # runs to max_tokens. That reads exactly like the KV corruption this gate
        # exists to catch. See the same fix and its measurement in needle.py.
        body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": p}],
                           "max_tokens": OSL,
                           "temperature": TEMPERATURE,
                           "top_p": TOP_P,
                           "top_k": TOP_K}).encode()
        req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        try:
            d = json.load(urllib.request.urlopen(req, timeout=3600))
            txt = d["choices"][0]["message"]["content"] or ""
            fin = d["choices"][0].get("finish_reason", "")
            rec = {"idx": idx, "expect": expect, "verdict": classify(txt, expect, fin),
                   "finish": fin, "latency_s": round(time.time() - t0, 2),
                   "prompt_tokens": d["usage"]["prompt_tokens"],
                   "completion_tokens": d["usage"]["completion_tokens"], "output": txt}
        except Exception as e:
            rec = {"idx": idx, "expect": expect, "verdict": "ERROR",
                   "latency_s": round(time.time() - t0, 2), "output": f"{type(e).__name__}: {e}"}
        with lock:
            out.append(rec)
            n = len(out)
            if rec["verdict"] not in ("CLEAN", "TAIL_REPEAT"):
                print(f"  [{n}/{N}] idx={idx} {rec['verdict']} lat={rec['latency_s']}s "
                      f"-> {rec['output'][:90]!r}", flush=True)
            elif n % 32 == 0:
                print(f"  [{n}/{N}] ...", flush=True)


print(f"== conc={CONC} n={N} isl={ISL} osl={OSL} salt={SALT} -> {OUT}", flush=True)
T0 = time.time()
ths = [threading.Thread(target=worker, daemon=True) for _ in range(CONC)]
[t.start() for t in ths]
[t.join() for t in ths]
dur = time.time() - T0

out.sort(key=lambda x: x["idx"])
json.dump({"conc": CONC, "n": N, "isl": ISL, "osl": OSL, "salt": SALT,
           "duration_s": round(dur, 1), "rows": out}, open(OUT, "w"), indent=1)

c = collections.Counter(x["verdict"] for x in out)
print(f"\n== conc={CONC} n={N} dur={dur:.1f}s")
for k, v in sorted(c.items()):
    print(f"   {k:18s} {v}")
bad = sum(v for k, v in c.items() if k in ("DIGIT_LOOP", "CORRUPT_REASONING"))
print(f"== BAD (digit_loop+corrupt) = {bad}/{len(out)}")
print(f"-> {OUT}")
