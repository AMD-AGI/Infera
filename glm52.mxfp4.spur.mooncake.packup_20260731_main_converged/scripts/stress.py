#!/usr/bin/env python3
"""Concurrent stress against a PD router. Exit criterion (4): conc=128 x 512 tokens.

Reports per-request outcome, dp_rank spread, and spec_accept_length so a pass
cannot be confused with "spec-dec silently off". Writes raw jsonl for evidence.
"""
import concurrent.futures as cf
import json
import sys
import time
import urllib.request

URL = sys.argv[1]
CONC = int(sys.argv[2]) if len(sys.argv) > 2 else 128
MAXTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 512
OUT = sys.argv[4] if len(sys.argv) > 4 else "/tmp/stress.jsonl"
TMO = float(sys.argv[5]) if len(sys.argv) > 5 else 600

PROMPTS = [
    "Explain the theory of relativity in detail.",
    "Write a detailed essay about the history of computing.",
    "Describe how photosynthesis works at the molecular level.",
    "What are the main causes and effects of climate change?",
    "Explain quantum entanglement and its applications.",
    "Describe the process of protein synthesis in cells.",
    "What is the significance of the Renaissance in European history?",
    "Explain how neural networks learn from data.",
]


def one(i):
    body = json.dumps(
        {
            "text": PROMPTS[i % len(PROMPTS)],
            "sampling_params": {"temperature": 0.0, "max_new_tokens": MAXTOK},
            "rid": f"stress-{i}",
        }
    ).encode()
    req = urllib.request.Request(
        URL + "/generate", data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=TMO) as r:
            d = json.loads(r.read())
        mi = d.get("meta_info", {})
        return {
            "i": i,
            "ok": True,
            "sec": round(time.time() - t0, 2),
            "dp": mi.get("dp_rank"),
            "tok": mi.get("completion_tokens"),
            "acc": mi.get("spec_accept_length"),
            "retr": mi.get("num_retractions"),
            "text": d.get("text", "")[:200],
        }
    except Exception as e:
        return {
            "i": i,
            "ok": False,
            "sec": round(time.time() - t0, 2),
            "err": f"{type(e).__name__}: {e}",
        }


t0 = time.time()
with cf.ThreadPoolExecutor(max_workers=CONC) as ex:
    res = list(ex.map(one, range(CONC)))
el = time.time() - t0

with open(OUT, "w") as f:
    for r in res:
        f.write(json.dumps(r) + "\n")

ok = [r for r in res if r["ok"]]
bad = [r for r in res if not r["ok"]]
full = [r for r in ok if r.get("tok") == MAXTOK]
accs = [r["acc"] for r in ok if r.get("acc")]
dps = sorted({r.get("dp") for r in ok})

print(f"conc={CONC} maxtok={MAXTOK} elapsed={el:.1f}s")
print(f"ok      : {len(ok)}/{CONC}")
print(f"full tok: {len(full)}/{CONC}")
print(f"dp ranks: {dps}")
if accs:
    print(f"acc_len : min={min(accs):.2f} mean={sum(accs)/len(accs):.2f} max={max(accs):.2f}")
for r in bad[:5]:
    print(f"  FAIL[{r['i']}] {r['sec']}s {r['err']}")
print(f"raw -> {OUT}")
sys.exit(0 if len(bad) == 0 else 1)
