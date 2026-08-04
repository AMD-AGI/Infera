#!/usr/bin/env python3
"""Split the AgentX profile export by turn index and report the SERVER-reported
cache hit rate for each group.

Why: the run-level cache figure (51.9 %) is far from the reference run's 88.1 %
on the same trace and the same deployment shape. A first-turn-heavy run would
explain it arithmetically; this asks the data instead of assuming.
"""
import json
import sys
from collections import defaultdict

path = sys.argv[1]
rows = [json.loads(l) for l in open(path) if l.strip()]


def flat(d, out=None, pre=""):
    out = {} if out is None else out
    for k, v in (d or {}).items():
        if isinstance(v, dict) and not {"value", "unit"} & set(v):
            flat(v, out, f"{pre}{k}.")
        else:
            out[f"{pre}{k}"] = v
    return out


# discover the key names once
sample = flat(rows[0])
print("sample keys:", [k for k in sorted(sample) if "cache" in k.lower() or "turn" in k.lower()
                       or "session" in k.lower() or "prompt" in k.lower()][:20])


def num(v):
    if isinstance(v, dict):
        v = v.get("value")
    return v


groups = defaultdict(lambda: {"n": 0, "read": 0, "prompt": 0})
per_session = defaultdict(int)

for r in rows:
    f = flat(r)
    sid = None
    turn = None
    read = None
    prompt = None
    for k, v in f.items():
        kl = k.lower()
        if "cache_read" in kl or "prompt_cache_read" in kl:
            read = num(v)
        elif kl.endswith("input_sequence_length") or kl.endswith("prompt_tokens"):
            prompt = num(v)
        elif "turn_index" in kl:
            turn = num(v)
        elif "session_id" in kl or "conversation_id" in kl:
            sid = v
    if read is None or not prompt:
        continue
    if turn is None and sid is not None:
        turn = per_session[sid]
        per_session[sid] += 1
    key = "first turn" if turn in (0, None) else "turn >= 1"
    g = groups[key]
    g["n"] += 1
    g["read"] += read or 0
    g["prompt"] += prompt

print()
print(f"{'group':12s} {'n':>5s} {'sum prompt':>12s} {'sum cached':>12s} {'hit %':>7s}")
tot_r = tot_p = 0
for k in ("first turn", "turn >= 1"):
    g = groups.get(k)
    if not g:
        continue
    tot_r += g["read"]
    tot_p += g["prompt"]
    print(f"{k:12s} {g['n']:5d} {g['prompt']:12,d} {g['read']:12,d} {100*g['read']/g['prompt']:6.1f}%")
if tot_p:
    print(f"{'ALL':12s} {sum(g['n'] for g in groups.values()):5d} {tot_p:12,d} {tot_r:12,d} {100*tot_r/tot_p:6.1f}%")
