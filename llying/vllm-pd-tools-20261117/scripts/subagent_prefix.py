#!/usr/bin/env python3
"""Can a prefix cache share what AIPerf counts as shared? For the sub-agent first turns of an AgentX run's measurement,
render the turn-0 messages with the model's chat template (as the engine does) and measure the token-level longest
common prefix (LCP) with turn-0 requests of the same trace:
  - dataset order: every earlier turn-0 of the trace (what AIPerf's theoretical hit assumes);
  - replay order: only conversations of the trace that this run actually sent earlier (sibling sub-agent turn 0s,
    and the main conversation's turn-0 messages once any main request was sent, since its replayed prompts carry the
    history from turn 0) -- the upper bound for a real prefix cache with unlimited capacity.
Run inside the engine image (transformers + the model's tokenizer).
  subagent_prefix.py DATASET_DIR MODEL PROFILE_EXPORT_JSONL [MAX_TRACES=60]
DATASET_DIR holds AIPerf's memory-mapped dataset (index.dat, dataset.dat) saved from a run with the same trace / seed.
"""
import collections
import json
import sys

from transformers import AutoTokenizer

ddir, model, prof = sys.argv[1:4]
max_traces = int(sys.argv[4]) if len(sys.argv) > 4 else 60
idx = json.load(open(f"{ddir}/index.dat"))
offs = idx["offsets"]
data = open(f"{ddir}/dataset.dat", "rb")
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)


def conv(cid):
    o = offs[cid]
    data.seek(o["offset"])
    return json.loads(data.read(o["size"]))


first_sent = {}  # conversation id -> start time of its first request in this run (any phase)
first_turn0 = {}  # conversation id -> start time of its turn-0 request, if replayed
measured = collections.defaultdict(list)  # trace -> sub-agent conversation ids with turn 0 in the measurement
for line in open(prof):
    r = json.loads(line)
    md = r.get("metadata", {})
    cid, t = md.get("conversation_id"), md["request_start_ns"] / 1e9
    first_sent[cid] = min(first_sent.get(cid, t), t)
    if md.get("turn_index") == 0:
        first_turn0[cid] = min(first_turn0.get(cid, t), t)
        if md.get("benchmark_phase") == "profiling" and md.get("source_kind") == "weka_subagent":
            measured[md["source_trace_id"]].append(cid)
print(f"{len(measured)} traces with sub-agent first turns in the measurement; analysing {min(len(measured), max_traces)}")
by_trace = collections.defaultdict(list)
for cid in offs:
    by_trace[cid.split("::")[0]].append(cid)


def render(msgs):
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok.encode(text, add_special_tokens=False)


def lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


tot = collections.Counter()
shown = 0
for trace, cids in list(measured.items())[:max_traces]:
    t0s = {}
    for cid in by_trace.get(trace, []):
        c = conv(cid)
        t = c["turns"][0]
        msgs = [m for m in (t.get("raw_messages") or []) if isinstance(m, dict)]
        if t.get("raw_system"):
            msgs = [{"role": "system", "content": t["raw_system"]}] + msgs
        t0s[cid] = dict(ts=t.get("timestamp") or 0.0, kind=t.get("source_kind"), hit=t.get("theoretical_prefix_cache_hit_blocks"),
                        tot=t.get("theoretical_prefix_cache_total_blocks"), ids=render(msgs))
    for cid in sorted(set(cids), key=lambda c: first_turn0[c]):
        f = t0s.get(cid)
        if f is None:
            continue
        sent = first_turn0[cid]
        ds = max((lcp(f["ids"], g["ids"]) for k, g in t0s.items() if k != cid and g["ts"] < f["ts"]), default=0)
        rp = 0
        for k, g in t0s.items():
            if k == cid:
                continue
            if g["kind"] == "weka_main":
                seen = first_sent.get(k, float("inf")) < sent
            else:
                seen = first_turn0.get(k, float("inf")) < sent
            if seen:
                rp = max(rp, lcp(f["ids"], g["ids"]))
        n = len(f["ids"])
        theo = int((f["hit"] or 0) * n / f["tot"]) if f["tot"] else 0
        tot["n"] += 1
        tot["tokens"] += n
        tot["dataset"] += ds
        tot["replay"] += rp
        tot["theo"] += theo
        if shown < 20:
            print(f"  {cid[:60]:60s} tokens {n:7d}  LCP dataset-order {ds:7d}  replay-order {rp:7d}  theoretical ~{theo}")
            shown += 1
T = max(tot["tokens"], 1)
print(f"sub-agent first turns in the measurement: {tot['n']}, tokens {tot['tokens']:,}; shareable prefix: AIPerf theoretical "
      f"~{tot['theo']:,} ({tot['theo'] / T:.1%}), dataset order {tot['dataset']:,} ({tot['dataset'] / T:.1%}), replay order "
      f"{tot['replay']:,} ({tot['replay'] / T:.1%})")
