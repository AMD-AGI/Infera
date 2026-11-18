#!/usr/bin/env python3
"""Can a prefix cache share what AIPerf counts as shared? For sub-agent first turns (and other turn-0 requests) of the
AgentX weka trace, render the turn-0 messages with the model's chat template (as the engine does) and measure the
token-level longest common prefix with every earlier turn-0 request of the same trace (main and sibling sub-agents,
in dataset timestamp order), next to AIPerf's theoretical hit (theoretical_prefix_cache_hit_blocks x block size).
Run inside the engine image (transformers + the model's tokenizer).
  subagent_prefix.py DATASET_DIR MODEL PROFILE_EXPORT_JSONL [MAX_TRACES=60]
DATASET_DIR holds AIPerf's memory-mapped dataset (index.dat, dataset.dat) saved from a run.
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


# traces with a sub-agent first turn in the measurement phase
traces = collections.OrderedDict()
for line in open(prof):
    r = json.loads(line)
    md = r.get("metadata", {})
    if md.get("benchmark_phase") == "profiling" and md.get("source_kind") == "weka_subagent" and md.get("turn_index") == 0:
        traces.setdefault(md["source_trace_id"], set()).add(md["conversation_id"])
replayed = set().union(*traces.values()) if traces else set()
print(f"{len(traces)} traces with sub-agent first turns in the measurement; analysing {min(len(traces), max_traces)}")
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
for ti, (trace, _) in enumerate(list(traces.items())[:max_traces]):
    firsts = []
    for cid in by_trace.get(trace, []):
        c = conv(cid)
        t0 = c["turns"][0]
        msgs = [m for m in (t0.get("raw_messages") or []) if isinstance(m, dict)]
        if t0.get("raw_system"):
            msgs = [{"role": "system", "content": t0["raw_system"]}] + msgs
        firsts.append(dict(cid=cid, ts=t0.get("timestamp") or 0.0, kind=t0.get("source_kind"),
                           hit=t0.get("theoretical_prefix_cache_hit_blocks"),
                           tot=t0.get("theoretical_prefix_cache_total_blocks"), ids=render(msgs)))
    firsts.sort(key=lambda x: x["ts"])
    for k, f in enumerate(firsts):
        if f["kind"] != "weka_subagent" or f["cid"] not in replayed:
            continue
        best = max((lcp(f["ids"], g["ids"]) for g in firsts[:k]), default=0)
        n = len(f["ids"])
        bs = (n / f["tot"]) if f["tot"] else 0
        theo = int((f["hit"] or 0) * bs)
        tot["n"] += 1
        tot["tokens"] += n
        tot["lcp"] += best
        tot["theo"] += theo
        if tot["n"] <= 25:
            print(f"  {f['cid'][:60]:60s} tokens {n:7d}  best LCP with earlier turn-0 {best:7d}  "
                  f"theoretical hit {f['hit']}/{f['tot']} blocks (~{theo} tokens)")
print(f"sub-agent first turns: {tot['n']}, tokens {tot['tokens']:,}, best LCP with an earlier turn-0 of the trace "
      f"{tot['lcp']:,} ({tot['lcp'] / max(tot['tokens'], 1):.1%}), theoretical ~{tot['theo']:,} "
      f"({tot['theo'] / max(tot['tokens'], 1):.1%})")
