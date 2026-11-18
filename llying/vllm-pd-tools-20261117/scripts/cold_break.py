import json, collections, sys
run = sys.argv[1]
recs = []
for line in open(f"{run}/agentx-c104/aiperf_artifacts/profile_export.jsonl"):
    r = json.loads(line); md = r.get("metadata", {}); m = r.get("metrics") or {}
    isl = (m.get("usage_prompt_tokens") or {}).get("value")
    if isl is None: continue
    recs.append(dict(conv=md.get("conversation_id"), turn=md.get("turn_index"), phase=md.get("benchmark_phase"),
        depth=md.get("agent_depth"), kind=md.get("source_kind"), trace=md.get("source_trace_id"), root=md.get("root_correlation_id"),
        parent=md.get("parent_correlation_id"), corr=md.get("x_correlation_id"), t0=md["request_start_ns"]/1e9, isl=int(isl)))
by = collections.defaultdict(list)
for r in recs: by[r["conv"]].append(r)
for v in by.values():
    v.sort(key=lambda x: (x["turn"], x["t0"]))
    for a, b in zip([None]+v[:-1], v): b["prev"] = a
c = collections.Counter(); t = collections.Counter()
for r in recs:
    if r["phase"] != "profiling": continue
    a = r["prev"]
    if a is None: key = (r["depth"], r["kind"], "first-turn-in-run", "turn0" if r["turn"] == 0 else "turn>0")
    elif a["isl"] > r["isl"]: key = (r["depth"], r["kind"], "shrunk", "")
    else: continue
    c[key] += 1; t[key] += r["isl"]
for k in sorted(c, key=lambda k: -t[k]):
    print(f"{str(k):70s} n {c[k]:5d}  ISL {t[k]:12,}  mean {t[k]//c[k]:7,}")
kinds = collections.Counter((r["depth"], r["kind"]) for r in recs if r["phase"]=="profiling")
print(kinds)
