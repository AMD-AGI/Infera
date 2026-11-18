import json, sys, collections
from transformers import AutoTokenizer
ddir, model, prof = sys.argv[1:4]
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
idx = json.load(open(f"{ddir}/index.dat")); offs = idx["offsets"]; data = open(f"{ddir}/dataset.dat", "rb")
def conv(cid):
    o = offs[cid]; data.seek(o["offset"]); return json.loads(data.read(o["size"]))
def render(msgs):
    return tok.encode(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), add_special_tokens=False)
def lcp(a, b):
    n = min(len(a), len(b)); i = 0
    while i < n and a[i] == b[i]: i += 1
    return i
for trace in sys.argv[4:]:
    sent = []
    for line in open(prof):
        r = json.loads(line); md = r.get("metadata", {})
        if md.get("source_trace_id", "").startswith(trace) and md.get("turn_index") == 0 and md.get("source_kind") == "weka_subagent":
            sent.append((md["request_start_ns"] / 1e9, md["conversation_id"], md.get("benchmark_phase")))
    sent.sort()
    ids = []
    for t, cid, ph in sent[:14]:
        c = conv(cid); m = [x for x in (c["turns"][0].get("raw_messages") or []) if isinstance(x, dict)]
        ids.append((t, cid, render(m), m))
    t0 = ids[0][0]
    print("trace", trace, "first turn-0 sub-agents in replay order:")
    for k, (t, cid, x, m) in enumerate(ids):
        best = max([(lcp(x, y[2]), j) for j, y in enumerate(ids[:k])], default=(0, -1))
        roles = [mm["role"] for mm in m]
        print(f"  +{t - t0:5.1f}s {cid[-30:]:30s} tokens {len(x):6d} roles {roles} best LCP with an earlier one {best[0]:6d} (#{best[1]})")
    # where do the ~3K ones diverge?
    small = [z for z in ids if 2800 < len(z[2]) < 3200]
    if len(small) >= 2:
        a, b = small[0][2], small[1][2]; n = lcp(a, b)
        print("  first two ~3K: LCP", n, "| next tokens:", repr(tok.decode(a[n:n + 30])), "VS", repr(tok.decode(b[n:n + 30])))
        print("  shared head:", repr(tok.decode(a[:200])))
