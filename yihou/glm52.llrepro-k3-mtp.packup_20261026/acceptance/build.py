"""Build AgentX-wire-like chat requests from the cc-traces-weka-062126 trace corpus.

The corpus holds no text: each turn is (in, out, hash_ids[block_size=64], stop). aiperf
synthesizes content by mapping every hash id to a deterministic slice of its Shakespeare
corpus (tokenized with the served model's tokenizer), shapes the context into
system/user/assistant/tool messages, and sends max_tokens=out with ignore_eos.
This builder mimics that: same corpus, same tokenizer, hash-id-keyed slices (sha256 offset;
not byte-identical to aiperf's RNG), message boundaries from earlier turns of the same trace,
user chunks after a 'tool_use' stop shaped as role=tool (as aiperf's tool shaping does).
"""
import glob, hashlib, json, random, sys
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
OUT = sys.argv[2] if len(sys.argv) > 2 else "/shared_nfs/yihou_glm52/llrepro_20261023/acc.yihou/req_agentx.jsonl"
MAXCTX, MAXOUT, MINOUT = 100_000, 1024, 32
L = "/shared_nfs/yihou_glm52/llrepro_20261023/work/cache"
d = L + "/agentx/hf/datasets/semianalysisai___cc-traces-weka-062126/default/0.0.0/23f152f6f0f9399a85901b89a6458def0ef16729"
ds = concatenate_datasets([Dataset.from_file(f) for f in sorted(glob.glob(d + "/*.arrow"))])
tok = AutoTokenizer.from_pretrained("/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4", trust_remote_code=True)
corpus = tok.encode(open(L + "/InferenceX/utils/aiperf/src/aiperf/dataset/generator/assets/shakespeare.txt").read(),
                    add_special_tokens=False)
C = len(corpus)

def slice_tokens(seed, n):
    off = int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "big") % (C - n - 1)
    return corpus[off:off + n]

def prompt_tokens(tid, req, bs):
    t = []
    for h in req["hash_ids"]:
        t += slice_tokens(f"{tid}:{h}", bs)
    if len(t) < req["in"]:
        t += slice_tokens(f"{tid}:tail:{req['t']}", req["in"] - len(t))
    return t[:req["in"]]

rng = random.Random(42)
rows = list(range(len(ds)))
rng.shuffle(rows)
reqs = []
for ri in rows:
    if len(reqs) >= N:
        break
    r = ds[ri]
    tid, bs = r["id"], r["block_size"]
    main = [q for q in r["requests"] if q.get("type") in ("n", "s")]
    early = len(reqs) % 2 == 0
    lo, hi = (0, max(1, int(0.3 * len(main)))) if early else (int(0.5 * len(main)), len(main))
    cand = [j for j in range(lo, hi) if 2000 <= main[j]["in"] <= MAXCTX and main[j]["out"] >= MINOUT]
    if not cand:
        continue
    j = rng.choice(cand)
    q = main[j]
    toks = prompt_tokens(tid, q, bs)
    # earlier turns whose (prompt + output) is a prefix of this prompt define the message boundaries
    nb = lambda x: len(x["hash_ids"])
    prev = [p for p in main[:j] if p["in"] + p["out"] <= q["in"] and nb(p) > 0
            and q["hash_ids"][:min(nb(p), p["in"] // bs)] == p["hash_ids"][:min(nb(p), p["in"] // bs)]]
    cuts, last = [], 0
    for p in prev:
        a, b = p["in"], p["in"] + p["out"]
        if a > last:
            cuts.append((last, a, "user" if cuts else "system", None)); cuts.append((a, b, "assistant", p.get("stop", "")))
            last = b
    cuts.append((last, q["in"], "user" if cuts else "system", None))
    if cuts[-1][2] == "system":
        cuts.append((q["in"], q["in"], "user", None))
    msgs, k = [], 0
    for (a, b, role, stop) in cuts:
        text = tok.decode(toks[a:b]) if b > a else "Continue."
        if role == "user" and msgs and msgs[-1]["role"] == "assistant" and msgs[-1].get("_stop") == "tool_use":
            cid = f"call_turn_{k}"; k += 1
            msgs[-1]["tool_calls"] = [{"id": cid, "type": "function", "function": {"name": "recorded_tool", "arguments": "{}"}}]
            msgs.append({"role": "tool", "tool_call_id": cid, "content": text})
        else:
            m = {"role": role, "content": text}
            if role == "assistant":
                m["_stop"] = stop
            msgs.append(m)
    for m in msgs:
        m.pop("_stop", None)
    reqs.append({"id": f"ax{len(reqs):02d}", "trace": tid, "turn": j, "nturns": len(main), "pos": "early" if early else "late",
                 "in": q["in"], "max_tokens": min(q["out"], MAXOUT), "ignore_eos": True, "messages": msgs})
    print(reqs[-1]["id"], reqs[-1]["pos"], f"turn {j}/{len(main)} in={q['in']} out={q['out']} msgs={len(msgs)} roles={''.join(m['role'][0] for m in msgs)[:40]}", flush=True)
with open(OUT, "w") as f:
    for x in reqs:
        f.write(json.dumps(x) + "\n")
print("wrote", len(reqs), OUT)
