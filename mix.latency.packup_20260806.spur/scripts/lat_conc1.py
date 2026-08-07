#!/usr/bin/env python3
# Task 2 — single-conc (conc=1) agentic LATENCY at pinned Case-A shapes.
# For each shape (P50/P90/P99 full ISL/OSL): build a fixed cacheable prefix of
# round(0.89*ISL) tokens (Case-A cache_hit_rate), WARM it once, then fire 10
# SEQUENTIAL requests, each = shared prefix + a small fresh suffix to reach ISL,
# max_tokens=OSL. conc=1, no think time (operator). Records per-request TTFT/E2E
# and derived TPOT; reports the 10-sample mean/median. Mirrors the agentic
# driver's "fixed cacheable prefix + variable random suffix" cache model so the
# prefill is a ~89% cache hit like Case A.
#
# Runs INSIDE the engine container (has sglang tokenizer + requests). Talks to the
# router. Streaming for a real TTFT.
import json, time, sys, urllib.request, os
from transformers import AutoTokenizer

URL   = os.environ.get("URL", "http://127.0.0.1:8100")
MODEL = os.environ.get("SERVED", "glm5.2-mxfp4")
TOK   = os.environ.get("TOK", "/shared_nfs/GLM-5.2-MXFP4")
REPEATS = int(os.environ.get("REPEATS", "10"))
CACHE_HIT = float(os.environ.get("CACHE_HIT", "0.89"))
OUT = os.environ.get("OUT", "/tmp/mix_lat")
os.makedirs(OUT, exist_ok=True)

SHAPES = [("p50", 74000, 320), ("p90", 155000, 3300), ("p99", 235000, 17000)]
if os.environ.get("SHAPES"):
    want = set(os.environ["SHAPES"].split())
    SHAPES = [s for s in SHAPES if s[0] in want]

tok = AutoTokenizer.from_pretrained(TOK, trust_remote_code=True)

# a big pool of deterministic real-ish tokens to slice prefixes/suffixes from
_POOL_TXT = ("The history of computing spans mechanical calculators, vacuum tubes, "
             "transistors, integrated circuits, and modern accelerators. ") * 20000
_POOL = tok(_POOL_TXT, add_special_tokens=False)["input_ids"]

def make_text(ntok, offset=0):
    ids = _POOL[offset:offset+ntok]
    while len(ids) < ntok:
        ids += _POOL[:ntok-len(ids)]
    return tok.decode(ids)

def post_stream(prompt, max_tokens):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "min_tokens": max_tokens, "ignore_eos": True,
        "temperature": 1.0, "top_p": 0.95,
        "stream": True, "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(URL + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; ntok_out = 0; usage = {}
    with urllib.request.urlopen(req, timeout=1200) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"): continue
            data = line[5:].strip()
            if data == "[DONE]": break
            try: chunk = json.loads(data)
            except: continue
            ch = chunk.get("choices") or []
            if ch:
                delta = ch[0].get("delta", {})
                if (delta.get("content") or delta.get("reasoning_content")):
                    if ttft is None: ttft = time.time() - t0
                    ntok_out += 1
            if chunk.get("usage"): usage = chunk["usage"]
    e2e = time.time() - t0
    comp = (usage.get("completion_tokens") or ntok_out or 1)
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    prompt_toks = usage.get("prompt_tokens")
    tpot = (e2e - (ttft or 0)) / max(1, comp - 1)
    return {"ttft_ms": (ttft or 0)*1000, "e2e_ms": e2e*1000, "tpot_ms": tpot*1000,
            "completion_tokens": comp, "prompt_tokens": prompt_toks, "cached_tokens": cached}

def pctile(xs, p):
    xs = sorted(xs);
    return xs[min(len(xs)-1, int(round((len(xs)-1)*p)))]

summary = []
for name, ISL, OSL in SHAPES:
    prefix_tok = int(round(CACHE_HIT * ISL))
    fresh_tok  = max(1, ISL - prefix_tok)
    prefix = make_text(prefix_tok)
    print(f"\n===== {name}: ISL={ISL} OSL={OSL} prefix={prefix_tok} fresh={fresh_tok} =====", flush=True)
    # warm the shared prefix once (fresh tail = warm marker); not measured
    _ = post_stream(prefix + make_text(fresh_tok, offset=999), 8)
    print("  warmed shared prefix", flush=True)
    recs = []
    for i in range(REPEATS):
        # each rep: same prefix (cache hit) + a DISTINCT fresh suffix (so it is a real new request)
        prompt = prefix + make_text(fresh_tok, offset=1000 + i*fresh_tok)
        r = post_stream(prompt, OSL)
        recs.append(r)
        print(f"  rep {i+1:2d}: TTFT={r['ttft_ms']:8.1f}ms  E2E={r['e2e_ms']:9.1f}ms  TPOT={r['tpot_ms']:6.2f}ms  "
              f"prompt={r['prompt_tokens']} cached={r['cached_tokens']} out={r['completion_tokens']}", flush=True)
    for k in ("ttft_ms","e2e_ms","tpot_ms"):
        vs=[x[k] for x in recs]
        print(f"  {k:8s} mean={sum(vs)/len(vs):9.1f}  p50={pctile(vs,0.5):9.1f}  p90={pctile(vs,0.9):9.1f}  min={min(vs):9.1f}  max={max(vs):9.1f}", flush=True)
    ch = [x["cached_tokens"] for x in recs if x["cached_tokens"] is not None]
    pt = [x["prompt_tokens"] for x in recs if x["prompt_tokens"] is not None]
    hit = (sum(ch)/sum(pt)) if ch and pt and sum(pt) else None
    print(f"  cache-hit (cached/prompt) = {hit*100:.1f}%" if hit is not None else "  cache-hit = n/a", flush=True)
    with open(f"{OUT}/lat_{name}.jsonl","w") as f:
        for r in recs: f.write(json.dumps(r)+"\n")
    summary.append({"shape":name,"ISL":ISL,"OSL":OSL,"n":len(recs),
        "ttft_ms_mean":sum(x["ttft_ms"] for x in recs)/len(recs),
        "e2e_ms_mean":sum(x["e2e_ms"] for x in recs)/len(recs),
        "tpot_ms_mean":sum(x["tpot_ms"] for x in recs)/len(recs),
        "cache_hit":hit})
with open(f"{OUT}/lat_summary.json","w") as f: json.dump(summary,f,indent=2)
print(f"\n=== lat done -> {OUT}/lat_*.jsonl + lat_summary.json ===", flush=True)
