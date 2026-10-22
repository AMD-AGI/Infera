#!/usr/bin/env python3
"""Task v0521cap2 correctness probes on a live GLM-5.2 PD stack (router OpenAI API). Not a quality benchmark: a few
deterministic checks. Uses /v1/chat/completions (server-side chat template) with (a) greedy T=0 and (b) the model's
generation_config sampling T=1.0, top_p=0.95. Records content, reasoning_content, usage, latency per probe -> results.jsonl."""
import json, sys, time, urllib.request, random
URL, MODEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
SAMPLED = {"temperature": 1.0, "top_p": 0.95}
GREEDY = {"temperature": 0.0}
out = open(f"{OUT}/results.jsonl", "w")

def chat(msg, params, max_tokens=4096, timeout=1200):
    body = {"model": MODEL, "messages": [{"role": "user", "content": msg}], "max_tokens": max_tokens, **params}
    t = time.time()
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=timeout))
        m = r["choices"][0]["message"]
        return {"content": m.get("content") or "", "reasoning": m.get("reasoning_content") or "", "usage": r.get("usage"),
                "finish": r["choices"][0].get("finish_reason"), "secs": round(time.time() - t, 1)}
    except Exception as e:
        return {"error": repr(e)[:500], "secs": round(time.time() - t, 1)}

def record(name, params_name, res, expect, passed):
    row = {"test": name, "sampling": params_name, "expect": expect, "pass": passed, **res}
    out.write(json.dumps(row, ensure_ascii=False) + "\n"); out.flush()
    print(f"{'PASS' if passed else 'FAIL'} {name} [{params_name}] secs={res.get('secs')} prompt_tok={(res.get('usage') or {}).get('prompt_tokens')} "
          f"finish={res.get('finish')} content={res.get('content','')[:160]!r} err={res.get('error','')[:120]}")

# 1. smoke
for pname, p in (("greedy", GREEDY), ("sampled", SAMPLED)):
    r = chat("What is 17 * 23? Reply with the number only.", p, 2048); record("smoke_arith", pname, r, "391", "391" in r.get("content", ""))
    r = chat("List the first five prime numbers, comma-separated, nothing else.", p, 2048)
    record("smoke_primes", pname, r, "2, 3, 5, 7, 11", all(x in r.get("content", "").replace(" ", "") for x in ["2,3,5,7,11"]))
    r = chat("What is the capital of France? One word.", p, 1024); record("smoke_capital", pname, r, "Paris", "paris" in r.get("content", "").lower())

# filler: varied, deterministic sentences (~12 tokens each)
random.seed(7)
SUBJ = ["The river", "A merchant", "The old library", "Every winter", "The engineer", "A small boat", "The council", "Our neighbour",
        "The mountain road", "A quiet student", "The orchestra", "The baker"]
VERB = ["described", "remembered", "carried", "repaired", "ignored", "painted", "measured", "studied", "visited", "counted"]
OBJ = ["the northern bridge", "a forgotten map", "seven copper lamps", "the evening market", "an unfinished letter",
       "the harbour wall", "three wooden chairs", "the eastern tower", "a basket of apples", "the narrow stairs"]
def filler(n_sent):
    return " ".join(f"{random.choice(SUBJ)} {random.choice(VERB)} {random.choice(OBJ)} on day {i}." for i in range(n_sent))

# 2. needle-in-a-haystack: ~tokens via sentences (≈11-12 tok/sentence); depths 10/50/90 %
for target in (8000, 32000, 64000, 128000, 256000):
    n = target // 12
    for depth in (0.1, 0.5, 0.9):
        key = str(random.randint(10000, 99999))
        hay = filler(n).split(". ")
        pos = int(len(hay) * depth)
        hay.insert(pos, f"Important: the secret passkey is {key}")
        doc = ". ".join(hay)
        q = doc + "\n\nWhat is the secret passkey mentioned in the text above? Reply with the passkey number only."
        r = chat(q, SAMPLED, 4096)
        record(f"needle_{target//1000}k_d{int(depth*100)}", "sampled", r, key, key in r.get("content", ""))

# 3. long-context multi-fact retrieval (~100k tokens, 5 facts)
facts = {"Alder": "violet", "Birch": "amber", "Cedar": "teal", "Dogwood": "crimson", "Elm": "silver"}
hay = filler(100000 // 12).split(". ")
for i, (k, v) in enumerate(facts.items()):
    hay.insert(int(len(hay) * (0.15 + 0.17 * i)), f"Note: the {k} house is painted {v}")
q = ". ".join(hay) + "\n\nFor each of the houses Alder, Birch, Cedar, Dogwood and Elm, state the colour it is painted, one per line as 'House: colour'."
for pname, p in (("greedy", GREEDY), ("sampled", SAMPLED)):
    r = chat(q, p, 4096)
    hits = sum(1 for k, v in facts.items() if v in r.get("content", "").lower())
    record("longctx_5facts_100k", pname, r, json.dumps(facts), hits == 5)
    print(f"  longctx facts recovered {hits}/5")
print("DONE")
