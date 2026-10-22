#!/usr/bin/env python3
"""check40.py BASE_URL OUT_JSON: 40 concurrent greedy requests (20 math variants + 10 passkey 8K + 10 passkey 34K),
then 4 concurrent long greedy generations (1500 tokens, thinking off) for a real accept-length window in the server log."""
import json, sys, time, random, urllib.request, concurrent.futures as cf
base, out = sys.argv[1], sys.argv[2]
MODEL = __import__("os").environ.get("CHECK_MODEL", "amd/GLM-5.2-MXFP4")
def chat(content, max_tokens, thinking=False):
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}], "temperature": 0,
            "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": thinking}}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        j = json.load(r)
    return j["choices"][0]["message"].get("content") or "", j.get("usage", {})
rng = random.Random(int(__import__("os").environ.get("CHECK_SEED", "7")))
tasks = []
for i in range(20):
    a, b, c, d = rng.randint(10, 60), rng.randint(1, 20), rng.randint(1, 9), rng.randint(1, 30)
    q = (f"A farmer has {a} sheep. He buys {b} more sheep, then {c} sheep run away, and finally he buys {d} more. "
         "How many sheep does he have now? Answer with only the number.")
    tasks.append(("math", q, str(a + b - c + d), 64))
filler = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
for name, ntok, n in (("pk8k", 8000, 10), ("pk34k", 34000, 10)):
    for i in range(n):
        key = str(rng.randint(10000, 99999)); k = ntok // 21; parts = [filler] * k
        parts.insert(rng.randint(k // 10, 9 * k // 10), f"The pass key is {key}. Remember it. {key} is the pass key. ")
        q = ("There is an important piece of information hidden inside a lot of irrelevant text. Find it and memorize it.\n\n"
             + "".join(parts) + "\n\nWhat is the pass key? Answer with only the number.")
        tasks.append((name, q, key, 32))
t0 = time.time(); res = []
with cf.ThreadPoolExecutor(40) as ex:
    futs = [ex.submit(chat, q, mt) for (_, q, _, mt) in tasks]
    for (kind, _, exp, _), f in zip(tasks, futs):
        try:
            c, u = f.result(); ok = exp in c if kind != "math" else c.strip().rstrip(".") == exp
        except Exception as e:
            c, u, ok = repr(e), {}, False
        res.append({"kind": kind, "expect": exp, "got": c[:80], "pass": ok, "prompt_tokens": u.get("prompt_tokens")})
summ = {}
for r in res:
    s = summ.setdefault(r["kind"], [0, 0]); s[0] += r["pass"]; s[1] += 1
print("concurrent40", {k: f"{v[0]}/{v[1]}" for k, v in summ.items()}, "secs", round(time.time() - t0, 1), flush=True)
for r in res:
    if not r["pass"]: print("FAIL", r, flush=True)
long_prompts = ["Write a detailed 1500-word essay on the history of the printing press.",
                "Explain, step by step and in great detail, how a compiler turns C source code into machine code.",
                "Write a long story about a lighthouse keeper who discovers a message in a bottle.",
                "Describe in detail the water cycle, photosynthesis, and the carbon cycle, and how they interact."]
t1 = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()); t0 = time.time()
with cf.ThreadPoolExecutor(4) as ex:
    outs = list(ex.map(lambda p: chat(p, 1500), long_prompts))
t2 = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
gen = [u.get("completion_tokens") for _, u in outs]
print("long_greedy window", t1, "->", t2, "completion_tokens", gen, "secs", round(time.time() - t0, 1), flush=True)
json.dump({"concurrent40": res, "summary": summ, "long_window": [t1, t2], "long_tokens": gen,
           "long_heads": [c[:200] for c, _ in outs]}, open(out, "w"), indent=1)
