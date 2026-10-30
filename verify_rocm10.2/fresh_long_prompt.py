#!/usr/bin/env python3
"""Send a fresh (never-cached) long needle prompt through the router and time it.

Usage: fresh_long_prompt.py ROUTER_URL MODEL [N_SENTENCES] [TAG]
"""
import json, random, sys, time, urllib.request

url, model = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 5200
tag = sys.argv[4] if len(sys.argv) > 4 else str(time.time_ns())
rng = random.Random(tag)
words = ["river", "stone", "lamp", "orbit", "maple", "signal", "copper", "harbor", "violet", "engine", "meadow", "quartz"]
code = f"{rng.choice(['ALDER','BIRCH','OAK','PINE'])}-{rng.randint(1000, 9999)}"
lines = [f"Note {i} [{tag}]: the " + " ".join(rng.choice(words) for _ in range(8)) + "." for i in range(n)]
lines.insert(rng.randint(n // 4, 3 * n // 4), f"IMPORTANT: the secret access code is {code}.")
prompt = "\n".join(lines) + "\n\nWhat is the secret access code? Answer with the code only."
body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 4096,
        "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
t = time.time()
out = json.load(urllib.request.urlopen(req, timeout=3600))
dt = time.time() - t
text = out["choices"][0]["message"]["content"] or ""
print(json.dumps({"tag": tag, "prompt_tokens": out["usage"]["prompt_tokens"], "elapsed_s": round(dt, 2),
                  "expected": code, "text": text.strip()[:80], "passed": code in text}))
