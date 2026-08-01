import json
import time
import urllib.request

BASE = "http://10.2.122.10:8100"
MODEL = "glm5.2-mxfp4"
# ~900-token stable shared prefix; must exceed the hicache prefetch threshold.
PREFIX = (
    "You are a helpful assistant. Answer directly from general knowledge. "
    "The reference material below is background only -- ignore it unless "
    "the question is about it. Reply with just the answer.\n\n=== BACKGROUND ===\n"
) + "\n".join(
    f"Section {i}: component {i} handles subsystem {i % 7}, mode {i % 3}, "
    f"controller {i % 5}, throughput {1000 + i * 7} units/s."
    for i in range(120)
)
Q = [
    ("What is the capital of France? Just the city name.", "paris"),
    ("What is 2+2? Just the number.", "4"),
    ("Largest planet in our solar system? One word.", "jupiter"),
    ("What is the capital of Japan? Just the city name.", "tokyo"),
]


def ask(s, q):
    b = json.dumps(
        {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": PREFIX},
                {"role": "user", "content": f"[session {s}] {q}"},
            ],
            "max_tokens": 128,
            "temperature": 0,
        }
    ).encode()
    r = urllib.request.Request(
        f"{BASE}/v1/chat/completions", data=b, headers={"Content-Type": "application/json"}
    )
    t = time.time()
    with urllib.request.urlopen(r, timeout=300) as f:
        d = json.load(f)
    return d["choices"][0]["message"]["content"], time.time() - t


for phase in ("PHASE1-cold", "PHASE2-reuse"):
    ok = n = 0
    lat = []
    for s in range(4):
        for q, want in Q:
            n += 1
            try:
                txt, dt = ask(s, q)
                lat.append(dt)
                ok += want in txt.lower()
            except Exception as e:
                print("  ERR", e)
    print(f"{phase}: {ok}/{n} correct, median {sorted(lat)[len(lat) // 2]:.2f}s", flush=True)
