#!/usr/bin/env python3
"""Send the SAME chat completion twice through the router and report the
server's own cached_tokens both times.

Point: smoke.sh printed `cached None` on its one-shot request. That is expected
for a cold, never-seen prefix -- it does NOT prove --enable-cache-report is
missing. Round 2 must report a nonzero cached_tokens if prefix reuse is working.

Runs INSIDE the prefill container (it is the one that can reach the router).
"""
import json
import urllib.request

URL = "http://10.2.122.78:8100/v1/chat/completions"
BODY = {
    "model": "glm5.2-mxfp4",
    "messages": [
        {
            "role": "user",
            # long enough to span more than one 64-token KV block
            "content": (
                "Here is a list of cities: "
                + ", ".join(f"city-{i}" for i in range(300))
                + ". How many cities are in that list? Answer with the number only."
            ),
        }
    ],
    "max_tokens": 24,
    "temperature": 1.0,
    "top_p": 0.95,
}

for rnd in (1, 2):
    req = urllib.request.Request(
        URL, data=json.dumps(BODY).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    u = d.get("usage", {})
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    fin = d["choices"][0].get("finish_reason")
    print(f"round {rnd}: prompt={u.get('prompt_tokens')} cached={cached} finish={fin}")
