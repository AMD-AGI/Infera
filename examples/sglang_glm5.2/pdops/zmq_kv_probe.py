"""Listen on every DP rank's kv-event port while a request is in flight.

Tells apart "the engine never published" from "the router never applied":
this subscriber is independent of Infera's client, so if it sees frames the
fault is downstream, and if it sees nothing the fault is in the engine.
"""

import json
import sys
import threading
import time
import urllib.request

import zmq

HOST = sys.argv[1] if len(sys.argv) > 1 else "10.32.17.210"
BASE_PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 32760
RANKS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
ROUTER = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "/wekafs/models/GLM-5.2-FP8"

ctx = zmq.Context()
poller = zmq.Poller()
socks = {}
for r in range(RANKS):
    s = ctx.socket(zmq.SUB)
    s.connect(f"tcp://{HOST}:{BASE_PORT + r}")
    # Empty prefix accepts every topic, so this also reveals what the engine
    # actually stamps on frame 0 (which is the thing a topic filter matches).
    s.subscribe(b"")
    socks[s] = r
    poller.register(s, zmq.POLLIN)

# PUB/SUB drops anything sent before the subscription lands, so settle first.
time.sleep(2)


def fire():
    # Salt the very first line so the whole prefix is new on every run; a repeat
    # prompt is served entirely from the radix tree and creates no store events,
    # which would make a "no frames" result meaningless.
    salt = f"Run {time.time():.6f}\n"
    body = salt + "\n".join(
        f"Line {i}: checksum {i * 31 % 101} for module {i % 17}." for i in range(900)
    )
    payload = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": body + "\n\nReply OK."}],
            "max_tokens": 400,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        ROUTER, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            d = json.loads(resp.read())
            u = d.get("usage", {})
            print(f"[req] prompt={u.get('prompt_tokens')} done", flush=True)
    except Exception as exc:
        print(f"[req] failed: {exc}", flush=True)


threading.Thread(target=fire, daemon=True).start()

counts = {r: 0 for r in range(RANKS)}
first_sample = None
deadline = time.time() + 90
while time.time() < deadline:
    for sock, _ in poller.poll(timeout=1000):
        rank = socks[sock]
        frames = sock.recv_multipart()
        counts[rank] += 1
        if first_sample is None:
            first_sample = (rank, [f[:120] for f in frames])

total = sum(counts.values())
print(f"\nframes received: {total}  per-rank: "
      f"{ {r: c for r, c in counts.items() if c} }")
if first_sample:
    print(f"first frame from dp{first_sample[0]}:")
    for f in first_sample[1]:
        print(f"  {f!r}")
else:
    print("NO FRAMES — the engine published nothing on these ports.")
