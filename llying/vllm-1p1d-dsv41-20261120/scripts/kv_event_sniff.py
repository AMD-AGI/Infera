#!/usr/bin/env python3
"""Passively subscribe to a vLLM KV-event ZMQ PUB endpoint and summarize what it publishes.

    kv_event_sniff.py tcp://HOST:PORT [SECONDS] [OUT_JSONL]

vLLM's publisher sends multipart [topic, seq (8-byte BE), payload (msgpack KVEventBatch)]; a batch is
[ts, [events...], dp_rank?] with each event a tagged array (BlockStored / BlockRemoved / AllBlocksCleared).
Prints per-event-type counts, the distinct field layouts and block sizes seen, and the first few raw events.
"""

import json
import sys
import time
from collections import Counter

import msgpack
import zmq

ep = sys.argv[1]
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 15
out = open(sys.argv[3], "w") if len(sys.argv) > 3 else None
ctx = zmq.Context()
s = ctx.socket(zmq.SUB)
s.setsockopt(zmq.SUBSCRIBE, b"")
s.setsockopt(zmq.RCVTIMEO, 1000)
s.connect(ep)
t_end = time.time() + secs
kinds, shapes, sizes, groups = Counter(), Counter(), Counter(), Counter()
samples, nbatch = [], 0


def short(x):
    if isinstance(x, (bytes, bytearray)):
        return f"<{len(x)}B {bytes(x[:8]).hex()}>"
    if isinstance(x, list):
        return [short(v) for v in x[:6]] + ([f"...+{len(x) - 6}"] if len(x) > 6 else [])
    return x


while time.time() < t_end:
    try:
        parts = s.recv_multipart()
    except zmq.Again:
        continue
    nbatch += 1
    batch = msgpack.unpackb(parts[-1], raw=False, strict_map_key=False)
    events = batch[1] if isinstance(batch, list) and len(batch) > 1 else batch
    for ev in events:
        tag = ev[0] if isinstance(ev, list) and ev and isinstance(ev[0], str) else type(ev).__name__
        kinds[tag] += 1
        shapes[(tag, len(ev) if isinstance(ev, list) else -1)] += 1
        if tag == "BlockStored" and isinstance(ev, list):
            # vLLM BlockStored: [tag, block_hashes, parent_block_hash, token_ids, block_size, lora_id, medium, ...]
            if len(ev) > 4:
                sizes[ev[4]] += 1
                ntok, nblk = len(ev[3] or []), len(ev[1] or [])
                groups[("tokens/blocks", ntok // max(nblk, 1))] += 1
            if len(ev) > 7:
                groups[("extra fields", tuple(type(v).__name__ for v in ev[7:]))] += 1
        if len(samples) < 4:
            samples.append(short(ev))
        if out:
            out.write(json.dumps(short(ev), default=str) + "\n")
print(f"endpoint {ep}: {nbatch} batches in {secs:.0f}s; topic={parts[0] if nbatch else None!r}")
print("event kinds:", dict(kinds))
print("event shapes (tag, n_fields):", dict(shapes))
print("BlockStored block_size field:", dict(sizes))
print("other:", dict(groups))
for smp in samples:
    print("sample:", json.dumps(smp, default=str)[:600])
