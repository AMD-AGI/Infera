#!/usr/bin/env python3
"""Read the RAW kv-event wire from each leg and report which token view it uses.

WHY. The merged branch carries a bigram-decode fix (`_flat_tokens`, and its Rust
twin) for the case where SGLang keys its radix tree on BIGRAMS: under EAGLE/MTP a
stored block's tokens arrive as overlapping pairs (t[i], t[i+1]) instead of bare
ints, and a router that hashes the pairs builds a view no request can ever match
-- the cache view reads 0.

The question this answers is not "is the fix present" (it is, in bytecode) but
"is the bigram PATH EXERCISED in this deployment". That is not obvious:

    kv_cache_builder.py:211   is_eagle=spec_algorithm.is_eagle()

reads each ENGINE's own speculative config, and in this deployment MTP is on the
DECODE leg only -- the prefill leg runs speculative_algorithm=None. If the
prefill leg emits plain ints, then its kv-events never take the bigram path, and
claiming the fix as "proven by a non-zero cache view" would be unfounded.

So: subscribe to the real socket, decode with the router's own msgspec structs,
and print what actually arrives.

Usage: kv_event_wire_probe.py <endpoint> [dp_ranks] [seconds]
   e.g. kv_event_wire_probe.py tcp://10.245.157.89:17568 8 25
"""
import sys
import time
import zmq
import msgspec

sys.path.insert(0, "/opt/infera")
from infera.router.kv_event.events import (  # noqa: E402
    SglangKVEventBatch, BLOCK_STORED_TYPES,
)

ENDPOINT = sys.argv[1]
RANKS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
SECS = float(sys.argv[3]) if len(sys.argv) > 3 else 25.0


def offset(ep, rank):
    if rank == 0:
        return ep
    head, _, port = ep.rpartition(":")
    return f"{head}:{int(port) + rank}"


ctx = zmq.Context()
socks = {}
poller = zmq.Poller()
for r in range(RANKS):
    s = ctx.socket(zmq.SUB)
    s.setsockopt(zmq.SUBSCRIBE, b"")
    s.setsockopt(zmq.RCVHWM, 0)
    s.connect(offset(ENDPOINT, r))
    socks[s] = r
    poller.register(s, zmq.POLLIN)

dec = msgspec.msgpack.Decoder(type=SglangKVEventBatch)
print(f"subscribed to {ENDPOINT} ranks 0..{RANKS-1}, listening {SECS:.0f}s")

stats = {}   # rank -> [batches, stored_events, int_blocks, pair_blocks]
samples = {}
t_end = time.time() + SECS
while time.time() < t_end:
    for s, _ in poller.poll(timeout=500):
        rank = socks[s]
        try:
            parts = s.recv_multipart(zmq.NOBLOCK)
        except zmq.Again:
            continue
        payload = parts[-1]
        try:
            batch = dec.decode(payload)
        except Exception as e:
            stats.setdefault(rank, [0, 0, 0, 0])
            print(f"  rank {rank}: DECODE ERROR {type(e).__name__}: {str(e)[:120]}")
            continue
        st = stats.setdefault(rank, [0, 0, 0, 0])
        st[0] += 1
        for ev in batch.events:
            if not isinstance(ev, BLOCK_STORED_TYPES):
                continue
            st[1] += 1
            tids = ev.token_ids
            if tids and isinstance(tids[0], (list, tuple)):
                st[3] += 1
                samples.setdefault(rank, ("PAIR", tids[:3], ev.block_size))
            elif tids:
                st[2] += 1
                samples.setdefault(rank, ("INT", tids[:3], ev.block_size))

print()
print(f"{'rank':<6} {'batches':>8} {'stored':>8} {'int-view':>9} {'pair-view':>10}  first sample")
tot = [0, 0, 0, 0]
for r in sorted(stats):
    b, s_, i, p = stats[r]
    for k, v in enumerate((b, s_, i, p)):
        tot[k] += v
    kind, samp, bs = samples.get(r, ("-", [], 0))
    print(f"{r:<6} {b:>8} {s_:>8} {i:>9} {p:>10}  {kind} bs={bs} {samp}")
print(f"{'TOTAL':<6} {tot[0]:>8} {tot[1]:>8} {tot[2]:>9} {tot[3]:>10}")
print()
if tot[3] and not tot[2]:
    print("VERDICT: BIGRAM view on the wire -> the _flat_tokens fix IS load-bearing here.")
elif tot[2] and not tot[3]:
    print("VERDICT: PLAIN-INT view on the wire -> the bigram path is NOT exercised by this leg.")
elif tot[2] and tot[3]:
    print("VERDICT: MIXED views across ranks -- report per rank, do not summarise.")
else:
    print("VERDICT: no BlockStored events seen (drive traffic while this runs).")
