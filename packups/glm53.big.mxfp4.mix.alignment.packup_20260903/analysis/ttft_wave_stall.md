# The TTFT p99 anomaly is a whole-wave stall, not queueing — and PD amplifies it ~3×

Answered from archived per-request JSONL on NFS. **No GPU, no node.** Open since
12:43; the node outage is what finally made someone read the files.

## The observation that pointed the right way

checkpoint-writer: *"TTFT p99 has sat between 11 s and 21 s regardless of
concurrency or configuration, while p50 moves between 0.54 s and 6.6 s. A p99
roughly constant while p50 varies by an order of magnitude does not look like
queueing."* Correct, and the JSONL says what it is instead.

## The decisive case: PD, conc 8

```
idx   8  17.8576      spread across the whole cluster: 0.1016 s
idx   9  17.9592
idx  10  17.8609      all 72 other requests: median 1.96 s, max 3.89 s
idx  11  17.9575
idx  12  17.8613
idx  13  17.9575
idx  14  17.8608
idx  15  17.8813
```

**Requests 8–15 are exactly the second wave of 8 at concurrency 8.** They stall
together for ~17.9 s with a spread of **0.1 s**, and nothing else in the run
exceeds 3.9 s.

Eight requests released within 100 ms of each other after an 18-second block is a
**single event that held every in-flight slot and then let go**. It is not
queueing — queueing is gradual and rises with load. It is not first-request
warm-up — index 0 is fine here; the stall is the *second* wave.

## It is not PD-specific, but PD makes it far worse

Same metric across every archived arm, `ttft > 5 s`:

| arm | conc | n slow | first idx | contiguous | value range | spread |
|---|---:|---:|---:|---|---|---:|
| **MIX8 dpa0** | 8 | **8** | **8** | **yes** | 5.02–6.95 s | 1.93 s |
| **PD dpa1** | 8 | **8** | **8** | **yes** | **17.86–17.96 s** | **0.10 s** |
| PD dpa1 | 16 | 24 | 33 | no — two blocks: **33–48** (16 = conc) and 89–96 | 5.31–11.42 s | |
| MIX8 dpa0 | 16 | 15 | 87 | no — scattered (87, 90, 94, 100, 107…) | 5.59–5.71 s | |
| PD dpa1 | 1 | 1 | **0** | yes | 11.31 s | — |
| PD dpa0 | 24 | 172 | 0 | no | 5.04–21.86 s | |
| MIX8 dpa0 | 24 | 80 | 1 | no | 5.72–12.33 s | |

Three things fall out:

1. **The second-wave stall happens on aggregated MIX too** — identical indices
   8–15 at conc 8. So it is **not caused by disaggregation**.
2. **PD's version is ~3× longer and ~19× tighter** — 17.9 s at 0.10 s spread
   against MIX's 5–7 s at 1.93 s. The tightness is the tell: PD's eight requests
   were released by one event, MIX's were merely all slow.
3. **At conc 16 the shapes diverge.** PD stalls *whole waves* — indices 33–48 is
   exactly 16 consecutive requests, one full concurrency width — while MIX8 is
   scattered across the run. Whatever the event is, PD couples it to the batch.

## What this does and does not establish

**Established:** the tail is a discrete blocking event affecting entire
concurrency waves, present on both topologies, and materially amplified by PD.
Not queueing, not warm-up, not load.

**Not established — and deliberately not guessed:** the mechanism. Candidates
that fit the shape and were *not* tested include CUDA-graph capture at a new
batch size, first-eviction or reorganisation of the radix cache, a mooncake
registration event on the first transfer to a new destination, and dynamo
recompilation. Naming them is not evidence for any of them.

**The check that would discriminate**, and it needs the node: correlate each
stall window against the engine log's `Prefill batch` / `Decode batch` lines and
`#queue-req` over the same seconds. The leg logs are archived alongside the JSONL
in `pd_hipon_artifacts/` and `mix8_artifacts/`, so this is *also* doable without a
GPU by anyone with an hour — the JSONL carry no timestamps, so it needs the run's
start time to align the two.

## Consequence for the reported numbers

`ttft_p99` on every arm in this campaign is **a measurement of this event, not of
tail latency under load.** Quoting it as a latency percentile overstates the tail
a real workload would see at these concurrencies — most requests are unaffected —
while understating how bad the event itself is when it lands.

Report `ttft_p50` and this event separately. They are not the same phenomenon.

---

# UPDATE 2026-09-03 — the conc-1 tail has a handle: decode-side DP-attention

Three PD arms on n01-21, same node, same image, archived side by side. Lead came
from checkpoint-writer noticing the tail survives on both hip arms; the axis is
not hip.

| arm | decode config | conc-1 TTFT p50 | conc-1 TTFT p99 |
|---|---|---:|---:|
| `n21_pd_dpa0` | `dp_size=1`, `enable_dp_attention=False` | 199 ms | **211 ms** |
| `n21_pd_hipon` | `dp_size=4`, `enable_dp_attention=True` | 193 ms | **10,197 ms** |
| `n21_pd_hipoff` | `dp_size=4`, `enable_dp_attention=True` | ~195 ms | **10,138 ms** |

**hip was varied between the two DPA-on arms and moved the tail by 0.6 %.** It is
ruled out. Decode DP-attention separates cleanly: with it off the conc-1 p99 is
211 ms against a 199 ms median — no tail at all.

## Why this is a strong reading

At concurrency 1 there is **no wave to stall and no queue to build**, so the
usual confounds are absent. One request, one slow event, and the only
configuration axis that tracks it across three arms is decode DPA.

## What it does NOT establish

- **No mechanism is offered.** At `dp_size=4` and concurrency 1 a single request
  occupies one DP rank while three idle, and a collective still runs each step —
  that is a *shape* consistent with the observation, not a measurement of it.
- **Whether this is the same event as the higher-concurrency wave stall is
  unknown.** The wave stall (requests 8–15 released within 0.1 s after ~17.9 s)
  was measured on arms that also had DPA on, so the two are consistent — but a
  conc-1 single-request stall and a conc-8 whole-wave stall have not been shown
  to be one phenomenon.
- **MIX arms are not covered.** The earlier finding that MIX at conc 8 shows the
  identical index signature was on a DPA-off MIX arm, which would argue *against*
  DPA being the whole story. That tension is unresolved and should not be smoothed
  over.

## The check that would settle the mechanism

Correlate each stall window against per-DP-rank `Decode batch` lines and
`#queue-req` over the same seconds, on the DPA-on arms. All three arms' leg logs
and JSONL are archived; this needs no GPU.
