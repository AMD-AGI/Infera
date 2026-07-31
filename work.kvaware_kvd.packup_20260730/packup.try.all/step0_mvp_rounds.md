# Round 0 — the Qwen3-1.7B MVP rounds (r1–r5)

Five rounds on **one** node (chi2879), before any GLM-5.2 run. Purpose: shake
out the kvaware/kvd wiring cheaply. Qwen3-1.7B is 4 GB and cold-starts in ~2 min
versus GLM-5.2's ~30 min, which is what made five rounds affordable in an
afternoon — and rounds 1–3 were *all* wiring bugs, which a 1.7B model surfaces
exactly as well as a 400 GB one.

**Topology:** 1P1D on a single host — prefill TP4 on GPU0-3, decode TP4 on
GPU4-7, infera router :8100, `infera.kvd` daemon on `/tmp/kvd/kvd.sock`.

> **Logs are gone.** This container was removed before the GLM-5.2 runs, so
> `/tmp/r1..r5/` went with it. Everything quoted below was captured in-session
> at the time. Recorded as a gap rather than reconstructed.

---

## r1 — first launch: two independent failures

`KVD=1 KVAWARE=1 POLICY=kv-aware`

**prefill died:**

```
zmq.error.ZMQError: Address already in use (addr='tcp://127.0.0.1:30235')
RuntimeError: sglang subprocess exited with code -9 before reporting ready
```

**decode tried to allocate 1.4 TB of host RAM:**

```
[DP0 TP0 EP0] max_total_num_tokens=1547424, ...
[DP0 TP0 EP0] Allocating 354.94 GB host memory for hierarchical KV cache.
[DP1..DP3] (same, x4 ranks = 1.4 TB on a 3 TB box)
```

Cause: `--hicache-ratio` defaults to 2.0 and sizes the host pool off
`max_total_num_tokens`. The model is small and VRAM is plentiful, so the KV pool
was 1.5 M tokens → an enormous host pool. **Not a bug** — a default that
misfires badly on a small model.

**Fixes applied:** `--hicache-size 16` (absolute GB, overrides the ratio) and
spacing the two legs' `--port` 1000 apart.

## r2 — hicache fixed; decode still dies

hicache now behaves, and the kvd adapter genuinely connects:

```
[DP1 TP1 EP1] Creating dynamic storage backend 'infera-kvd'
[DP1 TP1 EP1] infera-kvd adapter connected to /tmp/kvd/kvd.sock (model=qwen3, compat_key=tp0of1_pp0of1)
[DP1 TP1 EP1] Tree cache initialized: impl=HiRadixCache hierarchical=True
[DP*] Allocating 8.00 GB host memory for hierarchical KV cache.      <- was 354.94
```

prefill = ready. decode Rank-0 scheduler died:

```
zmq.error.ZMQError: Address already in use (addr='tcp://*:32765')
RuntimeError: Rank 0 scheduler died during initialization (exit code: -3)
```

Both legs' `--kv-events-config` carried the **same** port (32764). With
`dp_size=4` sglang binds `base+rank`, so decode collided with prefill's 32765.

**→ This is bug #1.** Root cause, the two rejected alternative fixes, and the
MVP that reproduced it: `patches/0001-note.md`.

## r3 — port fix works; same-host RDMA does not

Both legs ready, distinct kv-event endpoints (17213 / 31215) — the randomised
scan holds, and both workers register with the router. But a completion returned
HTTP 500:

```
Failed to get kvcache from prefill instance
worker_pool.cpp:408 ... local_nic: ionic_0, peer_nic: ...@ionic_4:
                        transport retry counter exceeded
rdma_endpoint.cpp:472  Invalid argument: received packet mismatch
```

Two legs on one host means mooncake RDMA has to loop back across rails
(`ionic_0` → `ionic_4`), which this fabric will not do. A same-host limitation,
**orthogonal to kvaware/kvd**.

**Workaround:** `MC_FORCE_TCP=1` — this repo's known correct-but-slower path.
Also gave each leg its own `--kv-events-bind` (5557 / 5657), which is infera's
*own* publisher socket and a separate default that is identical on both legs.

## r4 — full stack up over TCP; output garbled

kvaware ON + kvd ON. Both legs ready, no 500s, KV handoff completes. But:

```
Q "capital of France" -> 'v4 freddy\n\nAists. Log In andapace.a\n\nWho wouldin %%%%3...'
Q "17*23"             -> 'S情辣梯neig治\n\n杖\n\n及格...'
```

Directly probing a PD leg returns nothing (a PD leg only serves through the
pair), so a direct probe cannot isolate this. The discriminating experiment is a
**differential run**.

## r5 — 对拍: switches OFF, same setup

`KVD=0 KVAWARE=0 POLICY=round-robin`, everything else identical:

```
'v4ই\n\n脐猫\n\nument="">< t-tcan you-...'
```

**Equally garbled — and note both runs open with the same `v4` token.**

### Verdict from r4 vs r5

The garbling belongs to this **same-host PD + `MC_FORCE_TCP`** substrate, not to
kvaware or kvd. With both features off the output is just as wrong.

This is a *no-regression* observation, **not** a correctness pass. It is exactly
why the investigation moved to two nodes and real RDMA — see
`results/baseline_probe_4of4.txt`, where the same probe scores 4/4.

---

## What r1–r5 bought

| | |
|---|---|
| Bugs found | 1 real infera bug (#1, port collision) |
| Defaults that misfire | `--hicache-ratio` on a small model; `--kv-events-bind` shared across legs |
| Environment limits mapped | same-host mooncake RDMA cannot loop back across rails |
| Cost | ~5 rounds × ~2 min cold start, one node, one afternoon |

Doing this on GLM-5.2 would have been five 30-minute cold starts on two nodes to
learn the same three things. **When validating wiring, use the smallest model
that exercises the same code path.**
