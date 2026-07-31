# kvaware + kvd on the sglang engine — support survey & PD+DPA correctness experiment

Goal: (a) document what infera actually supports for **kv-aware routing** and **kvd**
on the **sglang** engine; (b) run a PD + DP-attention deployment with both switched on
and check correctness.

Node: chi2879 (10.2.122.10), 8x MI355X. Image `infera/engine-sglang:pd-unified`
(sglang 0.5.15.post1). Model: **Qwen3-1.7B** — deliberately tiny. The thing under test is
the kvaware/kvd wiring, not the model, and a 1.7B model turns a ~30 min GLM-5.2 cold start
into ~2 min, so the loop could run 5 rounds instead of 1.

Topology: single host, 1P1D, prefill TP4 on GPU0-3 :30000, decode TP4 on GPU4-7 :31000,
DP-attention on **both** legs, mooncake PD transport, infera router :8100, etcd :2379,
`infera.kvd` daemon on /tmp/kvd/kvd.sock (8 GiB RAM tier + 64 GB L3 on /tmp/kvd-long).

## Round 0 — static validation (no GPU spent)

Ran sglang's own `ServerArgs.from_cli_args` over the arg combos in a container.

| # | combo | verdict |
|---|-------|---------|
| 1 | mix baseline | OK |
| 2 | mix + hicache | OK |
| 3 | mix + DPA + hicache | OK |
| 4 | PD-prefill + DPA + hicache | OK |
| 5 | PD-decode + DPA + hicache | **FAIL** — hicache vs disable-radix-cache |
| 6 | PD-decode + DPA + hicache + `--disaggregation-decode-enable-radix-cache` | OK |
| 7 | + `--disable-radix-cache` | FAIL (same conflict) |
| 8 | + `--disaggregation-decode-enable-offload-kvcache` | FAIL (same conflict) |

Mechanism behind #5/#6, confirmed by probing `disable_radix_cache` directly:
a decode leg sets `disable_radix_cache=True` by itself ("KV cache is forced as chunk cache
for decode server"), and sglang rejects `enable-hierarchical-cache + disable-radix-cache`
(`server_args.py:_handle_cache_compatibility`). `--disaggregation-decode-enable-radix-cache`
flips it back to False, which is what makes kvd legal on a decode leg — and infera
auto-appends exactly that flag when kv-events are on and the backend is mooncake
(`infera/engine/sglang/args.py:251-263`). So **kvaware being on is what makes kvd work on
the decode leg**; turning kv-events off there re-breaks it.

## Round 1 — first live run: two failures

`MY_IP=10.2.122.10 KVD=1 KVAWARE=1 POLICY=kv-aware`

- prefill died: `ZMQError: Address already in use (tcp://127.0.0.1:30235)`.
- decode reached `Allocating 354.94 GB host memory for hierarchical KV cache` **per DP rank**
  (x4 = 1.4 TB on a 3 TB box). Cause: default `hicache_ratio=2.0` sizes the host pool off
  `max_total_num_tokens` (1,547,424 tokens for this small model).

Fixes: `--hicache-size 8` (absolute GB, overrides the ratio) and spaced the leg ports
30000/31000.

## Round 2 — hicache fixed, decode still dies

hicache now allocates 8.00 GB/rank and the kvd adapter connects
(`infera-kvd adapter connected to /tmp/kvd/kvd.sock`, `Tree cache initialized: ...
HiRadixCache ... hierarchical=True`). prefill = ready. decode Rank 0 scheduler died:
`zmq.error.ZMQError: Address already in use (addr='tcp://*:32765')`.

Both legs' `--kv-events-config` carried the **same** endpoint port (32764). With dp_size=4
sglang binds base+rank, so decode collided with prefill's 32765.

### Root cause — a real infera bug (kvaware path)

`infera/common/net.py:free_tcp_port_block()` scanned down from a **fixed**
`ip_local_port_range.low - count` and released the probe sockets in `finally`. Two engines
on one host therefore picked the *same* base deterministically. It is reached only from
`worker.py:77`, i.e. only when `enable_kv_events` is on — a kvaware-path bug.

MVP against the pre-fix code:

```
OLD bases: [32764, 32764, 32764, 32764, 32764, 32764, 32764, 32764, 32764, 32764]
```

Two candidate fixes were tested before choosing:

- *Hold the reservation until the child binds* — **rejected, verified harmful**: the probe
  binds `127.0.0.1:P`, the child binds `0.0.0.0:P` (zmq `tcp://*`), and a live MVP showed
  the child then gets `errno 98`. Holding would lock out our own subprocess.
- *`0.0.0.0` + `SO_REUSEADDR` reservation* — **rejected**: MVP showed a second probe can take
  the same port anyway, so the reservation isn't exclusive; it buys nothing.

Chosen fix: **randomise the scan start** (`_PORT_BLOCK_TRIES=64` random bases, then the
old exhaustive downward scan as fallback). Keeps the signature and the socket semantics,
removes the determinism that made the collision certain.

Regression test `tests/unit/common/test_net_port_block.py` (4 tests): block is free +
contiguous, sits below the ephemeral range, repeated calls don't all collide, count=1
delegates. Verified it **fails on the pre-fix code** (10/10 identical bases) and passes after.

## Round 3 — port fix confirmed; same-host RDMA fails

Both legs ready, distinct kv-event endpoints (17213 vs 31215) — the port bug is fixed and
both workers register with the router. Completion returned HTTP 500:

```
Failed to get kvcache from prefill instance
worker_pool.cpp:408 ... local_nic: ionic_0, peer_nic: ...@ionic_4: transport retry counter exceeded
rdma_endpoint.cpp:472 Invalid argument: received packet mismatch
```

Both legs are on one host, so mooncake RDMA has to loop back across rails (ionic_0 →
ionic_4) and can't. This is a same-host RDMA limitation, **orthogonal to kvaware/kvd**.
Worked around with `MC_FORCE_TCP=1` (this repo's known correct-but-slower path). Also gave
each leg its own `--kv-events-bind` port (5557/5657) — infera's own publisher socket,
separate from the sglang one, and its default is identical on both legs.

## Round 4 — full stack up, output garbled

kvaware ON + kvd ON, TCP transport. Both legs ready, no 500s, KV handoff completes. But:

```
Q1 "capital of France" -> 'v4 freddy\n\nAists. Log In andapace.a\n\nWho wouldin %%%%3...'
Q2 "17*23"             -> 'S情辣梯neig治\n\n杖\n\n及格...'
```

Directly probing a PD leg returns nothing (expected — a PD leg only serves through the
pair), so the discriminating experiment is a **differential run**.

## Round 5 — 对拍 baseline: kvaware OFF + kvd OFF

Same topology, same TCP transport, `KVD=0 KVAWARE=0 POLICY=round-robin`:

```
'v4ই\n\n脐猫\n\nument="">< t-tcan you-...'
```

**The baseline is garbled too** — and note both runs start with the same `v4` token.

## Conclusion

- The garbled output belongs to this **same-host PD + `MC_FORCE_TCP`** setup, not to
  kvaware or kvd. With both features off the output is equally wrong.
- **kvaware and kvd introduce no correctness regression** relative to the baseline: legs
  start, the kvd HiCacheStorage backend connects and serves, both workers register with
  the kv-aware router, KV handoff completes.
- What this run does **not** establish is coherent generation end-to-end — the transport
  substrate was already broken before the features were added. A clean coherence check
  needs a **two-node** PD pair (real RDMA between hosts), which is how this stack is
  normally run.

## Support matrix (what infera actually implements for sglang)

| | sglang | vLLM | ATOM |
|---|---|---|---|
| kv-aware (KV events) | yes, native — infera monkey-patches `RadixCache.insert/evict` (`kv_probe.py`), passes `--kv-events-config` to sglang (`worker.py:78-85`) | yes | yes, but via `.pth` site hooks and OFF by default |
| kvd | yes — `InferaKvdBackend(HiCacheStorage)` (`kvd_adapter.py`), wired by `--infera-kvd-socket` (`kvd_wiring.py`) | yes — `InferaKvdConnector(KVConnectorBase_V1)` | no |

Notes worth carrying forward:

- `manual/features/kv_cache_offload.md:14-18` still says KV-Cache Offload is **"vLLM only
  (for now)"**. That is stale: sglang has a working kvd path via HiCache. Reported, not fixed
  (out of scope here).
- `--infera-kvd-socket` fails **fast** if the daemon isn't reachable (5 s probe) — the engine
  refuses to start rather than serving with a silently dead cache backend.
- infera lowers sglang's prefetch threshold to 64 for short prompts, but on 0.5.15.post1 it
  logs `SGLang version has no recognized prefetch_threshold field` — the field was renamed
  upstream, so the override silently doesn't apply. It still passes `prefetch_threshold: 64`
  inside the dynamic-backend extra-config, which is probably what takes effect.
- `--hicache-ratio` < 1.5 silently disables L3 prefetch (`hicache_validate.py`); with a big
  KV pool prefer `--hicache-size <GB>`, or the host allocation explodes (round 1: 355 GB/rank).

## Files

- `leg.sh` — one PD leg via `python3 -m infera.engine.sglang` (**not** `sglang.launch_server`;
  the kvaware/kvd wiring lives in the infera wrapper, so the GLM-5.2 scripts in
  `glm5.2.mxfp4.packup_20260727/` bypass it entirely).
- `up.sh` — kvd daemon + both legs + kv-aware router.
- Logs on chi2879 in the `kvexp` container: `/tmp/r1..r5/`.
