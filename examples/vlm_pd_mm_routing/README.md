# VLM multimodal-aware routing on PD (image affinity)

A runnable, single-node demo that a **repeat image co-locates on one prefill
worker** under the Infera router's kv-aware policy — reusing that worker's warm
vision/prefix cache — while the same workload spreads evenly under round-robin.

It stands up a **PD-disaggregated** topology (2 prefill + 2 decode SGLang
`Qwen2.5-VL-7B` workers, KV cache moved over `mooncake_tcp` — no RDMA), fronts it
with the Infera router, and drives 20 requests that all carry the **same** image.

## Why this matters

For a vision request the router-side text hasher can't reproduce the engine's
image blocks (SGLang substitutes pad-values into token-ids, vLLM folds image
identity into block-hash extra-keys), and the image-placeholder token id is the
same regardless of which image was sent. So the router **can't** trust text
cache-locality for multimodal requests. Instead it keys a small per-worker LRU on
the image *reference* (`xxh3` of the http URL / `data:` URI) and routes a repeat
image back to the worker that already holds it. This is **engine-agnostic** — the
affinity keys the router's own image→worker map, never the engine's KV hashes, so
one code path serves SGLang, vLLM and ATOM. In PD it applies in the **prefill**
pool (the image is processed during prefill).

## Run

```bash
bash run.sh            # bring up, measure, tear down
KEEP=1 bash run.sh     # leave the containers up afterwards
```

Everything is env-driven (defaults in parentheses): `MODEL`
(`…/Qwen2.5-VL-7B-Instruct`), `P_GPUS` (`0 2`), `D_GPUS` (`4 6`), `NREQ` (`20`),
`IMG`, `NODE_IP`, `MOUNT`. Needs 4 free GPUs on one node and ~4 min to load.

## Expected output

```
== [5/5] verdict (prefill-pool routing of 20 identical-image requests) ==
   kv-aware:      <worker-A>×20    <- affinity co-locates
   round-robin:   <worker-A>×10 <worker-B>×10    <- control splits evenly
```

All 20 requests complete (`200`) through the full prefill → KV-transfer → decode
pipeline in both runs; only the *placement* differs.

## How it works

`run.sh` runs the router straight from this checkout
(`python -m infera.server --router-backend python`), so it exercises the source
here with no image rebuild. The router auto-selects PD dispatch when a model has
both prefill and decode workers and calls `pick(role_hint="prefill")` /
`pick(role_hint="decode")` per pool; the verdict is read from the `pick` log
lines (`role=prefill … picked=<worker> mm_affinity_hits=N`).

To enable this in your own deployment, see
[`manual/features/kv_aware_routing.md`](../../manual/features/kv_aware_routing.md)
(§ Multimodal (image) affinity).

## Files

| File           | Purpose                                                             |
| -------------- | ------------------------------------------------------------------- |
| `run.sh`       | One-command orchestrator: etcd + 2P2D workers + router + verdict.   |
| `mm_probe.py`  | Load probe — N requests carrying one stdlib-generated PNG.          |
