# Infera recipes

Ready-to-run Kubernetes deployments for specific models, in the shape you want to
serve them. Pick a model, pick a combo, `kubectl apply`.

| Model | Engine | Recipe |
|---|---|---|
| GLM-5.2-MXFP4 | SGLang | [`glm5.2/`](glm5.2/README.md) |
| Kimi-K3 | vLLM | [`kimi-k3/`](kimi-k3/README.md) |

## The four combos

Every recipe comes in the same four shapes. They compose two independent choices:
**how requests are split across GPUs**, and **whether KV survives past the GPU**.

| Combo | Serving | KV cache | Use it when |
|---|---|---|---|
| `mixed` | one worker does prefill + decode | GPU only | default; simplest thing that works |
| `mixed-kvd` | one worker does prefill + decode | + kvd L2 host RAM, L3 on a PVC | repeated/long prefixes — shared system prompts, multi-turn chat, RAG |
| `pd` | prefill and decode on separate nodes | GPU only | prefill and decode want different batching; you have a RoCE fabric |
| `pd-kvd` | prefill and decode on separate nodes | + kvd on each role | both of the above |

`pd` needs the two nodes on a **mutually routable RoCE fabric** — the KV handoff is
RDMA, and there is no TCP fallback. If you are on one box, use `mixed`.

## How these manifests are built

Every recipe is **stock vendor image + overlay + (optional) sidecar**. The vendor
image is never forked:

```
initContainer  infera-overlay   busybox carrying /payload  ──cp──▶ emptyDir
container      main             STOCK vllm/sglang image, runs /overlay/bin/infera-exec
container      kvd              same vendor image, also runs from /overlay   (kvd combos)
```

`infera-exec` picks the payload trees matching the container's CPython minor and
ROCm major, then execs the engine. So following an upstream vLLM or SGLang release
is an image-tag edit here — no rebuild of ours, and no repeat of the incident where
forking the base for one model broke every other model.

Build the overlay before deploying:

```bash
docker build -f deploy/overlay/Dockerfile.payload -t rocm/infera-overlay:latest .
```

**The payload must contain a `native/` tree** (Mooncake + hipFile) for the `kvd`
and `pd` combos. Build it from an image that has them — that is what the
`NATIVE_IMAGE` build arg is for. A payload without `native/` still runs `mixed`
perfectly, and merely *warns* on the others, so the kvd and pd manifests set
`INFERA_REQUIRE_NATIVE=1`: without it a wrong overlay tag deploys clean and
silently serves without the GPU-direct L3 or the KV transport you asked for.

See [`deploy/overlay/README.md`](../../deploy/overlay/README.md) for how the payload
is assembled.

## Source

- Manifests: [`examples/recipes/`](.) in [AMD-AGI/Infera](https://github.com/AMD-AGI/Infera)
- Overlay build: [`deploy/overlay/`](../../deploy/overlay)
- More deployment shapes: [`examples/k8s-deployments/`](../k8s-deployments)
