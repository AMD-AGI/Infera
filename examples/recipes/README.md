# Infera recipes

Ready-to-run Kubernetes deployments for specific models, in the shape you want to
serve them. Pick a model, pick a combo, `kubectl apply`.

| Model | Engine | Recipe |
|---|---|---|
| GLM-5.2-MXFP4 | SGLang | [`glm5.2/`](glm5.2/README.md) |
| GLM-5.2-FP8 (gfx942) | SGLang | [`glm5.2-fp8-gfx942/`](glm5.2-fp8-gfx942/README.md) |
| Kimi-K3 | vLLM | [`kimi-k3/`](kimi-k3/README.md) |


```{admonition} The directory is `aggregated`, the API field still says `mixed`
:class: note
`aggregated` / `disaggregated` is the industry vocabulary and what these directories
are named. The `role:` field inside each manifest is an **API value consumed by the
operator** — `mixed`, `prefill`, `decode` — and is deliberately left alone: renaming
it would break every deployed configuration and anything outside this repo using the
CRD. So `aggregated/deploy.yaml` legitimately contains `role: mixed`. Read `role:`
as the wire format, and the directory name as what it means.
```

## The four combos

Recipes come in the same four shapes, composing two independent choices: **how
requests are split across GPUs**, and **whether KV survives past the GPU**. A
recipe pinned to one validated configuration ships only the shapes it was run in —
`glm5.2-fp8-gfx942` is `disaggregated-kvd` alone.

| Combo | Serving | KV cache | Use it when |
|---|---|---|---|
| `aggregated` | one worker does prefill + decode | GPU only | default; simplest thing that works |
| `aggregated-kvd` | one worker does prefill + decode | + kvd L2 host RAM, L3 on a PVC | repeated/long prefixes — shared system prompts, multi-turn chat, RAG |
| `disaggregated` | prefill and decode on separate nodes | GPU only | prefill and decode want different batching; you have a RoCE fabric |
| `disaggregated-kvd` | prefill and decode on separate nodes | + kvd on each role | both of the above |

`disaggregated` needs the two nodes on a **mutually routable RoCE fabric** — the KV handoff is
RDMA, and there is no TCP fallback. If you are on one box, use `aggregated`.

## How these manifests are built

Recipes are **stock vendor image + overlay + (optional) sidecar**. The vendor image
is never forked:

```
initContainer  infera-overlay   busybox carrying /payload  ──cp──▶ emptyDir
container      main             STOCK vllm/sglang image, runs /overlay/bin/infera-exec
container      kvd              same vendor image, also runs from /overlay   (kvd combos)
```

`infera-exec` picks the payload trees matching the container's CPython minor and
ROCm major, then execs the engine. So following an upstream vLLM or SGLang release
is an image-tag edit here — no rebuild of ours, and no repeat of the incident where
forking the base for one model broke every other model.

**`glm5.2-fp8-gfx942` is the exception**, and its README §1 says why: GLM-5.2 on the
v0.5.16 gfx942 base needs a rebuilt Mooncake `engine.so` plus four SGLang source
patches, while the payload carries `deploy/docker/patches/vllm/` only and
`infera-exec` runs that loop just when `vllm` imports. A payload-mounted stock base
would come up green there and then die on the first cross-node KV transfer, corrupt
long prompts, or kill the prefill scheduler. Where the overlay can carry what a
deployment needs, it does; that recipe is what it looks like when it cannot.

Build the overlay before deploying:

```bash
docker build -f deploy/overlay/Dockerfile.payload -t inferaimage/infera-overlay:v0.2.2 .
```

The build produces **one native tree per ABI family**, by two different routes:
`NATIVE_IMAGE` supplies the vLLM one (CPython 3.12) by harvesting it, while the
SGLang one (3.10) is compiled during the build itself, so that its Mooncake is
known to carry the HIP-transport gate instead of inheriting whatever the engine
image shipped. Mooncake and hipFile bind both the ROCm major and the CPython
minor, so neither tree can stand in for the other.

The families do not carry the same capabilities, and that is by design:

| | `mooncake` (PD KV transport) | `hipfile` (kvd GPU-direct L3) |
|---|---|---|
| `rocm7-py312` (vLLM) | yes | yes |
| `rocm7-py310` (SGLang) | yes | **no — vLLM-only path** |

So each manifest names the capabilities it needs via `INFERA_REQUIRE_NATIVE`
(e.g. `mooncake`, `hipfile`, or `mooncake,hipfile`) and `infera-exec` refuses to
start without them. Combos that need nothing native — `aggregated` anywhere, and
`aggregated-kvd` on SGLang, which reaches its L2 and file L3 through pure-Python
HiCacheStorage — leave it unset.

See [`deploy/overlay/README.md`](../../deploy/overlay/README.md) for how the payload
is assembled.

## Source

- Manifests: [`examples/recipes/`](.) in [AMD-AGI/Infera](https://github.com/AMD-AGI/Infera)
- Overlay build: [`deploy/overlay/`](../../deploy/overlay)
- More deployment shapes: [`examples/k8s-deployments/`](../k8s-deployments)
