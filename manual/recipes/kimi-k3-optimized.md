# Kimi-K3 (optimized build)

Kimi-K3 on 8× MI355X (TP8) using `johnqin2025/kimi-k3-dspark` — an image carrying an
FP8 pre-route / shared-expert kernel cluster, an FP8 latent-MoE tail, a fused MLA
output gate and a tri-projection dispatch.

A separate entry from [Kimi-K3](kimi-k3) rather than a combination on it, because it
is a different artifact. Same vLLM commit (`g5f76ae224`) as the stock image, so the
delta is kernels, not engine version.

## 1. Choose the combination

::::{tab-set}

:::{tab-item} Mixed
:sync: mixed

No speculative decoding. **The right choice above concurrency 8**, where it is also
the faster of the two.

```bash
kubectl apply -f examples/recipes/kimi-k3-optimized/mixed/deploy.yaml
```

| 1024 in / 128 out | tok/s |
|---:|---:|
| c=4 | 67.11 |
| c=8 | 187.17 |
| c=16 | **426.51** |
:::

:::{tab-item} Mixed + DSpark
:sync: mixed-dspark

Speculative decoding with a block-diffusion draft that produces 7 tokens per
parallel pass. Worth **1.9–2.2×** at concurrency ≤ 8.

```bash
kubectl apply -f examples/recipes/kimi-k3-optimized/mixed-dspark/deploy.yaml
```

| 1024 in / 128 out | tok/s | vs mixed |
|---:|---:|---:|
| c=4 | 146.96 | 2.19× |
| c=8 | 359.72 | 1.92× |
| c≥16 | crashes | — |

```{admonition} Above concurrency 8 this is an outage, not a slowdown
:class: warning
At `c>=16` the engine dies with

    AssertionError: AiterMLA flattened verify requires a uniform decode query len

"verify" is the tell — the speculative path only. The identical config without
`--speculative-config` serves `c=16` at 426.51 tok/s, which is also *faster* than
this manifest's best result (359.72 at `c=8`).
```
:::

::::

## 2. Prerequisites

```bash
kubectl get nodes -o custom-columns=NODE:.metadata.name,GPU:.status.allocatable.'amd\.com/gpu'
helm install infera-operator deploy/operator/helm/infera-operator -n infera-system --create-namespace

hf download moonshotai/Kimi-K3      --local-dir <MODEL_DIR>/Kimi-K3
hf download Inferact/Kimi-K3-DSpark --local-dir <MODEL_DIR>/Kimi-K3-DSpark   # DSpark only
```

**`<MODEL_DIR>` must be local NVMe.** Kimi-K3's 96 shards load in ~8 min from local
disk and ~95 min from NFS — and the slow path does not merely run late, it exceeds
the ready timeout, so the worker restarts mid-load and never finishes.

First start is 10–14 min: the image rebuilds its AITER JIT modules in-container on
top of weight load and CUDA-graph capture.

## 3. Smoke test

```bash
kubectl -n infera port-forward svc/kimi-k3-opt-<base|dspark>-server 8000:8000 &
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"kimi-k3","messages":[{"role":"user","content":"What is the capital of France?"}],
       "max_tokens":200}' | jq -r '.choices[0].message.content'
```

Keep `max_tokens` generous — this model spends 70+ tokens on that sentence, and a
tight cap truncates it into something that reads like a broken deployment.

On the DSpark manifest, confirm speculation engaged rather than inferring it from
throughput later:

```bash
kubectl -n infera logs <worker-pod> -c main | grep -c 'running the draft eagerly'   # must be 0
kubectl -n infera logs <worker-pod> -c main | grep -oE 'Mean acceptance length: [0-9.]+' | tail -1
```

## 4. Settings that are not optional

| Setting | Why |
|---|---|
| the `KIMI_K3_*` / `VLLM_ROCM_*` env block | selects the optimized kernels. Without them the MoE asks aiter for a kernel that was never generated — `Invalid FlyDSL kernel name: flydsl_moe1_..._t16x64x256_...` — because there is no Kimi-K3 `tuned_fmoe.csv`, while dsv3/dsv4/glm5 all have one |
| `VLLM_ROCM_USE_KIMI_K3_PREROUTE_BF16=0` | must be `0`; a `1` shadows the FP8 cluster silently, at ~40% of throughput |
| `attention_backend: ROCM_AITER_MLA` | the ROCm counterpart of the upstream quick-start's `FLASHINFER_MLA`. Dropping the key instead of translating it is not the fix |
| `--gpu-memory-utilization 0.88` | the draft's weights land after the KV budget is computed; `0.95` dies with 998 MB free trying to allocate 2.32 GiB |
| `INFERA_ENGINE_READY_TIMEOUT=7200` | the 1800 s default is impossible on slow storage, and the worker then restarts mid-load forever — which reads as a crash loop, not as slow storage |

## 5. Validation status

Both manifests were **run end-to-end as written** on k3s (MI355X, 8 GPUs, 1M
context, BF16 KV), through the infera router.

Against the upstream quick-start for this image, three numbers reproduce and three
claims do not:

| | upstream | measured here |
|---|---|---|
| baseline step time, 8K/1K | 12.39 ms | 12.37 ms |
| DSpark step time, 8K/1K | 29.50 ms | 29.11 ms |
| acceptance length | 4.00 | 3.85 |
| concurrency assertion fires at | c=48 | **c=16** |
| c=16 throughput | 315.7 tok/s | **crashes** |
| c=32 throughput | 454.3 tok/s | **crashes** |

The upstream document states its concurrency table was measured on a sibling build
(`kimi-k3-dspark:wvsplitk-fix`) and not re-run on this image. These measurements are
that re-run: the concurrency fixes are not in this image.

PD and kvd combinations are not built for this image.

## Source

[`examples/recipes/kimi-k3-optimized/`](https://github.com/AMD-AGI/Infera/tree/main/examples/recipes/kimi-k3-optimized)
· [all recipes](index)
