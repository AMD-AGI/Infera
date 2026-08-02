# Kimi-K3 (optimized build)

Kimi-K3 on 8× MI355X (TP8) using `johnqin2025/kimi-k3-dspark`, an image carrying an
FP8 pre-route / shared-expert kernel cluster, an FP8 latent-MoE tail, a fused MLA
output gate and a tri-projection dispatch.

This is a **separate model entry** from [`kimi-k3/`](../kimi-k3/README.md) rather
than a combo on it, because it is a different artifact. It is the same vLLM commit
(`g5f76ae224`) as the stock `vllm/vllm-openai-rocm:kimi-k3`, so the delta is
kernels, not engine version.

## 1. Choose the combination

| Combination | Manifest | Use it when |
|---|---|---|
| **mixed** | [`mixed/`](mixed/deploy.yaml) | concurrency above 8 — and it is the faster recipe there |
| **mixed + DSpark** | [`mixed-dspark/`](mixed-dspark/deploy.yaml) | concurrency ≤ 8, where speculation is worth 1.9–2.2× |
| **pd** | [`pd/`](pd/deploy.yaml) | you have two nodes on a routable RoCE fabric and want prefill and decode batched separately |

DSpark is speculative decoding with a block-diffusion draft model that drafts 7
tokens in one parallel pass. The draft is a community checkpoint
([`Inferact/Kimi-K3-DSpark`](https://huggingface.co/Inferact/Kimi-K3-DSpark)) —
there is no official Moonshot draft for Kimi-K3.

```{admonition} With DSpark, concurrency above 8 is an outage, not a slowdown
:class: warning
At `c>=16` the engine dies with

    AssertionError: AiterMLA flattened verify requires a uniform decode query len

"verify" is the tell — this is the speculative path only. The identical config
without `--speculative-config` serves `c=16` at **426.51 tok/s**, which is also
*faster* than the DSpark recipe's best result (359.72 at `c=8`). Above 8, use
`mixed/`.
```

## 2. Prerequisites

```bash
# nodes must advertise amd.com/gpu
kubectl get nodes -o custom-columns=NODE:.metadata.name,GPU:.status.allocatable.'amd\.com/gpu'

# the operator (provides the InferaDeployment CRD)
helm install infera-operator deploy/operator/helm/infera-operator -n infera-system --create-namespace
```

Both models go in one directory, mounted at `/models`:

```bash
hf download moonshotai/Kimi-K3      --local-dir <MODEL_DIR>/Kimi-K3
hf download Inferact/Kimi-K3-DSpark --local-dir <MODEL_DIR>/Kimi-K3-DSpark   # DSpark only
```

**`<MODEL_DIR>` must be local NVMe.** Kimi-K3's 96 shards load in ~8 min from local
disk and ~95 min from NFS with two nodes on the same mount — and the slow path does
not merely run late, it exceeds the ready timeout, so the worker restarts mid-load
and never finishes.

First start is **10–14 min**: the image rebuilds its AITER JIT modules in-container
on top of weight load and CUDA-graph capture. DSpark adds ~4 min over the baseline.

## 3. Deploy

```bash
sed 's|<MODEL_DIR>|/your/local/nvme/models|' \
  examples/recipes/kimi-k3-optimized/<combo>/deploy.yaml | kubectl apply -f -
kubectl -n infera get pods -w
```

`pd/` takes four placeholders instead, because the two roles land on different
nodes and each reads its **own local** copy — the paths do not have to match, and
on this fleet they do not:

```bash
sed -e 's|<PREFILL_NODE>|chi2800|'      -e 's|<DECODE_NODE>|chi2866|' \
    -e 's|<PREFILL_MODEL_DIR>|/mnt/k3local/models/moonshotai|' \
    -e 's|<DECODE_MODEL_DIR>|/mnt/nvmeraid/models/moonshotai|' \
    examples/recipes/kimi-k3-optimized/pd/deploy.yaml | kubectl apply -f -
```

## 4. Smoke test

```bash
kubectl -n infera port-forward svc/kimi-k3-opt-<base|dspark>-server 8000:8000 &

curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"kimi-k3","messages":[{"role":"user","content":"What is the capital of France?"}],
       "max_tokens":200}' | jq -r '.choices[0].message.content'
```

Keep `max_tokens` generous: this model spends 70+ completion tokens on that
sentence, so a tight cap truncates it into something that reads like a broken
deployment.

On the DSpark manifest, confirm speculation actually engaged rather than inferring
it from throughput an hour later:

```bash
POD=$(kubectl -n infera get pod -o name \
  -l infera.amd.com/deployment=kimi-k3-opt-dspark,infera.amd.com/service=worker | head -1)

kubectl -n infera logs $POD -c main | grep -oE "cudagraph_mode': <CUDAGraphMode[^,]*"   # FULL_AND_PIECEWISE
kubectl -n infera logs $POD -c main | grep -c 'running the draft eagerly'               # must be 0
kubectl -n infera logs $POD -c main | grep -oE 'Mean acceptance length: [0-9.]+' | tail -1
```

`running the draft eagerly` means the draft fell back to Triton instead of being
captured into CUDA graphs, which costs about 5% of decode throughput.

On `pd`, the correct answer proves nothing on its own: if the KV handoff fails
open, the decoder simply re-prefills locally and returns the same text. Check the
**decode** side instead —

```bash
DEC=$(kubectl -n infera get pod -o name -l infera.amd.com/service=decode | head -1)
kubectl -n infera logs $DEC -c main | grep 'Engine 000' | tail -2
```

```
Avg prompt throughput: 0.1 tokens/s, Avg generation throughput: 7.8 tokens/s,
  ... External prefix cache hit rate: 100.0%
```

`External prefix cache hit rate: 100.0%` with `Avg prompt throughput` near zero is
the handoff working: the decoder computed no prefill and took the KV off the wire.
The prefill pod is the mirror image — all prompt throughput, no generation.

## 5. Measured results

8× MI355X, TP8, `--max-model-len 1048576`, BF16 KV, on k3s through the infera
router. Every number below was measured on the manifests in this directory.

**8K in / 1K out, batch 1** — read *step time* (median ITL), not TPOT: TPOT is step
time ÷ tokens-per-step, so it flatters speculation and swings with acceptance.

| | step time | acceptance |
|---|---:|---|
| mixed | 12.37 ms | — |
| mixed + DSpark | 29.11 ms | 3.85 of 7 |

A DSpark step costs 2.35× a baseline step and delivers ~3.85 tokens instead of 1.

**pd, 2139 in / 127 out, 1P1D over two nodes — 16 GPUs:**

| concurrency | output tok/s | P50 | P90 |
|---:|---:|---:|---:|
| 4 | 130.15 | 3.59 s | 4.27 s |
| 8 | 229.61 | 4.34 s | 4.59 s |
| 16 | 386.28 | 4.63 s | 6.09 s |

These are deliberately **not** in the table below. `pd` uses 16 GPUs against
`mixed`'s 8 *and* was measured at a 2139-token prompt against its 1024 — two
differences at once, so any ratio between the two tables measures nothing.

**mixed, 1024 in / 128 out, output token throughput — 8 GPUs:**

| concurrency | mixed | mixed + DSpark | speedup |
|---:|---:|---:|---:|
| 4 | 67.11 | **146.96** | 2.19× |
| 8 | 187.17 | **359.72** | 1.92× |
| 16 | **426.51** | crashes | — |
| 32, 64 | not measured | crashes | — |

The speedup decays with concurrency (2.19× → 1.92×) as the verify batch grows and
acceptance drops (3.85 at batch 1 → 3.62 at `c=8`). Extrapolating that trend, even
without the crash it is unlikely to still be winning at `c=16`.

## 6. Settings that are not optional

Each row is a failure that was hit and diagnosed on this hardware, not a preference.

| Setting | Why |
|---|---|
| the `KIMI_K3_*` / `VLLM_ROCM_*` env block | these select the optimized kernels. Without them the MoE asks aiter for a kernel that was never generated: `ValueError: Invalid FlyDSL kernel name: flydsl_moe1_..._t16x64x256_...` — there is no Kimi-K3 `tuned_fmoe.csv`, while dsv3/dsv4/glm5 all have one |
| `VLLM_ROCM_USE_KIMI_K3_PREROUTE_BF16=0` | must be `0`. The pre-route dispatch tries BF16 first and a `1` shadows the FP8 cluster entirely — silently, at ~40% of throughput |
| `attention_backend: ROCM_AITER_MLA` | in the speculative config. The upstream quick-start says `FLASHINFER_MLA`, which is CUDA-only; **omitting the key entirely is not the fix** — this is its ROCm counterpart |
| `--gpu-memory-utilization 0.88` | the draft's weights land after the KV budget is computed. At `0.95` the run dies with 998 MB free trying to allocate 2.32 GiB |
| `INFERA_ENGINE_READY_TIMEOUT=7200` | infera's 1800 s default is generous for local NVMe and impossible for anything slower; the worker then kills itself mid-load and restarts forever, which reads as a crash loop rather than as slow storage |
| `pd`: each node needs its **own local** copy of the weights | the `model` volume is a `hostPath`, so the two nodes need not use the same path — and on this fleet the tempting common name (`/mnt/shared`) is an NFS export of the *other* node's array. Mounting it on both sides turns one side's 8-minute load into ~95 minutes, which then exceeds the ready timeout and restarts forever. `df -hT <dir>` on each node and take the local device |
| `pd`: never point a client at it that sends requests the engine will reject | the engine validates **after** prefill, so a rejected request has already had its KV computed and queued for transfer. 84 requests rejected for one bad field (`min_tokens` equal to `max_tokens`, which fails because the server silently decrements `max_tokens` by 1) left 424 aborted Mooncake transfers — 53 requests × 8 TP ranks — and stalled *valid* traffic behind the backlog for ~20 minutes. It presents as `MooncakeXferMetadata transfer failed: Resource temporarily unavailable` and reads exactly like a broken fabric |

## 7. Validation status

| What | Status |
|---|---|
| `mixed/deploy.yaml` as written | **validated end-to-end** — ready in ~10 min, correct answer, c=4/8/16 measured |
| `mixed-dspark/deploy.yaml` as written | **validated end-to-end** — ready in ~14 min, correct answer, c=4/8 measured, c≥16 confirmed to crash |
| `pd/deploy.yaml` as written | **validated end-to-end across two nodes** — see below |
| `pd` + DSpark | not attempted. Speculation is decode-side and its behaviour in the `kv_consumer` role is a separate unknown |
| kvd combinations | not built for this image |
| fp8 KV cache | not measured here |

The `pd` run was 1P1D on chi2800 (prefill) and chi2866 (decode), 8 GPUs each:

| | |
|---|---|
| both roles Ready | 790 s, 0 restarts |
| RDMA devices | 8 on each side, via `ibv_get_device_list` |
| handoff verified | decode at `External prefix cache hit rate: 100.0%` with ~0 prompt throughput; prefill at 2010.5 tok/s prompt and ~0 generation |
| clean requests | 131 consecutive (47 correctness + 84 sweep), 0 failed transfers |

`ibv_devices` is **not installed** in this image. Running it and reading `not
found` as "no RDMA devices" is a mistake that cost two false TCP-fallback
diagnoses here; `ibv_get_device_list` reported 8 the whole time:

```bash
kubectl -n infera exec <pod> -c main -- python3 -c '
import ctypes; lib = ctypes.CDLL("libibverbs.so.1")
lib.ibv_get_device_list.restype = ctypes.POINTER(ctypes.c_void_p)
n = ctypes.c_int(0); lib.ibv_get_device_list(ctypes.byref(n)); print(n.value)'
```

Against the upstream quick-start for this image, three of its numbers reproduce
closely and three of its claims do not:

| | upstream | measured here |
|---|---|---|
| baseline step time, 8K/1K | 12.39 ms | 12.37 ms |
| DSpark step time, 8K/1K | 29.50 ms | 29.11 ms |
| acceptance length | 4.00 | 3.85 |
| concurrency assertion fires at | c=48 | **c=16** |
| c=16 throughput | 315.7 tok/s | **crashes** |
| c=32 throughput | 454.3 tok/s | **crashes** |

The upstream document states its own concurrency table was measured on a sibling
build (`kimi-k3-dspark:wvsplitk-fix`) and **not re-run on this image**. These
measurements are that re-run: the concurrency fixes are not in this image.

## Source

[`examples/recipes/kimi-k3-optimized/`](.) in [AMD-AGI/Infera](https://github.com/AMD-AGI/Infera)
· [all recipes](../README.md) · [stock Kimi-K3 recipe](../kimi-k3/README.md)
