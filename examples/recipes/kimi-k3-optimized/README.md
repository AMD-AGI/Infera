# Kimi-K3 (optimized build)

Kimi-K3 on MI355X using `johnqin2025/kimi-k3-dspark`, an image carrying an FP8
pre-route / shared-expert kernel cluster, an FP8 latent-MoE tail, a fused MLA
output gate and a tri-projection dispatch — plus an optional **DSpark** draft model
for speculative decoding.

This is a **separate model entry** from [`kimi-k3/`](../kimi-k3/README.md) rather
than a combo on it, because it is a different artifact. It is the same vLLM commit
(`g5f76ae224`) as the stock `vllm/vllm-openai-rocm:kimi-k3`, so the delta is
kernels, not engine version.

`mixed`, `mixed-dspark` and `pd` were **run end-to-end as written** on the image
they pin. `pd-dspark` needs `--speculative-config` on **both** roles, which is not
redundancy — see §7.

## 1. Choose the combination

|  | GPUs | Manifest | Reach for it when |
|---|---:|---|---|
| **mixed** | 8 | [`mixed/`](mixed/deploy.yaml) | the default. Best tokens per GPU at every concurrency measured |
| **mixed + DSpark** | 8 | [`mixed-dspark/`](mixed-dspark/deploy.yaml) | concurrency ≤ 16, where speculation is worth 1.1–2.6× on the same hardware |
| **pd** | 16 | [`pd/`](pd/deploy.yaml) | concurrency ≥ 32 **and** latency matters — at c=64 it is 1.25× the throughput at 22% lower TPOT |
| **pd + DSpark** | 16 | [`pd-dspark/`](pd-dspark/deploy.yaml) | PD's shape at high concurrency, plus speculation. Both roles must carry `--speculative-config` |

DSpark is speculative decoding with a block-diffusion draft that produces 7 tokens
in one parallel pass. The draft is a community checkpoint
([`Inferact/Kimi-K3-DSpark`](https://huggingface.co/Inferact/Kimi-K3-DSpark)) —
there is no official Moonshot draft for Kimi-K3.

```{admonition} The 20260801 image crashed above concurrency 8. This one does not.
:class: note
On `…-20260801`, any speculative combination died at `c>=16` with

    AssertionError: AiterMLA flattened verify requires a uniform decode query len

`…-20260802` fixes it: c=16/32/64 all complete, 0 restarts, and the assertion
appears zero times. Speculation is genuinely still on — `num_spec_tokens=7`,
CUDA-graph captured, `running the draft eagerly` count 0 — so this is a fix, not
speculation being quietly disabled.

If you are still on the 20260801 digest, the old ceiling still applies to you.
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
disk and ~95 min from NFS — and the slow path does not merely run late, it exceeds
the ready timeout, so the worker restarts mid-load and never finishes.

For the PD combinations, **each node needs its own copy** and the paths need not
match — hence the separate placeholders. On the fleet this was validated on, the
tempting common mount (`/mnt/shared`) turned out to be an NFS export of the *other*
node's array; using it on both sides puts one side on the 95-minute path. Check
each node with `df -hT <dir>` and take the local device, not the convenient common
name.

For `pd-dspark`, **the draft must be on the decode node** — that is the role that
loads it. Having it only on the prefill node fails with a missing path, which reads
like a typo in the manifest rather than a missing 6.7 GB directory.

First start is **12–14 min**: the image rebuilds its AITER JIT modules in-container
on top of weight load and CUDA-graph capture.

## 3. Deploy

```bash
sed 's|<MODEL_DIR>|/your/local/nvme/models|' \
  examples/recipes/kimi-k3-optimized/<mixed|mixed-dspark>/deploy.yaml | kubectl apply -f -
```

The PD combinations take four placeholders, because the roles land on different
nodes and each reads its own local copy:

```bash
sed -e 's|<PREFILL_NODE>|nodeA|'      -e 's|<DECODE_NODE>|nodeB|' \
    -e 's|<PREFILL_MODEL_DIR>|/local/nvme/models|' \
    -e 's|<DECODE_MODEL_DIR>|/local/nvme/models|' \
    examples/recipes/kimi-k3-optimized/<pd|pd-dspark>/deploy.yaml | kubectl apply -f -
```

## 4. Smoke test

```bash
kubectl -n infera port-forward svc/<name>-server 8000:8000 &

curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"kimi-k3","messages":[{"role":"user","content":"What is the capital of France?"}],
       "max_tokens":200}' | jq -r '.choices[0].message.content'
```

Keep `max_tokens` generous: this model spends 70+ completion tokens on that
sentence, so a tight cap truncates it into something that reads like a broken
deployment.

**On the DSpark manifests**, confirm speculation actually engaged rather than
inferring it from throughput later:

```bash
POD=$(kubectl -n infera get pod -o name -l infera.amd.com/service=worker | head -1)   # or =decode for PD

kubectl -n infera logs $POD -c main | grep -oE "speculative_config=SpeculativeConfig\([^)]*\)" | tail -1
kubectl -n infera logs $POD -c main | grep -c 'running the draft eagerly'   # must be 0
```

`running the draft eagerly` means the draft fell back to Triton instead of being
captured into CUDA graphs, which costs about 5% of decode throughput.

**On the PD manifests**, a correct answer proves nothing on its own: if the KV
handoff fails open, the decoder re-prefills locally and returns the same text —
faster. Check the decode side instead:

```bash
DEC=$(kubectl -n infera get pod -o name -l infera.amd.com/service=decode | head -1)
kubectl -n infera logs $DEC -c main | grep 'Engine 000' | tail -2
```

```
Avg prompt throughput: 0.1 tokens/s, Avg generation throughput: 7.8 tokens/s,
  ... External prefix cache hit rate: 100.0%
```

`External prefix cache hit rate` near 100% with `Avg prompt throughput` near zero
is the handoff working. The prefill pod is the mirror image — all prompt
throughput, no generation.

## 5. Measured results

MI355X, TP8 per worker, `--max-model-len 1048576`, BF16 KV, on k3s. Sweeps are
`vllm bench serve`, 1024 in / 128 out, `--random-range-ratio 0` `--ignore-eos`
`--seed 0`, `num-prompts = 4 × concurrency`.

**Everything below was measured on the same image**, so the ratios are attributable
to the manifests and not to a base-image difference. That matters: the previous
revision of this page compared a DSpark number against a `mixed` number taken on
the older image, and `mixed` itself moved by 15% between the two.

### 8 GPUs — output token throughput (tok/s)

| c | `mixed` | `mixed-dspark` | DSpark / mixed |
|---:|---:|---:|---:|
| 4 | 56.78 | **147.90** | **2.60×** |
| 8 | 241.69 | **328.58** | **1.36×** |
| 16 | 427.86 | **471.74** | **1.10×** |
| 32 | **689.22** | 685.25 | 0.99× |
| 64 | **972.37** | 879.84 | 0.90× |

Speculation's advantage decays monotonically with concurrency and crosses over
between c=16 and c=32. It is a low-concurrency optimisation, not a free win — and
the old "1.9–2.2×" figure on this page was inflated by an older, slower `mixed`
baseline.

### 16 GPUs — the PD pair

| c | `pd` | `pd-dspark` | dspark / pd | `pd` TPOT | `pd-dspark` TPOT |
|---:|---:|---:|---:|---:|---:|
| 4 | 50.38 | **135.34** | 2.69× | 23.20 ms | **7.81 ms** |
| 8 | 218.47 | **445.72** | 2.04× | 25.81 ms | **11.50 ms** |
| 16 | 428.98 | **718.64** | 1.68× | 27.28 ms | **12.37 ms** |
| 32 | 747.79 | **928.50** | 1.24× | 31.25 ms | **17.80 ms** |
| 64 | **1217.32** | 1079.32 | 0.89× | 38.12 ms | **17.44 ms** |

Speculation helps PD far more than it helps `mixed` (1.68× vs 1.10× at c=16),
because the decode role is not competing with prefill for the same GPUs. It decays
with concurrency the same way and crosses over at c=64 — but even there, where it
loses 11% of throughput, TPOT is less than half. If latency is what you are buying
16 GPUs for, `pd-dspark` is the endpoint at every concurrency measured.

### All four, per GPU

Absolute throughput favours the 16-GPU options; tokens **per GPU** never does:

| c | best absolute | tok/s | best per GPU | tok/s per GPU |
|---:|---|---:|---|---:|
| 4 | `mixed-dspark` (8) | 147.90 | `mixed-dspark` | 18.5 |
| 8 | `pd-dspark` (16) | 445.72 | `mixed-dspark` (8) | 41.1 |
| 16 | `pd-dspark` (16) | 718.64 | `mixed-dspark` (8) | 59.0 |
| 32 | `pd-dspark` (16) | 928.50 | `mixed` (8) | 86.2 |
| 64 | `pd` (16) | 1217.32 | `mixed` (8) | 121.5 |

So the PD pair is not a throughput-efficiency play. What it buys is headroom
(`pd` reaches 1217 tok/s where `mixed` tops out at 972) and latency
(`pd-dspark` holds TPOT at 17 ms where `mixed` is at 49 ms). Pick on those, not on
the tok/s column.

The decode side held `External prefix cache hit rate` at 99.9% across all five
concurrencies in both PD combinations, so the handoff never silently failed open —
checked per concurrency, not once at the start, because a handoff that fails open
under load makes throughput look *better*.

## 6. Settings that are not optional

Each row is a failure that was hit and diagnosed on this hardware, not a preference.

| Setting | Why |
|---|---|
| the `KIMI_K3_*` / `VLLM_ROCM_*` env block | these select the optimized kernels. Without them the MoE asks aiter for a kernel that was never generated: `ValueError: Invalid FlyDSL kernel name: flydsl_moe1_..._t16x64x256_...` — there is no Kimi-K3 `tuned_fmoe.csv`, while dsv3/dsv4/glm5 all have one |
| `VLLM_ROCM_USE_KIMI_K3_PREROUTE_BF16=0` | must be `0`. The pre-route dispatch tries BF16 first and a `1` shadows the FP8 cluster entirely — silently, at ~40% of throughput |
| `attention_backend: ROCM_AITER_MLA` | in the speculative config. The upstream quick-start says `FLASHINFER_MLA`, which is CUDA-only; **omitting the key entirely is not the fix** — this is its ROCm counterpart |
| `--gpu-memory-utilization 0.88` | the draft's weights land after the KV budget is computed. At `0.95` the run dies with 998 MB free trying to allocate 2.32 GiB |
| `INFERA_ENGINE_READY_TIMEOUT=7200` | infera's 1800 s default is generous for local NVMe and impossible for anything slower; the worker then kills itself mid-load and restarts forever, which reads as a crash loop rather than as slow storage |
| PD: each node needs its **own local** copy | the `model` volume is a `hostPath`. The fleet's obvious shared mount was an NFS export of the peer's array — one side then loads for ~95 min and restarts forever |
| PD: never point a client at it that sends requests the engine will reject | the engine validates **after** prefill, so a rejected request has already had its KV computed and queued for transfer. 84 requests rejected for one bad field left 424 aborted Mooncake transfers — 53 requests × 8 TP ranks — and stalled *valid* traffic for ~20 minutes with `MooncakeXferMetadata transfer failed: Resource temporarily unavailable`, which reads exactly like a broken fabric. It was a broken client |

`ibv_devices` is **not installed** in this image. Reading its `not found` as "no
RDMA devices" produced two false TCP-fallback diagnoses here; ask the library
instead:

```bash
kubectl -n infera exec <pod> -c main -- python3 -c '
import ctypes; lib = ctypes.CDLL("libibverbs.so.1")
lib.ibv_get_device_list.restype = ctypes.POINTER(ctypes.c_void_p)
n = ctypes.c_int(0); lib.ibv_get_device_list(ctypes.byref(n)); print(n.value)'
```


### Why `pd-dspark` configures speculation on *both* roles

The obvious reading is that speculation belongs only on decode — the prefiller
never samples, so a draft there looks like dead weight. That configuration was
tried. It fails in two independent ways, and the second is why the prefiller has
to be speculation-aware anyway.

**1. The layer lists disagree.** vLLM merges the draft's layers into the same
`kv_caches` dict as the target's, continuing the numbering. Kimi-K3 has 93 layers
and DSpark adds 5, so a speculating decoder registers `model.layers.0`…`97` and
sends all 98 names to the prefiller, which has nothing past 92:

```
KeyError: 'model.layers.93.self_attn'
```

`93` is exactly one past the target's last layer. The error travels back over ZMQ
and is logged on the **decode** side, pointing at the wrong host.

**2. The block counts disagree.** Forcing the layer lists to match — by keeping
draft layers out of the registration, which is semantically right since the
draft's KV is decode-local — gets past the KeyError and straight into:

```
pulling kv_caches for [...] failed: P num blocks less than D
```

`mooncake_connector.py` compares the blocks a request needs on each side
(`local blocks(N) < remote blocks(M)`). Speculation recomputes
`max_num_scheduled_tokens` to reserve slots for draft tokens, so the decoder's
block accounting differs from a prefiller that does not know speculation is
happening. No amount of layer filtering fixes that: the two sides have to compute
the count the same way.

So loading the draft on the prefiller is **the price of that agreement, not an
oversight**. It is never run there.

```{admonition} Both failures hang rather than error
:class: warning
The request never completes and never returns an error. All pods stay Ready,
health checks pass, restarts stay 0, and the decoder logs no inference at all. The
first symptom is a client waiting until its own timeout — 45 minutes in the run
that found this, against a deployment that `kubectl get pods` called healthy.

This is why the smoke test for PD checks the decode side's counters rather than
the answer text, and why every probe here carries an explicit timeout.
```

## 7. Validation status

| What | Status |
|---|---|
| `mixed/deploy.yaml` as written | **validated** — ready ~12 min, correct answer, c=4…64 swept |
| `mixed-dspark/deploy.yaml` as written | **validated** — ready ~12 min, c=4…64 swept, 0 restarts, assertion count 0 |
| `pd/deploy.yaml` as written | **validated cross-node** — ready ~12 min, c=4…64 swept, extcache 99.9% throughout |
| `pd-dspark/deploy.yaml` as written | **validated cross-node** — both roles Ready in ~14 min, 0 restarts; sweep below |
| `pd-dspark` with speculation on **decode only** | **hangs**, two separate ways — see above. Do not ship it |
| kvd combinations | not built for this image |
| fp8 KV cache | not measured here |

## Source

[`examples/recipes/kimi-k3-optimized/`](.) in [AMD-AGI/Infera](https://github.com/AMD-AGI/Infera)
· [all recipes](../README.md) · [stock Kimi-K3 recipe](../kimi-k3/README.md)
