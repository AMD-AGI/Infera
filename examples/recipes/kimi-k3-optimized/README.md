# Kimi-K3 (optimized build)

Kimi-K3 on MI355X using `johnqin2025/kimi-k3-dspark`, an image carrying an FP8
pre-route / shared-expert kernel cluster, an FP8 latent-MoE tail, a fused MLA
output gate and a tri-projection dispatch — plus an optional **DSpark** draft model
for speculative decoding.

This is a **separate model entry** from [`kimi-k3/`](../kimi-k3/README.md) rather
than a combo on it, because it is a different artifact. It is the same vLLM commit
(`g5f76ae224`) as the stock `vllm/vllm-openai-rocm:kimi-k3`, so the delta is
kernels, not engine version.

All four combinations were **run end-to-end** on the image they pin, with the
placeholders substituted — the files themselves are templates and cannot be applied
as-is. `disaggregated-dspark` needs `--speculative-config` on **both** roles, which is not
redundancy — see §7.

## 0. Placeholders you must fill in

**Every manifest in this directory is a template.** None of them can be applied
directly, and the API server accepts them anyway — placeholders pass CRD
validation. What happens next depends on which one you left in, and neither is
obvious:

- **`<MODEL_DIR>` left in** → the Pod is created and fails at mount time with
  `hostPath type check failed: <MODEL_DIR> is not a directory`.
- **`<NODE>` left in** → **no Pod is ever created.** `kubectl apply` prints
  `created` and exits 0, `kubectl get pods` shows nothing, and the
  InferaDeployment's status stays empty. The only error is in the *operator's* log,
  in a different namespace:
  ```
  kubectl -n infera-system logs deploy/infera-operator
  ERROR Reconciler error ... spec.template.spec.nodeSelector: Invalid value: "<NODE>":
    a valid label must be an empty string or consist of alphanumeric characters...
  ```
  Waiting for a Pod that will never appear is the expensive part. If `get pods`
  returns nothing a minute after apply, read the operator log.

Substitute first.

| Placeholder | In | What it is | How to find it |
|---|---|---|---|
| `<MODEL_DIR>` | `aggregated`, `aggregated-dspark` | directory holding `Kimi-K3/` (and `Kimi-K3-DSpark/` for the speculative variant), mounted at `/models` | see below — **it is site-specific and there is no default** |
| `<NODE>` | `aggregated`, `aggregated-dspark` | the node both pods are pinned to | any node with 8 free `amd.com/gpu` **that has the weights at `<MODEL_DIR>`** |
| `<PREFILL_NODE>` `<DECODE_NODE>` | `disaggregated`, `disaggregated-dspark` | the two nodes | two nodes on a mutually routable RoCE fabric |
| `<PREFILL_MODEL_DIR>` `<DECODE_MODEL_DIR>` | `disaggregated`, `disaggregated-dspark` | per-node model directory | **these do not have to be equal** — see below |

```{admonition} The model path is site-specific, and a mixed fleet may not agree with itself
:class: warning
There is no default and no value this document can supply. On the fleet these
recipes were validated on the two GPU nodes use **different** paths for the same
weights, which is why the PD manifests take two separate directory placeholders
rather than one.

Getting it wrong does not produce a clear error. If the path is absent, the pod
fails with `hostPath type check failed`. If the path *exists but is empty* — which
is what happens when you copy a working node's value onto a node that never had the
weights — the mount succeeds and the engine crashloops several minutes later with

    ValueError: '/models/Kimi-K3': not a local path and HF resolution failed:
      Repo id must be in the form 'repo_name' or 'namespace/repo_name'

which reads as a bad model name and points nowhere near the real cause.

Find it **per node**, rather than assuming — and note that for PD you usually
cannot read it off an existing deployment, because PD needs every GPU so that
deployment has to be deleted first.

If you have a shell on the node:

```bash
find /mnt -maxdepth 5 -name 'Kimi-K3' -type d 2>/dev/null
```

If you do not — the usual case — run a throwaway pod pinned to it:

```bash
kubectl -n infera run pathprobe --rm -it --restart=Never --image=busybox \
  --overrides='{"spec":{"nodeSelector":{"kubernetes.io/hostname":"<NODE>"},
    "containers":[{"name":"p","image":"busybox","stdin":true,"tty":true,
    "command":["sh","-c","find /host/mnt -maxdepth 5 -name Kimi-K3 -type d"],
    "volumeMounts":[{"name":"h","mountPath":"/host","readOnly":true}]}],
    "volumes":[{"name":"h","hostPath":{"path":"/","type":"Directory"}}]}}'
```

**Candidates usually look identical** — same shard count, same layout — while only
one is local. Confirm the one you pick by mounting *that directory* and reading its
backing device, which is the only reliable way:

```bash
kubectl -n infera run fsprobe --rm -it --restart=Never --image=busybox \
  --overrides='{"spec":{"nodeSelector":{"kubernetes.io/hostname":"<NODE>"},
    "containers":[{"name":"p","image":"busybox","stdin":true,"tty":true,
    "command":["sh","-c","df -PT /m | tail -1; du -sm /m/Kimi-K3"],
    "volumeMounts":[{"name":"m","mountPath":"/m","readOnly":true}]}],
    "volumes":[{"name":"m","hostPath":{"path":"<MODEL_DIR>","type":"Directory"}}]}}'
```

Mount the **directory itself**, not a parent: `df` on a path under a mounted `/`
reports the root filesystem and will call a network-mounted weights directory
local. Expect a local device (`xfs`, `ext4`, `/dev/...`) and ~1.4 TB.

That last check is not a formality. The obvious shared-looking mount on this fleet
(`/mnt/shared`) is an NFS export of the *other* node's array. Using it puts one side
on a ~95-minute weight load, which exceeds the ready timeout, so the worker restarts
mid-load and never finishes — presenting as a crash loop rather than as slow storage.
```

## 1. Choose the combination

|  | GPUs | Manifest | Reach for it when |
|---|---:|---|---|
| **aggregated** | 8 | [`aggregated/`](aggregated/deploy.yaml) | the default. Best tokens per GPU at every concurrency measured |
| **aggregated + DSpark** | 8 | [`aggregated-dspark/`](aggregated-dspark/deploy.yaml) | low concurrency, where speculation pays for itself on the same hardware |
| **disaggregated** | 16 | [`disaggregated/`](disaggregated/deploy.yaml) | you need more headroom than one node gives, or lower latency at high concurrency. It never wins per GPU |
| **disaggregated + DSpark** | 16 | [`disaggregated-dspark/`](disaggregated-dspark/deploy.yaml) | lowest TPOT of the four. Both roles must carry `--speculative-config` |

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

## 2. Pre-flight

Before deploying, especially for the disaggregated combinations, run the
repository's preflight tool. It covers RDMA device and link state, cross-node RoCE
bandwidth, Mooncake KV-transfer bandwidth measured separately over RDMA and TCP,
and whether the KV path is on local NVMe:

```bash
python -m infera.tools.preflight --dump-path output/preflight   # one node
python -m infera.tools.preflight --network                      # network probes only

NODES=<node-a>,<node-b> PARTITION=<partition> IMAGE=<image> \
  infera/tools/preflight/run_preflight_slurm.sh                 # both nodes, one report
```

GPU perf and `ais-check` only run **inside the engine container**. Full check list
and thresholds: [`infera/tools/preflight/README.md`](../../../infera/tools/preflight/README.md).

The Mooncake rows are the ones that matter here — they report KV-move bandwidth
over `rdma` and over `tcp` separately, so a fabric that will silently serve at TCP
speed appears as a number instead of as a mysteriously slow deployment.

A second check runs automatically: `infera/common/disagg_preflight.py` validates
the disaggregated config before the engine subprocess starts and fails fast rather
than hanging. It catches a worker advertising a non-routable host (`0.0.0.0`,
`127.0.0.1`) to etcd, and configurations prone to silent TCP fallback. Being pure
config validation it cannot tell you the NIC is healthy — that is the tool above.

## 3. Prerequisites

```bash
# nodes must advertise amd.com/gpu. NOTE this is CAPACITY, not availability — a
# fully occupied node still prints 8. For what is actually free:
#   kubectl describe node <NODE> | grep -A5 'Allocated resources'
# and check the node is schedulable: `kubectl get node <NODE>` must not say
# SchedulingDisabled, and `kubectl describe node <NODE> | grep Taints` must not
# show a NoSchedule taint -- the manifests here tolerate none.
kubectl get nodes -o custom-columns=NODE:.metadata.name,GPU:.status.allocatable.'amd\.com/gpu'

# every manifest here hardcodes `namespace: infera`, and nothing else creates it
kubectl create namespace infera --dry-run=client -o yaml | kubectl apply -f -

# the operator (provides the InferaDeployment CRD). Skip if already installed —
# `helm install` fails on name reuse; use `helm upgrade --install` to be idempotent.
helm upgrade --install infera-operator oci://docker.io/rocm/infera-operator --version 0.1.0 \
  -n infera-system --create-namespace
```

```{admonition} On k3s, helm needs KUBECONFIG spelled out
:class: tip
`kubectl` is a symlink to `k3s` and finds `/etc/rancher/k3s/k3s.yaml` implicitly.
`helm` does not, and fails with `Kubernetes cluster unreachable: ... dial tcp
[::1]:8080: connect: connection refused`. Export it first:

```bash
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
```
```

Both models go in one directory, mounted at `/models`:

```bash
hf download moonshotai/Kimi-K3      --local-dir <MODEL_DIR>/Kimi-K3
hf download Inferact/Kimi-K3-DSpark --local-dir <MODEL_DIR>/Kimi-K3-DSpark   # DSpark only
```

**Finding `<MODEL_DIR>` on a cluster where the weights already exist.** Nothing
here can tell you the path — it is site-specific, and on a mixed fleet the nodes
may not agree. Read it off a deployment that already works, or look for it:

```bash
# from an existing InferaDeployment
kubectl -n infera get deploy <any-worker-deploy> \
  -o jsonpath='{.spec.template.spec.volumes[?(@.name=="model")].hostPath.path}{"\n"}'

# or search the node (needs a shell or a privileged pod on it)
find /mnt -maxdepth 4 -name 'Kimi-K3' 2>/dev/null
```

**It must be local NVMe.** Kimi-K3's 96 shards load in ~8 min from local
disk and ~95 min from NFS — and the slow path does not merely run late, it exceeds
the ready timeout, so the worker restarts mid-load and never finishes.

For the PD combinations, **each node needs its own copy** and the paths need not
match — hence the separate placeholders. On the fleet this was validated on, the
tempting common mount (`/mnt/shared`) turned out to be an NFS export of the *other*
node's array; using it on both sides puts one side on the 95-minute path. Check
each node with `df -hT <dir>` and take the local device, not the convenient common
name.

For `disaggregated-dspark`, **the draft must be on BOTH nodes.** Both roles carry
`--speculative-config` and both load it — see §6 for why the prefiller needs it
even though it never runs the draft. Provisioning it on only one node fails with a
missing path, which reads like a typo in the manifest rather than a missing 6.7 GB
directory.

First start is **10–14 min**: the image rebuilds its AITER JIT modules in-container
on top of weight load and CUDA-graph capture.

## 4. Deploy

The 8-GPU combinations take **two** placeholders. `<NODE>` pins the server and the
worker to one node: both mount the weights by `hostPath` and are scheduled
independently, so without it they can land on different nodes and the one that
guesses wrong fails.

```bash
sed -e 's|<MODEL_DIR>|/mnt/local-nvme/models|' -e 's|<NODE>|node-a|' \
  examples/recipes/kimi-k3-optimized/aggregated/deploy.yaml | kubectl apply -f -

kubectl -n infera get pods -w
```

| Placeholder | What to put there |
|---|---|
| `<MODEL_DIR>` | Directory on that node containing `Kimi-K3/`, mounted into the pod at `/models`. A `hostPath` — the node must hold the weights locally. |
| `<NODE>` | Hostname of the node (`kubernetes.io/hostname`). Pins **both** the server and the worker to it — see below for why. |

Replace `aggregated` with `aggregated-dspark` for the speculative variant; for that one,
`<MODEL_DIR>` must also contain `Kimi-K3-DSpark/`.

If `get pods` shows nothing a minute after `apply`, a placeholder survived — see
§0; that failure reports only in the operator's log, in another namespace.

The PD combinations take four placeholders, because the roles land on different
nodes and each reads its own local copy:

```bash
sed -e 's|<PREFILL_NODE>|nodeA|'      -e 's|<DECODE_NODE>|nodeB|' \
    -e 's|<PREFILL_MODEL_DIR>|/mnt/local-nvme/models|' \
    -e 's|<DECODE_MODEL_DIR>|/mnt/array/models|' \
    examples/recipes/kimi-k3-optimized/disaggregated/deploy.yaml | kubectl apply -f -
```

| Placeholder | What to put there |
|---|---|
| `<PREFILL_NODE>` | Hostname of the node that runs **prefill** (`kubernetes.io/hostname`). It processes the prompt and produces the KV cache; the router pod is placed here too. |
| `<DECODE_NODE>` | Hostname of the node that runs **decode** — it receives that KV over RDMA and generates the tokens. Must be a *different* node, and the two must be able to reach each other over the RoCE fabric. |
| `<PREFILL_MODEL_DIR>` | Directory **on the prefill node** containing `Kimi-K3/`, mounted into the pod at `/models`. |
| `<DECODE_MODEL_DIR>` | The same, **on the decode node**. Frequently a different path. |

Both are `hostPath` mounts, so each node reads its **own local copy** — there is no
shared volume, and the two paths do not have to agree.

Replace `disaggregated` with `disaggregated-dspark` for the speculative variant; for that one,
`Kimi-K3-DSpark/` must sit alongside `Kimi-K3/` in **both** directories, because
both roles load the draft.

The two model directories are **deliberately different in that example.** They may
be equal on a uniform fleet, but assuming so is the single most likely way to get
the empty-mount crashloop described in §0 — determine each node's own path with the
probe in §0.

### Tearing down

The combinations cannot coexist — `aggregated`/`aggregated-dspark` take 8 GPUs, the disaggregated pair
takes all 16 — so switching between them means deleting the first:

```bash
kubectl -n infera delete inferadeployment <name>    # e.g. kimi-k3-opt-dspark
```

Use the name from the table below, not the directory name. GPUs are released when
the pods finish terminating; `kubectl -n infera get pods` shows when they are
actually gone.

**The deployment name is not the directory name**, which matters for every
`kubectl` command below:

| directory | deployment / service prefix |
|---|---|
| `aggregated/` | `kimi-k3-opt-base` |
| `aggregated-dspark/` | `kimi-k3-opt-dspark` |
| `disaggregated/` | `kimi-k3-opt-pd` |
| `disaggregated-dspark/` | `kimi-k3-opt-pd-dspark` |

## 5. Smoke test

```bash
# Check the forward came up. If the port is already held, port-forward fails, the
# curl below then silently returns nothing and exits 0 — a false negative on the
# one health check you are running.
# Any free local port will do — 8000 is often already taken by an earlier
# port-forward. If this reports failure, pick another and change the curl below.
kubectl -n infera port-forward svc/<name>-server 18000:8000 & PF=$!
sleep 3; kill -0 $PF 2>/dev/null || { echo "port-forward failed — try another local port"; exit 1; }

# --max-time is not optional. This system's failure mode is a HANG, not an error:
# a request that cannot be served does not come back, and an un-timed curl waits
# indefinitely while everything looks healthy.
curl -s --max-time 300 localhost:18000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"kimi-k3","messages":[{"role":"user","content":"What is the capital of France?"}],
       "max_tokens":200}' | jq -r '.choices[0].message.content'
```

Expect ~120–155 completion tokens and `finish_reason: "stop"`.

```{admonition} Use max_tokens 1024, not 200
:class: tip
This model emits a reasoning preamble before the answer, and its length varies. At
`max_tokens: 200` roughly one request in four comes back `"finish_reason":
"length"` with 199 tokens of *"The user is asking a simple factual question…"* and
never reaches "Paris" — which reads exactly like a broken deployment, on a
deployment that is fine. Raise the cap; the answer still stops on its own at
~150.
```

Keep `max_tokens` generous: this model spends 70+ completion tokens on that
sentence, so a tight cap truncates it into something that reads like a broken
deployment.

**On the DSpark manifests**, confirm speculation actually engaged rather than
inferring it from throughput later:

```bash
# Scope to the deployment. Without infera.amd.com/deployment, `head -1` may pick a
# NON-speculative worker from another deployment in the namespace, and the check
# then reports "speculation did not engage" against a perfectly good one.
POD=$(kubectl -n infera get pod -o name \
  -l infera.amd.com/deployment=kimi-k3-opt-dspark,infera.amd.com/service=worker | head -1)
# for the PD variants: -l infera.amd.com/deployment=kimi-k3-opt-pd-dspark,infera.amd.com/service=decode

kubectl -n infera logs $POD -c main | grep -oE "speculative_config=SpeculativeConfig\([^)]*\)" | tail -1
kubectl -n infera logs $POD -c main | grep -c 'running the draft eagerly'   # must be 0
```

`running the draft eagerly` means the draft fell back to Triton instead of being
captured into CUDA graphs, which costs about 5% of decode throughput.

**On the PD manifests**, a correct answer proves nothing on its own: if the KV
handoff fails open, the decoder re-prefills locally and returns the same text —
faster. Check the decode side instead:

```bash
DEC=$(kubectl -n infera get pod -o name \
  -l infera.amd.com/deployment=kimi-k3-opt-pd,infera.amd.com/service=decode | head -1)
  # for pd-dspark: deployment=kimi-k3-opt-pd-dspark
kubectl -n infera logs $DEC -c main | grep 'Engine 000' | tail -2
```

Run this **within about a minute of the request.** vLLM logs these lines only for a
couple of intervals after traffic stops, so later on `tail -2` returns two idle
`0.0 / 0.0` lines and tells you nothing.

```
Avg prompt throughput: 0.1 tokens/s, Avg generation throughput: 7.8 tokens/s,
  ... External prefix cache hit rate: 100.0%
```

`External prefix cache hit rate` near 100% with `Avg prompt throughput` near zero
is the handoff working. The prefill pod is the mirror image — all prompt
throughput, no generation.

## 6. Measured behaviour

**Throughput figures have been withdrawn from this page.** They did not reproduce.

An independent run of the exact sweep this section used to specify got 165.99 tok/s
on a fresh deployment and 270.4 tok/s once warm, against a published 241.69 — the
published number matched neither. The cause is the sweep itself: at
`num-prompts = 4 × concurrency` it is 32 requests over ~15 s, short enough that
first-request TTFT (~2500 ms cold against ~400 ms warm) dominates the aggregate. No
warm-up was specified, so a reader following the method literally lands ~31% below
the published figure with no way to know that is expected.

That also undermines the comparisons drawn from those numbers. The sweeps were run
back-to-back within one deployment, low concurrency first, so the early points were
colder than the late ones — which is the same direction as the trends that were
being reported. Warm, `disaggregated` and `aggregated` at c=8 measured 269.7 and 270.4, effectively
identical, where this page had claimed 0.90×.

Rather than republish numbers whose method is known to be unsound, what follows is
what survived independent checking.

### Latency reproduces

Median TPOT was reproduced independently within a few percent, on both
combinations checked — it is a per-step measure and does not care about warm-up:

| | published | independent |
|---|---:|---:|
| `aggregated` c=8 | 27.85 ms | 26.51 ms |
| `disaggregated` c=8 | 25.81 ms | 25.70 ms |
| `disaggregated-dspark` c=64 | 17.44 ms | — |

`disaggregated-dspark` holding TPOT near 17 ms at c=64 where `aggregated` is near 49 ms is the
largest latency effect on this page, and the one worth deploying 16 GPUs for.

### What is established without depending on throughput

- **The concurrency crash is fixed.** On the `…-20260801` image every speculative
  combination died at `c>=16` with `AssertionError: AiterMLA flattened verify
  requires a uniform decode query len`. On the image pinned here, c=16/32/64 all
  complete with 0 restarts and the assertion never appears. Binary, not a
  measurement.
- **Speculation is genuinely engaged**, not disabled to make the crash go away:
  `num_spec_tokens=7`, CUDA-graph captured, `running the draft eagerly` count 0.
- **The PD handoff is real.** The decode side holds `External prefix cache hit
  rate` at 98.9–99.9% with prompt throughput near zero, and prefill shows the
  mirror image. Independently confirmed on `disaggregated` and `disaggregated-dspark`.
- **`disaggregated-dspark` requires `--speculative-config` on both roles.** Decode-only fails
  two ways, both hangs rather than errors — see §6.

Re-establishing throughput figures needs a stated warm-up protocol and a sweep long
enough that the first request does not dominate. Until then this page does not
publish any.

## 7. Settings that are not optional

Each row is a failure that was hit and diagnosed on this hardware, not a preference.

| Setting | Why |
|---|---|
| the `KIMI_K3_*` / `VLLM_ROCM_*` env block | these select the optimized kernels. Without them the MoE asks aiter for a kernel that was never generated: `ValueError: Invalid FlyDSL kernel name: flydsl_moe1_..._t16x64x256_...` — there is no Kimi-K3 `tuned_fmoe.csv`, while dsv3/dsv4/glm5 all have one |
| `VLLM_ROCM_USE_KIMI_K3_PREROUTE_BF16=0` | must be `0`. A `1` makes the pre-route dispatch take BF16 and shadows the FP8 cluster. **Measured on this image** against the same manifest with `0`, each as the first sweep after its own fresh deployment, so the two sides are comparable to each other even though the absolute figures are not reproducible (see §5): roughly half the throughput at c=8 and about three quarters at c=16 — the penalty is worst at low concurrency, not a flat factor. It is silent in every other respect: the worker starts normally, logs no warning, and median TPOT barely moves (28.52 vs 27.85 ms at c=8), so latency monitoring will not show it |
| `attention_backend: ROCM_AITER_MLA` | in the speculative config. The upstream quick-start says `FLASHINFER_MLA`, which is CUDA-only; **omitting the key entirely is not the fix** — this is its ROCm counterpart |
| `--gpu-memory-utilization 0.88` | the draft's weights land after the KV budget is computed. At `0.95` the run dies with 998 MB free trying to allocate 2.32 GiB |
| `INFERA_ENGINE_READY_TIMEOUT=7200` | infera's 1800 s default is generous for local NVMe and impossible for anything slower; the worker then kills itself mid-load and restarts forever, which reads as a crash loop rather than as slow storage |
| the **server** pod mounts `/models` too, and `--router-tokenizer-path` is a **local path** | kv-aware tokenizes each request on the router to compare its prefix against what the workers cached, so the router needs the same tokenizer files the engines load. Point it at a hub id and it resolves an HF cache directory that may hold only `tokenizer_config.json`: the load fails, every request hashes to zero blocks, and kv-aware quietly becomes least-loaded with `--kv-overlap-weight` having no effect at any value. Nothing else shows it — the server starts, `/health` is green, requests succeed. Seen on a fleet whose router had been moved to a CPU node with no model volume. Watch for `kv-aware DEGRADED` in the server log and `infera_cache_locality_skipped_total{reason="no_tokenizer"}` |
| PD: each node needs its **own local** copy | the `model` volume is a `hostPath`. The fleet's obvious shared mount was an NFS export of the peer's array — one side then loads for ~95 min and restarts forever |
| PD: never point a client at it that sends requests the engine will reject | the engine validates **after** prefill, so a rejected request has already had its KV computed and queued for transfer. 84 requests rejected for one bad field left 424 aborted Mooncake transfers — 53 requests × 8 TP ranks — and stalled *valid* traffic for ~20 minutes with `MooncakeXferMetadata transfer failed: Resource temporarily unavailable`, which reads exactly like a broken fabric. It was a broken client |

`ibv_devices` is **not installed** in this image. Reading its `not found` as "no
RDMA devices" produced two false TCP-fallback diagnoses here; ask the library
instead:

Run it in a **prefill or decode** pod. The server pod runs the same image but does
not mount `/dev/infiniband` and is not privileged, so it reports `0` — reproducing
the very false negative this check exists to prevent:

```bash
POD=$(kubectl -n infera get pod -o name \
  -l infera.amd.com/deployment=kimi-k3-opt-pd,infera.amd.com/service=decode | head -1)
kubectl -n infera exec $POD -c main -- python3 -c '
import ctypes; lib = ctypes.CDLL("libibverbs.so.1")
lib.ibv_get_device_list.restype = ctypes.POINTER(ctypes.c_void_p)
n = ctypes.c_int(0); lib.ibv_get_device_list(ctypes.byref(n)); print(n.value)'
```


### Why `disaggregated-dspark` configures speculation on *both* roles

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

## 8. Validation status

| What | Status |
|---|---|
| `aggregated/deploy.yaml`, placeholders substituted | **validated** — ready ~12 min, correct answer, c=4…64 swept |
| `aggregated-dspark/deploy.yaml`, placeholders substituted | **validated** — ready ~12 min, c=4…64 swept, 0 restarts, assertion count 0 |
| `disaggregated/deploy.yaml`, placeholders substituted | **validated cross-node** — ready ~12 min, c=4…64 swept, extcache 99.9% throughout |
| `disaggregated-dspark/deploy.yaml`, placeholders substituted | **validated cross-node** — both roles Ready in ~14 min, 0 restarts; sweep below |
| `disaggregated-dspark` with speculation on **decode only** | **hangs**, two separate ways — see above. Do not ship it |
| kvd combinations | not built for this image |
| fp8 KV cache | not measured here |

## Source

[`examples/recipes/kimi-k3-optimized/`](.) in [AMD-AGI/Infera](https://github.com/AMD-AGI/Infera)
· [all recipes](../README.md) · [stock Kimi-K3 recipe](../kimi-k3/README.md)
