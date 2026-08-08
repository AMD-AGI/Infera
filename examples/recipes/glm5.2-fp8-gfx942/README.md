# GLM-5.2-FP8 on gfx942 (Kubernetes)

Serve GLM-5.2-FP8 across two gfx942 nodes (MI300X / MI325X): SGLang prefill/decode
disaggregation over Mooncake RDMA, DP-attention, MTP speculative decoding, kv-aware
routing, and KV offload to host RAM + node-local NVMe through `infera-kvd`.

Every flag here is lifted from a `docker` + shell deployment of the same topology,
validated on 2 × MI300X and referred to throughout as **the docker recipe**:
[`examples/glm5.2_gfx942/`](../../glm5.2_gfx942/README.md), which also carries the
bring-up, verification and benchmark scripts. Nothing was retuned for Kubernetes:
§6 lists every difference and why the substrate forced it, and there are no others.

| Combo | Serving | KV cache | Manifest |
|---|---|---|---|
| **disaggregated + kvd** | prefill and decode on separate nodes | + kvd L2 pinned host RAM, L3 on node NVMe | [`disaggregated-kvd/deploy.yaml`](disaggregated-kvd/deploy.yaml) |

The docker recipe's `KVD=0` A/B baseline is the same deployment minus the offload
tier — §5 has the four-line edit rather than a second copy of the manifest.

## 1. This recipe does not use the overlay

Every other recipe here runs a **stock vendor image** with the overlay payload
mounted in. This one runs an **infera-built engine image** instead, because GLM-5.2
on the v0.5.16 gfx942 base needs a rebuilt native library and four source patches,
none of which a mounted payload can supply. The overlay payload carries
`deploy/docker/patches/vllm/` only, and `infera-exec`'s patch loop is additionally
gated on `import vllm` — so on an SGLang base it does not merely miss them, it does
not run at all.

**The Mooncake rebuild is the part no payload could ever carry.** The base bundles
Mooncake at upstream #2682, which installs a HIP IPC transport unconditionally and
prefers it over RDMA — so cross-node PD dies on the first request inside
`hipIpcOpenMemHandle`, which cannot open a peer node's handle.
`Dockerfile.sglang.gfx942` rebuilds `engine.so` in place with the transport gated;
the build step is self-verifying and fails if the gate did not compile in.

**Four SGLang source patches**, all under `deploy/docker/patches/`:

| Fix | Patch | What happens without it |
|---|---|---|
| DSA indexer row count | `sglang_dsa/patch_dsa_indexer_hip_dp_padded_rows.py` | top-k dies with `Expected lengths.size(0) == B` as soon as concurrency > 1 |
| mooncake early-send KV wait event | `sglang_disagg/patch_mooncake_early_send_wait_event.py` | every prefill chunk but the last is RDMA-read while the forward is still writing it; prompts longer than one chunk come back **partially wrong, with nothing in any log** |
| hicache staged write-back gate | `sglang_rocm/patch_hicache_rocm_staged_write_back.py` | the prefill scheduler dies (exit −3, `Tensor match failed … device=rocm:0`) on the first request that reuses a prefix |
| hicache host allocator | `sglang_rocm/patch_hicache_rocm_host_alloc.py` | preventive on gfx942 rather than a fix for a crash seen here: hicache hands host `data_ptr()`s to GPU kernels, which is only correct while `hipHostRegister` maps the pages at the host VA. It did in every measurement on this base; the patch moves to `hipHostMalloc`, where that identity is an API guarantee instead of an accident of the driver. On gfx950 the two differ and the first kvd write-back aborts with `Memory access fault by GPU node-N` |

Build the image:

```bash
docker build -f deploy/docker/Dockerfile.sglang.gfx942 -t infera:sglang-gfx942-glm52 .
```

Load it on **both** nodes (or push it to a registry the cluster can pull; the
manifest uses `imagePullPolicy: IfNotPresent`, so a locally-loaded image is used
as-is). Three of the four patches leave greppable markers, which is the cheap way to
tell a correctly built image from one where a patch silently no-op'd against a moved
anchor:

```bash
docker run --rm --entrypoint bash infera:sglang-gfx942-glm52 -c '
P=$(python3 -c "import sglang, os; print(os.path.dirname(sglang.__file__))")
for m in GLM52_ROCM_HOST_ALLOC GLM52_ROCM_STAGED_WRITE_BACK GLM52_P1V3; do
  grep -rql "$m" "$P" && echo "ok      $m" || echo "MISSING $m"
done'
```

The mooncake wait-event patch leaves no named marker — its replacement text is its
own idempotency check — so it, and the Mooncake rebuild, are the two to confirm from
the build log.

## 2. Prerequisites

```{admonition} kv-aware needs the tokenizer on the ROUTER, not just the workers
:class: warning
The server pod mounts `/models` and passes `--router-tokenizer-path` a **local
path** — both manifests already do this, and both matter. kv-aware tokenizes each
request on the router to compare its prefix against what the workers cached, so
the router needs the same tokenizer files the engines load.

Point it at a hub id instead and it resolves an HF cache directory that may hold
only `tokenizer_config.json`. The load fails, every request hashes to zero
blocks, and kv-aware quietly becomes least-loaded — `--kv-overlap-weight` then
has no effect at any value. Nothing else shows it: the server starts, `/health`
is green, requests succeed. Watch for `kv-aware DEGRADED` in the server log and
`infera_cache_locality_skipped_total{reason="no_tokenizer"}`.
```

**Hardware.** Two nodes, 8× gfx942 each, on a mutually routable RoCE fabric — the
KV handoff is RDMA with no TCP fallback. The prefill node carries three Pods and so
needs ~**670 GiB** of free host RAM: 512 for the engine (256 GB of hicache host tier
plus load buffers), 136 for kvd (§6), 16 for the router. Those are requests; the
node's `/dev/shm` is on top of them and is charged to no limit. The prefill node
also needs a **node-local NVMe** directory for L3.

Check the fabric from inside the image on **both** nodes before deploying, because
zero visible ports is how RDMA fails here — `ibv_get_device_list()` returns nothing,
Mooncake falls back to TCP, and the deployment still comes up:

```bash
docker run --rm --network host --device=/dev/infiniband --cap-add=IPC_LOCK \
  --entrypoint bash infera:sglang-gfx942-glm52 -c 'ibv_devinfo | grep -c PORT_ACTIVE'
```

[`sglang_1p1d_glm5.2/preflight_rdma.sh`](../../sglang_1p1d_glm5.2/preflight_rdma.sh)
wraps this plus a cross-node bandwidth and mooncake KV probe. It takes the image as
`IMAGE=` and tests the hosts, so it applies here unchanged even though it ships with
the MI355X example.

That preflight report is also where the manifest's RDMA placeholders come from:

- `<RDMA_IB_DEVICES>` — the rail(s) Mooncake may use, comma-separated, from
  `ibv_devices`. A rail that is physically down must **not** be listed.
- `<PREFILL_GID_INDEX>` / `<DECODE_GID_INDEX>` — from `show_gids <dev>`, the index
  whose type is `RoCE v2`. There are two because the index is **per node, not per
  cluster**; two identical machines routinely expose different ones. They are
  usually equal, but check both — the wrong index pins KV to an interface that
  never carries it and the transfer simply times out.

The docker recipe's cluster answered `mlx5_0` and `3`, which is what its validation
ran on. Those are its numbers, not defaults: `leg.sh` takes both as required
variables with no fallback, so a manifest that hard-coded them would be less
faithful to the recipe, not more.

**Cluster.** Kubernetes **1.29+** — the kvd daemon is a native sidecar
(`initContainers` with `restartPolicy: Always`), which is what makes it reach a
healthy startupProbe *before* the engine starts. That ordering is load-bearing: the
engine probes the kvd socket once with a 5 s timeout and refuses to start if
nothing answers, without retrying. On an older cluster the sidecar has to be an
ordinary container and the ordering becomes a race the engine loses about as often
as it wins.

```bash
# AMD GPU device plugin — nodes must advertise amd.com/gpu
kubectl get nodes -o custom-columns=NODE:.metadata.name,GPU:.status.allocatable.'amd\.com/gpu'

# the infera operator (provides the InferaDeployment CRD)
helm install infera-operator deploy/operator/helm/infera-operator -n infera-system --create-namespace
kubectl -n infera-system rollout status deploy/infera-operator
```

**`memlock`.** The docker recipe passed `--ulimit memlock=-1` and Kubernetes has no per-Pod
equivalent, but the capability is what actually carries this: `CAP_IPC_LOCK` — granted
to the kvd sidecar explicitly and to the engines through `privileged` — makes the
kernel skip `RLIMIT_MEMLOCK` accounting for both `mlock(2)` and RDMA registration. If
you do hit the limit the failure is soft and quiet: kvd logs `mlock failed (errno=…)`
at INFO and runs its arena **unpinned** (still correct, slower DMA staging). Raising
the limit itself needs `default_ulimits` in the container runtime's config; there is
no Pod-spec field for it.

**Weights.** `hostPath`, not a PVC, at the **same path on both nodes**. If the path
is a HuggingFace cache symlink, mount the directory the links resolve into as well
— otherwise the inner relative links dangle and `transformers` rejects the model
with `Should have a model_type key in its config.json`, four minutes into startup
and far from its cause.

## 3. Deploy

```bash
kubectl create namespace infera --dry-run=client -o yaml | kubectl apply -f -

sed -e "s|<PREFILL_NODE>|node-a|"        -e "s|<DECODE_NODE>|node-b|" \
    -e "s|<MODEL_DIR>|/mnt/models|"      -e "s|<KVD_L3_DIR>|/mnt/nvme/kvd-l3|" \
    -e "s|<RDMA_IB_DEVICES>|mlx5_0|" \
    -e "s|<PREFILL_GID_INDEX>|3|"        -e "s|<DECODE_GID_INDEX>|3|" \
    examples/recipes/glm5.2-fp8-gfx942/disaggregated-kvd/deploy.yaml | kubectl apply -f -
```

The RDMA values above are the docker recipe's; substitute what §2 reported on your
own fabric. Any placeholder left unsubstituted fails loudly, which is the point —
`kubectl` rejects `<PREFILL_NODE>` outright, and a literal `<RDMA_IB_DEVICES>` is
not a device `ibv_open_device` will accept.

Cold start is 15–25 min: both legs load GLM-5.2 plus the MTP nextn layer, and the
log goes quiet while they do. That is why the workers use a `startupProbe` with a
90-minute budget (matching `INFERA_ENGINE_READY_TIMEOUT`) and no readiness probe.
Don't kill a slow load.

```bash
kubectl -n infera get pods -w
kubectl -n infera logs -f -c main \
  -l infera.amd.com/deployment=glm52-fp8-pd-kvd,infera.amd.com/service=prefill
```

## 4. Smoke test

```bash
kubectl -n infera port-forward svc/glm52-fp8-pd-kvd-server 8000:8000 &

curl -s localhost:8000/v1/workers | jq          # expect one prefill + one decode
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"/models/GLM-5.2-FP8",
       "messages":[{"role":"user","content":"What is 127 * 31? Answer with the number only."}],
       "max_tokens":128,"temperature":0,
       "chat_template_kwargs":{"enable_thinking":false}}' | jq -r '.choices[0].message.content'
```

`3937`, plus `installTransport, type=rdma` in the decode leg's log, means the router
paired the legs and KV moves over RDMA rather than silently falling back to TCP:

```bash
kubectl -n infera logs -c main \
  -l infera.amd.com/deployment=glm52-fp8-pd-kvd,infera.amd.com/service=decode \
  | grep -aE 'GID index|installTransport'
```

The manifest passes no `--served-model-name`, matching the docker recipe, so the
served name **is** the model path — `/models/GLM-5.2-FP8` in the request above.

Benchmarking needs nothing further: the router is an ordinary OpenAI-compatible
endpoint and does not care how the engines were started, so point a load generator
at this same forwarded port. For a KV-reuse benchmark specifically, read §5 first —
prompts shorter than a hicache page never reach kvd at all.

## 5. kvd

```bash
POD=$(kubectl -n infera get pod -o name \
  -l infera.amd.com/deployment=glm52-fp8-pd-kvd,infera.amd.com/service=prefill | head -1)
kubectl -n infera exec $POD -c kvd -- \
  python3 -m infera.kvd.statctl --socket /tmp/infera-kvd/kvd.sock
```

`sets_total` climbing means the engine writes to kvd; `gets_total` / `hits_total`
climbing means it reads back. Writes alone prove only half the path.

Send a prompt long enough to fill several hicache pages before reading these — one
shorter than a page produces no kvd traffic at all and proves nothing.

Two counters that will mislead you:

- **`misses_total` counts failed gets only.** SGLang never gets what `batch_exists`
  did not confirm, and a prefetch abandoned before its query reaches the backend
  leaves *every* counter untouched. `0 misses` is compatible with L3 having served
  nothing. Read the scorer's `cached tokens by tier` instead.
- **`entries: 0` with a healthy-looking deployment** means kvd rejected the KV
  layout. A value larger than the biggest tablespace pool is rejected, not split,
  and `--tablespace-pools 1M,4M` is sized for GLM-5.2-FP8's 2.74 MiB KV pages and
  624 KiB indexer pages at `page_size 64`. Grep the kvd container for
  `value_exceeds_largest_pool` (a single-pool tablespace says
  `value_exceeds_slot_bytes` instead).

**The `KVD=0` A/B baseline** is this manifest minus the offload tier: delete the
`kvd` entry under `prefill.extraPodSpec.initContainers`, drop
`--infera-kvd-socket /tmp/infera-kvd/kvd.sock --hicache-size 32` from the prefill
command, drop the `kvd-sock` / `kvd-l3` volumes, and drop the `kvd-sock`
`volumeMount` from the prefill `main` container — miss that last one and the Pod
is rejected for referencing a volume that no longer exists. Worth running only once the
deployment is above its pressure point — below it the 54 GB device pool per rank
answers everything and both arms are identical.

**That is what the docker recipe's agentic trace measured, and the result was
negative:** on a 32-conversation / 225-turn trace at `CONC=16`, `KVD=1` ran
**12.0% slower** than `KVD=0` and served **`gets_total = 0`** — 100.8 GB written
to L3 and not one page read back. Nothing was wrong with the offload path; the
trace simply had no misses left for it, its scorer efficiency against the
achievable ideal already sitting at ~100% on the GPU pool alone. Run this
manifest for a workload whose working set outgrows 54 GB/rank, and run the
`KVD=0` variant above to confirm yours does before paying for the tier.

## 6. What changed from the docker recipe, and why

Every engine and kvd flag is identical. The one router flag that differs is
`--router-backend`: the docker recipe defaults to `rust`, which this manifest
cannot use, because the Rust binary supports only `--discovery-backend etcd`
while the operator's backend is `kubernetes` (first row below). The two make
identical routing decisions request for request; `python` reaches them ~27%
slower end to end, which is the price of the substrate here.

The rest are substrate translations:

| `docker` form | Kubernetes form | Why |
|---|---|---|
| etcd container + `--etcd-endpoint` | `discoveryBackend: kubernetes` | the operator's own backend: workers self-register on their Pod annotation, so there is no etcd to run. Both publish the same `WorkerInfo` |
| `--advertise-host $PREFILL_IP` | `--advertise-host $(POD_IP)` | downward API. With `hostNetwork` this is the node IP, which is what the peer dials for the Mooncake bootstrap handshake. **Override it if your RoCE rail is on a different address than the node's primary IP** |
| `--network host` | `hostNetwork: true` | the RoCE rails are host interfaces; the pod network cannot reach them |
| `--ipc host --shm-size 128g` | `hostIPC: true` | `--ipc host` already makes `/dev/shm` the host's and docker ignores `--shm-size` alongside it, so this is the whole translation. The node's `/dev/shm` must be large enough for TP8 |
| `--device=/dev/infiniband` | `privileged: true` + `hostPath /dev/infiniband` | no unprivileged equivalent without an RDMA device plugin. Same pattern as the validated [`pd-1p1d-mooncake.yaml`](../../k8s-deployments/pd-1p1d-mooncake.yaml) |
| `--device=/dev/kfd --device=/dev/dri`, `HIP_VISIBLE_DEVICES=0..7` | `amd.com/gpu: 8` | the device plugin owns these. It exposes the allocated GPUs renumbered from 0, so the visible-device variables are dropped rather than translated — pinning them would silently mask GPUs on any node where the allocation is not 0–7 |
| `bash launch_kvd.sh` before `launch_prefill.sh` | native sidecar + `startupProbe` | the shell ordering becomes a scheduling guarantee instead of a convention |
| `-v $KVD_L3_DIR:$KVD_L3_DIR` | `hostPath` at `/kvd-l3` | still node-local NVMe. A PVC would work only with a node-local StorageClass, and anything shared classifies as buffered — 3.70 GB/s against 14.56 GB/s with `O_DIRECT`, measured on the LVM-over-7-NVMe xfs the docker recipe ran on |
| `--ulimit memlock=-1` | *nothing* | no Pod-spec equivalent, and mostly moot: `CAP_IPC_LOCK` already exempts both containers from `RLIMIT_MEMLOCK`. See §2 |
| `RDMA_IB_DEVICES` / `MC_GID_INDEX`, both `require_env` in `leg.sh` | `<RDMA_IB_DEVICES>`, `<PREFILL_GID_INDEX>`, `<DECODE_GID_INDEX>` | same contract — the recipe never had defaults for these — expressed as placeholders instead of required environment. Split in two because the GID index is per node, and `leg.sh` was invoked once per node while one manifest covers both |
| image ENTRYPOINT bypassed by `docker exec` | image ENTRYPOINT bypassed by `command:` | same net effect. The ENTRYPOINT only matches a host `libionic` ABI, and the image bakes the ABI-4 build (`INSTALL_LIBIONIC=1`), so skipping it costs nothing on Mellanox and nothing on ionic either unless the image was built with `INSTALL_LIBIONIC=0` |
| `KVD_IO_MODE=auto` default | `--io-mode direct` | what the docker recipe ran, and at the time it had to: the classifier resolved a mount by device *name*, and an LVM mount inside a container has no name it can open, so it took the conservative branch and picked buffered even on NVMe. That is fixed — the classifier now falls back to a `major:minor` walk through sysfs, which needs no device node and is not namespaced, so `auto` would reach `direct` on its own here. Kept pinned regardless: it is the validated value, and a misclassification is a silent 4x |

Resource limits are new; the docker recipe ran without any, which makes the kvd
sidecar's the one worth checking before you copy it. kvd holds **two** independent
budgets: `--max-bytes 64G` caps the inline store, and the shared arena is sized
separately — it defaults to `--max-bytes`, and it is `mmap`'d and `mlock`'d whole at
startup, so it is never reclaimable under pressure. 64 + 64 + headroom is the
136 GiB in the manifest; size it to one of them and the OOM killer takes the sidecar
mid-run, which reads as a kvd bug rather than a limit. `prefill` gets 512 GiB
because `--hicache-size 32` is **GB per DP rank**, so 8 ranks pin 256 GB of host
tier; `decode` runs no host tier and gets half. `cpu: 32` matches `num_threads` in
`--model-loader-extra-config`.

## Validation status

| What | Status |
|---|---|
| This configuration in its **`docker` form** | brought up and benchmarked on 2 × MI300X. The figures quoted in §5 and in the docker recipe come from that sweep, one axis at a time against a locked baseline; the kvd counters in §5 are from the single `KVD=1` run of it |
| **This manifest** | **not run.** Derived from that deployment flag for flag, and every deviation is in §6, but the Kubernetes form has not been brought up |
| Native kvd sidecar ordering | not run. The mechanism is standard k8s 1.29+; the claim that it removes the startup race is reasoned, not measured |

Two things worth re-reading before a first bring-up, because both fail quietly
rather than loudly: `--advertise-host` resolving to a node IP that is not on the
RoCE rail (§6), and an engine image missing any of the three source fixes (§1) —
which is why §1 ends with a command that reads the markers out of the built image
rather than trusting the build log.

## Source

[`examples/recipes/glm5.2-fp8-gfx942/`](.) in
[AMD-AGI/Infera](https://github.com/AMD-AGI/Infera) · [all recipes](../README.md) ·
[the same shape on MI355X, in `docker` form](../../sglang_1p1d_glm5.2/README.md)
