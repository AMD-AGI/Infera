# GLM-5.2-FP8 on gfx942 (Kubernetes)

Serve GLM-5.2-FP8 on gfx942 (MI300X / MI325X) with SGLang: TP8/DP8 with
DP-attention, MTP speculative decoding, fp8 KV, kv-aware routing, and optionally
prefill/decode disaggregation over Mooncake RDMA and KV offload to host RAM plus
node-local NVMe through `infera-kvd`.

The engine, router and kvd flags are lifted from a `docker` + shell deployment of
the disaggregated topology, validated on 2 × MI300X and referred to throughout as
**the docker recipe**: [`examples/glm5.2_gfx942/`](../../glm5.2_gfx942/README.md),
which also carries the bring-up, verification and benchmark scripts. Nothing was
retuned for Kubernetes: §7 lists every difference and why the substrate forced it,
and there are no others.

| Combo | Nodes | KV cache | MTP | Manifest |
|---|---|---|---|---|
| `aggregated` | 1 | GPU only | yes | [`aggregated/deploy.yaml`](aggregated/deploy.yaml) |
| `aggregated + kvd` | 1 | + kvd L2 pinned host RAM, L3 on node NVMe | **no** | [`aggregated-kvd/deploy.yaml`](aggregated-kvd/deploy.yaml) |
| `disaggregated` | 2 | GPU only | yes | [`disaggregated/deploy.yaml`](disaggregated/deploy.yaml) |
| `disaggregated + kvd` | 2 | + kvd on the prefill leg | yes | [`disaggregated-kvd/deploy.yaml`](disaggregated-kvd/deploy.yaml) |

All four are validated (§8) and all four run TP8 / DP8 with DP-attention. The grid
is otherwise clean on two axes — one node or two, GPU-only KV or tiered.

**That third column is the one surprise, and it is not a tuning preference: on a
worker that both prefills and decodes, MTP and hicache deadlock each other.** Six
configurations were needed to pin MTP as the trigger rather than a bystander, and
the last of them — MTP off, everything else unchanged — is what ships.
`disaggregated + kvd` keeps MTP because speculative decoding only happens on its
decode leg, so the combination never forms on the leg that carries kvd. §6 has
the hypothesised mechanism, both crash signatures, and what dropping MTP costs.

**One consequence for measurement:** `aggregated` is *not* the `KVD=0` sibling of
`aggregated + kvd`, because it also has MTP. Only the disaggregated pair differs
by kvd alone.

**Start with `aggregated`.** It needs one box, touches neither Mooncake nor
`/dev/infiniband` nor a GID index, and therefore cannot hit the entire RDMA class
of silent failures. That also makes it the right thing to bring up when something
is wrong and you do not yet know which layer it is in — it separates "can this
image load these weights and serve" from "can KV cross this fabric", and those two
have very different failure modes.

**Add `kvd` only once you have a reason.** hicache is `write_through`, so every
byte the tier absorbs is paid for on the prefill path whether or not anything ever
reads it back. It earns that when requests share long prefixes and the reuse
outgrows the ~54 GB/rank device pool; below that point the device pool answers
everything and the two arms serve identically. §6 has the counters that tell you
which case you are in, and measurements for both.

```{admonition} The directory says `aggregated`, the manifest says `mixed`
:class: note
`aggregated` / `disaggregated` is the vocabulary these directories use, and
`role: mixed` inside the manifest is the operator's API value — deliberately
unchanged, since renaming it would break deployed configurations. Same thing.
```

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

Build the image — but read the host's driver version first, because the base you
want depends on it:

```bash
dpkg -l | grep -E 'amdgpu-dkms|rocm-core'    # on BOTH nodes
```

A container brings its own ROCm userspace and cannot bring a kernel driver; it
talks to the host's amdgpu through `/dev/kfd`, and AMD supports that pairing only
within a bounded window. A **6.3.x** host driver takes ROCm userspace up to
**7.0.x**, a **6.4.x** host up to **7.2.x**. The Dockerfile's default base is
`rocm720`, for 6.4.x hosts:

```bash
# 6.4.x host driver (e.g. amdgpu 6.14.14) — the default
docker build -f deploy/docker/Dockerfile.sglang.gfx942 -t infera:sglang-gfx942-glm52 .

# 6.3.x host driver — you MUST override the base
docker build -f deploy/docker/Dockerfile.sglang.gfx942 \
  --build-arg SGLANG_BASE_IMAGE=lmsysorg/sglang:v0.5.16-rocm700-mi30x \
  -t infera:sglang-gfx942-glm52 .
```

**Get this wrong and nothing refuses to start.** The image initialises, loads
weights, captures graphs, and then faults with `Memory access fault by GPU
node-N` somewhere under load — on gfx942 the faults landed in the DSA indexer,
the EAGLE draft path and an aiter fp8 MoE kernel, three unrelated places each
with a plausible-looking stack. Two of them got a patch written for them before
anyone compared version numbers. `Dockerfile.sglang.gfx942`'s `PRECONDITION`
header carries the same table; two `dpkg -l` lines are cheaper than a rebuild.

**Build on one node and copy it, rather than running the build on each.** The base is
an `ARG` with a default, so the same command run on two nodes can produce two
different ROCm userspaces under the one tag `infera:sglang-gfx942-glm52`, and nothing
downstream compares them — the deployment comes up and only the leg on the mismatched
node faults, under load, in the shape described above. If both nodes have already
built, compare what is inside rather than the tag:

```bash
for N in <PREFILL_NODE> <DECODE_NODE>; do
  echo "-- $N"
  ssh "$N" 'docker run --rm --entrypoint cat infera:sglang-gfx942-glm52 \
    /opt/rocm/.info/version'
done
```

Three of the four patches leave greppable markers, which is the cheap way to tell a
correctly built image from one where a patch silently no-op'd against a moved
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

### Getting a locally built image where kubelet will look for it

`docker build` puts the image in the **docker daemon's** store. Kubelet never reads
that store — it asks **containerd**, and only inside containerd's `k8s.io`
namespace. So `docker images` listing the image on the node proves nothing about
whether a Pod can start, and what you get instead is `ErrImagePull` for a tag
sitting on the very machine that failed to pull it. A bare tag makes that message
more confusing, not less: with no registry prefix, `infera:sglang-gfx942-glm52`
resolves to `docker.io/library/infera`, which does not exist — so the error names
neither the image store nor the node, and reads as a credentials problem instead:

```text
failed to resolve reference "docker.io/library/infera:sglang-gfx942-glm52":
pull access denied, repository does not exist or may require authorization:
insufficient_scope: authorization failed
```

An `imagePullSecret` cannot fix that; nothing was ever supposed to be pulled.

`imagePullPolicy: IfNotPresent` — set on every container in all four manifests — is
what makes a local image usable at all: an image already in the `k8s.io` namespace
is used as-is and never pulled. Import it on **every node the manifest names**,
which is both of them for the disaggregated combos:

```bash
docker save infera:sglang-gfx942-glm52 -o /tmp/glm52.tar
sudo ctr -n k8s.io images import /tmp/glm52.tar     # k3s: sudo k3s ctr images import
sudo ctr -n k8s.io images ls -q | grep glm52
```

**`-n k8s.io` is the whole point.** An import into containerd's `default` namespace
succeeds, `ctr images ls` shows the image, and kubelet still cannot see it — which
looks exactly like a broken manifest rather than a misplaced image.

**No root on the nodes but cluster-admin on the cluster?** A privileged Pod is then
the one lever you have. Pin it to the node with `nodeName`, mount the host's
`docker` and `ctr` binaries as `hostPath` files plus both sockets, and let its
command pipe one into the other so the ~100 GB never lands on disk as a tarball:

```bash
/host/docker save "$IMAGE" \
  | /host/ctr -a /run/k3s/containerd/containerd.sock -n k8s.io images import -
```

Both binaries have to be mounted — with only `ctr` there is no `docker save` to run
— and the image must already be in *that* node's docker, since this copies within a
node rather than transferring between them.
[`deploy/scripts/import-image-to-containerd.sh`](../../../deploy/scripts/import-image-to-containerd.sh)
does this for a list of nodes at once, and waits for each to report `IMPORT_OK`
rather than leaving you to watch the Pods:

```bash
deploy/scripts/import-image-to-containerd.sh infera:sglang-gfx942-glm52 \
  <PREFILL_NODE> <DECODE_NODE>
```

**Taking a node list makes it look like a transfer, and it is not.** For every node
named, the image must already be in *that* node's docker, so on the disaggregated
combos there is a step before this one with no tooling here:

```bash
docker save infera:sglang-gfx942-glm52 | ssh <DECODE_NODE> 'docker load'
```

That step is the easiest of the two to skip, because the node you built on works and
nothing looks wrong until the second leg schedules. **One leg `Running` and the other
in `ImagePullBackOff` is its signature.**

**Confirm it by running the image, on every node, before deploying.** A Pod pinned
with `nodeName`, using the manifest's own image reference and `imagePullPolicy`, is
the only check that answers the question asked:

```bash
for N in <PREFILL_NODE> <DECODE_NODE>; do
  kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata: {name: img-probe-$N, namespace: infera}
spec:
  nodeName: $N
  restartPolicy: Never
  containers:
  - name: c
    image: infera:sglang-gfx942-glm52
    imagePullPolicy: IfNotPresent
    command: ["/bin/echo", "usable"]
EOF
done

for N in <PREFILL_NODE> <DECODE_NODE>; do
  kubectl -n infera wait --for=jsonpath='{.status.phase}'=Succeeded \
    pod/img-probe-$N --timeout=90s
  kubectl -n infera delete pod img-probe-$N
done
```

`kubectl get node -o jsonpath='{.status.images[*]}'` looks like the cheaper version
of that check and is not a substitute: a node has been seen listing
`docker.io/library/infera:sglang-gfx942-glm52` in its own status while the probe above
went to `ImagePullBackOff` on it. The kubelet reports names it cannot necessarily
unpack.

**One failure mode that reads as "the import silently failed".** Kubelet's image GC
reclaims *unreferenced* images once the node's disk is above its high threshold
(85% by default), honouring only a two-minute minimum age. A ~100 GB image imported
onto a nearly full node can therefore disappear within minutes of reporting success,
before any Pod refers to it. Render the manifest **first** and `kubectl apply` it the
moment the import succeeds, so something references the image inside that window.
Freeing disk is the real fix and may not be available: on the node where this was
hit, clearing hundreds of GB of build cache did not bring it under the threshold,
because the weights and the image are themselves most of the disk.

A registry the cluster can pull from avoids every one of these: retag, push, and
point `image:` at it. One caveat if you stand one up per node for convenience — a
`registry:2` published on `localhost:5010` **on each node** is two registries that
share a name, not one shared registry. Their catalogues drift, and `localhost:5010/…`
in a manifest then means a different image depending on where the Pod landed.

## 2. Prerequisites

**Hardware**, per combo. Every engine Pod wants 8× gfx942 and `cpu: 32`; what
differs is how many nodes, how much host RAM, and whether you need NVMe and a RoCE
fabric at all.

| Combo | Nodes | Host RAM on the busiest node | Node-local NVMe | RoCE fabric |
|---|---|---|---|---|
| `aggregated` | 1 | ~272 GiB (256 engine + 16 router) | no | no |
| `aggregated + kvd` | 1 | ~**670 GiB** (512 engine + 136 kvd + 16 router) | yes, for L3 | no |
| `disaggregated` | 2 | ~272 GiB (prefill node, incl. router) | no | **yes** |
| `disaggregated + kvd` | 2 | ~**670 GiB** (prefill node) | yes, on the prefill node | **yes** |

Those are requests, and `/dev/shm` sits on top of them charged to no limit — TP8
needs a large one, which is what `hostIPC` supplies. The kvd figure looks oversized
next to `--max-bytes 64G` and is not: kvd holds two independent budgets, and §7
explains why sizing the limit to one of them gets the sidecar OOM-killed mid-run.

**RoCE fabric — the disaggregated combos only.** Two nodes, mutually routable; the
KV handoff is RDMA with no TCP fallback. Check the fabric from inside the image on
**both** nodes before deploying, because zero visible ports is how RDMA fails here
— `ibv_get_device_list()` returns nothing, Mooncake falls back to TCP, and the
deployment still comes up:

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

**Cluster.** Kubernetes **1.29+** for the two kvd combos — the kvd daemon is a
native sidecar (`initContainers` with `restartPolicy: Always`), which is what makes
it reach a healthy startupProbe *before* the engine starts. That ordering is
load-bearing: the engine probes the kvd socket once with a 5 s timeout and refuses
to start if nothing answers, without retrying. On an older cluster the sidecar has
to be an ordinary container and the ordering becomes a race the engine loses about
as often as it wins. The two non-kvd combos have no such requirement.

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

**Weights.** `hostPath`, not a PVC, at the **same path on every node the deployment
uses**. The manifests mount `<MODEL_DIR>` at `/models` and read `/models/GLM-5.2-FP8`,
so download into that layout:

```bash
hf download zai-org/GLM-5.2-FP8 --local-dir <MODEL_DIR>/GLM-5.2-FP8
```

`--local-dir` is load-bearing. A bare `hf download` leaves a HuggingFace cache, in which
every file in the snapshot is a *relative* symlink into a sibling `blobs/`. The docker
recipe survives that because `host_container.sh` mounts the host path at the same path
inside the container, so the links resolve as they do outside; a `hostPath` volume
instead renames the directory to `/models`, and `config.json`'s `../../blobs/<sha>` then
points at `/blobs/<sha>`, which does not exist. Nothing reports a missing file:
`transformers` rejects the model with `Should have a model_type key in its config.json`,
four minutes into startup and far from its cause.

To serve a cache you already have without paying for a second copy, mount the
*repository* directory — it holds `blobs/` and `snapshots/` both, so the relative links
stay inside one mount — and repoint the engine at the snapshot:

```bash
NODE=          # kubectl get nodes
HF_HOME=       # the cache root, holding hub/models--zai-org--GLM-5.2-FP8
SHA=           # ls $HF_HOME/hub/models--zai-org--GLM-5.2-FP8/snapshots

: "${NODE:?}" "${HF_HOME:?}" "${SHA:?}"
sed -e "s|<NODE>|$NODE|" \
    -e "s|<MODEL_DIR>|$HF_HOME/hub/models--zai-org--GLM-5.2-FP8|" \
    -e "s|/models/GLM-5.2-FP8|/models/snapshots/$SHA|g" \
    examples/recipes/glm5.2-fp8-gfx942/aggregated/deploy.yaml | kubectl apply -f -
```

The `g` matters: that string is both `--model-path` and `--router-tokenizer-path`, and
the two have to agree. **It is also the served model name**, since the manifests pass no
`--served-model-name` — so on this route every `"model"` and `--model` below, in §4 and
§5, becomes `/models/snapshots/<sha>` too. Sending the old string gets a model-not-found
rather than anything that looks like a mount problem.

Either way, check the mount before paying for a cold start rather than four minutes into
one. This reads `config.json` through the same volume the engine will, which is the only
arrangement that reproduces the failure:

```bash
cat <<'EOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata: {name: hf-check, namespace: infera}
spec:
  nodeName: <NODE>
  restartPolicy: Never
  containers:
  - name: c
    image: busybox:1.36
    command: ["cat", "/models/GLM-5.2-FP8/config.json"]
    volumeMounts:
    - {name: model, mountPath: /models, readOnly: true}
  volumes:
  - {name: model, hostPath: {path: <MODEL_DIR>, type: Directory}}
EOF
kubectl -n infera wait --for=jsonpath='{.status.phase}'=Succeeded pod/hf-check --timeout=90s
kubectl -n infera logs hf-check | grep model_type    # a line printed is the pass
kubectl -n infera delete pod hf-check
```

Three details in that Pod are deliberate. It goes in `infera` because a `hostPath` volume
is denied under Pod Security `baseline` and `restricted`, and that namespace already runs
engine Pods with one. The tag on `busybox` is what makes the image reusable: `busybox`
alone means `:latest`, for which kubelet defaults to `imagePullPolicy: Always` and tries
to pull even when the node has it. And the `wait` is not politeness — without it `logs`
runs before the container does and reports that instead of the answer.

Run it for every node the combo uses, with the same `<MODEL_DIR>` you will deploy with —
and `cat` the path the engine will read, so `/models/snapshots/<sha>` if you took the
repository-mount route above. Any image with `cat` will do; substitute the engine image
where the nodes cannot reach Docker Hub. `kubectl debug node/<NODE>` is not a substitute:
it mounts the node's root filesystem, which is the one arrangement in which the links
*do* resolve.

## 3. Deploy

Each manifest ships placeholders rather than defaults. Any left unsubstituted fails —
but not all at the same moment, and the difference is worth knowing before you pick a
substitution method. `kubectl` rejects a literal `<NODE>` outright, at admission. A
literal `<RDMA_IB_DEVICES>` it accepts: that one only fails when the engine opens the
device, minutes later and after the weights have loaded. A *plausible but wrong*
fabric value fails in the second way too, which is why the renderer below is worth
preferring over `sed` for those two values specifically.

```bash
kubectl create namespace infera --dry-run=client -o yaml | kubectl apply -f -
```

Fill in what your combo needs — every one of these is a property of your cluster,
and §2 says where each is read from. They are left empty on purpose, and each block
below opens with a `${VAR:?}` guard so that pasting it unedited aborts on the first
value still missing.

The guard is not decoration. **An empty value is not caught downstream.** An empty
`nodeName` in particular is accepted by the CRD and by the API server, and hands the
Pod to the scheduler instead — which on the disaggregated combos can place both legs
on one node and quietly take the RDMA path out of the deployment.

```bash
NODE=            # aggregated combos: the one node.  kubectl get nodes
PREFILL_NODE=    # disaggregated combos: the two nodes
DECODE_NODE=
MODEL_DIR=       # absolute path, identical on every node used, holding GLM-5.2-FP8/
KVD_L3_DIR=      # kvd combos only: node-local NVMe
RAIL=            # disaggregated only: from ibv_devices, and ACTIVE on both nodes
PREFILL_GID=     # the RoCE v2 index on that rail -- per node, so check both
DECODE_GID=
```

**`aggregated`** — one node, two placeholders:

```bash
: "${NODE:?}" "${MODEL_DIR:?}"
sed -e "s|<NODE>|$NODE|" -e "s|<MODEL_DIR>|$MODEL_DIR|" \
    examples/recipes/glm5.2-fp8-gfx942/aggregated/deploy.yaml | kubectl apply -f -
```

**`aggregated + kvd`** — adds the L3 directory. Note this arm runs without MTP;
§6 says why and what it costs:

```bash
: "${NODE:?}" "${MODEL_DIR:?}" "${KVD_L3_DIR:?}"
sed -e "s|<NODE>|$NODE|" -e "s|<MODEL_DIR>|$MODEL_DIR|" \
    -e "s|<KVD_L3_DIR>|$KVD_L3_DIR|" \
    examples/recipes/glm5.2-fp8-gfx942/aggregated-kvd/deploy.yaml | kubectl apply -f -
```

**`disaggregated`** — two nodes, and the RDMA values from §2:

```bash
: "${PREFILL_NODE:?}" "${DECODE_NODE:?}" "${MODEL_DIR:?}" \
  "${RAIL:?}" "${PREFILL_GID:?}" "${DECODE_GID:?}"
sed -e "s|<PREFILL_NODE>|$PREFILL_NODE|" -e "s|<DECODE_NODE>|$DECODE_NODE|" \
    -e "s|<MODEL_DIR>|$MODEL_DIR|"       -e "s|<RDMA_IB_DEVICES>|$RAIL|" \
    -e "s|<PREFILL_GID_INDEX>|$PREFILL_GID|" \
    -e "s|<DECODE_GID_INDEX>|$DECODE_GID|" \
    examples/recipes/glm5.2-fp8-gfx942/disaggregated/deploy.yaml | kubectl apply -f -
```

**`disaggregated + kvd`** — both of the above:

```bash
: "${PREFILL_NODE:?}" "${DECODE_NODE:?}" "${MODEL_DIR:?}" "${KVD_L3_DIR:?}" \
  "${RAIL:?}" "${PREFILL_GID:?}" "${DECODE_GID:?}"
sed -e "s|<PREFILL_NODE>|$PREFILL_NODE|" -e "s|<DECODE_NODE>|$DECODE_NODE|" \
    -e "s|<MODEL_DIR>|$MODEL_DIR|"       -e "s|<KVD_L3_DIR>|$KVD_L3_DIR|" \
    -e "s|<RDMA_IB_DEVICES>|$RAIL|" \
    -e "s|<PREFILL_GID_INDEX>|$PREFILL_GID|" \
    -e "s|<DECODE_GID_INDEX>|$DECODE_GID|" \
    examples/recipes/glm5.2-fp8-gfx942/disaggregated-kvd/deploy.yaml | kubectl apply -f -
```

**`sed` cannot check the two values most worth checking.** A rail name or GID index
that is merely wrong for your fabric substitutes as readily as a right one, and only
fails once the engine opens the device — minutes later, after the weights have
loaded. A fabric that needs more than a swap, too — a non-default GID index, or
Mooncake pinned to one rail because your rails carry no IPv4 — changes the same few
keys in all four combos. Both are reasons to render from one source rather than
editing four files by hand.

[`examples/recipes/render.py`](../render.py) is that source, and takes the same
values as the `sed` above:

```bash
# add --pin-rail as well if your rails carry no IPv4
python3 examples/recipes/render.py glm5.2-fp8-gfx942/disaggregated \
  --set PREFILL_NODE=<PREFILL_NODE> --set DECODE_NODE=<DECODE_NODE> \
  --set MODEL_DIR=<MODEL_DIR> \
  --set RDMA_IB_DEVICES=<RAIL> \
  --set PREFILL_GID_INDEX=<PREFILL_GID_INDEX> \
  --set DECODE_GID_INDEX=<DECODE_GID_INDEX> \
  --check-rail \
  | kubectl apply -f -
```

It is `sed` plus the three things `sed` cannot do. It refuses to emit a manifest with
a placeholder still in it, re-scanning its own output for `<NAME>`; that is why the
block above is written in placeholders rather than plausible examples — pasted
unedited it fails in under a second and names what it wants, where a plausible wrong
value would have rendered silently. It refuses a placeholder name this combo does not
have, so a misspelled `--set` cannot pass as a no-op. And `--pin-rail` adds
`MC_MS_AUTO_DISC=0` plus `MC_MS_FILTERS=<rail>` to both legs — an insertion into
the env list rather than a substitution. Pin the rail when yours carry no IPv4:
every GID is then link-local `fe80::`, so every rail looks like the same `/64`
subnet and Mooncake can pair the prefill node's rail A with the decode node's
rail B, where the transfer times out. `--check-rail` reads the rail's state on
each target node over ssh before rendering — two seconds against the ~11 minutes
it otherwise takes for a down rail to surface as a GPU memory fault.

Each combo deploys under its own name, so the Service and label selectors below
differ per combo:

| Combo | `InferaDeployment` | Engine service(s) |
|---|---|---|
| `aggregated` | `glm52-fp8-mixed` | `worker` |
| `aggregated + kvd` | `glm52-fp8-mixed-kvd` | `worker` |
| `disaggregated` | `glm52-fp8-pd` | `prefill`, `decode` |
| `disaggregated + kvd` | `glm52-fp8-pd-kvd` | `prefill`, `decode` |

Cold start is **10–20 min for `aggregated`** and **15–25 min for the disaggregated
combos**, which load GLM-5.2 plus the MTP nextn layer on both legs in parallel. The
log goes quiet while they do: weights land in ~3.5 min, and everything after that —
draft weights, memory pools, `tilelang`/`aiter` JIT, graph capture — prints almost
nothing for ten minutes or more. That is why the workers use a `startupProbe` with a
90-minute budget (matching `INFERA_ENGINE_READY_TIMEOUT`) and no readiness probe.
Don't kill a slow load.

```bash
CR=glm52-fp8-mixed        # or glm52-fp8-pd-kvd, etc.
SVC=worker                # or prefill

kubectl -n infera get pods -w
kubectl -n infera logs -f -c main \
  -l infera.amd.com/deployment=$CR,infera.amd.com/service=$SVC
```

## 4. Smoke test

```bash
kubectl -n infera port-forward svc/$CR-server 8000:8000 &

# one `mixed` worker, or one prefill + one decode
curl -s localhost:8000/v1/workers | jq

curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"/models/GLM-5.2-FP8",
       "messages":[{"role":"user","content":"What is 127 * 31? Answer with the number only."}],
       "max_tokens":128,"temperature":0,
       "chat_template_kwargs":{"enable_thinking":false}}' | jq -r '.choices[0].message.content'
```

The manifests pass no `--served-model-name`, matching the docker recipe, so the
served name **is** the model path — `/models/GLM-5.2-FP8` in the request above.

`3937` is necessary and not sufficient, in two different ways depending on the
combo.

**On the disaggregated combos, check the transport.** RDMA that failed to
initialise does not stop the deployment; Mooncake falls back to TCP and everything
still answers. One line per DP rank:

```bash
kubectl -n infera logs -c main \
  -l infera.amd.com/deployment=$CR,infera.amd.com/service=decode \
  | grep -aE 'GID index|installTransport'
```

Eight `installTransport, type=rdma` is the pass. `type=tcp` means the run is
functional but proves nothing about the fabric.

**On every combo, send a prompt longer than one chunk.** Without the mooncake
early-send wait-event patch (§1), every prefill chunk but the last is read while
the forward pass is still writing it, and multi-chunk prompts come back *partially
wrong with nothing in any log*. A short prompt that answers correctly cannot see
that. Bury a distinctive needle at the head, middle and tail of a prompt several
times `--chunked-prefill-size` long and ask for it back — losing only the head
reads as "it works" if you happen to probe the tail.

## 5. Benchmark

Both workloads live with the docker recipe, in
[`examples/glm5.2_gfx942/`](../../glm5.2_gfx942/README.md), and neither needs
anything added for Kubernetes — the router is an ordinary OpenAI-compatible
endpoint and does not care how the engines were started. What changes is where the
client runs and which addresses it can dial.

**Two workloads answering different questions.** `bench.sh` sizes raw serving
throughput on random prompts. `run_agentic_trace.sh` replays the multi-turn trace
this recipe was tuned on, and is the only one of the two that can say anything about
the KV cache or the kv-aware router: random prompts share no prefix, so a cache-hit
line of ~0 there is correct by construction rather than a fault. For a KV-reuse
benchmark read §6 first — prompts shorter than a hicache page never reach kvd at
all, and a generator that gives every request a unique prompt measures the tier's
cost with none of its benefit.

### 5.1 Addresses: what is reachable, and what is not

**Only the router gets a Service.** The operator creates `svc/$CR-server`
(ClusterIP, port 8000) for the `server` component and **no Service at all** for the
engines, so each engine leg is reachable only at a Pod or node address:

| Target | Address | Why |
|---|---|---|
| router | `svc/$CR-server:8000`, or the server Pod's IP | the only operator-managed Service |
| `aggregated` worker | Pod IP, port 30000 | on the Pod network, and no Service fronts it |
| `disaggregated` prefill | **node** IP, port 30001 | `hostNetwork: true`, so the Pod IP *is* the node IP |
| `disaggregated` decode | **node** IP, port 31501 | same |

The engine addresses are not optional extras: the agentic runner flushes each leg
before the run and reads `page_size` off one of them, so a client that cannot reach
the engines cannot produce a cache number worth reading.

```bash
CR=glm52-fp8-pd
podip() { kubectl -n infera get pod \
            -l infera.amd.com/deployment=$CR,infera.amd.com/service=$1 \
            -o jsonpath='{.items[0].status.podIP}'; }

ROUTER_IP=$(podip server)
ROUTER=http://$ROUTER_IP:8000
PREFILL=http://$(podip prefill):30001    # hostNetwork, so this is the node IP
DECODE=http://$(podip decode):31501
```

**Do not benchmark through `kubectl port-forward`.** It is the right tool for §4's
smoke test and the wrong one here: it proxies every request through the API server
over a single connection, which caps concurrency and contributes its own latency to
whatever you are measuring. Two cluster-specific traps are worth ruling out before
trusting any address, because both were hit on the validating cluster: its
kube-proxy installed no host-side rules for Service VIPs, so the ClusterIP was
unreachable from the nodes themselves, and NodePort did not answer either — the Pod
IP was the only address that worked.

### 5.2 Random dataset — sizing

`bench.sh` is written for the docker arm: it dials `$PREFILL_IP:$ROUTER_PORT` and
requires resolvable node IPs, neither of which describes a router on the Pod
network. Under Kubernetes, call the same benchmark directly from a client container
off the engine image, so the tokenizer is the model's own:

```bash
docker run --rm --network host \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  -v "$MODEL_DIR":/models:ro --entrypoint python3 infera:sglang-gfx942-glm52 \
  -m sglang.bench_serving --backend sglang-oai-chat \
    --host "$ROUTER_IP" --port 8000 \
    --model /models/GLM-5.2-FP8 --tokenizer /models/GLM-5.2-FP8 \
    --dataset-name random --random-input-len 4096 --random-output-len 1024 \
    --random-range-ratio 0.8 --num-prompts 64 --max-concurrency 16 \
    --warmup-requests 1 --seed 42 --cache-report --output-details
```

The client wants the GPU device nodes even though it computes nothing: importing
`sglang` pulls in aiter, which shells out to `rocminfo` and fails without them.
`--warmup-requests 1` keeps graph capture and the first-token path out of the
measured window; it cannot pre-warm a prefix, because there is none.

**Vary the seed when you sweep concurrency.** At a fixed seed each point's prompt
set is a *superset* of the one below it and the radix tree still holds the smaller
one, so every point above the first reports a large cache-hit rate that is an
artefact of the sweep rather than reuse. `run_sweep.sh` takes a fresh seed per
point for that reason.

### 5.3 Agentic trace — the workload that chose the recipe

Build the trace once, anywhere the tokenizer is available; it is a data file, not a
deployment artefact. The corpus is
[`semianalysisai/cc-traces-weka-062126-256k`][corpus] (Apache-2.0) — Claude Code
agent traffic carrying per-turn token counts and KV block ids but no text, so
`weka_to_agentic_trace.py` synthesises filler while preserving what matters: exact
per-turn lengths and block-level prefix reuse.

[corpus]: https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k

```bash
cd examples/glm5.2_gfx942
hf download semianalysisai/cc-traces-weka-062126-256k --repo-type dataset
SRC=$(ls "$HF_HOME"/hub/datasets--semianalysisai--cc-traces-weka-062126-256k/snapshots/*/traces.jsonl)

python3 weka_to_agentic_trace.py "$SRC" -o data/cc_traces_100k.json \
  --output-len 220 --min-turns 4 --max-context 100000 \
  --verify 20 --tokenizer "$MODEL_DIR"/GLM-5.2-FP8
```

`--output-len` must be the value the run replays with. The converter sizes each
turn's filler with the reply length baked in, so a mismatch drifts every turn's
length and quietly invalidates the scorer's ideal. `--max-context` fits the corpus
to what the deployment can prefill, and `--dry-run` reports the resulting
distribution without writing, which is the cheap way to choose it.

Then drive it with `bench_client.sh`, which exists for exactly this case — a
deployment with no engine container to `docker exec` into. It runs the same client
in a throwaway container, so the tokenizer, dataset, concurrency limiter and scorer
stay identical to the docker arm rather than silently becoming part of the
comparison:

```bash
MODEL="$MODEL_DIR"/GLM-5.2-FP8 TRACE=$PWD/data/cc_traces_100k.json \
NUM_PROMPTS=60 CONC=16 \
  bash bench_client.sh k8s "$ROUTER" "$PREFILL" "$DECODE"
```

`MODEL` and `TRACE` are worth passing explicitly: `env.sh` defaults `MODEL` to a
placeholder path, and `bench_client.sh` bind-mounts it into the client, so the
default fails at the mount rather than at the tokenizer. Pass one engine URL for
the aggregated combos and both legs for the disaggregated ones. `NUM_PROMPTS` counts
**conversations**, not requests. The script reads the served model name off
`/v1/models` and the KV page size off the first engine URL rather than assuming
either, then flushes, replays and rescores.

Two traps it is covering for you. **The served name is the engine's
`--model-path`** — a mount path, so it differs between the arms and is
`/models/GLM-5.2-FP8` here — while the tokenizer must be a path that exists in the
client, so the two cannot be the same string. Sending the wrong one is not a clean
404: the router answers `no active mixed worker for model="…"` and every request
fails in milliseconds, which reads like a dead fleet. **The flush has to actually
land**, because blocks left by an earlier run inflate the hit rate, and
`flush_cache` is a no-op while requests are in flight *and still returns success*.

Read **efficiency** — actual hits over what a cache that evicted nothing could have
returned — rather than the raw hit rate, which is mostly a property of the dataset.
`sglang.benchmark.serving` also mis-reports the input side in multi-turn mode,
keeping the conversation-level `prompt_len` for every turn, which is why
`score_agentic_trace.py` recomputes against the dataset's verified per-turn lengths.
Efficiency at 100% with nothing evicted means the run is below its pressure point,
where kv-aware routing has nothing to distinguish itself on; raise `CONC` until
eviction appears. **Above** 100% means the flush did not take. A run with any failed
request is refused rather than scored, since a failure is recorded with
`cached_tokens=0` and would otherwise read as a cache problem.

## 6. kvd

kvd runs on the worker that prefills — the `mixed` worker on `aggregated + kvd`,
the prefill leg on `disaggregated + kvd`. There is deliberately none on a decode
leg: SGLang issues storage prefetch on its aggregated and prefill branches only, so
a decode-side tier would be written and never read, and infera refuses to wire it
up even if handed the socket.

### Why `aggregated + kvd` is the one arm without MTP

The empirical result is narrow and solid: **MTP and hicache both active on a
worker that prefills *and* decodes hangs.** Five configurations across three
failure shapes were tried with MTP on; dropping MTP and changing nothing else
passed on the first attempt. Either feature alone is fine — `aggregated` runs MTP
without kvd, and the no-MTP arm runs kvd with DP-attention and the overlap
scheduler still enabled.

The mechanism is a hypothesis, not a proven root cause. hicache's prefetch and
write-back bookkeeping is per-rank local state, while both SGLang's scheduler
control broadcast and the model's own collectives need every rank doing the same
thing on the same iteration; MTP adds a *second* per-rank host pool for draft
tokens — the same pool whose 132-byte stride forces `--hicache-io-backend direct`
below — and more per-rank state means more room to drift.

`disaggregated + kvd` keeps MTP and has served 224 requests across two sessions
because **speculative decoding only happens on the decode leg** — the prefill
leg's `spec_accept_length` is `0.0` on all 8 ranks — so the combination never
actually forms there even though the flags are present on both legs.

Worth recognising by shape, because neither signature says "cache":

- **The scheduler's control broadcast**, on the *first* prompt long enough to
  trigger write-back. Startup is clean and `127 × 31` returns `3937` first.
  Nothing at all is logged for the failing request — not even a `Prefill batch`
  line — and 600 s later the NCCL watchdog fires on all 8 ranks with
  `WorkNCCL(SeqNum=810, OpType=ALLREDUCE) ran for 600088 milliseconds`, every rank
  parked in `_broadcast_reqs_across_ranks`. The ranks' `last enqueued work`
  counters differ, which is the desync itself.
- **Inside the model forward**, under concurrency, after the deployment has been
  answering correctly for a while:
  `eagle_worker_v2.forward_batch_generation → deepseek_v2.forward`, stuck in an
  allreduce. Here the ranks composed *different batches*.

**Establishing that MTP is the trigger took six configurations**, and the wrong
turns are recorded so nobody retries them blind.
`--enable-dp-attention-local-control-broadcast` removes the per-iteration
all-ranks gloo sync and made things worse — two GPUs took a `Memory access fault`
during startup warmup. `--disable-overlap-schedule` faulted the same way. Dropping
DP-attention for plain TP8 got furthest and still failed: every correctness judge
passed, then it hung in the model forward on the fourth benchmark request — so
DP-attention was never the trigger. (One of the six was not a finding at all:
plain TP8 sized as if `--hicache-size 32` were 32 GB total got the container
OOMKilled at exit 137. It is per TP rank — see §7.) Dropping MTP and putting
DP-attention *back* is a single-variable change from the first attempt, and it
works.

Cross-recipe evidence agrees. [`glm5.2/`](../glm5.2/README.md) runs SGLang hicache
on a mixed worker as well and does not hang, and its engine line carries no
speculative flags at all. (It validated its kvd plumbing on Qwen3-0.6B rather than
on GLM-5.2-MXFP4, so treat it as evidence about the flag combination, not about
the model.)

**What it costs.** Output throughput lands well below `aggregated`'s with MTP and
no tier. Two variables differ there, so it isolates neither — read it as "this arm
trades MTP's decode throughput for the tier". If you need MTP on one node, use
`aggregated`. If you need the tier with MTP, use `disaggregated + kvd`.

**What it bought, and this is the only arm where it did.** This is the one
configuration in which kvd has been observed *serving reads* rather than only
absorbing writes: `gets_total 370 / hits_total 370 / misses_total 0`, with the
engine logging `HiCache prefetch success … loaded=185`. The reads came from three
9-chunk prompts sharing a haystack — i.e. from prefix reuse, which is exactly what
the tier is for and what §5's random dataset never generates.

**`--hicache-io-backend direct` is required, and the default is not.** SGLang's
`kernel` write-back path needs every host pool's stride to be a
multiple of 8. MTP registers a second host pool for its draft tokens at
`page_size 1`, and GLM-5.2's 128-wide indexer head with 128-element quant blocks
makes that stride 132 bytes — the main KV pool at `page_size 64` is 8448 and
passes, the draft pool does not. The guard raises on a pool that misses the
condition rather than falling back, so with kvd and MTP both on, the first
request to write back kills the prefill scheduler:

```
memory_pool_host.py  backup_from_device_all_layer
ValueError: Unsupported IO backend: kernel
```

`direct` is not a downgrade to work around this: it also switches `mem_layout`
from `page_first` to `page_first_direct`, which is what the indexer pool's direct
branch actually implements. It does mean the kvd numbers below and the `kernel`
path's are not directly comparable — `kernel` exists because it is faster, and
nothing here separates its cost from kvd's.

```bash
# kvd runs on the leg that prefills, so SVC is `worker` on aggregated-kvd and
# `prefill` on disaggregated-kvd -- the same SVC set in §3.
POD=$(kubectl -n infera get pod -o name \
  -l infera.amd.com/deployment=$CR,infera.amd.com/service=$SVC | head -1)
kubectl -n infera exec $POD -c kvd -- \
  python3 -m infera.kvd.statctl --socket /tmp/infera-kvd/kvd.sock
```

`sets_total` climbing means the engine writes to kvd; `gets_total` / `hits_total`
climbing means it reads back. Writes alone prove only half the path.

Send a prompt long enough to fill several hicache pages before reading these — one
shorter than a page produces no kvd traffic at all and proves nothing.

Four counters that will mislead you:

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
- **`sets_total` is not "what this run wrote".** On this deployment it keeps
  climbing at a steady ~24/s with **no traffic at all**: 18944 when the benchmark
  finished, 46588 twenty-four minutes later, nothing sent in between. Every
  SGLang-side counter (`prompt_tokens_total`, `hicache_host_used_tokens`,
  `backuped_tokens_total`) is frozen meanwhile, and the 8 schedulers are the
  socket's only peers, so the engine is issuing them. Root cause is not yet
  known. Treat `sets_total > 0` as "the write path works" and nothing more.
- **`long_bytes` is residency, not volume.** It goes down as well as up with
  `evictions_total` at 0, because a rewritten key overwrites its tablespace slot.
  Do not read it as cumulative bytes written.

That non-convergent write queue has one consequence beyond the counters:
**`POST /flush_cache` can never succeed on a kvd leg.** `is_fully_idle()` requires
hicache's `ongoing_backup` to be empty, so the endpoint returns `400` and the log
says `Deferred flush_cache timed out while waiting for idle state.` even with
`?timeout=90` and zero requests in flight. Anything you were planning to do by
flushing the device cache needs another route.

**One L2 limit worth knowing, because it is silent after the first line.** The
shared arena fixes its slot size on the *first* put and refuses everything larger
for the process lifetime, warning exactly once:

```
shared_arena: slot grid initialized — slot_size=10368 num_slots=6628035
shared_arena: blob size 44944 > slot_size 10368 (set by first put). Refusing oversize.
```

This model writes **two blob sizes**, both at one-token-by-all-78-layers
granularity: the main MLA latent at `576 B × 78 = 44928 B`, and the DSA indexer
at `132 B × 78 ≈ 10352 B`. **Whichever lands first sets the grid**: big-first
gives a 44944 B slot both sizes fit, small-first locks the main KV out for the
process lifetime. The mixed worker gets the small one first, on every cold start
observed so far. Independent of MTP — the no-MTP arm logs the same
`slot_size=10368`; the small pool is the DSA indexer, which this model has
regardless of speculative decoding.

Whether the prefill leg behaves differently is **unverified** — an earlier claim
that it wins the race was based on a misreading of the `sets`/`entries` counters
(`entries` counts inline fallbacks too, so the two being close says nothing about
what the arena accepted). Since the ordering looks deterministic rather than
raced, assume the prefill leg is affected too until someone reads
`slot grid initialized` out of its kvd sidecar.

Refused blobs are **not dropped** — `HostStore` falls back to storing them inline
in kvd's own heap, so this costs performance and RAM accounting, never
correctness. **L3 is entirely unaffected**: the long-region write is a separate
branch that resolves the value whether it lives in a slot or inline. What you
lose is the zero-copy read (the engine mmaps the arena and the socket carries
only `offset/length/version`; inline blobs are serialized over the UDS instead)
and the 64 GiB of `mlock`ed arena, which ends up holding only indexer blobs while
the main KV competes for the separate `--max-bytes` inline budget. Passing
`--shared-arena-bytes 0` disables the arena outright and reclaims that RAM.

**Running the A/B is a matter of deploying the sibling combo**, which is the main
reason `disaggregated` ships as a manifest rather than as a list of lines to
delete: it is the `KVD=0` arm of `disaggregated + kvd`, differing only by the kvd
sidecar, three engine flags, two volumes and the engine's memory limit.

On one node there is no such pair. `aggregated` has MTP and `aggregated + kvd`
cannot, so comparing them moves two variables at once. The clean single-node A/B
is `aggregated + kvd` against itself with the kvd sidecar, its three engine flags
and its two volumes removed.

Worth running only once the deployment is above its pressure point — below it the
54 GB device pool per rank answers everything and both arms serve identically.

**Every measurement of this tier so far has been negative, and for a reason worth
understanding before you read that as "kvd is slow".** On the docker recipe's
agentic trace, `KVD=1` was slower than `KVD=0` while serving **zero reads** —
every page it wrote to L3 stayed there. Nothing was wrong with the offload path:
the trace had no misses left for it, the GPU pool alone already covering
essentially all the reuse it had to offer.

The Kubernetes A/B found the same direction for a cruder reason: §5.2's random
dataset gives every request a unique prompt, so there is no prefix to reuse at all.
Against its `KVD=0` sibling, kvd cost throughput and TTFT with `gets_total = 0` —
full price, zero benefit.

Both results say the same thing: **hicache is `write_through`, so the cost lands on
the prefill path unconditionally, and the benefit only exists if something reads the
tier back.** Deploy the kvd combo when your traffic has prefix reuse that outgrows
the device pool, and deploy its non-kvd sibling first to confirm that it does.

## 7. What changed from the docker recipe, and why

Nothing, as of this round. Both recipes now run `--router-backend rust` and every
engine and kvd flag was already identical — which is worth stating as a measured
result rather than an assertion, because this section previously carried a
fabricated reason for the one flag that did differ.

That claim is checked, not eyeballed. The comparison runs the
docker recipe's launch scripts with a `python3` shim on PATH that records its argv
and exits, so what gets compared is the argv the engine would really have received
— everything `env.sh` computed, every conditional branch the script took — against
the `command:` arrays in these manifests. It also compares the engine-facing
environment, which a `command:`-only diff misses entirely and which carries things
that change performance rather than plumbing (`SGLANG_DSA_TRITON_PREFILL`,
`SGLANG_USE_AITER`, `HSA_NO_SCRATCH_RECLAIM`).

Result on `disaggregated`, before the switch: **zero real differences on either
leg**, with `--router-backend` the only one left on the router. Everything else it
reports is a substrate translation with a stated reason (`--advertise-host` env IP
vs downward API, `--discovery-backend` etcd vs kubernetes, `--model-path` mount
path). After the switch all three roles diff clean.

**The earlier reason for `python` was wrong.** It claimed the Rust binary supports
only `--discovery-backend etcd`. `launch_rust.py` lists `_SUPPORTED_DISCOVERY = ("etcd",
"kubernetes")`, `rust/router/src/main.rs` has the Kubernetes watch path, and the
only extra requirement — `--k8s-label-selector` — is injected by the operator as
`INFERA_K8S_LABEL_SELECTOR` on every server component, which
`--k8s-label-selector` already defaults to.

Measured, not inferred: flipping that one word and changing nothing else brings up
a server pod whose PID 1 is `/usr/local/bin/infera-router --discovery-backend
kubernetes --k8s-label-selector infera.amd.com/deployment=<name> ...`, with the
selector supplied entirely by the operator's env. On the `aggregated` shape scaled
to two workers, rust passed every judge python did — arithmetic, the 9-chunk
needle at three depths, MTP accepting on all 16 ranks, and a kv-aware routing
check confirming prefix stickiness with `cached_tokens` showing real reuse.

One thing that check made necessary, worth knowing before you repeat it:
**`replicas: 1` cannot test a routing policy at all.** With one worker there is
nothing to choose between, so a router whose kv view is completely empty scores
exactly like a working one. That is not hypothetical: the Rust ZMQ decoder once
returned `None` for every element of an MTP bigram batch and kv-aware silently
degraded to load balancing, logging nothing (`rust/router/tests/kv_event_zmq.rs`).
The rust router exposes no kv-view metric either, so the only honest test is
behavioural, with two workers on two nodes: send the same prefix twice and assert
it lands on the same worker with `cached_tokens` above zero.

### PD on rust, and the workload that settled it

The `aggregated` check above left the PD combos open, because they take a different
path through the router: PD dispatch plus bootstrap protocol injection, not
mixed-worker dispatch. So `disaggregated` was run on rust against a real agentic
workload — the multi-turn trace from `examples/glm5.2_gfx942_agentic_bench` on the
`llying/dev/glm5p2_fp8_kvd` branch, built from the Apache-2.0
`semianalysisai/cc-traces-weka-062126-256k` corpus, whose turns run to tens of
thousands of input tokens — and compared against the docker recipe serving the
identical trace from the identical client container.

Both arms completed every request, and both reached the growing-prefix cache ideal
exactly, differing by a handful of cached tokens out of tens of millions.

`aggregated-kvd` and `disaggregated-kvd` were **not** run on rust. The switch still
covers them, and the reason is structural rather than optimistic: all four
manifests' `server` commands are byte-identical, kvd is entirely engine-side
(`--infera-kvd-socket`, `--hicache-size`), and the router never sees it. The two
combos that were run cover both dispatch paths the router actually has.

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
because `--hicache-size 32` is **GB per TP rank** — every one of the 8 schedulers
logs `Allocating 32.00 GB host memory for hierarchical KV cache`, so the tier is
256 GB, not 32. Dropping DP-attention does not reduce it, which is worth knowing
because it looks like it should: sizing `aggregated + kvd` as if TP8 meant one
32 GB pool got that container OOMKilled (exit 137) immediately after graph
capture. `decode` runs no host tier and gets half. `cpu: 32` matches `num_threads`
in `--model-loader-extra-config`.

## 8. Validation status

**All four combos are validated on Kubernetes**, on the same cluster, image and
judges, so the arms are comparable to each other. Every one served 232 requests
across conc 1/8/16/32 with **zero failures, zero GPU faults and zero Pod
restarts**.

All of it was checked with `--router-backend python`, which is what these manifests
shipped at the time; §7 covers what changed and what was re-checked on `rust`.

| Combo | Status |
|---|---|
| `aggregated` | **validated.** Cold start 20.7 min, `3937` correct, MTP accepting on all 8 ranks |
| `aggregated + kvd` | **validated, without MTP** (§6). Cold start 21.5 min. The only arm observed *reading* from kvd: `gets_total 370 / hits_total 370 / misses_total 0`. Took six configurations to find; the five that failed are recorded in §6 |
| `disaggregated` | **validated.** Cold start 21.4 min, MTP accepting on the decode leg |
| `disaggregated + kvd` | **validated.** Cold start 21.2 min, MTP accepting. The write path works and L3 replayed its journal across a Pod restart — but read the counter caveats in §6 before quoting any volume from it |

What was checked rather than assumed, because each of these fails quietly:

- **Correctness past one chunk.** A 9-chunk (~9.8k token) prompt's needle came back
  intact from the head, middle *and* tail positions on all four combos. On the
  disaggregated arms that is multi-chunk KV crossing the fabric correctly, not
  "the deployment is green".
- **KV really moves over RDMA.** All 8 decode-leg ranks logged
  `installTransport, type=rdma` on the pinned rail and GID index. §2 explains why
  the TCP fallback is invisible if you don't look.
- **MTP is accepting, not silently degrading** to one token per step — hence
  `--enable-metrics-for-all-schedulers` in all four manifests. Note the prefill
  leg reports accept length `0.0` and that is correct: speculation happens on
  decode. (`aggregated + kvd` reports nothing, because it runs no MTP.)
- **kvd reads back, on the one arm where the workload allowed it.** 370 gets, 370
  hits, 0 misses on `aggregated + kvd`, with `HiCache prefetch success …
  loaded=185` in the engine log. Everywhere else `gets_total` stayed at 0 for the
  reason in §6, so this is the only end-to-end confirmation that the tier's read
  path works at all.
- **Native kvd sidecar ordering.** The sidecar reached a healthy `startupProbe`
  before `main` started, on every bring-up.
- **kvd's L3 survives the Pod.** A restarted Pod replayed the tablespace journal
  and came up with the previous run's `long_bytes` intact.

**Two shape trade-offs the arms make visible**, both worth knowing before you pick
one. PD buys inter-token latency and spends TTFT: the decode node no longer stops
to prefill, but the prompt must prefill on one node and ship its KV across the
fabric before the first token appears. And its throughput scaling is sublinear, so
PD is not a per-GPU efficiency win here — choose by which end your SLO sits on.
kvd, on a workload with no reuse, costs throughput and TTFT and returns nothing;
§6 says how to tell whether your traffic is the kind that gets it back.

**Four boundaries.** (1) The validating cluster's rails carry no IPv4, so every
rail's only RoCEv2 GID sits in the same `fe80::/64` and Mooncake cannot tell them
apart to pair them; KV was pinned to a **single rail** (`MC_MS_AUTO_DISC=0` plus
`MC_MS_FILTERS=<rail>`) rather than the multi-rail aggregate. Correctness is
unaffected, but nothing here supports a claim about KV transport *not* being the
bottleneck. (2) That cluster's host driver is 6.3.x, so it ran the `rocm700` base
per §1 — the `rocm720` default remains validated only in the `docker` form.
(3) **kvd's benefit is still unmeasured** — its read path is now confirmed to work,
but no benchmark here exercises it: §5.2's random dataset sends a distinct prompt per
request, so `gets_total` stayed at 0 through every sweep. (4) The sweep tops out at
conc 32,
so nothing here speaks to saturation; the PD comparison is not equal-hardware; and
the single-node arms differ by MTP as well as by kvd, so neither of those
comparisons isolates one variable.

Two things worth re-reading before a first bring-up, because both fail quietly
rather than loudly: `--advertise-host` resolving to a node IP that is not on the
RoCE rail (§7), and an engine image missing any of the three source fixes (§1) —
which is why §1 carries a command that reads the markers out of the built image
rather than trusting the build log.

## 9. Tear down

```bash
kubectl -n infera delete inferadeployment $CR
pkill -f 'port-forward.*8000' || true     # the tunnel from §4, if still up

# on every node the deployment used
rocm-smi --showpids     # a deleted Pod can leave processes holding VRAM
```

Deleting the `InferaDeployment` takes the Deployment or LWS and the Service with
it, so there is nothing else to chase. Two caveats:

- **Let the VRAM come back before re-applying.** The engines release it over tens
  of seconds; a re-apply that races them dies allocating its memory pool on a node
  that already looks idle to `kubectl get pods`. This is the same race the `docker`
  arm's `stop.sh` waits out before it relaunches.
- **`<KVD_L3_DIR>` survives, by design.** L3 is journalled and recovered on the
  next start rather than rebuilt — that is what let it replay across a Pod restart
  in §8. Delete the directory to reclaim the NVMe or to start from a cold tier.
  Nothing else survives: the JIT and CUDA-graph caches go with the Pod, so the next
  bring-up pays the full cold start from §3.

## Source

[`examples/recipes/glm5.2-fp8-gfx942/`](.) in
[AMD-AGI/Infera](https://github.com/AMD-AGI/Infera) · [all recipes](../README.md) ·
[the same shape on MI355X, in `docker` form](../../sglang_1p1d_glm5.2/README.md)
