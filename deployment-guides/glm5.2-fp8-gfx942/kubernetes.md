# GLM-5.2-FP8 on gfx942 — Kubernetes deployment

Serve GLM-5.2-FP8 on gfx942 (MI300X / MI325X) through the Infera operator: SGLang
TP8 with DP-attention, MTP speculative decoding, fp8 KV cache, kv-aware routing,
and optionally prefill/decode disaggregation over Mooncake RDMA and KV offload to
host RAM plus node-local NVMe.

The engine, router and offload flags are lifted unchanged from the shell
deployment in [`docker.md`](docker.md). That is checked rather than asserted: a
flag-by-flag comparison of the argv each side really produces found **zero
substantive differences** across prefill, decode and router, and the same workload
put decode-side latency within 1.2% across the two. **Tuning conclusions carry
between the two deployments.**

Manifests live in `examples/recipes/glm5.2-fp8-gfx942/`.

---

## 1. Pick a combo

| Combo | Nodes | KV cache | MTP | Manifest |
|---|---|---|---|---|
| `aggregated` | 1 | GPU only | yes | `aggregated/deploy.yaml` |
| `aggregated + kvd` | 1 | + host RAM (L2) and node NVMe (L3) | **no** | `aggregated-kvd/deploy.yaml` |
| `disaggregated` | 2 | GPU only | yes | `disaggregated/deploy.yaml` |
| `disaggregated + kvd` | 2 | + kvd on the prefill leg | yes | `disaggregated-kvd/deploy.yaml` |

All four run TP8 / DP8 with DP-attention, and all four are validated.

**Start with `aggregated`** even if you want PD. It needs one node and touches
neither Mooncake nor `/dev/infiniband` nor a GID index, so it cannot hit the
entire RDMA class of silent failures. That makes it the right thing to bring up
when something is wrong and you do not yet know which layer it is in: it separates
"can this image load these weights and serve" from "can KV cross this fabric", and
those two fail very differently.

**`aggregated + kvd` is the one arm without MTP.** On a worker that both prefills
and decodes, MTP and the hierarchical cache deadlock each other. It took six
configurations to establish that MTP was the trigger rather than a bystander, and
the one that ships is MTP off with everything else unchanged. `disaggregated + kvd`
keeps MTP because speculation only happens on its decode leg, so the combination
never forms on the leg carrying kvd.

**Add `kvd` only once you have a reason.** The cache is `write_through`, so every
byte the tier absorbs is paid for on the prefill path whether or not anything ever
reads it back. Measured on two nodes with a no-reuse workload it cost 43% of output
throughput and 3.45× TTFT while serving zero reads. It earns that back only when
requests share long prefixes and the reuse outgrows the ~54 GB/rank device pool.

One consequence for measurement: `aggregated` is *not* the `KVD=0` sibling of
`aggregated + kvd`, because it also has MTP. Only the disaggregated pair differs
by kvd alone.

---

## 2. Prerequisites

**Hardware**, per combo. Every engine Pod wants 8× gfx942 and `cpu: 32`.

| Combo | Nodes | Host RAM, busiest node | Node-local NVMe | RoCE fabric |
|---|---|---|---|---|
| `aggregated` | 1 | ~272 GiB | no | no |
| `aggregated + kvd` | 1 | ~670 GiB | yes, for L3 | no |
| `disaggregated` | 2 | ~272 GiB (prefill) | no | **yes** |
| `disaggregated + kvd` | 2 | ~670 GiB (prefill) | yes, on prefill | **yes** |

**Cluster.** Kubernetes **1.29+** for the two kvd combos — the kvd daemon is a
native sidecar (`initContainers` with `restartPolicy: Always`), which is what makes
it reach a healthy `startupProbe` *before* the engine starts. That ordering is
load-bearing: the engine probes the kvd socket once with a 5 s timeout and refuses
to start if nothing answers. On an older cluster the sidecar becomes an ordinary
container and the ordering becomes a race the engine loses about as often as it
wins. The two non-kvd combos have no such requirement.

```bash
# nodes must advertise amd.com/gpu
kubectl get nodes -o custom-columns=NODE:.metadata.name,GPU:.status.allocatable.'amd\.com/gpu'

# the operator, which provides the InferaDeployment CRD
helm install infera-operator deploy/operator/helm/infera-operator \
  -n infera-system --create-namespace
kubectl -n infera-system rollout status deploy/infera-operator
```

**Weights.** A `hostPath`, not a PVC, at the **same path on every node the
deployment uses**. If the path is a HuggingFace cache symlink, mount the directory
the links resolve into as well — otherwise the inner relative links dangle and
`transformers` rejects the model with `Should have a model_type key in its
config.json`, four minutes into startup and far from its cause.

**Read the host driver version before building — on every node.** It decides the
base image, and getting it wrong does not refuse to start:

```bash
dpkg -l | grep -E 'amdgpu-dkms|rocm-core'
```

| Host driver | ROCm userspace | Base image |
|---|---|---|
| 6.4.x | up to 7.2.x | `rocm720` — the Dockerfile default |
| 6.3.x | up to 7.0.x | `rocm700` — you **must** override |

A mismatched pair initialises, loads weights, captures graphs, then faults with
`Memory access fault by GPU node-N` somewhere under load, in a different place
each time.

---

## 3. Build the image and get it onto the nodes

**Do not use a vendor-preinstalled GLM-5.2 image.** This recipe runs an
infera-built engine image rather than a stock vendor one plus a mounted overlay,
because GLM-5.2 on this base needs a rebuilt Mooncake and four SGLang source
patches that no mounted payload can supply. The Mooncake rebuild in particular is
not optional: the base bundles a Mooncake that installs a HIP IPC transport
unconditionally and prefers it over RDMA, so cross-node PD dies on the first
request inside `hipIpcOpenMemHandle`, which cannot open a peer node's handle.

```bash
docker build -f deploy/docker/Dockerfile.sglang.gfx942 \
  -t infera:sglang-gfx942-glm52 .                            # 6.4.x host driver

docker build -f deploy/docker/Dockerfile.sglang.gfx942 \
  --build-arg SGLANG_BASE_IMAGE=lmsysorg/sglang:v0.5.16-rocm700-mi30x \
  -t infera:sglang-gfx942-glm52 .                            # 6.3.x host driver
```

The image is ~107 GB and takes about 40 minutes. Verify the patches landed before
trusting it — three of the four leave markers, which is the cheap way to catch a
patch that silently no-op'd against a moved anchor:

```bash
docker run --rm --entrypoint python3 \
  -v "$PWD/examples/glm5.2_gfx942/check_image.py:/check.py:ro" \
  infera:sglang-gfx942-glm52 /check.py
```

Three `YES` is the pass:

```text
sglang_rocm/host_alloc       srt/mem_cache/pool_host/common.py            YES
sglang_rocm/staged_wb        srt/mem_cache/pool_host/mla.py               YES
sglang_disagg/early_send     srt/disaggregation/mooncake/conn.py          YES
```

This is also how to check an image someone handed you. The GLM-5.2 image
preinstalled on the validated nodes reports `no` on the first line, which is why
these guides tell you to build your own.

### 3.1 The image has to be in the node's container runtime, not docker's

The manifests use `imagePullPolicy: IfNotPresent`, so a locally available image is
used as-is — but "locally available" means available to the **kubelet's** runtime.
On a containerd-based cluster (RKE2, k3s) a `docker build` result is invisible to
it. Push to a registry the cluster can pull from, or import directly:

```bash
docker save infera:sglang-gfx942-glm52 \
  | ctr -a /run/k3s/containerd/containerd.sock -n k8s.io images import -
```

Without root on the nodes but with cluster-admin, do the same from a privileged
Pod that mounts both the `docker` and the `ctr` binaries from the host — both, not
just `ctr`, since `docker save` has to run somewhere. See
[`mi325x-handoff/tools/import-image-to-containerd.sh`](../../mi325x-handoff/tools/import-image-to-containerd.sh)
for a working version.

### 3.2 The kubelet may delete it again within two minutes

An imported image that no Pod references is garbage collected as soon as the node
is above the image-GC high threshold — 85% disk by default, with a minimum-age
protection of only two minutes. A 107 GB image on a node at 90% disappears almost
immediately, and the import reports success first.

Check before you import:

```bash
df -h /var/lib/rancher     # or wherever the runtime's image store lives
```

If you cannot get under the threshold, **render the manifest before importing and
apply it the moment the import finishes**, so a Pod references the image inside the
two-minute window:

```bash
sed -e "s|<PREFILL_NODE>|node-a|" ... disaggregated/deploy.yaml > pd.yaml
bash import-image-to-containerd.sh ... && kubectl apply -f pd.yaml
```

---

## 4. Adapt to your fabric

Skip this section for the `aggregated` combos — they move no KV off the node and
carry no rail, no GID index and no Mooncake configuration at all.

The disaggregated manifests take three fabric values. Read them from the nodes
rather than copying them from anywhere, **including from this guide**:

| Placeholder | What it is |
|---|---|
| `<RDMA_IB_DEVICES>` | The rail(s) Mooncake may use, comma-separated, from `ibv_devices`. A rail that is physically down must **not** be listed. |
| `<PREFILL_GID_INDEX>` / `<DECODE_GID_INDEX>` | The index on that rail whose type is `RoCE v2`. Two placeholders because the index is per **node**, not per cluster — two identical machines routinely expose different ones. |

Check the fabric from inside the image on both nodes first, because zero visible
ports is how RDMA fails here — Mooncake falls back to TCP and the deployment still
comes up:

```bash
docker run --rm --network host --device=/dev/infiniband --cap-add=IPC_LOCK \
  --entrypoint bash infera:sglang-gfx942-glm52 -c 'ibv_devinfo | grep -c PORT_ACTIVE'
```

Then read the GID table on each node:

```bash
show_gids <rail>
# or:
for i in $(seq 0 7); do
  echo "$i $(cat /sys/class/infiniband/<rail>/ports/1/gid_attrs/types/$i 2>/dev/null)" \
       "$(cat /sys/class/infiniband/<rail>/ports/1/gids/$i 2>/dev/null)"
done
```

A wrong index does not error — it pins KV to an interface that never carries it,
and the transfer either times out or runs 4–18× slower with nothing in any log.

**Two fabric shapes have been validated, and they need different answers.** Find
which one you are on before substituting anything:

**A. Rails carry IPv4.** The GID table has a `RoCE v2` entry whose address is the
IPv4-mapped form of the node's data IP (`::ffff:10.115.43.101`). Take that index —
it was `3` on the validated MI325X pair, whose HCAs are named `rdma0..rdma7`
rather than `mlx5_N`. A plain placeholder substitution is all you need.

**B. Rails carry no IPv4.** Every GID is link-local `fe80::`, and the `RoCE v2`
entry is usually index `1` — index 3 in that case typically belongs to a
management NIC, so copying `3` from case A points Mooncake at an address the peer
cannot reach. This shape needs one thing more than a substitution: because every
rail then looks like the same `fe80::/64` subnet, Mooncake cannot tell them apart
and can pair the prefill node's rail A with the decode node's rail B, where the
transfer simply times out. Pin it to a single rail:

```yaml
- {name: MC_MS_AUTO_DISC, value: "0"}
- {name: MC_MS_FILTERS,   value: "<rail>"}
```

[`mi325x-handoff/tools/render-deploy.py`](../../mi325x-handoff/tools/render-deploy.py)
renders all four combos for exactly this case.

**Pinning KV to one rail is not a compromise.** Striping it across every NIC
measured 11.9% *slower* on this workload, which uses 4.5% of a single 200 Gb/s
port.

---

## 5. Deploy

Each manifest ships placeholders rather than defaults, so anything left
unsubstituted fails loudly — `kubectl` rejects `<NODE>` outright, and a literal
`<RDMA_IB_DEVICES>` is not a device `ibv_open_device` will accept.

```bash
kubectl create namespace infera --dry-run=client -o yaml | kubectl apply -f -
```

`aggregated` — one node, two placeholders:

```bash
sed -e "s|<NODE>|node-a|" -e "s|<MODEL_DIR>|/mnt/models|" \
    examples/recipes/glm5.2-fp8-gfx942/aggregated/deploy.yaml | kubectl apply -f -
```

`disaggregated` — two nodes, plus §4's fabric values:

```bash
sed -e "s|<PREFILL_NODE>|node-a|"   -e "s|<DECODE_NODE>|node-b|" \
    -e "s|<MODEL_DIR>|/mnt/models|" -e "s|<RDMA_IB_DEVICES>|rdma0|" \
    -e "s|<PREFILL_GID_INDEX>|3|"   -e "s|<DECODE_GID_INDEX>|3|" \
    examples/recipes/glm5.2-fp8-gfx942/disaggregated/deploy.yaml | kubectl apply -f -
```

The `kvd` variants add `<KVD_L3_DIR>`, which must be **node-local NVMe** — anything
shared (NFS, Weka) is classified as buffered I/O and the reload lands in the TTFT
budget instead of under it.

Each combo deploys under its own name:

| Combo | `InferaDeployment` | Engine service(s) |
|---|---|---|
| `aggregated` | `glm52-fp8-mixed` | `worker` |
| `aggregated + kvd` | `glm52-fp8-mixed-kvd` | `worker` |
| `disaggregated` | `glm52-fp8-pd` | `prefill`, `decode` |
| `disaggregated + kvd` | `glm52-fp8-pd-kvd` | `prefill`, `decode` |

**Cold start is 10–20 min aggregated, 15–25 min disaggregated.** The log goes
quiet: weights land in ~3.5 minutes and everything after that — draft weights,
memory pools, `tilelang`/`aiter` JIT, graph capture — prints almost nothing for ten
minutes or more. The workers therefore use a `startupProbe` with a 90-minute budget
and no readiness probe. Don't kill a slow load.

```bash
CR=glm52-fp8-pd        # or glm52-fp8-mixed, etc.
kubectl -n infera get pods -w
kubectl -n infera logs -f -c main \
  -l infera.amd.com/deployment=$CR,infera.amd.com/service=prefill
```

---

## 6. Reach the router

```bash
kubectl -n infera port-forward svc/$CR-server 8000:8000 &
```

`port-forward` goes through the API server and works everywhere. **Do not assume
the Service VIP works from a node's host shell** — on RKE2 neither the ClusterIP
nor a NodePort was reachable that way, because kube-proxy does not install
host-side forwarding rules for Service VIPs. The router's **Pod IP** works, and the
engine Pods use `hostNetwork: true`, so they are reachable at `nodeIP:port`.

For a benchmark client that has to sustain load, use the router's Pod IP rather
than a port-forward — the forward is a single API-server-proxied stream and
becomes the bottleneck.

---

## 7. Verify

```bash
curl -s localhost:8000/v1/workers | jq     # one mixed worker, or one prefill + one decode

curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"/models/GLM-5.2-FP8",
       "messages":[{"role":"user","content":"What is 127 * 31? Answer with the number only."}],
       "max_tokens":128,"temperature":0,
       "chat_template_kwargs":{"enable_thinking":false}}' | jq -r '.choices[0].message.content'
```

The manifests pass no `--served-model-name`, so the served name **is** the model
path — `/models/GLM-5.2-FP8` above.

`3937` is necessary and not sufficient. Two more checks, both for failures that
return HTTP 200:

**On the disaggregated combos, check the transport.** RDMA that failed to
initialise does not stop anything; Mooncake falls back to TCP and everything still
answers. Expect one line per DP rank:

```bash
kubectl -n infera logs -c main \
  -l infera.amd.com/deployment=$CR,infera.amd.com/service=decode \
  | grep -aE 'GID index|installTransport'
```

Eight `installTransport, type=rdma` is the pass. `type=tcp` means the run works and
proves nothing about the fabric.

**On every combo, send a prompt longer than one chunk.** Without the Mooncake
early-send wait-event patch, every prefill chunk but the last is read while the
forward pass is still writing it, and multi-chunk prompts come back *partially
wrong with nothing in any log*. A short prompt that answers correctly cannot see
that. Bury a distinctive needle at the head, middle **and** tail of a prompt
several times `--chunked-prefill-size` long and ask for all three back — losing
only the head reads as "it works" if you happen to probe the tail.

---

## 8. Benchmark

The router is an ordinary OpenAI-compatible endpoint and does not care how the
engines were started, so use the same client as the shell deployment. That is the
point: the tokenizer, dataset, concurrency limiter and scorer all sit on the client
side of the wire, so two different clients would put those into any comparison.

Build the dataset once (see [`docker.md`](docker.md) §7.2), then from a host with
the image and the weights:

```bash
cd examples/glm5.2_gfx942
cp cluster.env.example cluster.env    # MODEL, IMAGE and DATA_DIR are what matter here

NUM_PROMPTS=60 CONC=16 bash bench_client.sh k8s http://<router-pod-ip>:8000 \
  http://<prefill-node-ip>:30001 http://<decode-node-ip>:31501
```

The trailing URLs are the engines to flush before the run. Flushing matters:
blocks left by an earlier run inflate the hit rate, and `flush_cache` is a no-op
while requests are in flight — it still returns success. An efficiency **above**
100% is the signal that it did not take.

`NUM_PROMPTS` counts **conversations**, not requests: 60 conversations averaging
~7.5 turns is 448 requests. `docker.md` §7.3 explains how to read the score.

On the validated MI325X pair, this deployment served the workload identically to
the shell one:

| | Kubernetes | Docker + shell |
|---|---|---|
| Successful requests | 448 / 448 | 448 / 448 |
| Cache efficiency | 100.00% | 100.00% |
| Output throughput | 189.8 tok/s | 182.7 tok/s |
| Mean TPOT | 21.60 ms | 21.50 ms |
| Median TPOT | 18.34 ms | 18.57 ms |

---

## 9. The two operating points

The default recipe was tuned at concurrency 16, where DP-attention wins. At
concurrency 1 it inverts, and by a wide margin — the measurements are in
[`docker.md`](docker.md) §8, and they carry here because §0's flag comparison
showed the two deployments run the same engine.

| | Batch / high concurrency | Interactive / low concurrency |
|---|---|---|
| Aggregate throughput at concurrency 16 | **182.7 tok/s** | 139.4 tok/s |
| Per-user speed at concurrency 1, median | 62.5 tok/s/user | **133–159 tok/s/user** |
| Mean TTFT at concurrency 16 | 12.4 s | 22.2 s |

To move a deployment to the interactive point, edit both legs in the manifest:

- drop `--enable-dp-attention`
- set `--dp-size` to `1`
- set `--chunked-prefill-size` to `2048`

**The third is not optional.** `--chunked-prefill-size` is a global budget that
SGLang divides by `dp_size` **only while DP-attention is on**. At the default,
`8192` is 1,024 per rank; with DP-attention off the same `8192` is 8,192 per rank,
eight times the activation memory, and a long prefill at concurrency 16 goes
straight to `HSA_STATUS_ERROR_OUT_OF_RESOURCES`.

This ablation was run on the shell deployment, not on Kubernetes. To confirm it
here, apply the three edits and re-run §8 — expect the `docker.md` §8 figures.

---

## 10. Clean up

```bash
kubectl -n infera delete inferadeployment $CR
```

Deletion is asynchronous. Wait for `kubectl -n infera get pods` to come back empty
before starting anything else on those GPUs, or the two fight over them.

---

## 11. Troubleshooting

**Pods stay `Pending`.** Usually the `nodeSelector` hostname or `amd.com/gpu`
allocatable. `kubectl -n infera describe pod` says which.

**`ErrImageNeverPull` / `ImagePullBackOff` on an image you just built.** The
kubelet's runtime cannot see docker's image store, or the image was garbage
collected after import — §3.1 and §3.2.

**Everything answers, but `installTransport, type=tcp`.** Mooncake did not get
RDMA. Check the visible port count from inside the image, then the GID index —
§4.

**KV transfers time out on a fabric that looks healthy.** If your rails carry no
IPv4, Mooncake is probably pairing mismatched rails. Pin it to one — §4, case B.

**A long prompt comes back partially wrong, with nothing in any log.** The image
is missing the Mooncake early-send wait-event patch. Rebuild and re-run
`check_image.py` — §3.

**`Memory access fault by GPU node-N` after a clean-looking startup.** Either the
base image does not match the host driver (§2), or the pinned rail is down on one
node. For the latter, look thousands of lines earlier for `topology.cpp … is not
active`, `has no active ports, skipping`, `Skipping unavailable device`: Mooncake
skipped the only rail it was given, and the failure waits for the first real KV
transfer to surface as a memory fault.

**The kvd sidecar is OOMKilled (exit 137) right after graph capture.** kvd holds
two independent budgets; sizing the container limit to one of them is not enough.
Use the manifest's figure.

**Requests fail under sustained load, but single requests are fine.** Seen once,
on RKE2: 44 of 448 requests failed on a long-context workload with the router on
the Pod network, and failures went to zero when the router was moved to
`hostNetwork: true`. **The mechanism was never established** — an MTU explanation
was investigated and disproved, and the same recipe on a different cluster did not
reproduce it at all. So this is not a recommended default: `hostNetwork` claims
port 8000 on the node and steps outside NetworkPolicy. Treat it as a remedy to try
if you reproduce the symptom, and prefer finding the cause in your own CNI.

**`no active mixed worker for model="..."`.** A model-name mismatch, not a dead
fleet — the served name is the model path (§7). Every request fails in
milliseconds, which reads like an outage.

---

## Where these numbers come from

The four combos were validated on a 2× MI300X cluster; the MI325X figures in §8
and §9, the flag-by-flag comparison against the shell deployment, and the tuning
record are in [`mi325x-handoff/`](../../mi325x-handoff/README.md). Recipe-level
detail — the source patches, kvd sizing, and what each combo was checked
against — is in
[`examples/recipes/glm5.2-fp8-gfx942/`](../../examples/recipes/glm5.2-fp8-gfx942/README.md).
