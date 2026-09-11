# GLM-5.3-MXFP4 on Kubernetes — PD disaggregated, stock image + overlay

Serve GLM-5.3 "big" across two GPU nodes as one prefill leg and one decode leg,
with KV moving between them over RDMA. The engine image is an **unmodified vendor
image**; everything infera adds arrives as a mounted overlay payload.

Validated on 2× MI355X (gfx950) under self-installed k3s. Recipe:
[`examples/recipes/glm5.3/disaggregated/deploy.yaml`](../../examples/recipes/glm5.3/README.md).

```{admonition} This page is for the `big` family only
:class: warning
This page covers `model_type: glm_moe_dsa` (`GLM-5.3` and `GLM-5.3-MXFP4`), which a
stock SGLang release already serves — its `config.json` is identical to GLM-5.2's
field for field except `transformers_version`. The overlay presupposes a stock base
that can already load the model.
```

## What you need

| | |
|---|---|
| nodes | two, on a **mutually routable RoCE fabric**. The KV transfer is RDMA; there is no TCP fallback. |
| GPUs | 4 free cards per node (TP4 per leg) |
| weights | `GLM-5.3-MXFP4/` (408 GB) at the **same resolved path** on both nodes |
| cluster | any Kubernetes. If you have none, see [standing up k3s safely](#if-you-have-no-cluster) |
| operator | the infera operator and the `InferaDeployment` CRD |
| engine base | **`lmsysorg/sglang:v0.5.18-rocm720-mi35x` — pinned, not a suggestion.** See below |

```{admonition} The overlay payload is version-bound to the engine base
:class: danger
The payload carries `patches/sglang_dsa/`, which is `--fuzz=0` diffs cut against
**sglang v0.5.18** and nothing else. GNU patch has no atomicity across files, so
on a different release the set can land some files and reject others — and the
engine then **starts, serves, and returns 200s on a half-patched tree**, with
nothing in any log saying it is inconsistent. Measured on a 0.5.15 base: six of
seven files applied, all seven hunks of one file rejected.

A baked image catches this at build time, because bumping its pinned base fails
the build. **The overlay has no such gate** — it patches whatever base it is
mounted over, at container start. Holding the pin is therefore your job, not the
tooling's. Tracked as a TODO in `deploy/overlay/Dockerfile.payload`.
```

## 1. The overlay payload — build it, do not pull it

This is the step that most obviously looks skippable and is not.

The published overlay image ships `patches/vllm/` only, and its `infera-exec` gates
the runtime patch loop on `import vllm`. On an SGLang base that gate is closed, so
**no engine patch is applied at all** — while the baked `Dockerfile.sglang` applies
six. On gfx950 the two that matter are a DP-attention host-sync deadlock and an
aiter paged-MQA row-sizing bug; the first hangs, the second returns wrong output
with a 200.

```bash
docker build -f deploy/overlay/Dockerfile.payload -t infera-overlay:glm53-sglang-patches .
```

Then **stage the payload tree onto every node that will run a leg** and point
`PAYLOAD_DIR` at it; the manifest mounts it as a `hostPath`:

```bash
id=$(docker create infera-overlay:glm53-sglang-patches)
docker cp "$id:/payload/." /var/tmp/infera-payload/   # then rsync to each node
docker rm "$id"
```

```{admonition} Node-local, not a shared mount
:class: warning
The payload carries `.so` files that get `mmap`'d. Put it on node-local disk.

A `hostPath` rather than an initContainer copying the image into an `emptyDir`:
the initContainer is tidier but needs a registry the **cluster** can pull from,
and the image built above lives in the local docker store. Point it at a tag the
cluster cannot resolve and every pod sits in `Init:ImagePullBackOff`.
```

The engine image, by contrast, *is* pulled by the nodes — and on k3s that is a
**separate image store** from docker's, which does not read
`/etc/docker/daemon.json`, so any mirrors configured there are invisible to it:

```bash
sudo k3s ctr -a /run/k3s/containerd/containerd.sock -n k8s.io \
     images pull docker.io/lmsysorg/sglang:v0.5.18-rocm720-mi35x
sudo k3s ctr -a /run/k3s/containerd/containerd.sock -n k8s.io images label \
     docker.io/lmsysorg/sglang:v0.5.18-rocm720-mi35x io.cri-containerd.pinned=pinned
```

```{admonition} Pin large images
:class: tip
Kubelet garbage-collects any image no Pod references once the node is above the
image-GC threshold, and the minimum-age protection is only **two minutes** — a
large image can disappear shortly after an import reports success. The `pinned`
label is a hard guarantee where a threshold is not.
```

## 2. Render and apply

```bash
cd examples/recipes
python3 render.py glm5.3/disaggregated \
  --set PREFILL_NODE=<node-a> --set DECODE_NODE=<node-b> \
  --set MODEL_DIR=/path/to/models \
  --set PAYLOAD_DIR=/var/tmp/infera-payload \
  --set RDMA_IB_DEVICES=<rails> \
  --set PREFILL_GID_INDEX=<n> --set DECODE_GID_INDEX=<m> \
  -o /tmp/glm53-pd.yaml

kubectl create namespace infera
kubectl apply -f /tmp/glm53-pd.yaml
```

`render.py` refuses to emit a manifest with a placeholder left in it. That matters
more than it sounds: `kubectl` rejects a literal `<NODE>` outright, but it
*accepts* `--disaggregation-ib-device <RDMA_IB_DEVICES>`, and that one only fails
once the engine opens the device — minutes later, in a message about a device
rather than about a placeholder.

**The two GID indexes are separate placeholders on purpose.** The index is per
node, and two identical machines routinely differ because an empty slot on one
shifts everything after it. Read them with `show_gids` on each node. A wrong one
fails loudly with `GID is NULL` on every DP rank.

**`MODEL_DIR` must be the resolved path.** A symlink that crosses an NFS mount
boundary gives the container an empty directory, and the failure surfaces much
later as something unrelated.

## 3. Verify — in this order

```{admonition} Do not trust `/health`
:class: warning
With only the prefill leg registered, `/health` returns **200** and every
completion returns **503**. Check `/v1/workers` instead.
```

```bash
# two workers, prefill and decode, both dp_size=4
kubectl -n infera exec deploy/glm53-mxfp4-pd-server -- \
  curl -s http://127.0.0.1:8110/v1/workers | python3 -m json.tool | grep -E 'disagg_mode|dp_size'

# the overlay actually patched the engine: eight patches, ending in a
# bytecode verification for the DSA group
kubectl -n infera logs <prefill-pod> | head -18

# KV moved over RDMA rather than silently falling back
kubectl -n infera logs <prefill-pod> | grep -c MC_FORCE_TCP    # want 0
kubectl -n infera logs <prefill-pod> | grep -ci 'gid.*null'    # want 0
```

Then check the **output**, not the status code. Forward the router port and run
your own correctness probe against it — one that repeats a request and includes a
small concurrent load round, for the reason in the warning below:

```bash
kubectl -n infera port-forward svc/glm53-mxfp4-pd-server 18110:8110 &
```

```{admonition} DP-attention must be symmetric, and asymmetry fails silently
:class: danger
`--dp-size 4` on **both** legs. Measured on this checkpoint: prefill dp1 → decode
dp4 gives a clean *first* completion and then garbage on every subsequent one —
streams of repeated `</think>` and digit noise, every request finishing on
`length`. Zero faults, zero HIP errors, decode batches rolling with cuda graphs on.
**Nothing in any log says the tokens are wrong.** That is why the probe must repeat
a request and run a small load round; a single-shot check passes a broken
deployment.
```

Reference result:

```
capital 19 tok stop | arith '391' | needle41 'ORANGE-4417-DELTA'
repeat, repeat2 stop | long7845 7845 tok stop | load 32@conc8 degenerate 0/32
==> VERDICT: PASS
```

Finally, confirm **MTP is accepting tokens** — a deployment with speculative
decoding mis-wired serves correctly and simply gains nothing, so nothing above
would catch it:

```bash
kubectl -n infera logs <decode-pod> | grep -o 'accept len: [0-9.]*' \
  | awk '{print $3}' | sort -n \
  | awk '{a[NR]=$1; if ($1>=4) f++} END {printf "n=%d median=%s at4.00=%d\n", \
      NR, a[int(NR*0.5)+1], f+0}'
```

Reference: `n=45 median=2.85 at4.00=0`.

```{admonition} Read the median, and know why 4.00 is bad
:class: warning
Healthy is a **median of 2–3** against a ceiling of 4, which is
`--speculative-num-draft-tokens`. A median at **4.00 is a failure**, not a win:
the draft is predicting a repetition loop perfectly, i.e. the model has
degenerated and MTP is faithfully extending garbage.

Do **not** read `tail -5`. A few percent of batches sit at 4.00 even on a healthy
leg, so the last few lines land on them routinely and misread as degenerate.

And an empty result does **not** mean MTP is off: `decode_log_interval` defaults
to 40, so a leg that has served fewer than 40 decode iterations prints no such
line at all. Disambiguate with `grep -c 'Decode batch'` — non-zero with no
`accept len:` means MTP really is off; zero means send more traffic.
```

## 4. What this shape is, and what it is not

| | |
|---|---|
| TP4 per leg | the shape the symmetric-DP-attention result was measured in. TP8 untested here. |
| EAGLE MTP(3,1,4), decode leg only | validated **on this manifest**: 6/6 correct, 0/32 degenerate over three load rounds, acceptance-length median 2.85 (n=45) with none at the 4.00 ceiling. Also validated on the baked and plain-overlay docker paths |
| kvd / hicache off | the overlay carries the ROCm hicache patches, but this arm does not enable them |
| `mem-fraction-static` 0.70 / 0.85 | deliberately asymmetric; a prefill OOM is fixed by *lowering* it |
| `hostNetwork: true` | required — RoCE rails are host interfaces and a CNI pod IP has no route to a peer's rail |
| `privileged: true` | required, and **not** redundant with `IPC_LOCK`: there is a recorded case where `IPC_LOCK` plus `memlock=-1` was insufficient for RDMA registration and `--privileged` was the entire fix |
| no `amd.com/gpu` | GPUs are pinned with `HIP_VISIBLE_DEVICES` + `nodeSelector`; see the recipe README before changing this |
| startup budget 90 min | 408 GB on NFS. Do not size it from a warm-cache observation. |

## Performance

ISL 8192 / OSL 1024, concurrency 8, 80 prompts, 2× MI355X:

| | docker + overlay | k3s + overlay |
|---|---|---|
| output throughput | 343.5 tok/s | 348.3 tok/s |
| median TTFT | 1336 ms | 1507 ms |
| mean TPOT | 20.62 ms | 20.64 ms |

```{admonition} Read the cache-hit rate before believing a TTFT number
:class: tip
`bench_serving --dataset-name random` reuses its dataset across runs. A repeat run
against a warm prefix cache gives a 449 ms TTFT — a 4× "improvement" that is
entirely an artifact. The figures above are at a 1.2 % cache-hit rate. Change
`--seed` between runs.
```

## If you have no cluster

Any Kubernetes will do. Standing k3s up on a node that is **already doing someone
else's work** — running docker workloads, under a scheduler that can drain it,
alongside another kubelet — is possible, but read the installer's shell source
first: several of its hazards are visible only there. Take a health snapshot of
the node before and after, so "I did no harm" is evidence rather than a claim.
