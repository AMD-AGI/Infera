# GLM-5.3-MXFP4 — SGLang — Kubernetes recipe

GLM-5.3 "big" (`model_type: glm_moe_dsa`) served from a **stock** SGLang image with
the infera overlay mounted over it. One combo today:

| Combo | Serving | KV cache | Status |
|---|---|---|---|
| [`disaggregated/`](disaggregated/deploy.yaml) | 1 prefill + 1 decode, TP4 each | GPU only | **validated** on 2× MI355X |

Not here: aggregated — use `examples/sglang_mix_glm5.3/` under docker for now.

---

## What was validated

Two MI355X nodes, 4 cards per leg, `lmsysorg/sglang:v0.5.18-rocm720-mi35x`
unmodified, self-installed k3s v1.36.4, KV over mooncake RDMA on 8 `ionic` rails.

**Do not substitute another sglang base.** `patches/sglang_dsa/` is `--fuzz=0`
diffs cut against v0.5.18, and GNU patch is not atomic across files: on a
different release the set can land some files and reject others, and the engine
then starts and serves on a half-patched tree with nothing in any log saying so.
Measured on a 0.5.15 base. The baked image fails its build on a base bump; the
overlay has no such gate, so the pin is yours to hold.

```
/v1/workers -> prefill dp_size=4, decode dp_size=4
correctness  -> PASS, including a 7845-token needle recovered verbatim
                and 0/32 degenerate at concurrency 8
throughput   -> 348 tok/s output, ISL 8192 / OSL 1024 / conc 8
```

Identical, case for case, to the same recipe run under plain docker and to the
baked-image reference.

---

## Two things you must do before `kubectl apply`

### 1. Build the overlay payload from this repo

**The published `inferaimage/infera-overlay` tag will not work.** It ships
`patches/vllm/` only, and its `infera-exec` gates the patch loop on
`import vllm` — so on an SGLang base it applies **nothing**, and the engine runs
unpatched. On gfx950 the two that matter are a DP-attention host-sync deadlock and
an aiter paged-MQA row-sizing bug: a hang, or wrong output returned with a 200.

```bash
docker build -f deploy/overlay/Dockerfile.payload -t infera-overlay:glm53-sglang-patches .
```

Then **stage the payload tree onto every node that will run a leg**, and point
`PAYLOAD_DIR` at it. The manifest mounts it as a `hostPath`:

```bash
id=$(docker create infera-overlay:glm53-sglang-patches)
docker cp "$id:/payload/." /var/tmp/infera-payload/   # then rsync to each node
docker rm "$id"
```

**It must be node-local, not a shared mount.** The payload carries `.so` files
that get mmap'd, and an NFS-backed path is the wrong place for that.

Why a hostPath and not an initContainer copying the image into an `emptyDir`:
the initContainer is the tidier pattern, but it needs a registry the **cluster**
can pull from, and the image built above lives in the local docker store. Point
it at a tag the cluster cannot resolve and every pod sits in
`Init:ImagePullBackOff`. The hostPath form is the one with a measurement behind
it; switching to an initContainer against a real registry is a reasonable change,
but it is untested here.

Confirm it worked after the pods start — the head of a worker's log is
`infera-exec`'s own output and must show eight patches ending in
`=== all sglang DSA patches verified in bytecode ===`.

### 2. Decide about the GPU device plugin

This manifest ships **without** `amd.com/gpu` requests. It pins GPUs with
`HIP_VISIBLE_DEVICES` + `nodeSelector`, which is what the docker recipe does and is
therefore the lower-variance choice against the run this was validated in. It also
lets you place a leg on *specific* cards, which the device plugin cannot express.

If your cluster has the plugin and you would rather use it, add
`resources.requests/limits: {amd.com/gpu: 4}` to both worker legs and drop
`HIP_VISIBLE_DEVICES`. **Do not do half of that**: leaving `amd.com/gpu` in with no
plugin installed parks both pods in `Pending` forever on an extended resource
nothing provides, and leaving `HIP_VISIBLE_DEVICES` in *with* the plugin silently
masks GPUs, because the plugin renumbers the allocation from 0.

---

## Render and apply

Use `render.py`, not `sed` — it refuses to emit a manifest with a placeholder left
in it, and `kubectl` will happily accept `--disaggregation-ib-device
<RDMA_IB_DEVICES>` and fail minutes later in a message about a device.

```bash
cd examples/recipes
python3 render.py glm5.3/disaggregated \
  --set PREFILL_NODE=<node-a> --set DECODE_NODE=<node-b> \
  --set MODEL_DIR=/path/to/models \
  --set PAYLOAD_DIR=/var/tmp/infera-payload \
  --set RDMA_IB_DEVICES=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7 \
  --set PREFILL_GID_INDEX=1 --set DECODE_GID_INDEX=1 \
  -o /tmp/glm53-pd.yaml

kubectl create namespace infera
kubectl apply -f /tmp/glm53-pd.yaml
```

`MODEL_DIR` must hold `GLM-5.3-MXFP4/` at the **same path on both nodes**, and must
be the **resolved** path. A symlink that crosses an NFS mount boundary gives the
container an empty directory, and the failure surfaces much later as something
unrelated.

The two GID indexes are separate placeholders because the index is **per node** —
two identical machines routinely differ. Read them with `show_gids` on each node;
a wrong one fails loudly with `GID is NULL` on every DP rank.

Add `--pin-rail <rail> --check-rail` only if your rails carry **no IPv4** (every
GID link-local `fe80::`, so Mooncake cannot tell them apart). This recipe assumes
rails with IPv4, so it omits both — and correspondingly leaves `MC_TE_FILTERS`
unset, matching the docker recipe's mode-A configuration.

---

## The one thing that will silently ruin this

**DP-attention must be symmetric — `--dp-size 4` on BOTH legs.** Measured on this
checkpoint:

| shape | result |
|---|---|
| prefill dp1 → decode dp4 | first completion clean, then **every one garbage** |
| prefill dp4 → decode dp4 | 6/6 clean, 7845-token needle verbatim, 0/32 degenerate |

The failure is silent: zero faults, zero HIP errors, `MC_FORCE_TCP` 0, decode
batches rolling with cuda graphs on. Nothing in any log says the tokens are wrong.
Cause not diagnosed — the working configuration was adopted rather than the
mechanism chased.

**So the acceptance test must read the output, not the status code.** Use your own
correctness probe, and give it repeat cases and a small concurrent load round —
this failure appears *after* the first request, so a single-shot check passes a
broken deployment.

---

## Verifying a bring-up

```bash
# 1. both roles registered — /health returns 200 with only one leg up,
#    and every completion then returns 503
kubectl -n infera exec deploy/glm53-mxfp4-pd-server -- \
  curl -s http://127.0.0.1:8110/v1/workers | python3 -m json.tool | grep -E 'disagg_mode|dp_size'

# 2. the overlay patched the engine
kubectl -n infera logs <prefill-pod> | head -18

# 3. KV really moved over RDMA rather than falling back
kubectl -n infera logs <prefill-pod> | grep -c MC_FORCE_TCP     # want 0
kubectl -n infera logs <prefill-pod> | grep -ci 'gid.*null'     # want 0

# 4. correctness -- run your own probe against the forwarded port, and read the
#    generated text rather than the status code
kubectl -n infera port-forward svc/glm53-mxfp4-pd-server 18110:8110 &

# 5. MTP is actually accepting. Nothing above catches a mis-wired draft: the
#    deployment stays correct and simply gains nothing. Want a MEDIAN of 2-3.
kubectl -n infera logs <decode-pod> | grep -o 'accept len: [0-9.]*' \
  | awk '{print $3}' | sort -n \
  | awk '{a[NR]=$1; if ($1>=4) f++} END {printf "n=%d median=%s at4.00=%d\n", \
      NR, a[int(NR*0.5)+1], f+0}'          # reference: n=45 median=2.85 at4.00=0
```

Three ways step 5 misleads. A median at **4.00 is a failure**, not a win — the
ceiling is `--speculative-num-draft-tokens` and a draft pinned there is
predicting a repetition loop. Do **not** read `tail -5`: a few percent of batches
sit at 4.00 on a perfectly healthy leg. And **no output does not mean MTP is
off** — `decode_log_interval` defaults to 40, so a leg under 40 decode iterations
prints no such line at all; `grep -c 'Decode batch'` disambiguates, non-zero with
no `accept len:` means it really is off.

## Notes on the shape

- **TP4 per leg, not TP8.** That is the shape the symmetric-DP-attention result was
  measured in. TP8 is untested for GLM-5.3 in this configuration.
- **EAGLE MTP(3,1,4) on the decode leg.** Validated **on this manifest**: 6/6
  correct, 0/32 degenerate over three load rounds, acceptance-length median 2.85
  (n=45) with none at the 4.00 ceiling, and all seven DSA bytecode markers
  verified at container start. Also validated on the two docker paths, where the
  larger samples give a median of 2.35 (n≈505 each) and a clean single-variable
  A/B: **+36 % output throughput, −30 % TPOT** against the same deployment with
  MTP off. Read the acceptance *median*, never the last few lines: a few percent
  of batches sit at the 4.00 ceiling even on a healthy leg, and a median at 4.00
  is a repetition loop. Upstream's cookbook disables EAGLE on AMD; that line does
  not govern this architecture, whose PD+DPA+MTP path is exactly what
  `deploy/docker/patches/sglang_dsa/` exists for.
- **`mem-fraction-static` is deliberately asymmetric** (0.70 prefill / 0.85 decode)
  and the direction is counter-intuitive: a prefill OOM is fixed by *lowering* it.
- **`hostNetwork: true` is required, not tuning.** The RoCE rails are host
  interfaces; a CNI pod IP has no route to a peer's rail.
- **`privileged: true` is not redundant with `IPC_LOCK`.** There is a recorded case
  in this repo where `IPC_LOCK` plus `memlock=-1` was insufficient for RDMA
  registration and `--privileged` was the entire fix.
- The startup budget is **90 min** because 408 GB of weights on NFS can take that
  long. Do not size it from a warm-cache observation.

## If you have no cluster yet

Any Kubernetes will do. If you stand k3s up on a GPU node that is already running
other work, read the installer's shell source first — several of its hazards
(what it stops, what it rewrites) are visible only there.
