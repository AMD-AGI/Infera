# GLM-5.2-FP8 (gfx942)

Serve GLM-5.2-FP8 on gfx942 (MI300X / MI325X) with SGLang: TP8/DP8 with
DP-attention, MTP speculative decoding, fp8 KV, kv-aware routing, and optionally
prefill/decode disaggregation over Mooncake RDMA and KV offload to host RAM plus
node-local NVMe through `infera-kvd`.

```{admonition} This is the one recipe that does not use the overlay
:class: important
Every other recipe runs a stock vendor image with the infera overlay mounted in.
This one runs an **infera-built engine image**, because GLM-5.2 on the SGLang
v0.5.16 gfx942 base needs a rebuilt Mooncake `engine.so` and four SGLang source
patches — and a mounted payload can supply neither. The base bundles a Mooncake
that installs a HIP IPC transport and prefers it over RDMA, so cross-node PD dies
inside `hipIpcOpenMemHandle` on the first request; the image rebuild gates that
transport off and fails the build if the gate did not compile in.
```

## Which combo

| Combo | Nodes | RoCE fabric | MTP | Reach for it when |
|---|---|---|---|---|
| `aggregated` | 1 | no | yes | the default; start here |
| `aggregated + kvd` | 1 | no | **no** | one node, and requests share long prefixes |
| `disaggregated` | 2 | yes | yes | prefill and decode want different batching |
| `disaggregated + kvd` | 2 | yes | yes | both of the above |

```{admonition} `aggregated + kvd` is the one arm that runs without MTP
:class: warning
Not a tuning preference. **MTP and hicache both active on a worker that prefills
*and* decodes hangs** — in the scheduler control broadcast on the first long
prompt, or inside the model forward once concurrency starts. Five configurations
across three failure shapes were tried with MTP on; dropping MTP and changing
nothing else passed on the first attempt, with DP-attention and the overlap
scheduler still enabled. The manifest header lists all six.

The likely mechanism, though not proven: hicache keeps its prefetch and
write-back bookkeeping as per-rank local state while the collectives need every
rank in lockstep, and MTP adds a *second* per-rank host pool for draft tokens.

`disaggregated + kvd` keeps MTP because **speculative decoding only happens on
the decode leg** — the prefill leg's `spec_accept_length` is `0.0` on every rank —
so the combination never forms there. **If you need MTP on one node, use
`aggregated`. If you need the tier with MTP, use `disaggregated + kvd`.**
```

**Start with `aggregated`.** It needs one box and touches neither Mooncake nor
`/dev/infiniband` nor a GID index, so the whole RDMA class of silent failures
cannot apply. That also makes it the right thing to bring up when something is
wrong and you do not yet know which layer it is in: it separates "can this image
load these weights and serve correctly" from "can KV cross this fabric", and those
two fail in very different ways.

**Add `kvd` only once you have a reason.** hicache is `write_through`, so every
byte the tier absorbs is paid for on the prefill path whether or not anything reads
it back. Every measurement of it so far — on this recipe and on the `docker`
deployment it came from — has been a net loss, in each case because the workload
had no reuse left for the tier to serve: the A/B in §7 cost **−43% throughput and
×3.45 TTFT** while serving zero reads. That says nothing about a workload with
prefix reuse, and everything about deploying it without one. Read §6 first.

All four combos run TP8 / DP8 with DP-attention; the grid otherwise differs only
on one node vs two and GPU-only KV vs tiered, plus the MTP exception above.

## 1. Read the host driver version before you build

A container brings its own ROCm userspace but **cannot** bring a kernel driver — it
talks to the host's amdgpu through `/dev/kfd`, and AMD supports that pairing only
inside a bounded window. Which base image you build on is therefore a property of
your nodes, not a preference:

```bash
dpkg -l | grep -E 'amdgpu-dkms|rocm-core'    # on BOTH nodes
```

| Host amdgpu | Supported ROCm userspace | Base to build on |
|---|---|---|
| 6.3.x | ≤ 7.0.x | `v0.5.16-rocm700-mi30x` |
| 6.4.x (e.g. 6.14.14) | ≤ 7.2.x | `v0.5.16-rocm720-mi30x` (the Dockerfile default) |

```{admonition} Outside the window, nothing refuses to start
:class: danger
The image initialises, loads weights, captures CUDA graphs, serves a health check —
and then faults with `Memory access fault by GPU node-N` somewhere under load. On
gfx942 those faults landed in three unrelated places (the DSA indexer, the EAGLE
draft path, an aiter FP8 MoE kernel), each with a stack that reads like a genuine
engine bug. Two of them had a patch written for them before anyone compared
version numbers. Several unrelated code paths faulting at once is itself the
signal that the problem is environmental.
```

## 2. Build and verify the engine image

```bash
# 6.4.x host driver — the Dockerfile default
docker build -f deploy/docker/Dockerfile.sglang.gfx942 -t infera:sglang-gfx942-glm52 .

# 6.3.x host driver — override the base
docker build -f deploy/docker/Dockerfile.sglang.gfx942 \
  --build-arg SGLANG_BASE_IMAGE=lmsysorg/sglang:v0.5.16-rocm700-mi30x \
  -t infera:sglang-gfx942-glm52 .
```

Load it on **both** nodes, or push it somewhere the cluster can pull from; the
manifest uses `imagePullPolicy: IfNotPresent`, so a locally-loaded image is used
as-is. Then read the patch markers back **out of the built image** rather than
trusting the build log — a patch whose anchor moved no-ops silently:

```bash
docker run --rm --entrypoint bash infera:sglang-gfx942-glm52 -c '
P=$(python3 -c "import sglang, os; print(os.path.dirname(sglang.__file__))")
for m in GLM52_ROCM_HOST_ALLOC GLM52_ROCM_STAGED_WRITE_BACK GLM52_P1V3; do
  grep -rql "$m" "$P" && echo "ok      $m" || echo "MISSING $m"
done'
```

Confirm RDMA is visible from inside the image too, on both nodes. Zero visible
ports is the nastiest failure here: `ibv_get_device_list()` returns nothing,
Mooncake falls back to TCP, and the deployment still comes up green.

```bash
docker run --rm --network host --device=/dev/infiniband --cap-add=IPC_LOCK \
  --entrypoint bash infera:sglang-gfx942-glm52 -c 'ibv_devinfo | grep -c PORT_ACTIVE'
```

## 3. Prerequisites

Every engine Pod wants 8× gfx942 and `cpu: 32`. What differs per combo is how many
nodes, how much host RAM, and whether NVMe and a RoCE fabric are needed at all:

| Combo | Nodes | Host RAM on the busiest node | Node-local NVMe | RoCE fabric |
|---|---|---|---|---|
| `aggregated` | 1 | ~272 GiB | no | no |
| `aggregated + kvd` | 1 | ~670 GiB | yes | no |
| `disaggregated` | 2 | ~272 GiB | no | yes |
| `disaggregated + kvd` | 2 | ~670 GiB | yes | yes |

The kvd combos want roughly 512 GiB for the engine, 136 GiB for kvd and 16 GiB for
the router.

```{admonition} `--hicache-size` is per TP rank, not per worker
:class: warning
At `--hicache-size 32` every one of the 8 schedulers logs `Allocating 32.00 GB host
memory for hierarchical KV cache`, so the host tier is **256 GB**. Dropping
DP-attention does not reduce it, which is worth knowing because it looks like it
should. Sizing the container as if TP8 meant one 32 GB pool gets it OOMKilled
(exit 137) immediately after graph capture — a crash that reads like an engine bug
and is a budget mistake.
```

kvd's own 136 GiB looks oversized next to its `--max-bytes 64G` and is not: it
holds two independent budgets, an inline store and an `mlock`'d arena that is never
reclaimable. Size the limit to one of them and the OOM killer takes the sidecar
mid-run, which reads as a kvd bug.

`/dev/shm` sits on top of those and is charged to no limit — TP8 needs a large one,
which is what `hostIPC` supplies.

**Kubernetes 1.29+, for the two kvd combos.** The kvd daemon runs as a *native*
sidecar (`initContainers` with `restartPolicy: Always`), which is what makes it
pass its `startupProbe` before the engine starts. That ordering is load-bearing,
not tidiness: the engine probes the kvd socket once with a 5 s timeout and refuses
to start if nothing answers. As an ordinary container it becomes a race. The two
non-kvd combos have no such requirement.

**Weights on a `hostPath`** at the same path on every node the deployment uses.

```{admonition} Mount the directory the HuggingFace symlinks resolve into
:class: warning
A HF cache snapshot is a tree of relative symlinks into `blobs/`. Mount only the
snapshot directory and every link dangles, and `transformers` rejects the model
with `Should have a model_type key in its config.json` — four minutes into
startup, and nowhere near its actual cause.
```

## 4. Deploy

The manifests ship placeholders rather than defaults, because the RDMA values have
no correct default. Any placeholder left unsubstituted fails loudly, which is the
intent.

```bash
kubectl create namespace infera --dry-run=client -o yaml | kubectl apply -f -
```

::::{tab-set}

:::{tab-item} Aggregated
:sync: aggregated

One node, one worker doing prefill and decode. Two placeholders, no fabric.

```bash
sed -e "s|<NODE>|node-a|" -e "s|<MODEL_DIR>|/mnt/models|" \
    examples/recipes/glm5.2-fp8-gfx942/aggregated/deploy.yaml | kubectl apply -f -
```

Deploys as `glm52-fp8-mixed`, with one engine service called `worker`.
:::
:::{tab-item} Aggregated + kvd
:sync: aggregated-kvd

Adds the kvd tiered cache: an L2 arena in pinned host RAM and L3 on node-local
NVMe. This arm runs **without MTP** — see the warning at the top of this page for
why, and what it costs.

```bash
sed -e "s|<NODE>|node-a|" -e "s|<MODEL_DIR>|/mnt/models|" \
    -e "s|<KVD_L3_DIR>|/mnt/nvme/kvd-l3|" \
    examples/recipes/glm5.2-fp8-gfx942/aggregated-kvd/deploy.yaml | kubectl apply -f -
```

Deploys as `glm52-fp8-mixed-kvd`.
:::
:::{tab-item} Disaggregated
:sync: disaggregated

Prefill and decode on separate nodes, KV handed over by Mooncake RDMA.

```{admonition} Two nodes on a routable RoCE fabric
:class: warning
The KV handoff is RDMA with **no TCP fallback**. On one box this cannot work — use
`aggregated`.
```

```bash
sed -e "s|<PREFILL_NODE>|node-a|"   -e "s|<DECODE_NODE>|node-b|" \
    -e "s|<MODEL_DIR>|/mnt/models|" -e "s|<RDMA_IB_DEVICES>|mlx5_0|" \
    -e "s|<PREFILL_GID_INDEX>|3|"   -e "s|<DECODE_GID_INDEX>|3|" \
    examples/recipes/glm5.2-fp8-gfx942/disaggregated/deploy.yaml | kubectl apply -f -
```

Deploys as `glm52-fp8-pd`, with engine services `prefill` and `decode`.
:::
:::{tab-item} Disaggregated + kvd
:sync: disaggregated-kvd

Disaggregated, with kvd on the prefill leg. There is deliberately none on decode:
SGLang issues storage prefetch on its aggregated and prefill branches only, so a
decode-side tier would be written and never read.

```bash
sed -e "s|<PREFILL_NODE>|node-a|"        -e "s|<DECODE_NODE>|node-b|" \
    -e "s|<MODEL_DIR>|/mnt/models|"      -e "s|<KVD_L3_DIR>|/mnt/nvme/kvd-l3|" \
    -e "s|<RDMA_IB_DEVICES>|mlx5_0|" \
    -e "s|<PREFILL_GID_INDEX>|3|"        -e "s|<DECODE_GID_INDEX>|3|" \
    examples/recipes/glm5.2-fp8-gfx942/disaggregated-kvd/deploy.yaml | kubectl apply -f -
```

Deploys as `glm52-fp8-pd-kvd`.
:::

::::

On the disaggregated combos, `<RDMA_IB_DEVICES>` comes from `ibv_devices` — a rail
that is physically down must not be listed. The two GID indices come from
`show_gids <dev>`, the entry whose type is `RoCE v2`. There are **two** placeholders
because the index is per node, not per cluster; two identical machines routinely
expose different ones. They are usually equal, but check both: a wrong index pins KV
to an interface that never carries it and the transfer simply times out.

```{admonition} Cold start is 10–25 minutes, and the log goes quiet
:class: tip
Weights land in about 3.5 minutes. Everything after that — draft weights, memory
pools, `tilelang` and `aiter` JIT, CUDA-graph capture — prints almost nothing for
ten minutes or more, on a fresh container every time, since the JIT cache does not
survive the Pod. The workers use a `startupProbe` with a 90-minute budget and no
readiness probe for exactly this reason. Don't kill a slow load.
```

Each combo deploys under its own name, so the Service and label selectors below
take the deployment name and engine service as variables:

| Combo | `InferaDeployment` | Engine service(s) |
|---|---|---|
| `aggregated` | `glm52-fp8-mixed` | `worker` |
| `aggregated + kvd` | `glm52-fp8-mixed-kvd` | `worker` |
| `disaggregated` | `glm52-fp8-pd` | `prefill`, `decode` |
| `disaggregated + kvd` | `glm52-fp8-pd-kvd` | `prefill`, `decode` |

```bash
CR=glm52-fp8-mixed        # or glm52-fp8-pd-kvd, etc.
SVC=worker                # or prefill

kubectl -n infera get pods -w
kubectl -n infera logs -f -c main \
  -l infera.amd.com/deployment=$CR,infera.amd.com/service=$SVC
```

## 5. Smoke test

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

The manifests pass no `--served-model-name`, so the served name **is** the model
path. Expect `3937` — necessary, and not sufficient in two different ways.

**On the disaggregated combos, check the transport.** RDMA that failed to
initialise does not stop the deployment: Mooncake falls back to TCP and everything
still answers. Eight `installTransport, type=rdma`, one per DP rank, is the pass.

```bash
kubectl -n infera logs -c main \
  -l infera.amd.com/deployment=$CR,infera.amd.com/service=decode \
  | grep -aE 'GID index|installTransport'
```

```{admonition} On every combo, send a prompt longer than one chunk
:class: warning
Without the mooncake early-send wait-event patch, every prefill chunk but the last
is read while the forward pass is still writing it — multi-chunk prompts come back
**partially wrong, with nothing in any log**. A short prompt that answers correctly
cannot see that. Bury a distinctive needle at the head, middle *and* tail of a
prompt several times `--chunked-prefill-size` long and ask for it back: losing only
the head reads as "it works" if you happen to probe the tail.
```

## 6. kvd

kvd runs on the worker that prefills — the `worker` on `aggregated + kvd`, the
`prefill` leg on `disaggregated + kvd`. There is deliberately none on a decode
leg: SGLang issues storage prefetch on its aggregated and prefill branches only,
so a decode-side tier would be written and never read, and infera refuses to wire
it up even if handed the socket.

```bash
SVC=worker    # or prefill, on the disaggregated combo
POD=$(kubectl -n infera get pod -o name \
  -l infera.amd.com/deployment=$CR,infera.amd.com/service=$SVC | head -1)
kubectl -n infera exec $POD -c kvd -- \
  python3 -m infera.kvd.statctl --socket /tmp/infera-kvd/kvd.sock
```

`sets_total` climbing means the engine writes to kvd; `gets_total` / `hits_total`
climbing means it reads back. Writes alone prove half the path. Two counters that
mislead: `misses_total` counts *failed* gets only, so `0 misses` is fully
compatible with L3 having served nothing — read the scorer's `cached tokens by
tier` instead. And `entries: 0` on a healthy-looking deployment means kvd rejected
the KV layout; oversize values are rejected rather than split, so grep the kvd
container for `value_exceeds_largest_pool`.

```{admonition} `sets_total` is not "what this run wrote"
:class: warning
On this deployment it keeps climbing at a steady ~24/s with **no traffic at all**:
18944 when the benchmark finished, 46588 twenty-four minutes later, nothing sent in
between, while every SGLang-side counter stayed frozen. The 8 schedulers are the
socket's only peers, so the engine is issuing them; the root cause is not yet
known. Read `sets_total > 0` as "the write path works" and nothing more, and read
`long_bytes` as current residency rather than cumulative volume — it decreases too,
with `evictions_total` at 0, because a rewritten key overwrites its slot.

The same queue makes **`POST /flush_cache` permanently unavailable on a kvd leg**:
`is_fully_idle()` requires hicache's `ongoing_backup` to be empty, so the endpoint
returns 400 even with `?timeout=90` and nothing in flight.
```

```{admonition} L2's slot size is fixed by the first put
:class: note
The shared arena sizes its slot grid on the *first* blob it accepts and refuses
everything larger for the process lifetime, warning exactly once. This model
writes two sizes — the main MLA latent at 44928 B and the DSA indexer at
~10352 B — and whichever lands first wins. The mixed worker gets the small one
first on every cold start observed, which locks the main KV out of L2 for the
process lifetime. Independent of MTP: the no-MTP arm logs the same
`slot_size=10368`. Whether the prefill leg differs is unverified — assume it does
not until its kvd sidecar log says otherwise.

This costs performance, not correctness. Refused blobs fall back to inline
storage in kvd's heap, and **L3 is unaffected** — the long-region write is a
separate branch that reads the value from either location. What is lost is the
zero-copy read path and the usefulness of the `mlock`ed arena, which ends up
holding only indexer blobs. `--shared-arena-bytes 0` disables the arena and
reclaims that RAM.
```

```{admonition} kvd plus MTP needs `--hicache-io-backend direct`
:class: important
The manifest already passes it. It is not tuning: SGLang's default `kernel`
write-back path requires every host pool's stride to be a multiple of 8, and MTP's
draft pool at `page_size 1` has a 132-byte stride. The guard raises instead of
falling back, so without the flag the first request to write back kills the
prefill scheduler with `ValueError: Unsupported IO backend: kernel`. Note the
shape of that failure — a clean Python exception with a line number — as against
the `Memory access fault` class in §1. Telling those two apart matters, because
this deployment has been misdiagnosed in the other direction once already.
```

## 7. What was validated, and what was not

All four combos are validated on Kubernetes, on 2 × 8 MI300X under RKE2 1.35, with
the same image and the same judges — so the arms are comparable to each other.
Each served 232 requests across conc 1/8/16/32 with **zero failures, zero GPU
faults and zero Pod restarts**.

| Combo | Status |
|---|---|
| `aggregated` | **validated.** Cold start 20.7 min, `3937` correct, 407.5 output tok/s at conc 32, MTP accept length 3.31–5.11 on all 8 ranks |
| `aggregated + kvd` | **validated, without MTP** (see the top of this page). Cold start 21.5 min, 233.8 output tok/s at conc 32. The only arm observed *reading* from kvd: `gets_total 370 / hits_total 370 / misses_total 0` |
| `disaggregated` | **validated.** Cold start 21.4 min, 740.9 output tok/s at conc 32, decode-leg accept length 3.00–5.50 |
| `disaggregated + kvd` | **validated.** Cold start 21.2 min, 421.2 output tok/s at conc 32, accept length 2.30–4.81. The write path works and L3 replayed its journal across a Pod restart — see the counter caveats in §6 before quoting a volume from it |

Each of these fails quietly, so each was checked rather than assumed: a 9-chunk
(~9.8k token) prompt's needle came back intact from the head, middle **and** tail
on all four; all 8 decode ranks logged `installTransport, type=rdma` on the pinned
rail; MTP was confirmed to be *accepting* rather than silently degrading to one
token per step; and the kvd sidecar passed its `startupProbe` before `main`
started on every bring-up.

The read confirmation is worth singling out. `gets_total` stayed at 0 on every
other arm, which is compatible with two very different things — "the workload had
no reuse" and "the read path is broken and nobody noticed". Three 9-chunk prompts
sharing a haystack separated them: 370 gets, 370 hits, 0 misses, and
`HiCache prefetch success … loaded=185` in the engine log.

Two A/Bs came out of it, each isolating one axis on the same day, image, rail and
judges, at ~1700 in / 256 out with **no prefix reuse**:

```{admonition} PD buys ITL and spends TTFT — and scales sublinearly
:class: note
At conc 32, `disaggregated` on 16 GPUs against `aggregated` on 8: throughput
740.9 vs 407.5 tok/s (**×1.82** for 2× the hardware), ITL 24.60 vs 65.09 ms
(**2.6× better**), TTFT p50 3.476 vs 1.419 s (**2.4× worse**). The decode node no
longer stops to prefill, which is the ITL win; the prompt must prefill elsewhere
and ship its KV across the fabric first, which is the TTFT cost. PD is not a
per-GPU efficiency win here — choose by which end your SLO sits on, not by total
throughput.
```

```{admonition} kvd's cost, measured; its benefit, still not
:class: warning
At conc 32, `disaggregated + kvd` against `disaggregated`: throughput 421.2 vs
740.9 tok/s (**−43.2%**), TTFT p50 11.984 vs 3.476 s (**×3.45**), and
`gets_total` **0** for the whole run. `write_through` makes the prefill leg write
every page it produces, so the cost lands on TTFT and lands consistently (×2.26 /
×1.57 / ×3.83 / ×3.45 at conc 1 / 8 / 16 / 32). ITL *improves* 18%, and that is
not a benefit — the decode leg is receiving fewer requests per second, which is
the same fact as the −43%. This pair is the price of the tier.
```

Four boundaries on all of the above:

- **kvd's benefit is unmeasured** — only its cost is. Its read path is confirmed
  to work, but no benchmark here exercises it: the load generator sends a distinct
  prompt per request, so the tier had nothing to serve during any sweep. Read the
  A/B above as what kvd costs, not as what it does.
- **Aggregate RDMA bandwidth is unmeasured.** The validating cluster's eight 400G
  rails carry no IPv4, so all their RoCEv2 GIDs sit in one `fe80::/64` and
  Mooncake cannot pair them; KV ran pinned to a single 233 Gb/s rail. Correctness
  is unaffected, but nothing here supports a claim about KV transport not being
  the bottleneck.
- **The `rocm720` default is validated only in `docker` form.** That cluster's
  host driver is 6.3.x, so every Kubernetes result above ran the `rocm700` base
  from §1.
- **Neither comparison isolates one variable, and the sweep tops out at conc 32.**
  The PD pair is not equal-hardware (8 GPUs against 16; a 4+4 split would
  introduce variables of its own and was not run), and the single-node arms differ
  by MTP as well as by kvd. Nothing here speaks to saturation either.

## Source

[`examples/recipes/glm5.2-fp8-gfx942/`](https://github.com/AMD-AGI/Infera/tree/main/examples/recipes/glm5.2-fp8-gfx942)
— its README carries the full reasoning: why each of the four source patches
exists and what breaks without it, and a row-by-row table of every difference
between this manifest and the `docker` deployment it came from.

[The same topology on MI355X, in `docker` form](https://github.com/AMD-AGI/Infera/tree/main/examples/sglang_1p1d_glm5.2)
· [all recipes](index)
