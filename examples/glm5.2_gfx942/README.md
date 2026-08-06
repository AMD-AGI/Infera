# GLM-5.2-FP8 SGLang PD on gfx942 — bring-up, verification, benchmark

Runnable package for GLM-5.2-FP8 on two gfx942 (MI300X) nodes: SGLang
prefill/decode disaggregation over Mooncake RDMA, DP-attention, EAGLE speculative
decoding, and Infera kv-aware routing. It covers bringing the service up, proving
it is actually correct, and sizing it with a simple serving benchmark.

Every script is driven by environment variables, so you should not need to edit
any of them; `env.sh` holds the defaults, and they are the tuned recipe below
rather than SGLang's defaults. The node names (`node-0`, `node-1`) are
placeholders — substitute your own.

KV offload below the GPU cache (`kvd`) ships here too, but **off by default** —
on the workload this was tuned against it cost 12% and served zero reads. §6
covers turning it on and how to tell whether your workload is one that wants it.

The agentic multi-turn benchmark that produced the numbers below is out of scope
here, and is being generalised into a standalone tool.

## Scripts

Half of these run on the host and half inside the long-lived engine container;
the column says which, and mixing the two is the most common way to get stuck.

| Script | Runs on | Purpose |
| --- | --- | --- |
| `build_image.sh` | host | Build the engine image from `deploy/docker/Dockerfile.sglang.gfx942`. |
| `preflight_rdma.sh` | host | RDMA preflight: container port visibility, plus an optional cross-node fabric check. |
| `host_container.sh` | host | Create / inspect / remove the long-lived engine container. |
| `launch/launch_etcd.sh` | host | Start etcd on the prefill node (PD's shared registry). |
| `launch/launch_kvd.sh` | container | Optional KV offload daemon; a no-op at the default `KVD=0` (§6). |
| `launch/launch_prefill.sh` | container | SGLang prefill leg, TP8 / DP8 + dp-attention. |
| `launch/launch_decode.sh` | container | SGLang decode leg, same shape. |
| `launch/launch_router.sh` | container | Infera kv-aware router. |
| `verify.sh` | container | Correctness checks; exits non-zero on any failure. |
| `bench.sh` | container | SGLang `bench_serving` on a random dataset, through the router. |
| `stop.sh` | container | Stop the router and engine processes on this node. |

## Topology

| Node | Role |
| :--- | :--- |
| `node-0` | etcd + router + prefill leg + verify/benchmark entrypoint |
| `node-1` | decode leg |

Both legs run TP8 / DP8 with `--enable-dp-attention` on 8 GPUs, and move the KV
cache to each other over one RDMA rail with Mooncake.

## The tuned recipe

These are the `env.sh` defaults. They were chosen on an **agentic multi-turn
trace** (32 conversations / 225 turns, concurrency 16, ~68k-token median input),
one axis at a time against a locked baseline:

| Setting | Value | What it bought |
| --- | --- | --- |
| `ROUTER_BACKEND` | `rust` | Same routing decisions as the python backend request for request, 27% faster end to end. |
| `CHUNK` | `8192` (1,024/rank) | The largest single lever: −23.8% duration, −34.4% TTFT against 16,384/rank. |
| `MTP_STEPS`/`TOPK`/`DRAFT_TOKENS` | `5`/`1`/`6` | −8.5% duration, −13.8% TPOT. Acceptance 4.64 against a 4.00 break-even. |
| `IB_DEVICE` | one rail | Striping KV over every NIC measured 11.9% *slower*; KV uses 4.5% of one 200 Gb/s port. |
| dp-attention | on both legs | Pure TP8 prefill measured 25.9% slower: concurrency beats per-request latency here. |
| `KVD` | `0` | The offload tier cost 12% and served zero reads on this trace (§6). |
| `MEM_FRAC` / `MAX_RUNNING` | `0.85` / `128` | Baseline values, unchanged by the sweep. |

Together those took that trace from 764 s to 420 s (−45%) and output throughput
from 64.8 to 118.0 tok/s. One metric moved the wrong way: ITL p90 rose 13.7%,
because a deeper draft emits tokens in burstier groups. That is free for batch
work and worth weighing for interactive streaming.

`bench.sh` runs a *different*, simpler workload and will not reproduce those
figures — see §5.

## 1. Prerequisites

### 1.1 Hardware / software

```text
Hardware: 2 nodes, 8x gfx942 (MI300X) each, RoCE between them, Docker with GPU access
Model:    GLM-5.2-FP8  (glm_moe_dsa, MLA + DSA indexer, 78 layers)
Image:    built here from deploy/docker/Dockerfile.sglang.gfx942
```

### 1.2 Model weights

`MODEL` must be a **local directory** — the scripts bind-mount it read-only:

```bash
export MODEL=/your/path/GLM-5.2-FP8
```

A HuggingFace cache path works too; `host_container.sh` detects the snapshot
symlinks and mounts the blobs alongside so they still resolve in the container.

### 1.3 Build the image

On both nodes (or build once and push). The image carries the ROCm hicache fixes
and the `infera-router` binary the rust backend needs, so do not substitute a
stock SGLang image:

```bash
bash build_image.sh
```

## 2. Adapt to your cluster

Export these on **both** nodes before running anything:

```bash
export PREFILL_IP=10.0.0.1        # node-0, on the data network
export DECODE_IP=10.0.0.2         # node-1, on the data network
export MODEL=/your/path/GLM-5.2-FP8
export IB_DEVICE=mlx5_0           # the RDMA rail the two nodes share
export MC_GID_INDEX=3             # RoCE GID index on that device
```

If your nodes resolve by name, `PREFILL_NODE` / `DECODE_NODE` derive the IPs
instead. The addresses must be the ones the peers can reach on the data network,
not a management NIC.

## 3. Verify the RDMA fabric

Cross-node PD moves the KV cache over the fabric on every request, and a mismatch
between the container's RDMA provider and the host driver degrades it to TCP
*silently* — the pair still answers, just slower than one node would be. Run this
on each node's host shell before bringing anything up:

```bash
bash preflight_rdma.sh
```

The reported count of active RDMA ports must match the node's, not be `0`. For
the cross-node netperf and Mooncake probes, set a shared `DUMP_PATH` and run one
task per node (see `infera/tools/preflight/README.md`).

## 4. Bring-up

```text
node-0 (host):      host_container.sh -> launch/launch_etcd.sh
node-1 (host):      host_container.sh
node-0 (container): [launch/launch_kvd.sh] -> launch/launch_prefill.sh -> launch/launch_router.sh
node-1 (container): launch/launch_decode.sh
node-0 (container): verify.sh -> bench.sh
```

The two legs discover each other through etcd, so the decode leg can start in
parallel with the prefill leg. The router only needs both to be registered by the
time it takes traffic. The bracketed step is a no-op unless you set `KVD=1`.

### 4.1 Containers

On both nodes' host shells:

```bash
bash host_container.sh
```

It checks the image, the weight mounts and the in-container imports before
reporting success, so a failure here is cheap compared to finding the same
problem four minutes into engine startup. Then, on `node-0` only:

```bash
bash launch/launch_etcd.sh
```

### 4.2 Engines and router

Enter the container on each node (`docker exec -it infera-glm52-gfx942 bash`),
then on `node-0`:

```bash
bash launch/launch_prefill.sh
bash launch/launch_router.sh
```

and on `node-1`:

```bash
bash launch/launch_decode.sh
```

**Cold start takes 15–25 minutes** — weights, then CUDA-graph capture. Do not
kill a slow launch. Follow it with `tail -f logs/prefill.log`.

### 4.3 Verify

On `node-0`, inside the container:

```bash
bash verify.sh
```

Every check targets a failure this stack produces *without* returning an error:

1. **Workers** — both legs registered in etcd.
2. **Correctness** — a padded prompt with a known answer. A broken KV hand-off
   does not return an HTTP error; the decode leg reads a corrupt prefix and
   produces fluent text unrelated to the prompt, which only an answer check
   catches.
3. **kv-aware steering** — the router logged a prefill pick with
   `request_blocks > 0`. Without block hashes it routes on load alone and looks
   perfectly healthy doing it.
4. **MTP** — the decode leg's `/metrics` carries `sglang:spec_accept_length`.
   Speculative decoding is dropped silently if the two legs disagree on its shape.
5. **kvd** — skipped at the default `KVD=0`; with the tier on, a fresh
   multi-page prompt must leave writes behind in the daemon's counters.
6. **RDMA hand-off** — Mooncake transport lines in the decode log.

### 4.4 Stop

Inside the container on both nodes:

```bash
bash stop.sh
```

Then, from the host, `bash host_container.sh --rm` and
`docker rm -f infera-glm52-etcd`. Remove the engine processes before relaunching
or the next run OOMs against VRAM the old one still holds.

## 5. Benchmark

```bash
bash bench.sh                              # defaults from env.sh
ISL=8192 OSL=512 CONC=32 bash bench.sh
for C in 8 16 32 64; do CONC=$C bash bench.sh; done   # concurrency sweep
```

Defaults are `ISL=4096 OSL=1024 CONC=16`, and `NUM_PROMPTS` follows `CONC` at four
waves (64 prompts by default), so raising the concurrency alone keeps the run
length roughly constant. Results land in `results/<tag>.json` and `.log`.

Two things to read correctly:

- **The cache-hit line will be ~0, and that is right.** `--dataset-name random`
  generates prompts that share no prefix, so a kv-aware router has nothing to
  reuse. This benchmark sizes raw serving throughput; measuring cache reuse needs
  a workload with real shared prefixes.
- **It will not reproduce the numbers in "The tuned recipe".** Those came from
  the agentic trace described there, whose inputs are ~17× longer and heavily
  prefix-shared. Use this to check the deployment is healthy and to compare
  concurrencies against each other, not against those figures.

## 6. Optional: KV offload below the GPU cache (`kvd`)

`KVD=1` runs an `infera-kvd` daemon beside the prefill engine and points SGLang's
hierarchical cache at it, so a prefix evicted from the GPU survives in pinned host
RAM (L2) and on node-local NVMe (L3) instead of being recomputed. Prefill leg
only: SGLang issues storage prefetch on its aggregated and prefill branches, never
on the decode branch, so a decode-side daemon would be write-only.

**It is off by default because it measured slower here.** On the agentic trace
above, `KVD=1` ran 12.0% slower and served *zero* reads while writing 100.8 GB:
the 54 GB-per-rank device pool already answered ~100% of the reuse that trace had
to offer, so every byte the tier stored was pure write cost. That is a property of
the workload, not of the tier — it earns its keep when the reuse horizon is longer
than the GPU pool can hold. Look at the prefill leg's cache hit rate first: if it
is already near its ceiling, offload has nothing left to catch.

To turn it on, set these before creating the container, on the prefill node:

```bash
export KVD=1
export KVD_L3_DIR=/mnt/nvme/kvd-l3   # node-local NVMe; NFS/weka classifies as buffered
export KVD_IO_MODE=direct            # pin it when you know that path is local NVMe
```

`KVD_L3_DIR` is bind-mounted, so an already-running container has to be recreated
(`bash host_container.sh --rm && bash host_container.sh`). Then, inside it, the
daemon goes up **before** the engine — the engine probes the socket at startup and
refuses to run without an answer:

```bash
bash launch/launch_kvd.sh
bash launch/launch_prefill.sh
```

`verify.sh` then asserts the tier actually stores pages, and `bench.sh` writes the
daemon's counters next to each result as `<tag>.kvd.json`. You can read them at any
time with:

```bash
python3 -m infera.kvd.statctl --socket /tmp/infera-kvd/kvd.sock
```

`hits_total` against `sets_total` is the whole question: writes with no hits over a
full run is exactly the 12% regression above. The remaining knobs — `KVD_RAM_BYTES`,
`KVD_LONG_BYTES`, `KVD_TABLESPACE_POOLS`, `HICACHE_SIZE` — are documented in
`env.sh`. For the same tier under Kubernetes, see
[`examples/recipes/glm5.2-fp8-gfx942/`](../recipes/glm5.2-fp8-gfx942/README.md).

## Notes & gotchas

1. **`CHUNK` is an aggregate, not a per-rank value.** Under dp-attention SGLang
   splits it `CHUNK / dp_size`, so the default `8192` runs 1,024 per rank. The
   engine log says `adjusted from … to …`, which reads like the setting was
   rejected — it was not, it was divided. This is the single most misread value
   in the recipe, so read back what actually took effect:

   ```bash
   curl -s $PREFILL_URL/get_server_info \
     | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["chunked_prefill_size"], "per rank x", d["dp_size"], "ranks")'
   ```
2. **Both legs must agree on the MTP shape.** SGLang rejects a disaggregated pair
   whose speculative config differs, so change `MTP_STEPS` / `MTP_DRAFT_TOKENS`
   in `env.sh` (which both read) rather than on one leg's command line.
3. **The rust router requires etcd discovery.** `infera.server` validates the
   supported subset before it execs the binary and fails with a pointer to
   `--router-backend python`. This example uses etcd, so it is inside that subset;
   a Kubernetes deployment is not, which is why the k8s recipe runs the python
   backend.
4. **kv-aware fails soft.** Without a tokenizer it warns once and routes on load
   alone. `launch_router.sh` refuses to start on that warning and `verify.sh`
   re-checks it, because a run scored against load balancing while labelled
   kv-aware is worse than a launch that stops.
5. **Advertise the data-network IP.** `--advertise-host` is what the peer dials
   for the Mooncake bootstrap handshake; a management-NIC address there fails at
   hand-off time, not at startup.
6. **`Ctrl-C` on a `tail -f` does not stop an engine.** The launch scripts run
   them under `nohup`; use `stop.sh`.
7. **`kvd` outlives a restart, on disk.** `stop.sh` kills the daemon after the
   engines, so nothing is pulled from under a live one, but L3 is journalled and
   is recovered on the next start. Delete `KVD_L3_DIR` to start cold.
