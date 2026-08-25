# GLM-5.2-FP8 SGLang PD on gfx942 — bring-up, verification, benchmark

Runnable package for GLM-5.2-FP8 on two gfx942 (MI300X) nodes: SGLang
prefill/decode disaggregation over Mooncake RDMA, DP-attention, EAGLE speculative
decoding, and Infera kv-aware routing. It covers bringing the service up, proving
it is actually correct, and sizing it with a simple serving benchmark.

Every script is driven by environment variables, so you should not need to edit
any of them; `env.sh` holds the defaults, and they are the tuned recipe below
rather than SGLang's defaults. Your cluster's values go in one file, `cluster.env`
(§2), which `env.sh` reads first.

KV offload below the GPU cache (`kvd`) ships here too, but **off by default** —
on the workload this was tuned against it cost 12% and served zero reads. §6
covers turning it on and how to tell whether your workload is one that wants it.

The agentic multi-turn benchmark that chose the recipe ships here too (§5.2), and
drives either this deployment or the Kubernetes one in
[`examples/recipes/glm5.2-fp8-gfx942/`](../recipes/glm5.2-fp8-gfx942/README.md)
through the same client, so the two are comparable.

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
| `bench.sh` | container | SGLang `bench_serving` on a random dataset, through the router. Sizes the deployment (§5.1). |
| `run_sweep.sh` | container | `bench.sh` across concurrency 1–128, with a fresh seed and a flush per point. |
| `weka_to_agentic_trace.py` | container | Build the agentic multi-turn dataset from the public corpus (§5.2). |
| `run_agentic_trace.sh` | container | Replay that dataset through the router and score it (§5.2). |
| `score_agentic_trace.py` | container | Recompute the cache metrics `bench_serving` gets wrong in multi-turn mode. |
| `bench_client.sh` | host | Run the agentic bench from outside any engine container — for the k8s deployment, or a third machine. |
| `stop.sh` | container | Stop the router and engine processes on this node. |
| `check_image.py` | container | Read the patch markers out of a built image before trusting it. |

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
| `ROUTER_BACKEND` | `rust` | Same routing decisions as the python backend request for request, 27% faster end to end **on this shape** — the same comparison on the k8s recipe's aggregated shape came out within ±5% (note 4 below). |
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

Everything cluster-specific lives in one file. On **both** nodes:

```bash
cp cluster.env.example cluster.env    # then edit
```

`env.sh` reads it before its own defaults, and every script sources `env.sh`, so
there is nothing to remember at the call site and nothing else to edit. The repo
is bind-mounted at the same path inside the container, so one copy serves the host
and the container both.

`cluster.env.example` carries, for each value, the command that finds it rather
than a value to guess. The three that fail *silently* when wrong are
`MC_GID_INDEX` (§3), the advertised IPs (gotcha 7) and `ETCD_ENDPOINT` — worth the
two minutes each.

If your nodes resolve by name, `PREFILL_NODE` / `DECODE_NODE` derive the IPs
instead. The addresses must be the ones the peers can reach on the data network,
not a management NIC.

Every script that dials one of them refuses to start until both resolve, rather
than defaulting the missing one — see gotcha 7 for what that default would cost.

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

Two things to confirm by hand before the 20-minute bring-up, because both fail the
same unhelpful way — see gotcha 10.

**`IB_DEVICE` must be ACTIVE on both nodes**, not just present. Both legs pin to
one rail, so a rail that is down on either node takes the deployment with it:

```bash
for d in /sys/class/infiniband/*; do
  echo "$(basename "$d") $(cat "$d/ports/1/state")"
done
```

Then confirm the rail you picked carries traffic between the two nodes, which
liveness alone does not tell you:

```bash
ib_write_bw -d "$IB_DEVICE" -x "$MC_GID_INDEX"            # on the decode node
ib_write_bw -d "$IB_DEVICE" -x "$MC_GID_INDEX" "$DECODE_IP"  # on the prefill node
```

**Your fabric may need more than `IPC_LOCK`.** `host_container.sh` grants
`IPC_LOCK`, `SYS_PTRACE` and `memlock=-1`, which is enough on some clusters and not
on others. If PD warmup dies where gotcha 10 describes, pass
`EXTRA_DOCKER_ARGS=--privileged` when creating the container.

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

Two workloads, and they answer different questions. `bench.sh` sizes raw serving
throughput on random prompts; `run_agentic_trace.sh` replays the multi-turn trace
the recipe was tuned on and is the only one of the two that can say anything about
the cache or the kv-aware router.

### 5.1 Random dataset — sizing

```bash
bash bench.sh                              # defaults from env.sh
ISL=8192 OSL=512 CONC=32 bash bench.sh
bash run_sweep.sh                          # concurrency 1..128, fresh seed per point
```

Defaults are `ISL=4096 OSL=1024 CONC=16`, and `NUM_PROMPTS` defaults to four waves
of `CONC` — derived inside `bench.sh` on each call, so raising `CONC` re-derives
it. Results land in `results/<tag>.json` and `.log`.

Two things to read correctly:

- **The cache-hit line will be ~0, and that is right.** `--dataset-name random`
  generates prompts that share no prefix, so a kv-aware router has nothing to
  reuse. This benchmark sizes raw serving throughput; measuring cache reuse needs
  a workload with real shared prefixes.
- **It will not reproduce the numbers in "The tuned recipe".** Those came from
  the agentic trace below, whose inputs are ~17× longer and heavily prefix-shared.
  Use this to check the deployment is healthy and to compare concurrencies against
  each other, not against those figures.

Sweep with `run_sweep.sh` rather than a bare loop: at a fixed seed each point's
prompt set is a *superset* of the one below it and the radix tree still holds the
smaller one, which reads as ~50% cache hits at every point from `CONC=16` up.

### 5.2 Agentic trace — the workload that chose the recipe

The corpus is [`semianalysisai/cc-traces-weka-062126-256k`][corpus] (Apache-2.0),
Claude Code agent traffic. It carries per-turn token counts and KV block ids but
no text, so `weka_to_agentic_trace.py` synthesises filler while preserving what
matters: exact per-turn lengths and block-level prefix reuse. Build it once, in
the container on the prefill node:

[corpus]: https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k

```bash
hf download semianalysisai/cc-traces-weka-062126-256k --repo-type dataset
SRC=$(ls "$HF_HOME"/hub/datasets--semianalysisai--cc-traces-weka-062126-256k/snapshots/*/traces.jsonl)

python3 weka_to_agentic_trace.py "$SRC" -o "$TRACE" \
  --output-len "$OUTPUT_LEN" --min-turns 4 --max-context 100000 \
  --verify 20 --tokenizer "$MODEL"
```

`--max-context` fits the corpus to what the deployment can prefill; `--dry-run`
reports the resulting distribution without writing, which is the cheap way to pick
it. At 100000 this yields 295 conversations with a p50 peak context of 78,848
tokens, and `--verify` reproduces every checked turn's length exactly.

Then run it:

```bash
NUM_PROMPTS=60 CONC=16 bash run_agentic_trace.sh docker
```

`NUM_PROMPTS` counts **conversations**, not requests — 60 conversations averaging
~7.5 turns is 448 requests. The script reads the served model name and the KV page
size off the server rather than assuming them, flushes both legs, replays the
trace and rescores it.

To drive a deployment you cannot `docker exec` into — the Kubernetes one, or from
a third machine — use the host-side wrapper, which starts the same client in a
throwaway container so the tokenizer, dataset, concurrency limiter and scorer stay
identical across arms:

```bash
bash bench_client.sh k8s http://<router-ip>:8000 \
     http://<prefill-node>:30001 http://<decode-node>:31501
```

### 5.3 Read the agentic result

`sglang.benchmark.serving` mis-reports the input side in multi-turn mode: it keeps
the conversation-level `prompt_len` for every turn, so its own summary can print
`Total input tokens: 0` next to a cache hit rate above 100%. Its per-request
`cached_tokens` come from the server and are correct, so `score_agentic_trace.py`
recomputes against the dataset's verified per-turn lengths and reports the number
worth comparing across tools — **efficiency**, actual hits over what a cache that
evicted nothing could have returned:

```text
  actual hit rate              84.61 %
  ideal  hit rate              84.61 %
  efficiency (a/i)            100.00 %
  tokens lost to evict              0 (0.00% of ideal)
```

Efficiency at 100% with no eviction means the run is below the pressure point and
kv-aware routing has nothing to distinguish itself on; raise `CONC` until eviction
appears. Efficiency **above** 100% means the flush did not take — `flush_cache` is
a no-op while requests are in flight and still returns success.

A run with any failed request is refused rather than scored: a failure is recorded
with `cached_tokens=0`, so including it drags the hit rate down and a dead worker
reads as a cache problem.

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
3. **The rust router validates its subset before exec.** `infera.server` checks
   the requested config and fails with a pointer to `--router-backend python` for
   anything outside it. This example's etcd setup is inside that subset — and so
   is Kubernetes discovery: `_SUPPORTED_DISCOVERY` is `("etcd", "kubernetes")`,
   it just additionally requires `--k8s-label-selector`, which the operator
   injects. (An earlier version of this note claimed the rust backend was
   etcd-only and used that to explain the k8s recipe's `--router-backend python`.
   That reason was wrong, and the k8s recipe now ships `rust` on all four combos:
   it passed every judge on the `aggregated` shape and then served the agentic
   trace 448/448 at 100.00% cache efficiency on `disaggregated`, matching this
   example arm-for-arm.)
4. **The 27% in the table above is this example's number, not a portable one.** The
   same comparison on the k8s recipe's aggregated shape measured within ±5% either
   way, with +4.4% at concurrency 32 — the router only shows up when it is the
   bottleneck rather than the GPUs, and a 2-worker sweep to concurrency 32 never
   gets there. Re-measure before quoting it for a different shape.
5. **kv-aware fails soft.** Without a tokenizer it warns once and routes on load
   alone. `launch_router.sh` refuses to start on that warning and `verify.sh`
   re-checks it, because a run scored against load balancing while labelled
   kv-aware is worse than a launch that stops. Note that this only covers the
   tokenizer being absent — a router that loads a tokenizer and then keeps an empty
   kv view looks healthy in every log. Testing for that needs two workers and a
   behavioural check: the same prefix sent twice must land on the same worker,
   with `cached_tokens` above zero the second time.
6. **Advertise the data-network IP.** `--advertise-host` is what the peer dials
   for the Mooncake bootstrap handshake; a management-NIC address there fails at
   hand-off time, not at startup.
7. **`Ctrl-C` on a `tail -f` does not stop an engine.** The launch scripts run
   them under `nohup`; use `stop.sh`.
8. **A missing IP is refused, not defaulted.** Both legs find each other only
   through the addresses they register in etcd, and a wrong one costs a full cold
   start to discover: registration happens *after* the weights load, on the other
   node. Setting only `PREFILL_IP` is the trap worth naming — the decode leg would
   advertise the prefill node's address, both legs would register, and only a real
   request would find the hole. `require_ips` in `env.sh` stops that at launch.
9. **`kvd` outlives a restart, on disk.** `stop.sh` kills the daemon after the
   engines, so nothing is pulled from under a live one, but L3 is journalled and
   is recovered on the next start. Delete `KVD_L3_DIR` to start cold.
10. **A fabric problem surfaces as a GPU bug, 11 minutes late.** Both of the §3
    checks were added after hitting this twice with different causes and an
    identical symptom. The decode leg loads all 8 ranks' weights, allocates KV,
    starts uvicorn, answers `/model_info` — and only then dies:

    ```text
    Start of pd disaggregation warmup ...
    Memory access fault by GPU node-3 (Agent handle: 0x...) on address 0x... Reason: Unknown.
    ```

    After that `/health` returns `503` forever, the surviving ranks spin at 100%
    GPU waiting on a peer that is gone, and the wrapper keeps printing
    `waiting for SGLang HTTP` without timing out. Nothing in that picture points at
    the fabric, and `GPU node-3` is an HSA agent id, not a GPU index.

    The two causes seen so far:

    - **The pinned rail was down on one node.** The evidence is three info-level
      lines thousands of lines earlier, among the `[aiter]` autotune noise:
      `topology.cpp:93] <rail>:1 is not active (state: 1)`, then
      `has no active ports, skipping`, then `Skipping unavailable device`. Mooncake
      skips the only rail it was given and comes up with no usable transport, so
      the error waits for the first real KV transfer to appear.
    - **The container lacked `--privileged`.** `IPC_LOCK` plus `memlock=-1` was not
      enough for this cluster's RDMA registration path. Identical flags,
      environment and rail; adding `EXTRA_DOCKER_ARGS=--privileged` was the whole
      fix.

    Neither logs anything about permissions or link state at warning level or
    above, so telling them apart means changing one thing at a time.
