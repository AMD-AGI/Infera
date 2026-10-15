# GLM-5.2-MXFP4 — ATOM 4P4D with DCP on one MI355X node

Runnable kit for **GLM-5.2-MXFP4** served the infera way on a single 8×MI355X
(gfx950) node, PD-disaggregated: one **TP4 + DCP4 prefill** on GPUs 0-3 and one
**TP4 + DCP4 decode** on GPUs 4-7, KV moved by **Mooncake over the ionic RDMA
rails**, a **1 TiB LMCache CPU tier** behind the prefill, and the infera Python
router (round-robin with **`--session-affinity prefill`**) in front.

One file is site-specific. Fill it in, then:

```bash
$EDITOR env.sh               # the only file you edit
bash engine/up.sh            # etcd -> prefill + decode -> router, waiting on health
bash engine/smoke.sh         # PD registration, completion/chat/stream, 66k-token prefix reuse
bash engine/down.sh          # save logs, remove containers, wait for VRAM to drain
```

## Image

The engines need ATOM `d9f0720` (`rocm/atom-dev:nightly_202609221542`) plus strict
ATOM and LMCache patches, and the router needs Infera's `--session-affinity`.
Build the image from the repo root:

```bash
docker build -f deploy/docker/Dockerfile.atom \
  --build-arg ATOM_BASE_IMAGE=rocm/atom-dev:nightly_202609221542 \
  --build-arg BUILD_MOONCAKE=0 \
  --build-arg ATOM_DCP_PATCHES=1 \
  -t <infera-atom-image> .
```

- `ATOM_DCP_PATCHES=1` applies the patches in
  [`deploy/docker/patches/atom_dcp/`](../../deploy/docker/patches/atom_dcp/):
  the DCP block table for the first MTP draft after prefill (without it the
  first request crashes), MoE decode padding under eager mode, and the
  LMCache#5098 lookup-pin fix. A drifted anchor fails the build. It also pins
  `protobuf<7`, which the nightly's OpenTelemetry needs, and prebuilds the AITER
  kernel of gotcha 12.
- `BUILD_MOONCAKE=0` keeps the base's Mooncake v0.3.14-rc1. That build routes
  same-host targets by upstream `MC_DISABLE_HIP`, which this kit sets (gotcha 1).
- The default `Dockerfile.atom` build (ATOM 0.1.4 base) is not validated with this
  kit; the recipe relies on the nightly's DCP, `lmcache_offload` and MTP support.

## Contents

| path | what |
|---|---|
| [`env.sh`](env.sh) | **the only file you edit** — node IP, weights, image, scratch dir, load shape, ports |
| [`engine/worker.sh`](engine/worker.sh) | starts ONE engine container (`prefill` or `decode`); carries the tuned recipe |
| [`engine/up.sh`](engine/up.sh) | etcd → both workers → router; checks the router sees 1 prefill + 1 decode |
| [`engine/smoke.sh`](engine/smoke.sh), [`engine/smoke.py`](engine/smoke.py) | correctness checks through the router; need real MTP acceptance |
| [`engine/eval.sh`](engine/eval.sh) | GSM8K (5-shot, 200 questions) with `lm_eval` from the image; needs real MTP acceptance |
| [`engine/agentx/agentx.sh`](engine/agentx/agentx.sh), [`engine/agentx/aggregate.py`](engine/agentx/aggregate.py) | one InferenceX AgentX point, pinned at `918524ff`, with the single-node PD GPU count fixed |
| [`engine/agentx/constraints.txt`](engine/agentx/constraints.txt) | AgentX client package pins of the reference runs |
| [`engine/down.sh`](engine/down.sh) | save logs + inspect, remove the kit's containers by name, wait for VRAM |
| [`engine/common.sh`](engine/common.sh) | sourced helpers: placeholder check, run directory, health and VRAM waits |

Each `up.sh` creates `$WORK_DIR/runs/<RUN_ID>/` with `engine-config.env` (the
settings actually started); the other scripts read it and write `smoke/`,
`gsm8k/`, `agentx/` and `logs/` there.

## The recipe

| | prefill | decode |
|---|---|---|
| GPUs | 0-3 | 4-7 |
| parallelism | TP4 + DCP4, no DP-attention | TP4 + DCP4, no DP-attention |
| max-num-seqs | 512 | 2 × CONC |
| KV connector | `multi`: Mooncake `kv_producer` + `lmcache_offload` | Mooncake `kv_consumer` |
| shared | FP8 KV, block 64, prefix caching, MTP (`--method mtp`), online PTPC-FP8 with MXFP4 experts and BF16 MTP layer, max-num-batched-tokens 16384, mem util 0.90, CUDA graphs 1, 2, 4, 8, 12 … 2 × CONC | |

MTP depth follows `CONC`: K3 at CONC ≥ 48, K4 below. LMCache holds 256 GiB per
prefill rank (`PREFILL_OFFLOAD_GB`), chunk 256, loads from 8192 tokens, and
looks up all four DCP ranks because each holds a distinct KV shard.

## Validation status

Stated plainly. The reference is the experiment this kit was ported from: the
same engine recipe and environment, on an image built from the same base, ATOM
patches and Infera changes, on one 8×MI355X node, amdgpu 6.14.14, 2.7 TiB DRAM,
`MOONCAKE_DISABLE_HIP_DMABUF=1`, P/D mem util 0.90.

| item | status |
|---|---|
| correctness (the five `smoke.py` content checks) | **passed** on the reference; 66k-token needle 8.8 s cold, 0.7 s repeated; a ~200k-token prompt also passed |
| GSM8K | **0.99** (198/200) with this kit and the image from `Dockerfile.atom` |
| AgentX C24-C80 | **passed**, AIPerf `submission_valid=true` at every point (table below) |
| this kit's scripts and the image from `Dockerfile.atom` | smoke **passed**. AgentX C48 3600 s: 23992.63 total / 165.54 output tok/s/GPU, TTFT p99 105 s, from prefill stalls on a JIT compile during the run. With that kernel compiled before the run (and diagnostic timing hooks): 25904.67 / 180.26, TTFT p50 / p90 / p99 1.953 / 8.992 / 28.981 s (gotcha 12) |

## Reference numbers

AgentX agentic-coding replay, InferenceX `918524ff`, cold start per point,
10 warm-up requests per lane, 3600 s measured, forced MTP acceptance
(`MTP_AL=3.33` with K4, `2.99` with K3). Throughput is per GPU over all 8 GPUs and
counts cache-hit input tokens. Interactivity is the inverse of P90 inter-token
latency.

| CONC | MTP K / AL | total tok/s/GPU | output tok/s/GPU | P90 interactivity tok/s/user | TTFT p50 / p90 s | errors / tail cancels |
|---:|---|---:|---:|---:|---|---|
| 24 | 4 / 3.33 | 15414.73 | 105.67 | 83.75 | 1.074 / 4.436 | 0 / 2 |
| 36 | 4 / 3.33 | 19426.37 | 133.00 | 71.94 | 1.417 / 13.567 | 1 / 26 |
| 48 | 3 / 2.99 | 24900.07 | 172.78 | 66.45 | 2.121 / 15.460 | 1 / 9 |
| 64 | 3 / 2.99 | 22114.39 | 181.11 | 54.59 | 6.872 / 62.619 | 1 / 7 |
| 80 | 3 / 2.99 | 23053.40 | 200.21 | 57.87 | 25.758 / 75.218 | 2 / 22 |

To reproduce a point (cold start each time):

```bash
CONC=48 MTP_AL=2.99 bash engine/up.sh    # CONC < 48: MTP_AL=3.33 (K4)
bash engine/agentx/agentx.sh             # DURATION=3600 WARMUP_PER_LANE=10 by default
bash engine/down.sh
```

Correctness and accuracy instead need real acceptance: `MTP_AL=` (the default),
then `engine/smoke.sh` and `engine/eval.sh`. Both refuse to run against forced acceptance.

## Notes and gotchas

**1. `MC_DISABLE_HIP=1` is load-bearing.** P and D see disjoint GPU sets
(`HIP_VISIBLE_DEVICES`), and the base Mooncake's same-host HIP IPC path then
fails with "invalid device context". The flag makes it use the registered RDMA
buffers. Infera does not set it; its preflight tools only know the B.2 spelling
`MC_DISABLE_HIP_TRANSPORT`, which this Mooncake build ignores.

**2. `MOONCAKE_DISABLE_HIP_DMABUF` depends on the node's driver.** `1`
(peer-memory) was validated at mem util 0.90. Another node only registered GPU
memory with `0` (DMA-BUF), and there long prefills OOMed at 0.75; it passed at
P/D 0.70/0.72. The base image bakes in `1`, and Infera treats any set value as
an operator decision, so the kit always passes it explicitly.

**3. Forced MTP acceptance produces wrong text.** `MTP_AL` makes decode accept
drafts at a fixed rate whether they are right or not. Use it only for
performance runs; the default (empty) is real acceptance.

**4. 1 TiB of pinned host memory.** The four prefill ranks each pin
`PREFILL_OFFLOAD_GB` GiB. Lower it on smaller hosts, or set `0` to drop the CPU
tier; the prefill then registers a plain Mooncake producer. Startup also waits
a few minutes in an RCCL barrier after the pools are pinned.

**5. Two workers on one host network need two readiness ports.** Infera's
worker readiness probe defaults to port 30090; `worker.sh` sets
`INFERA_READINESS_PORT` per leg (19011 / 19012).

**6. Ports avoid two ranges.** Listeners stay below the ephemeral range
(32768-60999) and outside Mooncake's random RPC ports (15000-17000): an outgoing
socket can take the port before the listener binds. Mooncake handshakes use
base + TP rank, so 21301-21304 and 21311-21314.

**7. The Python router is required.** `--router-backend rust` cannot route ATOM
PD, and ATOM speaks neither NATS requests nor NATS KV events, so `up.sh` passes
`--router-backend python --request-transport http --kv-event-transport zmq`.
`kv-aware` would need engine KV events, so the policy is `round-robin`.

**8. Session affinity reads request headers.** `SESSION_AFFINITY` (default
`prefill`) is passed as `--session-affinity`: requests with an `x-session-id`
keep their session's prefill (`both`: and decode), others are round-robin.
`agentx.sh` has AIPerf send it (`AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID`).
The router also forwards `x-session-id` and `x-correlation-id` to both legs.
With a single prefill and decode it only carries the headers through; it
matters once more instances are added. Tune the idle TTL (default 3600 s) through
`ROUTER_EXTRA_ARGS` (`--session-affinity-ttl-seconds`). `SESSION_AFFINITY=off`
is the stateless alternative.

**9. VRAM is released late.** `docker rm` returns long before the driver frees
the engines' memory; 15-40 minutes were observed. `down.sh` and `up.sh` both wait
until every GPU holds less than 4 GiB (`VRAM_RELEASE_TIMEOUT`).

**10. Same-numbered rails only.** The fabric routes rail *n* to rail *n*, so the
prefill sets `ATOM_MOONCAKE_MATCHED_RAILS=auto` and writes on the rail of the
decode rank it serves. DCP size must also match on both legs in this ATOM.

**11. Cold start takes about 10 minutes**, mostly weights, AITER/FlyDSL JIT and
graph capture. JIT caches persist in per-role, per-image Docker volumes
`<CTR>-cache-<role>-<image-id>` and `<CTR>-aiter-jit-<role>-<image-id>`; remove
them with `docker volume rm` when done with an image.

**12. A JIT compile on live traffic stalls the prefill.** The base does not
prebuild the AITER varlen attention that chunked-MLA prefill calls once a prompt
extends a cached prefix; its first call compiled it for ~30 s, which fixed
AgentX warm-up requests need not reach. The image now prebuilds it (see
`ATOM_DCP_PATCHES`). AITER keeps such modules in `/app/aiter-test/aiter/jit`,
not `/root/.cache`, so `worker.sh` persists that directory too, for any other
kernel compiled at run time.

## What this kit exposes on the network

| port | service |
|---|---|
| 18000 | router (OpenAI API, `/metrics`, `/v1/workers`) |
| 19001 / 19002 | prefill / decode engine HTTP |
| 19011 / 19012 | prefill / decode readiness probe |
| 21301-21304 / 21311-21314 | Mooncake handshakes |
| 23379 / 23380 | etcd client / peer (peer on 127.0.0.1) |

All bind on the host network without authentication. `--trust-remote-code` runs
the checkpoint's Python; serve only weights you trust.

## Source

- Walkthrough: [`manual/examples/atom_4p4d_glm5.2_dcp.md`](../../manual/examples/atom_4p4d_glm5.2_dcp.md)
- Concepts: [PD disaggregation](../../manual/features/pd_disaggregation.md)
- Other ATOM PD launchers: [`deepseek_v4`](../deepseek_v4/)
