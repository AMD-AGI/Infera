# GLM-5.2-MXFP4 — ATOM 4P4D on one MI355X node

Runnable kit for **GLM-5.2-MXFP4** served the infera way on a single 8×MI355X
(gfx950) node, PD-disaggregated: one **TP1 × PP4 prefill** on GPUs 0-3, set up
as in ATOM's GLM-5.2 AgentX recipe, and one **TP4 + DCP4 decode** on GPUs 4-7,
KV moved by **Mooncake over the ionic RDMA rails**, a **1.5 TiB Mooncake Store
CPU tier** behind the prefill, and the infera Rust router (round-robin with
**`--session-affinity prefill`**) in front. Engine settings follow ATOM's GLM-5.2
AgentX recipe and InferenceX PR #4045's 1P1D. `PREFILL_PP=1` runs a TP4 + DCP4
prefill instead ([below](#the-dcp-prefill)).

One file is site-specific. Fill it in, then:

```bash
$EDITOR env.sh               # the only file you edit
bash engine/up.sh            # etcd -> prefill + decode -> router, waiting on health
bash engine/smoke.sh         # PD registration, completion/chat/stream, 66k-token prefix reuse
bash engine/down.sh          # save logs, remove containers, wait for VRAM to drain
```

## Image

The engines need the ATOM nightly `661ef766` (`rocm/atom-dev:nightly_202610131505`)
plus an ATOM DCP fix and an LMCache fix, and the router needs Infera's
`--session-affinity`. Build the image from the repo root:

```bash
docker build -f deploy/docker/Dockerfile.atom \
  --build-arg ATOM_BASE_IMAGE=rocm/atom-dev:nightly_202610131505 \
  --build-arg BUILD_MOONCAKE=0 \
  --build-arg ATOM_NIGHTLY=1 \
  -t <infera-atom-image> .
```

- `ATOM_NIGHTLY=1` runs
  [`deploy/docker/patches/atom_nightly/`](../../deploy/docker/patches/atom_nightly/)
  instead of the release-base `patches/atom/`: the ATOM sparse DCP padded-row
  fix (without it the prefill hits a `Memory access fault` under long-prefix
  load), the LMCache ROCm pinned-pool fix (the CPU tier becomes GTT, gotcha 4),
  the AITER kernel prebuild of gotcha 12 and the `protobuf<7` pin the nightly's
  OpenTelemetry needs. This nightly already has the DCP MTP-draft and LMCache
  lookup-pin fixes the previous image patched in.
  A drifted anchor fails the build.
- `BUILD_MOONCAKE=0` keeps the base's Mooncake 0.3.13. That build routes
  same-host targets by upstream `MC_DISABLE_HIP`, which this kit sets (gotcha 1).
- The default `Dockerfile.atom` build (ATOM 0.1.4 base) is not validated with this
  kit; the recipe relies on the nightly's DCP, `lmcache_offload` and MTP support.

## Contents

| path | what |
|---|---|
| [`env.sh`](env.sh) | **the only file you edit** — node IP, weights, image, scratch dir, load shape, ports |
| [`engine/worker.sh`](engine/worker.sh) | starts ONE engine container (`prefill` or `decode`); carries the tuned recipe |
| [`engine/up.sh`](engine/up.sh) | etcd → Store → both workers → router; checks the router sees 1 prefill + 1 decode |
| [`engine/store.sh`](engine/store.sh) | the Mooncake Store CPU tier: a master and one owner per NUMA node |
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
| parallelism | TP1 × PP4, layers per stage `20,20,20,18` | TP4 + DCP4, no DP-attention |
| execution | `--enforce-eager` (ATOM runs PP without CUDA graphs) | CUDA graphs 1, 2, 4, 8, 12 … 2 × CONC |
| max-num-seqs | 512 | 2 × CONC |
| max-num-batched-tokens | 8192 | 2048 |
| mem util | 0.85 | 0.94, communicator groups shared (`AITER_REUSE_IDENTICAL_COMM_GROUPS=1`) |
| KV connector | `multi`: Mooncake `kv_producer` (16 send workers per stage) + `mooncake_store` offload | Mooncake `kv_consumer` |
| shared | FP8 KV, block 16, prefix caching, MTP (`--method mtp`), online PTPC-FP8 with MXFP4 experts (the MTP layer's quantized online, `MTP_EXPERTS`) and the rest of the MTP layer in BF16 | |

The settings are those of ATOM's GLM-5.2 AgentX recipe
(`/app/ATOM/recipes/Agentic-GLM-5.2.md` in the image) and of InferenceX PR
#4045's 1P1D (`MC_NUM_QP_PER_EP=1` on every Mooncake peer, `TOPK_FORCE_PATH=one`,
`--timeout-keep-alive 900`, the Store). The decode's memory settings give it
~196K KV blocks instead of ~156K at 0.90 / 16384 with separate communicators:
AgentX lanes reuse their whole context every turn, and once C × mean prompt
length outgrows the decode's KV pool its prefix hits collapse, the prefill
re-sends whole prompts and TTFT explodes (about C85 before, C105 now). Not
taken from them: `OFFLOAD_PROFILE=1` (logging only), the decode's graphs (here
every 4 up to 2 × CONC, there every 8 up to 256) and BF16 MTP experts
(`MTP_EXPERTS`).

MTP depth follows `CONC`: K3 at CONC ≥ 48, K4 below. The Store
(`PREFILL_CPU_TIER=store`, gotcha 14) is one pool of 2 × 768 GiB that the four
PP stages share, each through its own NIC; `PREFILL_CPU_TIER=lmcache` instead
holds 256 GiB of LMCache per stage (`PREFILL_OFFLOAD_GB`), chunk 256, and looks
up all four stages because each holds the KV of its own layers. Both load hits
of any length.

KV moves from each PP stage (its own layers) to every decode DCP rank: the
producer gathers the tokens a rank owns into a staging slot
(`ATOM_PD_MLA_STAGING`, default on) and the decode lands it
(`ATOM_PD_MLA_LANDING`, default on). This assumes one block size on both legs,
so `BLOCK_SIZE` sets both. AgentX metadata reports the prefill as TP1 × PP4;
throughput per GPU is over all 8 GPUs.

### The DCP prefill

`PREFILL_PP=1` starts the prefill as TP4 + DCP4 with CUDA graphs up to
2 × CONC, as the decode. It was validated with these settings, which are not
the defaults; set them with it:

```bash
PREFILL_PP=1 PREFILL_MEM_UTIL=0.90 PREFILL_MAX_BATCHED_TOKENS=16384 BLOCK_SIZE=64 \
  OFFLOAD_MIN_LOAD_TOKENS=8192 PREFILL_SEND_WORKERS= bash engine/up.sh
```

Each DCP rank then writes to one decode rank without staging, and the LMCache
lookup covers the four DCP ranks, each holding a distinct KV shard. Under AgentX
load it is slower than the PP prefill (reference numbers below).

## Validation status

The image above, on one 8×MI355X node:

| item | PP prefill (default) | DCP prefill |
|---|---|---|
| correctness (`smoke.py`, real MTP acceptance) | **passed**, 6/6 | **passed**, 6/6 |
| GSM8K (200 questions) | **0.98** (16 send workers) | **0.975** |
| AgentX, 3600 s | **passed** at C64 and C96, no transfer failure | **passed** at C48, C64, C96, no `Memory access fault` |

On shutdown each PP stage logs a few `TypeError` tracebacks from aiter
`shm_broadcast`; none appear while serving.

## Reference numbers

These are the previous defaults: Python router, decode at 0.90 / 16384 with
separate communicators, 4 send workers, LMCache.

AgentX agentic-coding replay, InferenceX `918524ff`, cold start per point, 10
warm-up requests per lane, 3600 s measured, forced MTP acceptance (`MTP_AL=2.99`
with K3). Throughput is per GPU over all 8 GPUs and counts cache-hit input
tokens; TTFT in s, ITL in ms.

PP and DCP prefill on the same node (crsuse2-m2m-140), image
`infera-atom:nightly-202610131505-pr198fix`, one run per point:

| CONC | prefill | input tok/s/GPU | output tok/s/GPU | TTFT p50 / p90 / p99 | ITL p50 / p90 | prefill hit GPU / CPU / total % | errors |
|---:|---|---:|---:|---|---|---|---|
| 64 | PP4 | **33389.60** (+8.6%) | 262.13 | 2.09 / 5.17 / 15.74 | 16.46 / 17.59 | 77.7 / 17.2 / 95.0 | 1 / 7682 |
| 64 | DCP | 30739.05 | 242.78 | 3.23 / 17.81 / 49.41 | 16.03 / 17.30 | 79.8 / 16.7 / 96.5 | 1 / 7024 |
| 96 | PP4 | **30115.94** (+44.1%) | 274.59 | 24.98 / 51.53 / 76.31 | 16.75 / 18.74 | 34.6 / 52.0 / 86.6 | 0 / 8142 |
| 96 | DCP | 20902.07 | 193.03 | 51.39 / 103.55 / 161.52 | 13.75 / 16.48 | 16.3 / 77.6 / 93.9 | 0 / 5664 |

DCP prefill, image above, C48 and C64 on two different nodes:

| CONC | MTP K / AL | total tok/s/GPU | output tok/s/GPU | input tok/s/GPU | P90 interactivity tok/s/user | TTFT p50 / p90 / p99 s | errors / tail cancels |
|---:|---|---:|---:|---:|---:|---|---|
| 48 | 3 / 2.99 | 26325.24 | 183.30 | 26141.94 | 66.52 | 1.759 / 7.662 / 24.695 | 2 / 4 |
| 64 | 3 / 2.99 | 30820.57 | 241.22 | 30579.35 | 56.83 | 3.114 / 17.256 / 47.358 | 1 / 27 |

Interactivity is the inverse of P90 inter-token latency.

To reproduce a point (cold start each time):

```bash
CONC=64 MTP_AL=2.99 bash engine/up.sh    # CONC < 48: MTP_AL=3.33 (K4)
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

**4. LMCache: 1 TiB of pinned host memory, as GTT.** With `PREFILL_CPU_TIER=lmcache` the four prefill stages (or DCP ranks) each pin
`PREFILL_OFFLOAD_GB` GiB. Lower it on smaller hosts, or set `0` to drop the CPU
tier; the prefill then registers a plain Mooncake producer. Startup also waits
a few minutes in an RCCL barrier after the pools are pinned. With the CPU tier
on, `worker.sh` sets `HSA_USERPTR_FOR_PAGED_MEM=0`, so the image's LMCache patch
allocates the pool with `hipHostMalloc` (GTT): the kernel can no longer migrate
its pages, and each migration used to stall all GPU queues of the prefill for
seconds. GTT has a machine-wide cap (`mem_info_gtt_total`, about 1376 GiB on the
reference nodes) and lands on the NUMA node of the first GPU, so a node fits at
most one 4 × 256 GiB prefill; `worker.sh` refuses to start a prefill whose
pools do not fit in the free GTT. The prefill logs `LMCache CPU pool uses
hipHostMalloc (GTT` when the fix is active.

**5. Two workers on one host network need two readiness ports.** Infera's
worker readiness probe defaults to port 30090; `worker.sh` sets
`INFERA_READINESS_PORT` per leg (19011 / 19012).

**6. Ports avoid two ranges.** Listeners stay below the ephemeral range
(32768-60999) and outside Mooncake's random RPC ports (15000-17000): an outgoing
socket can take the port before the listener binds. Mooncake handshakes use
base + PP stage on the prefill and base + TP rank on the decode, so 21301-21304
and 21311-21314.

**7. Router backend and transports.** `ROUTER_BACKEND` picks the Rust router
(default) or the Python one (used for the reference numbers below); both route
ATOM PD (`atom-mooncake`) and hand the prefill's prompt token ids to the decode,
so the decode does not tokenize the prompt again. ATOM speaks neither NATS requests nor NATS KV events, so
`up.sh` passes `--request-transport http --kv-event-transport zmq`. `kv-aware`
would need engine KV events, so the policy is `round-robin`.

**8. Session affinity reads request headers.** `SESSION_AFFINITY` (default
`prefill`) is passed as `--session-affinity`: requests with an `x-session-id`
keep their session's prefill (`both`: and decode), others are round-robin.
`agentx.sh` has AIPerf send it (`AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID`).
The router also forwards `x-session-id` and `x-correlation-id` to both legs.
With a single prefill and decode it only carries the headers through; it
matters once more instances are added.
Tune the idle TTL (default 3600 s) through `ROUTER_EXTRA_ARGS` (`--session-affinity-ttl-seconds`). `SESSION_AFFINITY=off`
is the stateless alternative.

**9. VRAM is released late.** `docker rm` returns long before the driver frees
the engines' memory; 15-40 minutes were observed. `down.sh` and `up.sh` both wait
until every GPU holds less than 4 GiB (`VRAM_RELEASE_TIMEOUT`).

**10. Same-numbered rails only.** The fabric routes rail *n* to rail *n*, so the
prefill sets `ATOM_MOONCAKE_MATCHED_RAILS=auto` and writes on the rail of the
decode rank it serves. A DCP prefill must also match the decode's DCP size in
this ATOM; the PP prefill stages per decode DCP rank instead (see the recipe).

**11. Cold start takes about 10 minutes**, mostly weights, AITER/FlyDSL JIT and
graph capture. JIT caches persist in per-role, per-image Docker volumes
`<CTR>-cache-<role>-<image-id>` and `<CTR>-aiter-jit-<role>-<image-id>`; remove
them with `docker volume rm` when done with an image.

**12. A JIT compile on live traffic stalls the prefill.** The base does not
prebuild the AITER varlen attention that chunked-MLA prefill calls once a prompt
extends a cached prefix; its first call compiled it for ~30 s, which fixed
AgentX warm-up requests need not reach. The image now prebuilds it (see
`ATOM_NIGHTLY`). On gfx950 this nightly's prefill loads the ASM
`fmha_v3_varlen_fwd` module instead, so the prebuild is a fallback for other
archs (gfx942) or paths. AITER keeps such modules in `/app/aiter-test/aiter/jit`,
not `/root/.cache`, so `worker.sh` persists that directory too, for any other
kernel compiled at run time.

**13. The PP prefill needs fewer Mooncake send workers.** With ATOM's default
16 send workers per stage, every AgentX run on MI355X (crsuse2-m2m-140) failed
within its first four requests: one RDMA write from a stage to one decode rank
never completed (`staged-mla` / `landed-mla RDMA chunk error -1 ... Mooncake's
sync timeout`, Mooncake `CQ completion timeout diagnostic`, then `Failed to poll
CQ` and ionic `wqe is not posted` without end). Every later transfer to that
rank failed, so every request did. It hit `ionic_4` and `ionic_5`, with forced
and with real MTP acceptance. One send worker removed the hang, but a 443k-token
prompt then missed ATOM's fixed 120 s transfer deadline (`PD_TRANSFER_DEADLINE_S`,
queueing included). Four workers ran C64 and C96 without a transfer failure.
All of that was with Mooncake's default two QPs per endpoint. ATOM's offload
README reports the same symptom for concurrent reads at two QPs, and ATOM's CI
and InferenceX PR #4045 run the 16 workers with `MC_NUM_QP_PER_EP=1`, which the
kit now sets on every Mooncake peer; `PREFILL_SEND_WORKERS=4` is the fallback.
The DCP prefill, whose ranks each write to one decode rank without staging,
never showed the hang.

**14. The Store owners need 2 MiB pages.** Each owner pins `STORE_OWNER_GB`
(768 GiB) of its NUMA node and registers it with the NICs in 64 GiB regions,
which the ionic NICs only accept when backed by huge pages. `store.sh` gets them
without root through glibc's THP malloc (`GLIBC_TUNABLES=glibc.malloc.hugetlb=1`,
THP `enabled` = `madvise` or `always`) and prints each owner's `AnonHugePages`;
a fragmented node can leave an owner on 4 KiB pages, and it then fails to mount.
Each NUMA node needs `STORE_OWNER_GB` free plus the engines' host memory; lower
it or use `PREFILL_CPU_TIER=lmcache` on smaller hosts. The prefill refuses to
start when its startup probe reads back other bytes than it wrote ("the chunk
read back differs"): on the crusoe MI355X VMs every Store read into HBM
completed without moving data (puts from HBM and transfers to host buffers were
fine, and so is the P→D path), with peer memory and DMA-BUF alike; use
`PREFILL_CPU_TIER=lmcache` there. The prefill needs one
Store load worker (`OFFLOAD_LOAD_WORKERS=1`): more hung its pipeline at C96 in
ATOM's tests.

## What this kit exposes on the network

| port | service |
|---|---|
| 18000 | router (OpenAI API, `/metrics`, `/v1/workers`) |
| 19001 / 19002 | prefill / decode engine HTTP |
| 19011 / 19012 | prefill / decode readiness probe |
| 21301-21304 / 21311-21314 | Mooncake handshakes |
| 23379 / 23380 | etcd client / peer (peer on 127.0.0.1) |
| 26051 / 26080 / 26090 | Store master RPC / metadata HTTP / metrics |
| 26052-26053 | Store owners |

All bind on the host network without authentication. `--trust-remote-code` runs
the checkpoint's Python; serve only weights you trust.

## Source

- Walkthrough: [`manual/examples/atom_4p4d_glm5.2.md`](../../manual/examples/atom_4p4d_glm5.2.md)
- Concepts: [PD disaggregation](../../manual/features/pd_disaggregation.md)
- Other ATOM PD launchers: [`deepseek_v4`](../deepseek_v4/)
