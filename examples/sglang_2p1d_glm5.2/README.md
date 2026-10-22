# GLM-5.2-MXFP4 — SGLang 2P1D (2 × TP4 prefill, TP8 decode) on two MI355X nodes

Runnable kit for **GLM-5.2-MXFP4** served the infera way, PD-disaggregated
across two 8×MI355X (gfx950) nodes: **two TP4 + DP4-attention prefills** with a
**250 GB-per-rank host-memory HiCache tier** on one node, **one TP8 +
DP8-attention decode** with EAGLE MTP on the other, KV moved by **Mooncake over
the eight ionic RDMA rails**, and the infera router in front (**kv-aware**,
**session affinity on prefill picks**). Tuned for the InferenceX AgentX
agentic-coding replay. `PREFILL_WORKERS=1` runs the same recipe with one
prefill (1P1D, 12 GPUs), the better shape below about C88.

One file is site-specific. Fill it in, then, on the prefill node:

```bash
$EDITOR env.sh                 # the only file you edit
bash engine/up.sh              # etcd -> prefills + decode (over ssh) -> router
bash engine/smoke.sh           # PD registration, completion/chat/stream, 66k-token prefix reuse
bash engine/eval.sh            # GSM8K as InferenceX runs it, 1,319 questions
bash engine/down.sh            # save logs, remove containers on both nodes
```

## Image

Build from the repo root on both nodes (or build once and `docker save |
docker load` it across); `deploy/docker/Dockerfile.sglang` is pinned to the
SGLang v0.5.22 nightly `lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261018`
and carries the GLM-5.2 DSA, HiCache, KV-events and Mooncake RDMA patches this
kit relies on:

```bash
docker build -f deploy/docker/Dockerfile.sglang -t <infera-sglang-image> .
```

## Contents

| path | what |
|---|---|
| [`env.sh`](env.sh) | **the only file you edit** — node IPs, ssh target, weights, image, scratch dir, RDMA rails, prefill count, GPUs, ports |
| [`engine/worker.sh`](engine/worker.sh) | starts ONE engine container on its own node (`prefill 0`, `prefill 1` or `decode`); carries the tuned recipe |
| [`engine/up.sh`](engine/up.sh) | etcd → `PREFILL_WORKERS` prefills here and the decode over ssh → router, waiting on each health endpoint and on the router listing every prefill |
| [`engine/smoke.sh`](engine/smoke.sh), [`engine/smoke.py`](engine/smoke.py) | correctness checks through the router; need real MTP acceptance |
| [`engine/eval.sh`](engine/eval.sh) | GSM8K through the router with InferenceX's `run_eval` (its lm-eval revision, `utils/evals/gsm8k.yaml`, full test set) |
| [`engine/agentx/agentx.sh`](engine/agentx/agentx.sh) | one InferenceX AgentX point, pinned at `918524ff`, counted over all the deployment's GPUs (16, or 12 with one prefill) |
| [`engine/agentx/constraints.txt`](engine/agentx/constraints.txt) | AgentX client package pins |
| [`engine/down.sh`](engine/down.sh) | save logs + inspect, remove the kit's containers by name on both nodes |
| [`engine/common.sh`](engine/common.sh) | sourced helpers: placeholder check, run directory, health wait, InferenceX checkout |

`up.sh` creates `$WORK_DIR/runs/<RUN_ID>/` on the prefill node with
`engine-config.env` (the settings started, also handed to the decode node); the
other scripts read those settings back, so an override given to `up.sh` alone
still applies to them, and write `smoke/`, `gsm8k/`, `agentx/` and `logs/` there.

## Topology

```
prefill node                                      decode node
├─ etcd, infera router :28300  ◀── clients
├─ prefill 0  GPUs 0-3  TP4/DP4  ══ KV over 8 ionic rails ══▶  decode  GPUs 0-7
└─ prefill 1  GPUs 4-7  TP4/DP4                                TP8/DP8, EAGLE MTP
   250 GB host HiCache per rank                                 no KV offload
```

Both nodes need the same `MODEL` path, the image under the same tag, and this
kit at the same path (the decode is started as `ssh $DECODE_SSH bash
<kit>/engine/worker.sh decode`).

## The recipe

| | prefill (× 2) | decode |
|---|---|---|
| parallelism | TP4, DP4 attention, EP1 | TP8, DP8 attention, EP1 |
| all-reduce | RCCL: `--disable-custom-all-reduce` (note 6) | same |
| memory | `--mem-fraction-static 0.88` | `--mem-fraction-static 0.92` |
| scheduling | chunk 32768, max running 256, `--schedule-policy shortest-prefill-first` | max running 256, decode graphs up to 256, 4 tokenizer workers |
| KV | FP8, HiCache 250 GB per rank, ratio 1.5, `write_back`, `kernel` I/O, `page_first` | FP8, no offload |
| MTP | — | EAGLE 5 steps / top-k 1 / 6 draft tokens; `index_share_for_mtp_iteration=false` |
| DSA | Triton prefill and decode backends, v2 top-k kernel (`SGLANG_OPT_USE_TOPK_V2=true`) | same |
| router | Rust backend, `kv-aware` with prefill / decode overlap weights 20 / 2, `--session-affinity prefill`, `--pd-prefill-guard-release completion`, `--decode-input-demand on` | |

Shortest-prefill-first stops short continuation turns from queueing behind a
long cold first turn pinned to the same prefill rank; `write_back` keeps the
prefix cache on GPU plus host instead of host alone. The decode keeps no prefix
cache, so session affinity pins prefill picks only.

## Validation status

The image above on two 8×MI355X nodes, prefill on crsuse2-m2m-138 and decode
on crsuse2-m2m-140:

| item | result |
|---|---|
| correctness (`smoke.sh`, real MTP acceptance) | 6/6 pass (66k-token prefix: 40.0 s cold, 1.9 s repeated) |
| GSM8K (`eval.sh`: InferenceX `run_eval`, 1,319 questions, concurrency 64, real MTP acceptance) | 0.9697 strict / 0.9704 flexible; no template-tag leakage, no empty answer |
| AgentX C176, 3600 s, `SIM_ACCEPT_LEN=3.61` | 35,637 tok/s/GPU, 15,881 requests, 0 failed, 11 invalid responses |
| with the prefills' EAGLE flags (note 10), `PREFILL_WORKERS=1`: `smoke.sh` | 6/6 pass (66k-token prefix: 14.2 s cold, 0.7 s repeated) |
| same: GSM8K as above | 0.9704 strict / 0.9697 flexible; no template-tag leakage, no empty answer; real acceptance length median 4.60 (3.86 without the prefills' flags) |

## Reference numbers

AgentX agentic-coding replay, InferenceX `918524ff`, 10 warm-up requests per
lane, 3600 s measured, forced MTP acceptance 3.61, every point on a freshly
started deployment. Throughput is per GPU over all of the deployment's GPUs
and counts cache-hit input tokens; interactivity is per-request
tokens/s/user (InferenceX plots p90).

| shape | point | GPUs | tok/s/GPU | requests (per s) | interactivity p50 / p90 | TTFT p50 / p90 | ITL p50 / p90 | prompt cache hit |
|---|---|---|---|---|---|---|---|---|
| 2P1D | C176 | 16 | 35,637 | 15,881 (4.38) | 40.4 / 35.7 | 4.3 / 8.1 s | 24.8 / 28.0 ms | 95.9% (trace bound 96.8%) |
| 2P1D | C128 | 16 | 26,592 | 13,202 (3.64) | 43.8 / 36.7 | 2.4 / 4.6 s | 22.8 / 27.2 ms | 95.7% (trace bound 96.4%) |
| 2P1D | C96 | 16 | 23,515 | 11,248 (3.10) | 44.9 / 40.8 | 1.6 / 3.7 s | 22.3 / 24.5 ms | 96.1% (trace bound 96.6%) |
| 2P1D | C64 | 16 | 18,202 | 8,158 (2.25) | 63.2 / 49.4 | 1.0 / 2.9 s | 15.8 / 20.3 ms | 96.7% (trace bound 96.9%) |
| 1P1D | C88 | 12 | 27,453 | 9,970 (2.75) | 49.1 / 42.8 | 4.9 / 9.5 s | 20.4 / 23.4 ms | 95.8% (trace bound 96.6%) |
| 1P1D | C64 | 12 | 23,241 | 7,831 (2.16) | 65.0 / 51.4 | 2.7 / 5.2 s | 15.4 / 19.4 ms | 96.6% (trace bound 96.9%) |
| 1P1D | C32 | 12 | 14,153 | 3,881 (1.07) | 91.7 / 73.2 | 1.0 / 2.9 s | 10.9 / 13.7 ms | 97.4% (trace bound 97.7%) |

The 1P1D points ran with prefill 0 on GPUs 4-7 of the prefill node.

- **Interactivity is set by the busiest decode rank.** Under DP attention
  every decode step waits for the busiest of the eight ranks; while it runs
  seven or more requests a step takes 80-90 ms, so 2P1D from C176 down to
  C96 stays near 36-41 at p90 and C128 / C96 buy almost nothing for the
  throughput they give up.
- **Up to C88 one prefill is enough.** Prefill load follows the sessions per
  prefill instance: one instance at C64 carries what each instance carries
  at 2P1D C128 (busy about 78% of the time, ranks computing about a third of
  it), and at C88 what each carries at C176. Dropping a prefill there costs
  some TTFT and gives each GPU about a quarter more throughput: 1P1D C64
  matches 2P1D C96's throughput at 26% higher p90 interactivity, and 1P1D C88
  beats 2P1D C128 and C96 on both throughput and p90 interactivity. At C88
  the prefill is saturated (waiting requests in 74% of 10 s windows, TTFT
  p50 4.9 s), so 2P1D takes over above it.

To reproduce a point the way InferenceX does, start the whole deployment
afresh for it:

```bash
SIM_ACCEPT_LEN=3.61 bash engine/up.sh                     # PREFILL_WORKERS=1 for 1P1D
CONC=176 bash engine/agentx/agentx.sh      # DURATION=3600 WARMUP_PER_LANE=10 by default
bash engine/down.sh                        # then wait for VRAM to drain on both nodes
```

## Decode request balancing (optional)

`--decode-input-demand on` balances the decode ranks by in-flight input tokens.
That evens their KV, but not their request counts: one rank can hold many short
requests for another's one long request, and the busiest rank sets every step.
`--decode-request-weight W` adds `W` tokens per request a rank already holds to
its demand (the shape of Dynamo's `router-decode-active-request-weight`); 0, the
default, is token-only demand. Enable it with
`ROUTER_EXTRA_ARGS="--decode-request-weight 1000000"`.

1P1D C88 on two AUS MI355X nodes (smci355-ccs-aus-n04-25 / n06-33), prefills
with the EAGLE flags (note 10), one deployment, prefill cache flushed and router
restarted before each point, 10 warm-up requests per lane, **1800 s measured**:

| weight | tok/s/GPU | requests (per s) | interactivity p50 / p90 | TTFT p50 / p90 / mean | ITL p50 / p90 | busiest rank running, median / p90 |
|---|---|---|---|---|---|---|
| 0 | 26,462 | 5,160 (2.87) | 75.8 / 55.7 | 5.5 / 11.1 / 8.3 s | 13.2 / 18.0 ms | 8 / 11 |
| 1,000,000 | 27,186 | 5,184 (2.88) | 86.5 / 79.6 | 5.6 / 12.0 / 9.5 s | 11.6 / 12.6 ms | 6 / 8 |

- p90 interactivity +43% at the same throughput. Both points take the same time
  per step at the same busiest-rank load (40.5 / 40.8 ms at 6 requests, 43.6 /
  43.5 ms at 8): the gain is the load distribution, the busiest rank no longer
  reaching 10-14 requests.
- TTFT rises (mean +15%, p90 +8%): the ranks' context is less even (busiest /
  mean 1.23 → 1.52 at the median), so more KV lands on fewer ranks.
- On this point one more request on the busiest rank cost as much as about
  1.05M more context tokens there (least-squares fit of step time against the
  busiest rank's requests and tokens and the totals over 3 s bins of the decode
  log), which is why the weight is a million; on 2P1D C176 with the v0.5.21
  image the same fit gave 0.17-0.19M. A 180,000 weight on another node pair
  gave 27,577 tok/s/GPU at 85.0 / 74.4.
- 1800 s windows understate tok/s/GPU against the canonical 3600 s by about 6%
  (sessions reach their longer turns later), and leave p90 interactivity within
  2%.

## Notes

**1. Forced MTP acceptance produces wrong text.** `SIM_ACCEPT_LEN` makes the
decode accept drafts at a fixed rate whether they are right or not. Use it only
for performance runs; run `smoke.sh` and `eval.sh` on a deployment started
without it.

**2. Cold start takes 10-15 minutes**: weights, the 8 × 250 GB pinned host
pools on the prefill node, AITER JIT and the decode's graph capture. JIT caches
persist in per-container named volumes `<container>-cache-<image-id>` and
`<container>-aiter-jit-<image-id>` on each node; remove them with `docker
volume rm` when done with an image.

**3. VRAM is released late.** `docker rm` returns long before the driver frees
the engines' memory; 10-65 minutes were observed on the prefill node, a few
on the decode node. Check `rocm-smi` on both nodes before the next `up.sh`.
The delay comes from the prefills' pinned HiCache host pools: a deployment
whose prefill ran without `--enable-hierarchical-cache` released its VRAM on
both nodes within 20 s, which makes bring-up debugging much quicker.
A container that `docker stop` cannot end ("did not receive an exit event")
is left for the next `down.sh`, which removes it once its process exits.

**4. Two engines on one node.** The prefills run with `--ipc host`, and RCCL
names its shared-memory segments after process ids, which start low in every
container; two engines started side by side fail in `ncclCommInitRank` with
`HIP failure: 'invalid argument'`. `worker.sh` has prefill 1 burn 30,000
process ids before it starts, so the two sets of ranks never collide.

**5. RDMA settings are fabric properties.** `RDMA_DEVICES`, `MC_GID_INDEX`
(the routable RoCEv2 GID) and `MOONCAKE_DISABLE_HIP_DMABUF` in `env.sh` depend
on the NICs: on a provider whose dma-buf registration fails, the prefill ranks
crash in `HIPStream::stream()` at startup; set `MOONCAKE_DISABLE_HIP_DMABUF=1`
there.

**6. Keep `--disable-custom-all-reduce`.** With DP attention and decode CUDA
graphs, the AITER custom all-reduce garbles outputs under load on every
upstream image from v0.5.19 to v0.5.22; on prefill ranks, whose tensor sizes
differ per rank, it can also pick a different all-reduce backend than its
peers and deadlock. RCCL has neither problem.

**7. Session affinity reads request headers.** `--session-affinity prefill`
keeps requests with the same `x-session-id` on the prefill rank their session
used last; `agentx.sh` has AIPerf send it
(`AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID`).

**8. KV buffers are registered in 1 GiB chunks (`MC_MAX_MR_SIZE`).** Mooncake
splits each KV buffer at the NIC's `max_mr_size`, 2 GiB on ionic. When a buffer's
allocation happens to start 1 GiB-aligned, ionic rejects that 2 GiB dma-buf
chunk (`Failed to register memory ...: Invalid argument [22]`), and the rank
behind it can neither send nor receive KV: with the decode's 79 buffers per
rank this hit 2 of 8 ranks on the first bring-up, and infera's start-up PD
probe then stopped both prefills. A standalone registration test reproduces it
on the first aligned allocation and registers the same buffer in 1 GiB chunks.

**9. A prefill rank can hit a GPU memory access fault (open).** Once so far,
in about 19 prefill-instance hours, a prefill rank died with `Memory access fault
by GPU node-N ... on address 0xfffffffff000. Reason: Unknown.` while
extending a long cached prefix by a few hundred tokens; the rest of its
instance hung in `process_batch_result_disagg_prefill` until SGLang killed
it, and the run continued on the other prefill. The same signature appears on
the v0.5.21 image, with and without the custom all-reduce, and the address
(page −1) points at a kernel indexing with −1. Afterwards that GPU stays busy
with no process on it, and every new process hangs or segfaults at its first
allocation there (`parallel_state.py`), until an administrator resets it
(`sudo rocm-smi --gpureset -d <gpu>`). Check each GPU with a one-line
allocation before the next `up.sh` after such a fault.

**10. The prefills carry the decode's EAGLE flags.** A prefill started
without them has no NextN layer, so the KV transfer carries no draft-layer KV
or DSA index-K, and the decode's draft layer reads uninitialised KV for every
prompt token. In a fresh process those slots are zero, the draft indexer's
scores tie, and its top-k scatter runs 15-170x slower: on the v0.5.21 image
the decode took 48.7 ms a step at 6 running requests on the busiest rank,
36.5 ms with the flags. The prefill pays one more layer (about 4.6 GB per GPU,
6.6% less KV). InferenceX's AMD multi-node recipe passes the flags to both
roles for this reason. The reference numbers above were measured before this
change.

## What this kit exposes on the network

| port | node | service |
|---|---|---|
| 28300 | prefill | router (OpenAI API, `/metrics`, `/v1/workers`) |
| 27201, 27457 | prefill | prefill engine HTTP (SGLang adds 27434-27444, 27690-27700) |
| 28398, 28399 | prefill | PD bootstrap |
| 27757-27760, 27765-27768 | prefill | KV events, one per prefill DP rank |
| 29051, 29052 | prefill | KV snapshot |
| 24701 / 24702 | prefill | etcd client / peer (peer on 127.0.0.1) |
| 27301 | decode | decode engine HTTP (SGLang adds 27534-27548) |

All bind on the host network without authentication. `--trust-remote-code` runs
the checkpoint's Python; serve only weights you trust.
