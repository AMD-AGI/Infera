# GLM-5.2-MXFP4 — SGLang 2P1D (2 × TP4 prefill, TP8 decode) on two MI355X nodes

Runnable kit for **GLM-5.2-MXFP4** served the infera way, PD-disaggregated
across two 8×MI355X (gfx950) nodes: **two TP4 + DP4-attention prefills** with a
**250 GB-per-rank host-memory HiCache tier** on one node, **one TP8 +
DP8-attention decode** with EAGLE MTP on the other, KV moved by **Mooncake over
the eight ionic RDMA rails**, and the infera router in front (**kv-aware**,
**session affinity on prefill picks**). Tuned for the InferenceX AgentX
agentic-coding replay.

One file is site-specific. Fill it in, then, on the prefill node:

```bash
$EDITOR env.sh                 # the only file you edit
bash engine/up.sh              # etcd -> 2 prefills + decode (over ssh) -> router
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
| [`env.sh`](env.sh) | **the only file you edit** — node IPs, ssh target, weights, image, scratch dir, RDMA rails, GPUs, ports |
| [`engine/worker.sh`](engine/worker.sh) | starts ONE engine container on its own node (`prefill 0`, `prefill 1` or `decode`); carries the tuned recipe |
| [`engine/up.sh`](engine/up.sh) | etcd → both prefills here and the decode over ssh → router, waiting on each health endpoint |
| [`engine/smoke.sh`](engine/smoke.sh), [`engine/smoke.py`](engine/smoke.py) | correctness checks through the router; need real MTP acceptance |
| [`engine/eval.sh`](engine/eval.sh) | GSM8K through the router with InferenceX's `run_eval` (its lm-eval revision, `utils/evals/gsm8k.yaml`, full test set) |
| [`engine/agentx/agentx.sh`](engine/agentx/agentx.sh) | one InferenceX AgentX point, pinned at `918524ff`, counted over the 16 GPUs |
| [`engine/agentx/constraints.txt`](engine/agentx/constraints.txt) | AgentX client package pins |
| [`engine/down.sh`](engine/down.sh) | save logs + inspect, remove the kit's containers by name on both nodes |
| [`engine/common.sh`](engine/common.sh) | sourced helpers: placeholder check, run directory, health wait, InferenceX checkout |

`up.sh` creates `$WORK_DIR/runs/<RUN_ID>/` on the prefill node with
`engine-config.env` (the settings started, also handed to the decode node); the
other scripts write `smoke/`, `gsm8k/`, `agentx/` and `logs/` there.

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

## Reference numbers

AgentX agentic-coding replay, InferenceX `918524ff`, 10 warm-up requests per
lane, 3600 s measured, forced MTP acceptance 3.61, every point on a freshly
started deployment. Throughput is per GPU over all 16 and counts cache-hit
input tokens; interactivity is per-request tokens/s/user.

| point | tok/s/GPU | requests (per s) | interactivity p50 / p90 | TTFT p50 / p90 | ITL p50 / p90 | prompt cache hit |
|---|---|---|---|---|---|---|
| C176 | 35,637 | 15,881 (4.38) | 40.4 / 35.7 | 4.3 / 8.1 s | 24.8 / 28.0 ms | 95.9% (trace bound 96.8%) |

To reproduce a point the way InferenceX does, start the whole deployment
afresh for it:

```bash
SIM_ACCEPT_LEN=3.61 bash engine/up.sh
CONC=176 bash engine/agentx/agentx.sh      # DURATION=3600 WARMUP_PER_LANE=10 by default
bash engine/down.sh                        # then wait for VRAM to drain on both nodes
```

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
the engines' memory; 10-40 minutes were observed. Check `rocm-smi` on both
nodes before the next `up.sh`.

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
