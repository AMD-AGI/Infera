# GLM-5.2-MXFP4 — SGLang aggregated TP4 + DP4 attention on four MI355X GPUs

Runnable kit for **GLM-5.2-MXFP4** served the infera way by one aggregated
(not PD-disaggregated) SGLang worker on four MI355X (gfx950) GPUs: **TP4 with
DP4 attention**, EAGLE MTP, a **270 GB-per-rank host-memory HiCache tier**, and
the infera router in front (**kv-aware** with **session affinity**). It is the
MI355X counterpart of the InferenceX B300 Dynamo + SGLang aggregated TP4
recipe, tuned for the AgentX agentic-coding replay.

One file is site-specific. Fill it in, then:

```bash
$EDITOR env.sh                 # the only file you edit
bash engine/up.sh              # etcd -> engine -> router
bash engine/smoke.sh           # registration, completion/chat/stream, 66k-token prefix reuse
bash engine/eval.sh            # GSM8K, 200 questions
bash engine/down.sh            # save logs, remove containers
```

## Image

Build from the repo root; `deploy/docker/Dockerfile.sglang` is pinned to the
SGLang v0.5.22 nightly `lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261018`
and applies the GLM-5.2 DSA, HiCache and KV-events patches this kit relies on:

```bash
docker build -f deploy/docker/Dockerfile.sglang -t <infera-sglang-image> .
```

## Contents

| path | what |
|---|---|
| [`env.sh`](env.sh) | **the only file you edit** — node IP, weights, image, scratch dir, GPUs, ports |
| [`engine/worker.sh`](engine/worker.sh) | starts the engine container; carries the tuned recipe |
| [`engine/up.sh`](engine/up.sh) | etcd → engine → router, waiting on each health endpoint |
| [`engine/smoke.sh`](engine/smoke.sh), [`engine/smoke.py`](engine/smoke.py) | correctness checks through the router |
| [`engine/eval.sh`](engine/eval.sh) | GSM8K (5-shot, 200 questions) with `lm_eval` 0.4.13 through the router |
| [`engine/agentx/agentx.sh`](engine/agentx/agentx.sh) | one InferenceX AgentX point, pinned at `918524ff` |
| [`engine/agentx/constraints.txt`](engine/agentx/constraints.txt) | AgentX client package pins |
| [`engine/down.sh`](engine/down.sh) | save logs + inspect, remove the kit's containers by name |
| [`engine/common.sh`](engine/common.sh) | sourced helpers: placeholder check, run directory, health wait |

`up.sh` creates `$WORK_DIR/runs/<RUN_ID>/` with `engine-config.env` (the
settings started); the other scripts write `smoke/`, `gsm8k/`, `agentx/` and
`logs/` there.

## The recipe

| | |
|---|---|
| GPUs | 4 (`GPUS`, default 0-3) |
| parallelism | TP4, DP4 attention (`--enable-dp-attention --enable-dp-lm-head`), EP1 |
| MTP | EAGLE 5 steps / top-k 1 / 6 draft tokens; `index_share_for_mtp_iteration=false` |
| KV | FP8 (`fp8_e4m3`), `--mem-fraction-static 0.85`, DSA prefill and decode on the Triton backend |
| scheduling | chunk 32768 (8192 per DP rank), `--max-prefill-tokens 8192`, max running 128, decode graphs up to 64, prefill delayer up to 60 passes |
| HiCache | 270 GB host pool per rank, ratio 1.5, `write_through`, `kernel` I/O, `page_first` layout |
| router | Rust backend, `kv-aware` with `--kv-overlap-weight 1.0`, `--session-affinity both` |

These are the settings of the best MI355X point on the InferenceX axes
(tokens/s/GPU against interactivity p90) among the aggregated runs that
preceded this kit; `write_back`, a larger memory fraction or a longer delayer
cap raised throughput by at most noise at C64 while lowering interactivity.

## Validation status

The image above, on one 8×MI355X node (crsuse2-m2m-138), GPUs 0-3:

| item | result |
|---|---|
| correctness (`smoke.py`, real MTP acceptance) | TODO |
| GSM8K (200 questions, real MTP acceptance) | TODO |
| AgentX C48, 3600 s, `SIM_ACCEPT_LEN=3.61` | TODO |

## Reference numbers

AgentX agentic-coding replay, InferenceX `918524ff`, 10 warm-up requests per
lane, 3600 s measured, forced MTP acceptance 3.61. Throughput is per GPU over
the 4 GPUs and counts cache-hit input tokens; interactivity is per-request
tokens/s/user.

TODO

To reproduce a point:

```bash
SIM_ACCEPT_LEN=3.61 bash engine/up.sh
CONC=48 bash engine/agentx/agentx.sh       # DURATION=3600 WARMUP_PER_LANE=10 by default
bash engine/down.sh
```

## Notes

**1. Forced MTP acceptance produces wrong text.** `SIM_ACCEPT_LEN` makes the
engine accept drafts at a fixed rate whether they are right or not. Use it only
for performance runs; run `smoke.sh` and `eval.sh` on an engine started without
it.

**2. Cold start takes 10-15 minutes**: weights, the 4 × 270 GB pinned host pools
(plus the DSA indexer's), AITER JIT and graph capture. JIT caches persist in the
named volumes `<CTR>-cache-<image-id>` and `<CTR>-aiter-jit-<image-id>`; remove
them with `docker volume rm` when done with an image.

**3. VRAM is released late.** `docker rm` returns long before the driver frees
the engine's memory; 10-40 minutes were observed. Check `rocm-smi` before the
next `up.sh`.

**4. Session affinity reads request headers.** `--session-affinity both` keeps
requests with the same `x-session-id` on the DP rank their session used last;
`agentx.sh` has AIPerf send it (`AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID`).

## What this kit exposes on the network

| port | service |
|---|---|
| 28200 | router (OpenAI API, `/metrics`, `/v1/workers`) |
| 27101 | engine HTTP |
| 27657-27660 | KV events, one per DP rank |
| 28951 | KV snapshot |
| 24601 / 24602 | etcd client / peer (peer on 127.0.0.1) |

All bind on the host network without authentication. `--trust-remote-code` runs
the checkpoint's Python; serve only weights you trust.
