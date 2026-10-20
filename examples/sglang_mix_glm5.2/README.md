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
| all-reduce | RCCL: `--disable-custom-all-reduce` (see note 6) |
| MTP | EAGLE 5 steps / top-k 1 / 6 draft tokens; `index_share_for_mtp_iteration=false` |
| KV | FP8 (`fp8_e4m3`), `--mem-fraction-static 0.85`, DSA prefill and decode on the Triton backend, v2 top-k kernel (`SGLANG_OPT_USE_TOPK_V2=true`) |
| scheduling | chunk 32768 (8192 per DP rank), `--max-prefill-tokens 8192`, max running 128, decode graphs up to 64, prefill delayer up to 60 passes |
| HiCache | 270 GB host pool per rank, ratio 1.5, `write_through`, `kernel` I/O, `page_first` layout |
| router | Rust backend, `kv-aware` with `--kv-overlap-weight 1.0`, `--session-affinity both` |

These are the settings of the best MI355X point on the InferenceX axes
(tokens/s/GPU against interactivity p90) among the aggregated runs that
preceded this kit; `write_back`, a larger memory fraction or a longer delayer
cap raised throughput by at most noise at C64 while lowering interactivity.
The v2 top-k kernel then added 8.6% throughput and 18% interactivity p90 on
the same node; leaving `index_share_for_mtp_iteration` at its default changed
nothing beyond noise.

## Validation status

The image above, GPUs 0-3 of an 8×MI355X node (crsuse2-m2m-138 unless noted).
Rows marked * ran before `--disable-custom-all-reduce` and the v2 top-k kernel
were added, and were not repeated.

| item | result |
|---|---|
| correctness (`smoke.py`, real MTP acceptance)* | 6/6 pass (66k-token prefix: 9.8 s cold, 1.0 s repeated) |
| GSM8K, concurrency 1 (24 questions, real MTP acceptance)* | 0.958 |
| GSM8K, concurrency 64 (`eval.sh`'s 200 prompts, real MTP acceptance, last number in the answer scored; host pool 64 GB instead of 270 GB) | 0.980, no garbled answer (about 30% garbled without `--disable-custom-all-reduce`, note 6) |
| long context (128 requests of ~14k tokens at concurrency 32: needle retrieval, GSM8K behind filler; crsuse2-m2m-140) | needle 64/64, GSM8K 63/64, nothing garbled |
| AgentX C48, 3600 s, `SIM_ACCEPT_LEN=3.61` (crsuse2-m2m-140) | 34,662 tok/s/GPU, 3,699 requests, 1 invalid response |

## Reference numbers

AgentX agentic-coding replay, InferenceX `918524ff`, 10 warm-up requests per
lane, 3600 s measured, forced MTP acceptance 3.61. Throughput is per GPU over
the 4 GPUs and counts cache-hit input tokens; interactivity is per-request
tokens/s/user.

| point | tok/s/GPU | requests (per s) | interactivity p50 / p90 | TTFT p50 / p90 | ITL p50 / p90 | prompt cache hit |
|---|---|---|---|---|---|---|
| C48 | 34,662 | 3,699 (1.020) | 42.3 / 22.7 | 3.8 / 10.6 s | 23.6 / 44.1 ms | 96.5% (trace bound 97.2%) |

Measured on crsuse2-m2m-140. On that node the recipe without the v2 top-k
kernel gave 31,931 tok/s/GPU, interactivity p90 19.3; the identical
configuration ran 3.5% faster on crsuse2-m2m-138, so differences within a few
percent between nodes are noise.

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

**4. One kit per node.** The engine runs with `--ipc host`, and RCCL names its
shared-memory segments after process ids, which start low in every container.
A second engine started on the other four GPUs fails in `ncclCommInitRank` with
`HIP failure: 'invalid argument'`; run a second deployment on another node.

**5. Session affinity reads request headers.** `--session-affinity both` keeps
requests with the same `x-session-id` on the DP rank their session used last;
`agentx.sh` has AIPerf send it (`AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID`).

**6. Keep `--disable-custom-all-reduce`.** With the AITER custom all-reduce
on, about 30% of GSM8K answers at concurrency 64 turn into token soup
(template tags such as `</think>` as plain text, or runs of `!`); at
concurrency 1 the model is fine. It takes DP attention together with decode
CUDA graphs, and every upstream image from v0.5.19 to v0.5.22
(`lmsysorg/sglang-rocm`) shows it. Neither disabling QuickReduce
(`ROCM_QUICK_REDUCE_QUANTIZATION=NONE`) nor the narrower switches
`SGLANG_AITER_CAPTURE_COPY_IN=1`, `SGLANG_DP_USE_REDUCE_SCATTER=0` or
`SGLANG_USE_AITER_AG=0` removes it. Collectives on RCCL cost no throughput at
C48 but lowered interactivity p90 by about 15% in one comparison.

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
