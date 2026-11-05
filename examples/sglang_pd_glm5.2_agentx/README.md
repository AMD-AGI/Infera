# GLM-5.2-MXFP4 — SGLang PD, tuned for InferenceX AgentX, on MI355X

Runnable kit for **GLM-5.2-MXFP4** on the InferenceX AgentX agentic-coding
replay, PD-disaggregated the infera way: **TP1 × PP4 prefills** with a
**250 GB-per-stage host-memory HiCache tier**, a **TP + DP-attention decode**
with EAGLE MTP, KV moved by **Mooncake over the eight ionic RDMA rails**, and
the infera router in front (**kv-aware**, **session affinity on prefill
picks**). It is [`sglang_2p1d_glm5.2`](../sglang_2p1d_glm5.2/README.md) plus the
tuned AgentX recipe: KernelForge (KF) kernels and tuned GEMM rows on the
decode, the NextN draft MoE in FP8, three prefill scheduling / HiCache
patches, and the DSA decode-graph fix that makes AITER's collectives safe
under DP attention (README "Recipe").

The default is the **4+4 shape** (one PP4 prefill on 4 GPUs of the prefill
node, a TP4 + DP4-attention decode on 4 GPUs of the decode node; 8 GPUs);
`PREFILL_WORKERS=2` with an 8-GPU decode is **2P1D** (16 GPUs).

One file is site-specific. Fill it in, then, on the prefill node:

```bash
$EDITOR env.sh                 # the only file you edit
bash engine/up.sh              # check the image, etcd -> prefill(s) + decode (over ssh) -> router
bash engine/smoke.sh           # PD registration, completion/chat/stream, 66k-token prefix reuse
bash engine/eval.sh            # GSM8K as InferenceX runs it, 1,319 questions
bash engine/down.sh            # save logs, remove containers on both nodes
bash engine/agentx/sweep.sh 64 104   # AgentX points, a fresh deployment each (SIM_ACCEPT_LEN=2.99)
```

## Image

Build `deploy/docker/Dockerfile.sglang` from the repo root on both nodes (or
build once and `docker save | docker load` it across). It is pinned to the
SGLang v0.5.22 nightly `lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261018`
and carries the GLM-5.2 DSA, HiCache, KV-events, pipeline-stage and Mooncake
RDMA patches:

```bash
docker build -f deploy/docker/Dockerfile.sglang -t <infera-sglang-image> .
```

`patches/` holds whole-file copies of eleven SGLang files of that image.
`up.sh` first checks the image on both nodes against
[`patches/BASE_MD5SUMS`](patches/BASE_MD5SUMS) and stops on any difference,
since a copy laid over another version of a file would silently drop that
version's changes (re-cut the patches from the `.diff` files then).

## Contents

| path | what |
|---|---|
| [`env.sh`](env.sh) | **the only file you edit** — node IPs, ssh target, weights, image, scratch dir, RDMA rails, shape, GPUs, the KF / draft-FP8 switches, ports |
| [`engine/worker.sh`](engine/worker.sh) | starts ONE engine container on its own node (`prefill 0`, `prefill 1` or `decode`); carries the tuned recipe and its mounts |
| [`engine/up.sh`](engine/up.sh) | image check → etcd → `PREFILL_WORKERS` prefills here and the decode over ssh → router, waiting on each health endpoint and on the router listing every prefill |
| [`engine/smoke.sh`](engine/smoke.sh), [`engine/smoke.py`](engine/smoke.py) | correctness checks through the router; need real MTP acceptance |
| [`engine/eval.sh`](engine/eval.sh) | GSM8K through the router with InferenceX's `run_eval` (its lm-eval revision, `utils/evals/gsm8k.yaml`, full test set) |
| [`engine/agentx/agentx.sh`](engine/agentx/agentx.sh) | one InferenceX AgentX point, pinned at `918524ff`, counted over all the deployment's GPUs (8, 12 or 16) |
| [`engine/agentx/sweep.sh`](engine/agentx/sweep.sh) | AgentX points one after another, each on a fresh deployment, with a stall watchdog; `summary.md` at the end |
| [`engine/agentx/summarize.py`](engine/agentx/summarize.py) | one Markdown row per point from run directories |
| [`engine/agentx/constraints.txt`](engine/agentx/constraints.txt) | AgentX client package pins |
| [`engine/down.sh`](engine/down.sh) | save logs + inspect, remove the kit's containers by name on both nodes |
| [`engine/common.sh`](engine/common.sh) | sourced helpers: placeholder check, run directory, health wait, image check, InferenceX checkout |
| [`patches/`](patches) | whole-file SGLang patches mounted into the engines, a `.diff` per group for review, `BASE_MD5SUMS` (README "Source patches") |
| [`kf_overlay/`](kf_overlay) | the KF kernels and the `sitecustomize.py` that installs them (README "KF kernels") |
| [`aiter_csv/`](aiter_csv) | tuned decode GEMM rows (README "Tuned GEMM rows") |

`up.sh` creates `$WORK_DIR/runs/<RUN_ID>/` on the prefill node with
`engine-config.env` (the settings started, also handed to the decode node); the
other scripts read those settings back, so an override given to `up.sh` alone
still applies to them, and write `smoke/`, `gsm8k/`, `agentx/` and `logs/` there.
`sweep.sh` writes its logs and `summary.md` to `$WORK_DIR/sweeps/<SWEEP_ID>/`.

## Topology

```
prefill node                                      decode node
├─ etcd, infera router :28300  ◀── clients
├─ prefill 0  GPUs 0-3  TP1×PP4  ══ KV over 8 ionic rails ══▶  decode  GPUs 0-3 (4+4)
└─ prefill 1  GPUs 4-7  TP1×PP4  (2P1D only)                   or 0-7 (2P1D)
   250 GB host HiCache per stage                               TP4/DP4 or TP8/DP8, EAGLE MTP
```

Both nodes need the same `MODEL` path, the image under the same tag, and this
kit at the same path (the decode is started as `ssh $DECODE_SSH bash
<kit>/engine/worker.sh decode`), readable by the docker daemon: the engines
bind-mount files from `patches/`, `kf_overlay/` and `aiter_csv/`, which fails
from a root-squashed NFS home. With `DECODE_NODE_IP` equal to
`PREFILL_NODE_IP` and `PREFILL_WORKERS=1`, the 4+4 shape runs on one node
(prefill and decode on different GPUs, note 4); the reference numbers below are
from two nodes.

## Recipe

| | prefill (× 1 or 2) | decode |
|---|---|---|
| parallelism | TP1 × PP4, layers 20 / 20 / 20 / 18 (the last stage also holds the LM head and the NextN layer), EP1 (note 11) | TP4 (or TP8), DP attention over all ranks, EP1 |
| all-reduce | RCCL: `--disable-custom-all-reduce` (note 6) | AITER custom all-reduce; MoE gather / combine through AITER's collectives (`SGLANG_USE_ROCM700A=0`); both need `patches/dsa-graph-variant` (note 6) |
| memory | `--mem-fraction-static 0.88` | `--mem-fraction-static 0.95` |
| scheduling | chunk 8192, overlap scheduler off (PP needs it), `--pp-async-batch-depth 1`, stage sends overlapped with compute (`SGLANG_PP_COMM_OVERLAP=1`), max running 256, `--schedule-policy shortest-prefill-first` | max running 256, decode graphs up to 256, 4 tokenizer workers, `--stream-interval 30`, max_new_tokens estimate clipped to 8 steps (`SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=8`) |
| KV | FP8, one pool for the four stages, HiCache 250 GB per stage, ratio 1.5, `write_back`, `kernel` I/O, `page_first`; evictions free at least 512K tokens per call, the device-to-host backup copy runs 16 workgroups (patches) | FP8, no offload |
| MTP | the decode's EAGLE flags, so the transfer carries draft-layer KV (note 10) | EAGLE 3 steps / top-k 1 / 4 draft tokens; `index_share_for_mtp_iteration=true` (the checkpoint's own value); draft MoE in FP8 (`DRAFT_MOE_FP8=1`) |
| DSA | Triton prefill and decode backends, v2 top-k kernel | Triton backends with the KF sparse-MLA and indexer kernels (`KF_KERNELS=1`), v1 top-k kernel |
| MoE | aiter MXFP4 | KF MXFP4 MoE (`KF_KERNELS=1`; TP4 only, at TP8 its shapes fall back to aiter) |
| router | Rust backend, `kv-aware` with prefill / decode overlap weights 20 / 2, `--session-affinity prefill`, `--pd-prefill-guard-release completion`, `--decode-input-demand on`, `--decode-request-weight 0` | |

Three draft steps instead of five: at AgentX's acceptance the fourth and fifth
draft tokens cost more decode time than they return, and InferenceX scores a
three-step deployment at its golden acceptance length 2.99
(`SIM_ACCEPT_LEN=2.99`).

What each part was worth, 4+4 at C104 on one pair of AUS MI355X nodes
(prefill smci355-ccs-aus-n06-25, decode n06-33), forced acceptance 2.99, 3600 s
measured, one fresh deployment per row (tok/s/GPU):

| configuration | tok/s/GPU | interactivity p50 / p90 | TTFT p50 / p90 s |
|---|---|---|---|
| the recipe without the DSA graph patch, `SGLANG_USE_ROCM700A=1` | 50,797 | 63.7 / 55.7 | 4.55 / 11.41 |
| + `patches/dsa-graph-variant` | 51,477 | 63.9 / 56.2 | 4.29 / 10.93 |
| + `SGLANG_USE_ROCM700A=0`, `DRAFT_MOE_FP8=0` | 52,830 | 68.3 / 61.3 | 4.30 / 10.17 |
| + `DRAFT_MOE_FP8=1` (this kit's default) | **53,666** | 70.8 / 61.9 | 4.12 / 9.28 |

The first two rows ran with the draft MoE FP8 switch on but without
`patches/draft-moe-fp8`. The first row garbles output under load with real
acceptance (note 6), so it is a throughput reference only. Forced acceptance hides what the draft
precision does to real acceptance; with real acceptance the default recipe
accepts 3.62 tokens per step (README "Validation status").

## Source patches

Each group is a set of whole files mounted read-only over
`/sgl-workspace/sglang/python/sglang/srt` in one role, with its change as a
`.diff` (paths relative to that directory) next to it.

| group | role | files | what | upstream |
|---|---|---|---|---|
| [`dsa-graph-variant`](patches/dsa-graph-variant.diff) | decode | `layers/attention/graph_variants.py`, `model_executor/runner/decode_cuda_graph_runner.py`, `model_executor/forward_batch_info.py`, `managers/schedule_batch.py`, `managers/scheduler_components/dp_attn.py`, `batch_overlap/two_batch_overlap.py` | one TARGET_VERIFY graph; the decode graph variant from the maximum sequence length over all DP ranks (note 6) | [sglang#48302](https://github.com/sgl-project/sglang/pull/48302) (open) |
| [`draft-moe-fp8`](patches/draft-moe-fp8.diff) | decode, `DRAFT_MOE_FP8=1` | `models/glm4_moe.py` | Quark's leaf excludes kept the fused NextN experts unquantized while the loader cast their weights to FP8 (real acceptance about 1.2); quantize them as FP8 | backport of [sglang#45800](https://github.com/sgl-project/sglang/pull/45800) (merged after v0.5.22) |
| [`pp-chunk-wait`](patches/pp-chunk-wait.diff) | prefill | `disaggregation/prefill.py` | with PP the scheduler runs ahead of the forward stream (async batch depth, comm overlap), so the KV sender waits on an event recorded after the chunk's launch instead of reading KV the chunk may still be writing | Infera only |
| [`evict-headroom`](patches/evict-headroom.diff) | prefill | `mem_cache/unified_radix_cache.py` | an eviction frees at least `SGLANG_EVICT_HEADROOM_TOKENS`: a `write_back` eviction blocks the scheduler until the running forward ends, so fewer, larger ones; the victims are demoted to host, not dropped | Infera only |
| [`hicache-backup-quota`](patches/hicache-backup-quota.diff) | prefill | `mem_cache/pool_host/mla.py`, `mem_cache/pool_host/dsa.py` | the all-layer device-to-host write-back copy runs `SGLANG_HICACHE_BACKUP_BLOCK_QUOTA` workgroups (the kernel's default is 2) | Infera only |

## KF kernels

[`kf_overlay/`](kf_overlay) is a KernelForge kernel set for GLM-5.2's decode:
sparse MLA (`YIHOU_KF_SPARSE`), the DSA indexer (`YIHOU_KF_INDEXER`) and the
MXFP4 MoE (`YIHOU_KF_MOE`). With `KF_KERNELS=1` the decode container mounts it
and puts it first on `PYTHONPATH`; its `sitecustomize.py` runs the image's own
first, then installs the kernels over SGLang's in every python process when one
of the switches is set. The MoE kernel covers the TP4 decode's shapes; at TP8
its gate-up GEMM falls back to aiter. Provenance: the KernelForge campaigns
catalogued in Infera [#221](https://github.com/AMD-AGI/Infera/pull/221); SGLang
ports of the attention kernels are
[sglang#47278](https://github.com/sgl-project/sglang/pull/47278) (sparse MLA),
[sglang#47897](https://github.com/sgl-project/sglang/pull/47897) and
[sglang#47872](https://github.com/sgl-project/sglang/pull/47872) (indexer), and
[sglang#47276](https://github.com/sgl-project/sglang/pull/47276) (fused
indexer, not used here); the MoE kernel has none.

## Tuned GEMM rows

[`aiter_csv/`](aiter_csv) holds BF16 GEMM rows tuned for the decode's shapes,
mounted into aiter's `configs/model_configs/`, which aiter merges with its own
tables. aiter looks a GEMM up by M, then M padded to 128, then the next power
of two, so a row covers a range of batch sizes. The prefill uses none of them.

## Validation status

From a copy of this kit (the same flags and mounted files) on an image built
from Infera `fdbafb60`, whose `Dockerfile.sglang` differs from this branch's
only by the pipeline-stage patch (mounted into its prefills instead), on the
AUS pair above, with real MTP acceptance:

| item | result |
|---|---|
| GSM8K probe (1,319 questions through the router) | 0.971; 0 answers garbled or wrong from the first token (note 6), 16 at the 16,384-token limit |
| long context: 16k-token needle × 128, 16k-token prefix + GSM8K × 64; 100k × 32 each | 64/64, 62/64; 16/16, 16/16 (the stock image misses the same two prefixed questions) |
| GSM8K (`eval.sh`: InferenceX `run_eval`, concurrency 64) | 0.9697 strict / 0.9689 flexible |
| real acceptance length (decode log, every DP rank) | 3.62 tokens per step (about 1.2 with `DRAFT_MOE_FP8=1` before `patches/draft-moe-fp8`) |
| the stock SGLang v0.5.22 image, same checkpoint (BF16 KV), same probe | 0.973, and 0.966 on a rerun: the questions this recipe misses are the model's (batch composition flips borderline ones) |
| AgentX, `SIM_ACCEPT_LEN=2.99` | "Reference numbers" |

## Reference numbers

AgentX agentic-coding replay, InferenceX `918524ff`, 10 warm-up requests per
lane, 3600 s measured, forced MTP acceptance 2.99, every point on a freshly
started deployment, same AUS pair and image as "Validation status".
Throughput is per GPU over all of the deployment's GPUs and counts cache-hit
input tokens; interactivity is per-request tokens/s/user (InferenceX plots
p90).

| shape | point | GPUs | tok/s/GPU | requests (per s) | interactivity p50 / p90 | TTFT p50 / p90 s | ITL p50 / p90 ms | prompt cache hit |
|---|---|---|---|---|---|---|---|---|
| 2P1D | C208 | 16 | 54,094 | 23,073 (6.36) | 77.2 / 65.7 | 4.16 / 8.01 | 12.9 / 15.2 | 95.8% (trace bound 96.9%) |
| 4+4 | C112 | 8 | 52,478 | 12,321 (3.40) | 70.1 / 61.8 | 5.53 / 14.74 | 14.3 / 16.2 | 96.0% (trace bound 96.6%) |
| 4+4 | C104 | 8 | 53,666 | 12,661 (3.49) | 70.8 / 61.9 | 4.12 / 9.28 | 14.1 / 16.2 | 96.1% (trace bound 96.7%) |

C104 is the 4+4 peak: at C112 both legs are saturated. 2P1D at C208 is
prefill-bound (requests waiting for a prefill 92% of the time) while its decode
has room. Lower points of the Pareto line: `sweep.sh` with smaller CONC.

```bash
# 4+4 (env.sh defaults)
bash engine/agentx/sweep.sh 104
# 2P1D
PREFILL_WORKERS=2 DECODE_GPUS=0,1,2,3,4,5,6,7 bash engine/agentx/sweep.sh 208
```

## InferenceX

InferenceX accepts the recipe's flags but not the rest as is:

- `DRAFT_MOE_FP8=1`: the draft model must run in the precision it ships in
  ([CONTRIBUTING, "Draft model precision"](https://github.com/SemiAnalysisAI/InferenceX/blob/57116fff418d7bb651567e722907253a12b75a67/CONTRIBUTING.md#draft-model-precision));
  run `DRAFT_MOE_FP8=0` (52,830 at 4+4 C104).
- `patches/`, `kf_overlay/` and `aiter_csv/`: a submission runs a released
  image without engine patches or overlays; each part needs to land upstream
  (see the tables above) or in the image first.

## Notes

**1. Forced MTP acceptance produces wrong text.** `SIM_ACCEPT_LEN` makes the
decode accept drafts at a fixed rate whether they are right or not. Use it only
for performance runs; run `smoke.sh` and `eval.sh` on a deployment started
without it.

**2. Cold start takes 10-15 minutes**: weights, the 250 GB pinned host pool
of every prefill stage (four per prefill), AITER JIT and the decode's graph
capture. JIT caches
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

**6. Custom all-reduce: the decode only, and only with
`patches/dsa-graph-variant`.** With DP attention, SGLang v0.5.22 picks the DSA
decode graph variant (dense or sparse) and the TARGET_VERIFY variant per DP
rank, from that rank's own batch. AITER's collectives (the custom all-reduce,
and with `SGLANG_USE_ROCM700A=0` the MoE token gather / combine) register their
buffers per captured graph, so ranks replaying different variants exchange
data through mismatched buffers: garbled output under load (200-question
GSM8K probes: 26 answers garbled or wrong from the first token over two
probes without the patch, 53 garbled with `SGLANG_USE_ROCM700A=0`; none in
1,000 with it). The patch
captures one TARGET_VERIFY graph and picks the decode variant from the maximum
sequence length over all DP ranks (a host-side bound carried in the existing
DP sync), so every rank replays the same graph (sglang#48302). The prefills
keep `--disable-custom-all-reduce`: their tensor sizes differ per rank, and
the all-reduce can pick a different backend than its peers and deadlock.

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
in about 19 instance hours of the previous TP4 + DP4-attention prefills (none
yet with PP4), a prefill rank died with `Memory access fault
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
roles for this reason.

**11. Why TP1 × PP4 prefills, and what they cost.** A TP4 + DP4-attention
prefill steps all four ranks together, so one rank computing holds the whole
instance (busy 81% of the time at 1P1D C88 while its ranks computed 57%), and
session affinity pins a long session to one hot rank. Four pipeline stages
share one KV pool and have no such barrier: one PP4 prefill computes about 26k
new tokens/s on cold prompts and sustains about 3.0 AgentX requests/s, where a
TP4 + DP4-attention one saturated near 2.75. The costs:
- A batch crosses all four stages before its KV can go, so a single request
  is not faster: an isolated 66k-token cold prompt takes 37.5 s against 14.2 s.
  8k chunks keep each stage short: at 1P1D C88, TTFT p90 was 18.7 s with 16k
  chunks and 12.1 s with 8k, small continuation turns no longer waiting behind
  long ones.
- With the prefill saturated, shortest-prefill-first keeps pushing long cold
  first turns back (TTFT p90 of prompts with 32k or more new tokens 224 s at
  1P1D C88).
- SGLang releases a transferred request's KV only after two rounds of
  consensus across the stages, so finished requests briefly hold prefill KV.
- Middle stages hold no embedding; the image's pipeline-stage patch keeps
  `deepseek_v2.py` from reading one there.
**12. A decode on the prefill node needs `MC_DISABLE_HIP=1`.** Mooncake
(`faae8dd4`) always installs its HIP transport next to RDMA and prefers it for
a destination on the same host: the KV then moves as XGMI peer copies, which
under serving load ran at ~1 GB/s per transfer. One-node 4+4 at C104 (prefill
GPUs 0-3, decode 4-7, the pre-`MC_DISABLE_HIP` package): 40,807 tok/s/GPU,
prefill full chunk 572 ms, 6.9 s per request waiting for a decode slot; with
`MC_DISABLE_HIP=1` the KV goes over RDMA loopback on the destination GPU's
rail (both ends use the same ionic device, as `MC_ENABLE_DEST_LOCAL_RAIL`
picks it; two different devices of one host do not connect): 48,635 over a
1500 s window, prefill full chunk 322 ms as on two nodes. Idle, HIP copies run
at 63 GB/s and RDMA loopback at 24-34 GB/s, so the loss comes from contention
under load. `MC_DISABLE_HIP_TRANSPORT=1`, which earlier kits set, does not
turn the HIP transport off: with it the ionic RDMA counters stayed near zero
during the 40,807 run. Transfers between nodes never take the HIP transport.

## What this kit exposes on the network

| port | node | service |
|---|---|---|
| 28300 | prefill | router (OpenAI API, `/metrics`, `/v1/workers`) |
| 27201, 27457 | prefill | prefill engine HTTP |
| 28398, 28399 | prefill | PD bootstrap |
| 27757, 27765 | prefill | KV events, one per prefill (its first stage publishes) |
| 29051, 29052 | prefill | KV snapshot |
| 24701 / 24702 | prefill | etcd client / peer (peer on 127.0.0.1) |
| 27301 | decode | decode engine HTTP (SGLang adds 27534-27548) |

All bind on the host network without authentication. `--trust-remote-code` runs
the checkpoint's Python; serve only weights you trust.
