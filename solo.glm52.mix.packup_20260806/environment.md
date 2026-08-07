# Environment — the exact HW/SW these numbers came from

**This is the same live deployment that produced Phase 1.** One container, created
`2026-08-06T07:01:26Z`, brought up once and never restarted: the fixlen sweep ran
07:13:48 → 09:45:14, and the three solo arms in this packup ran 09:53:41 → 11:52:03
on that same process. Nothing was reconfigured in between.

Consequently the hardware/software table below is **carried across from**
[`../fixlen.glm52.mix.packup_20260806/environment.md`](../fixlen.glm52.mix.packup_20260806/environment.md),
which holds the fuller derivation (global-vs-per-rank flag analysis, source line
references, derived KV-pool state). The essentials are restated here so this
packup stands alone.

Raw snapshot: [`env/env_chi2835.txt`](env/env_chi2835.txt), captured by
`scripts/envsnap.sh` at `2026-08-06T08:12:14Z` — i.e. **before** these three arms,
during the Phase-1 sweep, on the same unrestarted process. The engine's own
resolved `server_args` (not what was requested) is
[`env/resolved_server_args.txt`](env/resolved_server_args.txt); read that file,
not the launch flags.

**Gap, stated rather than guessed:** no env snapshot was taken *between* the solo
arms. The container identity was re-verified while assembling this packup —
`/v1/workers` still returned exactly one active worker,
`disagg_mode: "mixed"`, `dp_size: 8`, `kv_block_size: 64`, at
`http://10.2.122.78:30000` — which establishes the deployment was unchanged, but
does not re-snapshot per-arm VRAM.

## Hardware

| item | value |
|---|---|
| node | `chi2835`, single node (no second leg — this is mix, not PD) |
| GPU | 8 × **AMD Instinct MI355X**, `gfx950`, card model `0x75a3` |
| VRAM | 2,016 GB total across the 8 cards (~252 GB/card) |
| amdgpu driver | **6.16.13** |
| ROCm | **7.2.0** (from the container's torch build string) |
| host kernel | **6.8.0-107-generic** |
| CPU | 2 × **AMD EPYC 9575F** 64-Core, 256 logical CPUs |
| host RAM | 3,023 GB |
| data-plane IP | **10.2.122.78** on `enp193s0f1np1` |
| mgmt IP | 45.76.23.123 — **not** the one to use |
| RDMA fabric | 8 × `ionic_0..7` present, **not load-bearing** |

RDMA is present but carries nothing here: mix means prefill and decode share the
same 8 GPUs and KV never crosses a wire. `NCCL_IB_DISABLE=1` is set.

## Software

| item | value |
|---|---|
| image tag | `infera/engine-sglang:merged-e` |
| **image id (pin on this, not the tag)** | `sha256:27667ee43291bed2bddb9caf44a63217fdb994d6f423f6ed3bf7e807340fae7a` |
| base image | `lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x` |
| sglang | **0.5.15.post1** |
| torch | **2.9.1+rocm7.2.0.git7e1940d4** |
| amd-infera | `0.0.0` at `/opt/venv/lib/python3.10/site-packages` (in-image build; the version string is not meaningful — the image id is the real pin) |
| etcd | `quay.io/coreos/etcd:v3.5.14` |
| container binds | `["/mnt/vast:/mnt/vast"]` |
| container created | `2026-08-06T07:01:26Z` |

**Repo state:** assembled from the working tree of `infera.glm52.mix.experiment`
on branch `dev.yihou.glm52.mix.experiment`. The scripts and workload YAMLs under
`scripts/` and `specs/` are **md5-verified byte-identical** to the copies that
executed on the cluster at `/mnt/vast/c_huggingface/glm52_mix_20260806/` — so the
files in this packup *are* the ones that ran, and no repo checkout is needed.

| file | md5 (this packup == cluster copy) |
|---|---|
| `specs/mix_solo_p50.yaml` | `ec7a0b57287be86affaa689cca1065ad` |
| `specs/mix_solo_p90.yaml` | `2dcb2db45106af9f5e13542b6852cd00` |
| `specs/mix_solo_p99.yaml` | `b12efc58bf2356906ddd9836b6b12c82` |
| `scripts/run_agentic.sh` | `7a711f780cd067b77c08961444d4c4d8` |
| `scripts/analyze_solo.py` | `d3cdda3782103d464c17d9a8748295f8` |
| `scripts/envsnap.sh` | `bba5f20cedb82b48be4c0855e406ef9e` |
| `scripts/accept_len.sh` | `4dcb0f64a98a4c8e58e82655fd06e1ce` |

No git command was run while assembling this packup, so **no commit SHA is
recorded**. Gap, stated rather than guessed: read it off the branch tip dated
2026-08-06 if you need it.

## The benchmark driver — this is the part unique to Phase 2

The agentic driver is **not** in this repo. It is a staged copy on the shared
mount, and it carries a local patch that the reported numbers depend on.

| item | value |
|---|---|
| staged driver | `/mnt/vast/c_huggingface/bench_20260801/agbench` |
| python | `/mnt/vast/c_huggingface/bench_20260801/venv/bin/python` — **3.12.3** |
| upstream repo | `Optimus-AgenticBench` |
| upstream branch | `fix/realistic-profile-session-driver` |
| upstream commit | **`1cf01cbf169d9370a0bc8fe574055c5e975d1be9`** (`1cf01cb`) |
| local patch | **SOLO_M1**, staged-only — never committed upstream |
| entry point | `python -m agent.agent_throughput` |

**The patch baseline is provable, not assumed.** The staged tree keeps
`agent/agent_throughput.py.orig` alongside the patched file, and that `.orig`
md5s to `2aa74d1d983984c1b53a3f27d51ebbaa` — **identical** to
`agent/agent_throughput.py` in the pristine local checkout at `1cf01cb`. So the
diff between them is exactly the patch and nothing else. Patched file md5:
`8f482b8fba8ba69d02c767a3618d1a36`.

The patch itself is [`patches/solo_m1_per_request_e2e_tpot.patch`](patches/solo_m1_per_request_e2e_tpot.patch)
(33 changed lines) with its rationale in
[`patches/README.md`](patches/README.md). It was verified to apply cleanly to the
pristine checkout with `patch -p1 --dry-run` while assembling this packup.
**Without it the E2E column in the results table cannot be measured at all** —
only back-solved from TTFT and TPOT.

## Deployment shape

One mix worker — `disagg_mode: "mixed"` in the registry,
`disaggregation_mode='null'` in `server_args` — launched through
`python3 -m infera.engine.sglang`, TP8, both phases on the same 8 GPUs. Router on
`:8100`, engine on `:30000`, etcd on `:2379`. **All benchmark traffic goes through
the router**, not straight to the engine — otherwise kv-aware routing is bypassed
and the deployment under test is not the one configured.

```
TP=8  GMU=0.80  CHUNK=65536  CTX=262144  MAX_RUNNING=256  CUDA_GRAPH_BS=128
DPA=1 (dp8)  MTP=1 (EAGLE)  KVAWARE=1  KVD=1  HICACHE_GB=32
ROUTER_POLICY=kv-aware  ROUTER_BACKEND=rust
--kv-cache-dtype fp8_e4m3   --ep-size 8   --disable-custom-all-reduce
--reasoning-parser glm45    --enable-cache-report
--nsa-prefill-backend tilelang  --nsa-decode-backend tilelang
--speculative-algorithm EAGLE --speculative-num-steps 3
--speculative-eagle-topk 1 --speculative-num-draft-tokens 4
--enable-prefill-delayer --prefill-delayer-max-delay-ms 5000
kvd: --max-bytes 64G --long-path /tmp/kvd-long --long-bytes 64G
```

**Cold start to `/health`: 390 s.** Weights + aiter/tilelang JIT + CUDA-graph
capture. Not a hang — do not kill it.

### Two flags DP-attention divides — read the resolved value, not the request

Both are **global** budgets that SGLang divides by `dp_size`:

| flag | requested (global) | **per rank** | how it surfaces |
|---|---|---|---|
| `--chunked-prefill-size` | 65536 | **8192** | explicit `WARNING` in the engine log |
| `--max-running-requests` | 256 | **32** | **no warning** — only the per-rank scheduler line |

Confirmed in the image's own source (`srt/server_args.py:4902`, gated on
`enable_dp_attention`; `srt/model_executor/pool_configurator.py:541-543`). Full
analysis in the Phase-1 packup's `notes.md` §3. It matters here because at
concurrency 1 neither budget binds — but if you reproduce at a different
concurrency, they will.

### Derived engine state, from the engine's log

| item | value |
|---|---|
| KV pool | `max_total_num_tokens = 2,812,672`; **144.66 GB** main KV per rank + 1.85 GB (MTP draft pool) |
| KV dtype | `torch.float8_e4m3fn` |
| page size | 64 |
| attention backend | `dsa`, prefill+decode via `tilelang` |
| decode CUDA graphs | captured, `max_bs=128`; prefill graphs disabled |
| hicache | `enable_hierarchical_cache=True`, `hicache_size=32`, backend `dynamic` → `InferaKvdBackend`, `prefetch_threshold=64`, `write_through` |
| KV events | zmq `tcp://*:24142`, topic `kv-events` |

The 235K-token p99 prompt sits comfortably inside `context_len=262144` — but only
just. A larger Case-A shape would need `--context-length` raised.

### The DSA-on-ROCm env block — mandatory on gfx950

Without these the model still serves and still returns HTTP 200 — it returns
**garbage**, because the sparse-attention indexer takes a path not ported to ROCm.
Verified present in the running process's env:

```
SGLANG_USE_AITER=1                  SGLANG_ROCM_FUSED_DECODE_MLA=0
SGLANG_OPT_USE_TILELANG_INDEXER=1   SGLANG_OPT_USE_TOPK_V2=0
SGLANG_OPT_USE_JIT_NORM=0           SAFETENSORS_FAST_GPU=1
HIP_FORCE_DEV_KERNARG=1             PYTHONHASHSEED=0
SGLANG_DP_USE_GATHERV=1             (DPA only)
NCCL_IB_DISABLE=1  NCCL_IGNORE_CPU_AFFINITY=1  HSA_NO_SCRATCH_RECLAIM=1
```

`PYTHONHASHSEED=0` is there so block hashes — and therefore kvd keys — are stable
across restarts. That matters directly to this phase: the whole measurement rests
on a shared prefix staying resident.

## External dependencies (absolute paths, NOT in this repo)

| what | where | notes |
|---|---|---|
| model weights | `/mnt/vast/xiaobo/models/GLM-5.2-MXFP4` | shared VAST mount. Also the tokenizer path, the EAGLE draft path, **and** the tokenizer the agentic driver loads to build prompts (vocab 154,820). |
| agentic driver | `/mnt/vast/c_huggingface/bench_20260801/agbench` | staged copy at `1cf01cb` + SOLO_M1. See above. |
| driver venv | `/mnt/vast/c_huggingface/bench_20260801/venv/bin/python` | python 3.12.3. Note it reports `PyTorch was not found` on startup — harmless, the driver only uses `tokenizers`. |
| staging dir | `/mnt/vast/c_huggingface/glm52_mix_20260806/` | scripts / specs / logs / results. Identity-masked on purpose. |
| kvd L3 spill | `/tmp/kvd-long` **inside the container** | `--long-bytes 64G` is ABSOLUTE. The ratio-based default can ask for hundreds of GB per rank and fill the node's root fs. |

`/mnt/vast` is shared to the compute nodes; `/tmp` is **not** — the engine, router
and kvd logs live inside the container and must be `docker exec`'d out.

## Required secrets — names and sources only, no values

| secret | source |
|---|---|
| cluster SSH | `ssh root@149.28.124.225` (jump host chi2866), then `ssh chi2835`. Key-based; arrange your own. |
| docker registry | the image was **already present on the node**; nothing was pulled. If you must pull, use the team registry credentials from the team vault. |
| etcd / router / kvd | none — no auth configured; all bound to the node. |
| model weights | none — readable on the shared mount. |

No API key, token, or password is used by any script in this packup, and none
appears in any copied file or log.
