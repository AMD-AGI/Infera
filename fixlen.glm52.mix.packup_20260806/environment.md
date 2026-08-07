# Environment — the exact HW/SW these numbers came from

Captured **on the node while the deployment was live**, by
`scripts/envsnap.sh`, at `2026-08-06T08:12:14Z` — i.e. mid-sweep, between the
p90 and p99 arms. Raw output: [`env/env_chi2835.txt`](env/env_chi2835.txt).

The `server_args` the engine actually resolved (not what was requested) is in
[`env/resolved_server_args.txt`](env/resolved_server_args.txt), one field per
line. **Read that file, not the launch flags** — two values differ, see
"Requested vs resolved" below.

## Hardware

| item | value |
|---|---|
| node | `chi2835`, single node (no second leg — this is mix, not PD) |
| GPU | 8 × **AMD Instinct MI355X**, `gfx950`, card model `0x75a3` |
| VRAM | 2,016 GB total across the 8 cards (~252 GB/card) |
| amdgpu driver | **6.16.13** |
| ROCm | **7.2.0** (from the container's torch build string) |
| host kernel | **6.8.0-107-generic** |
| CPU | 2 × **AMD EPYC 9575F** 64-Core, 256 logical CPUs, 1 thread/core |
| host RAM | 3,023 GB |
| data-plane IP | **10.2.122.78** on `enp193s0f1np1` |
| mgmt IP | 45.76.23.123 on `enp193s0f0np0` — **not** the one to use |
| RDMA fabric | 8 × `ionic_0..7` present |

**RDMA is present but not load-bearing for this run.** This is a mix
deployment: prefill and decode share the same 8 GPUs and KV never crosses a
wire. `NCCL_IB_DISABLE=1` is set. The fabric is listed for completeness and
because a PD comparison would depend on it.

`ENTRYPOINT_KEEP=0` in `mix_site.sh` — the image's entrypoint (which injects the
host's `libionic` so in-container `libibverbs` speaks the host `ionic_rdma`
kmod's ABI) is bypassed. Harmless here for the same reason.

## Software

| item | value |
|---|---|
| image tag | `infera/engine-sglang:merged-e` |
| **image id (pin on this, not the tag)** | `sha256:27667ee43291bed2bddb9caf44a63217fdb994d6f423f6ed3bf7e807340fae7a` |
| base image | `lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x` |
| sglang | **0.5.15.post1** |
| torch | **2.9.1+rocm7.2.0.git7e1940d4** |
| amd-infera | version `0.0.0` at `/opt/venv/lib/python3.10/site-packages` (in-image build; the version string is not meaningful — the image id is the real pin) |
| etcd | `quay.io/coreos/etcd:v3.5.14` |
| container binds | `["/mnt/vast:/mnt/vast"]` |
| container created | `2026-08-06T07:01:26Z` |

**Repo state:** the packup was produced from the working tree of
`infera.glm52.mix.experiment` on branch `dev.yihou.glm52.mix.experiment`. The
scripts under `scripts/` are byte-identical (md5-verified) to the copies that
executed on the cluster at
`/mnt/vast/c_huggingface/glm52_mix_20260806/scripts/`, so **the scripts in this
packup ARE the ones that ran** — no repo checkout is needed to reproduce.
No git command was run while assembling this packup, so no commit SHA is
recorded here. **Gap, stated rather than guessed:** if you need the SHA, read it
off the branch tip dated 2026-08-06.

## Deployment shape

One mix worker — `disagg_mode: "mixed"` in the registry, `disaggregation_mode='null'`
in `server_args` — launched through `python3 -m infera.engine.sglang`, TP8, both
phases on the same 8 GPUs. Router on `:8100`, engine on `:30000`, etcd on `:2379`.
All benchmark traffic goes **through the router**, not straight to the engine.

Config as requested by `mix_site.sh`:

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
capture. This is not a hang — do not kill it.

### Global budgets vs per-rank values — two flags DP-attention divides

Both are **global** budgets that SGLang divides by `dp_size` to get the per-rank
value. The machine-wide budget is what was requested; the per-rank number is what
each scheduler reports. Anyone reproducing must expect both numbers.

| flag | requested (global) | **per rank** | evidence |
|---|---|---|---|
| `--chunked-prefill-size` | 65536 | **8192** | log: `WARNING:sglang.srt.server_args:DP attention is enabled. The chunked prefill size is adjusted to 8192 to avoid MoE kernel issues.` |
| `--max-running-requests` | 256 | **32** | `server_args` records 256; the per-rank scheduler line reads `max_running_requests=32`. |

Confirmed in this image's source rather than inferred from the arithmetic:

- `srt/server_args.py:4902`, inside `if self._resolved().enable_dp_attention:`
  → `self.chunked_prefill_size = self.chunked_prefill_size // self.dp_size`.
  A division **gated on DP-attention**, not a clamp — the warning's wording
  ("adjusted to 8192 to avoid MoE kernel issues") is what makes it read like one.
- `srt/model_executor/pool_configurator.py:541-543` →
  `server_args.max_running_requests // mr.dp_size`.

No warning is emitted for the `max_running_requests` split; it is visible only in
the per-rank scheduler line.

### Derived engine state, from the engine's log

| item | value |
|---|---|
| KV pool | `max_total_num_tokens = 2,812,672`; **144.66 GB** main KV per rank + 1.85 GB (second pool, MTP draft) |
| KV dtype | `torch.float8_e4m3fn` |
| page size | 64 |
| attention backend | `dsa`, prefill+decode via `tilelang` |
| free GPU mem after alloc | 54.63 GB/card |
| decode CUDA graphs | captured, `max_bs=128`; prefill graphs disabled |
| hicache | `enable_hierarchical_cache=True`, `hicache_size=32`, backend `dynamic` → `InferaKvdBackend`, `prefetch_threshold=64`, write policy `write_through` |
| KV events | zmq `tcp://*:24142`, topic `kv-events` |

### The DSA-on-ROCm env block — mandatory on gfx950

Without these the model still serves and still returns HTTP 200 — it just
returns **garbage**, because the sparse-attention indexer takes a path not
ported to ROCm. Verified present in the running process's env:

```
SGLANG_USE_AITER=1                  SGLANG_ROCM_FUSED_DECODE_MLA=0
SGLANG_OPT_USE_TILELANG_INDEXER=1   SGLANG_OPT_USE_TOPK_V2=0
SGLANG_OPT_USE_JIT_NORM=0           SAFETENSORS_FAST_GPU=1
HIP_FORCE_DEV_KERNARG=1             PYTHONHASHSEED=0
SGLANG_DP_USE_GATHERV=1             (DPA only)
NCCL_IB_DISABLE=1  NCCL_IGNORE_CPU_AFFINITY=1  HSA_NO_SCRATCH_RECLAIM=1
```

`PYTHONHASHSEED=0` is there so block hashes — and therefore kvd keys — are
stable across restarts.

## External dependencies (absolute paths, NOT in this repo)

| what | where | notes |
|---|---|---|
| model weights | `/mnt/vast/xiaobo/models/GLM-5.2-MXFP4` | shared VAST mount, visible on compute nodes. Also serves as the tokenizer path and as the EAGLE draft-model path. |
| tokenizer | same path | the kv-aware router loads it to hash prefixes; without it kv-aware degrades silently to load-only routing |
| staging dir | `/mnt/vast/c_huggingface/glm52_mix_20260806/` | scripts / specs / logs / results. Identity-masked on purpose. |
| kvd L3 spill | `/tmp/kvd-long` **inside the container** | `--long-bytes 64G` is ABSOLUTE. The ratio-based default can ask for hundreds of GB per rank and fill the node's root fs. |

`/mnt/vast` is shared to the compute nodes. `/tmp` is **not** — the engine,
router and kvd logs live inside the container and must be `docker exec`'d out.

## Required secrets — names and sources only, no values

| secret | source |
|---|---|
| cluster SSH | `ssh root@149.28.124.225` (jump host chi2866), then `ssh chi2835`. Key-based; arrange your own. |
| docker registry | the `infera/engine-sglang` image was **already present on the node** for this run; nothing was pulled. If you must pull, use the team registry credentials from the team vault. |
| etcd / router / kvd | none — no auth configured; all bound to the node. |
| model weights | none — readable on the shared mount. |

No API key, token, or password is used by any script in this packup, and none
appears in any copied file or log.
