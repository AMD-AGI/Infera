# GLM-5.2-FP8 SGLang PD on gfx942 — agentic bench

Runnable package for GLM-5.2-FP8 on two gfx942 nodes: SGLang prefill/decode
disaggregation over Mooncake RDMA, DP-attention, MTP speculative decoding, Infera
kv-aware routing, and KV offload to host RAM + local NVMe through `infera-kvd` —
then an agentic multi-turn benchmark driven entirely by tooling that ships inside
the image.

Offload is on by default (`KVD=1`) and `KVD=0` gives the same deployment without
it, which is the A/B baseline. See §5.

§7 is the record of what this was measured to do on two MI300X nodes: what it cost
to make hicache work on ROCm at all, 448/448 requests at `CONC=16`, and the honest
answer on kvd — the path works in both directions, and it supplied 0.4% of the hits
because a 54 GB device pool per rank already serves this trace at ~100% of the
achievable ideal.

The engine image is built directly from `deploy/docker/Dockerfile.sglang.gfx942`.
No runtime patching of SGLang, Mooncake, or Infera is applied on top of it.

The benchmark uses SGLang's built-in `sglang.benchmark.serving --dataset-name
agentic-trace` against a public Apache-2.0 corpus, so reproducing it needs no
private tooling.

## Topology

Two nodes, 8 GPUs each. The node names and IPs below are placeholders —
substitute your own.

| Node | Role |
|---|---|
| `node-0` | etcd + router + prefill + kvd + bench driver |
| `node-1` | decode |

kvd runs only on the prefill node. SGLang issues storage prefetch on its
aggregated and prefill branches but not on the decode branch, so a decode-side L3
is written and never read; Infera detects a PD decode leg and refuses to wire kvd
there even if you pass the socket.

Every script is driven by environment variables; you should not need to edit any
of them. `env.sh` holds the defaults. Export at least these on **both** nodes:

```bash
export PREFILL_IP=10.0.0.1                    # node-0, on the data network
export DECODE_IP=10.0.0.2                     # node-1, on the data network
export MODEL=/your/path/GLM-5.2-FP8           # local weights dir
export DATA_DIR=/your/path/agentic-data       # where the trace dataset lives
export IMAGE=infera:sglang-gfx942-glm52
export KVD_L3_DIR=/your/nvme/kvd-l3           # node-local NVMe, prefill node
export KVD_IO_MODE=direct                     # if that path is really NVMe; see §5.1
```

If your nodes resolve by name, `PREFILL_NODE` / `DECODE_NODE` derive the IPs
instead. `IB_DEVICE` (default `mlx5_0`) and `MC_GID_INDEX` (default `3`) must
match the RDMA rail the two nodes share.

## Scripts

| Script | Where | Purpose |
|---|---|---|
| `build_image.sh` | host | Build `IMAGE` from `Dockerfile.sglang.gfx942`. |
| `preflight_rdma.sh` | host, both nodes | RDMA device visibility + optional cross-node fabric check. |
| `host_container.sh` | host, both nodes | Start/remove the long-lived engine container. |
| `launch/launch_etcd.sh` | host, prefill node | etcd, from the official etcd image. |
| `launch/launch_kvd.sh` | container, prefill node | infera-kvd daemon (L2 host RAM + L3 NVMe). |
| `launch/launch_router.sh` | container, prefill node | Infera kv-aware router. |
| `launch/launch_prefill.sh` | container, prefill node | SGLang prefill leg. |
| `launch/launch_decode.sh` | container, decode node | SGLang decode leg. |
| `smoke.sh` | container, prefill node | Worker list + one chat request + RDMA hand-off + kvd writes. |
| `weka_to_agentic_trace.py` | container, prefill node | Public traces → SGLang `agentic-trace` dataset. |
| `run_agentic_trace.sh` | container, prefill node | Run the benchmark and rescore it. |
| `run_kvd_reuse.sh` | container, prefill node | Two-pass experiment that isolates what kvd recovers (§5.3). |
| `score_agentic_trace.py` | container, prefill node | Recompute cache metrics (see §4.3). |
| `probe_hip_host_ptr.py` | container, either node | Does `hipHostRegister` return the host VA on this arch? Evidence for the hicache allocator patch (gotcha 7). |
| `stop.sh` | container, both nodes | Stop router/engine/kvd processes. |

## 1. Build the image

Run on both nodes, or build once and push to a registry:

```bash
cd examples/glm5.2_gfx942_agentic_bench
bash build_image.sh
```

The build context is the repository root and the Dockerfile is
`deploy/docker/Dockerfile.sglang.gfx942` — that single file is the whole image
definition.

## 2. Bring up

Cross-node PD moves KV over the fabric on every request, so check RDMA first. On
both hosts:

```bash
bash preflight_rdma.sh     # active RDMA port count must be non-zero
bash host_container.sh
```

On the prefill host — etcd runs on the host, the rest inside the container. kvd
must be up before the prefill leg: the engine probes the socket at startup and
refuses to load if nothing answers.

```bash
bash launch/launch_etcd.sh
docker exec -it infera-glm52-gfx942 bash
  bash launch/launch_kvd.sh
  bash launch/launch_router.sh
  bash launch/launch_prefill.sh
```

On the decode host:

```bash
docker exec -it infera-glm52-gfx942 bash
  bash launch/launch_decode.sh
```

Cold start takes 15–25 min — both legs load GLM-5.2 and the MTP nextn layer, and
the log goes quiet while they do. Follow `logs/prefill.log` on the prefill node
and `logs/decode.log` on the decode node; don't kill a slow load.

## 3. Smoke test

Once both workers are registered, on the prefill node inside the container:

```bash
bash smoke.sh
```

One prefill plus one decode worker, a coherent answer, and `installTransport,
type=rdma` in the decode log means the router paired the legs and KV moves over
RDMA rather than falling back to TCP.

## 4. Agentic benchmark

### 4.1 Build the dataset

The corpus is [`semianalysisai/cc-traces-weka-062126-256k`][corpus] (Apache-2.0),
Claude Code agent traffic. It carries only per-turn token counts and KV block ids
— no text — so `weka_to_agentic_trace.py` synthesizes filler text while preserving
the real structure: exact per-turn lengths and the block-level prefix reuse.

[corpus]: https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k

```bash
source env.sh                       # for TRACE, MODEL and OUTPUT_LEN
export HF_HOME=/your/path/hf_cache
huggingface-cli download semianalysisai/cc-traces-weka-062126-256k --repo-type dataset
SRC=$HF_HOME/hub/datasets--semianalysisai--cc-traces-weka-062126-256k/snapshots/*/traces.jsonl

python3 weka_to_agentic_trace.py "$SRC" -o "$TRACE" \
  --output-len "$OUTPUT_LEN" --min-turns 4 --max-context 100000 \
  --verify 20 --tokenizer "$MODEL"
```

`--max-context` fits the corpus to what the deployment can prefill; `--dry-run`
reports the resulting distribution without writing, which is the cheap way to pick
it. At 100000 this yields 295 conversations, p50 peak context 78,848 tokens, and
`--verify` reproduces every checked turn's length exactly.

### 4.2 Run

```bash
NUM_PROMPTS=60 CONC=4 bash run_agentic_trace.sh
```

`NUM_PROMPTS` counts **conversations**, not requests — 60 conversations averaging
8 turns is ~448 requests. The script flushes both legs' caches, runs
`sglang.benchmark.serving` against the router, and then rescores the result.
Sweep capacity by varying `CONC`.

### 4.3 Read the result

`sglang.benchmark.serving` mis-reports the input side in multi-turn mode: it keeps
the conversation-level `prompt_len` for every turn, so its summary can print
`Total input tokens: 0` next to a cache hit rate above 100%. Its per-request
`cached_tokens` come from the server and are correct.

`score_agentic_trace.py` therefore recomputes against the dataset's verified
per-turn lengths, and reports the number worth comparing across tools —
**efficiency**, actual cache hits over what this engine could reuse at best. The
ideal is page-aligned and drops the one page a match can never include, so a gap
means eviction rather than arithmetic (the docstring derives it). Real output, from
the `CONC=16` run in §7.4:

```
  actual hit rate              86.62 %
  ideal  hit rate              84.61 %
  efficiency (a/i)            102.37 %

  turns short of ideal              4 / 448
  tokens lost to evict            832 (0.00% of ideal)
  turns ABOVE ideal                24 (+615,488 tokens) — reuse the ideal does not model,
                       almost always a prefix shared with a different conversation

  cached tokens by tier
    device       26,489,600   99.97 %
    host                  0    0.00 %
    storage           8,320    0.03 %
```

Efficiency near 100% with almost no eviction means the run is below the pressure
point and kv-aware routing has nothing to distinguish itself on. Raise `CONC`
until eviction appears; that is where the routing policy starts to matter. Over
100% is not a broken meter — the ideal models one conversation growing on its own,
and this corpus shares prefixes between conversations too. `cached tokens by tier`
splits the hits across GPU / host / storage and is the only direct evidence that
the kvd tier did anything.

## 5. KV offload (kvd)

Three tiers under the GPU. The GPU KV pool is L1; SGLang's hierarchical cache adds
a host-RAM L2; `infera-kvd` owns an L3 on node-local NVMe with its own pinned RAM
arena in front of it. A prefix evicted from the GPU therefore survives, and comes
back over the `hipMemcpy`/NVMe path instead of being recomputed.

The seam is one flag. `--infera-kvd-socket` makes Infera probe the daemon, refuse
to start if it is down, and append SGLang's `--enable-hierarchical-cache`,
`--hicache-storage-backend dynamic` and the extra config that points `dynamic` at
`infera.engine.sglang.kvd_adapter`. Do not hand-write those; the launch script
passes the socket and `--hicache-size` and nothing else.

### 5.1 Configuration

| Variable | Default | Why |
|---|---|---|
| `KVD` | `1` | `0` runs the same deployment with no offload — the A/B baseline. |
| `KVD_L3_DIR` | placeholder | L3 directory. Must be node-local NVMe. |
| `KVD_IO_MODE` | `auto` | `direct` / `buffered` / `auto`. See below. |
| `KVD_RAM_BYTES` | `64G` | kvd's pinned arena, which is also the zero-copy window the engine mmaps. |
| `KVD_LONG_BYTES` | `512G` | L3 budget under `KVD_L3_DIR`. |
| `KVD_TABLESPACE_POOLS` | `1M,4M` | L3 slot sizes. Must cover one whole hicache page. |
| `HICACHE_SIZE` | `32` | SGLang's host tier, **GB per DP rank** (so ×8 here). |

Three of these are easy to get wrong in ways that leave the deployment looking
healthy while L3 does nothing:

- **`KVD_IO_MODE`.** `auto` runs kvd's classifier, which walks the mount through
  LVM/mdraid down to the block device. Inside a container that walk ends at a
  `/dev/mapper` node the container cannot see, so `auto` takes its conservative
  branch and picks buffered even on NVMe. On the LVM-over-7-NVMe xfs mount here
  that cost 4× write throughput: kvd's startup self-check measured **3.70 GB/s
  buffered against 14.56 GB/s with O_DIRECT**. Set `direct` when you know the path.
- **`KVD_TABLESPACE_POOLS`.** L3 stores one value per hicache page, and with the
  default `page_first` layout a page holds every layer: GLM-5.2-FP8 at page size 64
  writes 2.74 MiB per KV page and 624 KiB per DSA-indexer page. A value larger than
  the biggest pool is **rejected, not split**, so L3 stays empty while nothing else
  complains. `smoke.sh` fails on that rather than leaving you to find it.
- **`HICACHE_SIZE`.** Sized per DP rank, and deliberately smaller than this
  deployment's 54 GB device pool per rank — matching it would pin ~870 GB of host
  RAM for an L2 that L3 already backs. SGLang warns that the ratio costs L2 hit
  rate, which is the intended trade here. It does not disable prefetch: on this
  0.5.16 base the prefetch budget is half the host pool, not a function of the
  host/device difference. On sglang ≤ 0.5.15 it *was* that difference, and a ratio
  near 1.0 there switches L3 reads off entirely.

### 5.2 Check it is actually working

`smoke.sh` covers this, and the checks are worth knowing individually:

```bash
python3 -m infera.kvd.statctl --socket /tmp/infera-kvd/kvd.sock
```

`sets_total` climbing means the engine is writing to kvd; `gets_total` and
`hits_total` climbing means it is reading back. Writes alone prove only half the
path — a decode-leg misconfiguration produces exactly that.

The prefill log should show the wiring and the negotiated zero-copy arena:

```
kvd client sglang-startup-probe: shared arena negotiated (64.00 GiB, ...)
--infera-kvd-socket appends --hicache-storage-backend dynamic
infera-kvd HiCacheStorage backend ready (socket=/tmp/infera-kvd/kvd.sock)
```

### 5.3 Measuring what kvd is worth

A single agentic pass will not show it. With a 54 GB device pool per rank this
deployment already serves the trace at ~99.9% of the growing-prefix ideal, so
there is almost nothing left for a lower tier to recover, with or without kvd.
What kvd uniquely provides is a prefix that survives leaving the GPU — so measure
that directly:

```bash
NUM_PROMPTS=20 CONC=4 bash run_kvd_reuse.sh
```

Pass 1 runs the trace cold and fills L3. Then `flush_cache` resets the radix tree
and the host pool but explicitly does **not** touch the storage backend, so pass 2
replays the same trace with every tier above kvd empty. Pass 2 is scored with
`--warm`, since pass 1 stored every turn's whole prompt and the growing-prefix
ideal — where turn 0 reuses nothing — is no longer the ceiling.

Read the scorer's **`cached tokens by tier`** block, not pass 2's hit rate. Starting
the pass with cold upper tiers does not keep them cold: pass 2 replays whole
conversations, so from turn 2 on each conversation has refilled the GPU tier from
its own traffic and serves itself. Measured here, 99.6% of pass 2's hits came from
the device tier. The isolated number is the **conversation-opening turns** line —
those are the only turns no tier above kvd can hold — and §7.3 has what it said.

## 6. Stop

Inside each engine container:

```bash
bash stop.sh
```

Then on the hosts:

```bash
docker rm -f infera-glm52-gfx942
docker rm -f infera-glm52-etcd    # prefill host only
```

## 7. Experiment record

What was actually run to validate this recipe, on 2 × MI300X (`gfx942:sramecc+:xnack-`,
amdgpu 6.14.14, ROCm 7.2.0), image built from `Dockerfile.sglang.gfx942` on
`lmsysorg/sglang:v0.5.16-rocm720-mi30x`, GLM-5.2-FP8 with `--kv-cache-dtype fp8_e4m3`,
TP8/DP8 with DP-attention, MTP on, kv-aware routing on, kvd on. One prefill node,
one decode node, Mooncake over `mlx5_0`. Numbers are one cluster's shape, not a spec.

### 7.1 Bring-up

Weights come off a shared filesystem, so cold start is dominated by whatever is
not in page cache: 15–25 min on a first load, ~10 min once warm (both legs in
parallel). The kvd daemon's startup self-check measured its L3 mount
(xfs on LVM over 7 local NVMe) at **14.51 GB/s write, 14.13 GB/s read** with
`O_DIRECT`, against **3.70 GB/s write** buffered — hence `KVD_IO_MODE=direct`
in `logs/local.env` rather than `auto` (gotcha in §5.1).

`smoke.sh` then passed on the built image: two registered workers, `127 * 31`
answered `3937`, `installTransport, type=rdma` on the decode leg, the allocator
patch marker live, and **54 kvd writes** from one filler prompt.

### 7.2 The blocker: hicache write-back kills the scheduler on ROCm

Enabling kvd on this base does not work out of the box, and the failure is not
where you would look. Startup, weight load, worker registration and a short
request all succeed; the prefill scheduler then dies on the **first request that
reuses a prefix**:

```
File "mem_cache/pool_host/mla.py", line 403, in backup_from_device_all_layer
  jit_transfer_hicache_all_layer_mla_staged_lf_pf(
tvm.error.InternalError: Tensor match failed for
  Tensor<1152>[strides=<1>, dtype=int64, device=rocm:0]
- Root cause: Device value [rocm:0] not in the allowed options: [cpu, rocm_host]
Subprocess scheduler_0 crashed with exit code -3
```

Root cause and fix are in `patches/sglang_rocm/patch_hicache_rocm_staged_write_back.py`
and summarized in gotcha 7: two gates for one decision disagree on ROCm, so the
controller puts the destination indices on the GPU while the MLA pool launches the
JIT kernel that needs them on the host. Upstream `main` is affected; v0.5.15.post1
is not, which is why the MI355 image never hit it.

Cost of not knowing this: it reads as an unstable deployment. An earlier `CONC=16`
run in this series voided with 314 of 448 requests returning
`503 no active mixed worker` — the prefill leg had died exactly this way, and the
same run passes 448/448 after the patch (§7.4).

### 7.3 What kvd is worth here (`run_kvd_reuse.sh`, `NUM_PROMPTS=20 CONC=4`)

| | pass 1 (cold) | pass 2 (replay, tiers above kvd flushed) |
|---|---|---|
| requests | 138 | 138 |
| duration | 836 s | 797 s |
| mean TTFT | 16.4 s | 14.8 s |
| hit rate (actual / ideal) | 83.33% / 83.34% | 83.70% / 99.91% |
| efficiency | **99.99%** | 83.78% |
| hits from device / storage | 100% / 0% | **99.57% / 0.43%** |
| kvd lookups → hits | 0 → 0 | **5,677 → 5,677, 0 misses** |
| kvd writes | 72,096 | (continues) |

Read it in this order:

1. **The path works, both ways.** kvd took 72k writes in pass 1 and answered 5,677
   lookups in pass 2 with **zero misses**. L3 held 100 GB by the end.
2. **Pass 1 leaves nothing for a lower tier to win.** 99.99% of the achievable
   ideal came from the GPU tier alone — with a 54 GB device pool per rank, this
   trace never pressures L1 enough to need an L3.
3. **kvd supplied 0.43% of pass 2's hits.** Not because it failed, but because
   pass 2 replays whole conversations: from turn 2 on, each conversation has
   refilled the GPU tier from its own traffic and serves itself.
4. **On the turns kvd is the only possible source it served 2.8%.** For the 20
   conversation-opening turns, `device` was 0 — the isolation held — and kvd
   returned 32,768 of 1,176,384 ideal tokens, in units of exactly 8,192 tokens for
   4 of the 20. So prefetch stops early rather than kvd missing: the cap is on
   SGLang's side (`hicache_storage_prefetch_policy` defaults to `timeout`, and
   `prefetch_rate_limited()` gates on `prefetch_tokens_occupied` against
   `0.5 × host pool`), and pinning it exactly was not chased here.

Pass 2's 39 s / 1.6 s TTFT advantage is **not** evidence for kvd: 0.43% of hits
cannot pay for it, and a 138-request run varies by more than that.

**So: kvd is validated as functional, not as a win in this configuration.** To make
it a win, remove the reason it is idle — more concurrent distinct sessions than
54 GB/rank can hold, a smaller `MEM_FRAC`, or `--hicache-storage-prefetch-policy
wait_complete` to trade TTFT for prefix reuse. The A/B against `KVD=0` is only
worth running once one of those is true.

### 7.4 High concurrency

Same trace, same fleet, `NUM_PROMPTS=60` (448 requests):

| | CONC=4 (no kvd) | CONC=16 (kvd on) |
|---|---|---|
| successful | 448/448 | **448/448** |
| duration | 2,155 s | **1,287 s** |
| mean / p99 TTFT | 12.8 s / 48.1 s | 34.1 s / 108.7 s |
| efficiency | 99.90% | **102.37%** |
| turns short of ideal | — | 4 / 448 (832 tokens) |

1.67× the throughput for 2.7× the mean TTFT, no failures, and the decode leg ran
at 0.93 token usage with **0 retracted requests** — so c16 is near the capacity
knee without falling over. MTP held up under it: accept length 2.9–3.4 of 4.
The two rows differ in kvd as well as concurrency (the c4 run predates the daemon),
so read the duration column as a concurrency ladder only loosely.

Efficiency **above** 100% is not a broken meter: the ideal models one conversation
growing on its own, while this corpus also shares prefixes *between* conversations.
24 turns beat the per-conversation ceiling by 615,488 tokens, and 606,528 of those
landed on conversation-**opening** turns, which by construction cannot reuse their
own conversation. That is kv-aware routing paying off — cross-session prefixes are
exactly what it is for — and it is why the scorer prints the excess instead of
clamping it.

## Notes & gotchas

1. **`--output-len` must equal `--sharegpt-output-len`.** The converter solves each
   turn's filler size with the reply length baked in (`ignore_eos` is on by
   default, so replies are exactly that long). `OUTPUT_LEN` in `env.sh` feeds the
   run side; pass the same value when building the dataset or lengths drift turn
   over turn.
2. **`--warmup-requests` replays a whole conversation** in multi-turn mode, which
   pre-warms the cache the run is measuring. `run_agentic_trace.sh` defaults it
   to 0.
3. **`flush_cache` is a no-op while requests are in flight** and still returns
   success. Make sure nothing is running before a measured run.
4. **`--enable-cache-report` is required on the engines** for `--cache-report` to
   see `cached_tokens`; both launch scripts already pass it.
5. **kv events are on for prefill only.** Prefill-side prefix locality is the win;
   enabling them on the decode leg can make SGLang reject the speculative disagg
   flags. Set `KV_EVENTS=1` on the decode launch to override.
6. **Advertise the data-network IP.** `--advertise-host` / `SGLANG_HOST_IP` must be
   the address the peer node can reach, and `MC_GID_INDEX=3` with
   `--disaggregation-ib-device mlx5_0` keeps Mooncake on the RDMA rail.
7. **hicache on ROCm needs two patches, and the image bakes both.** They live in
   `deploy/docker/patches/sglang_rocm/` and each has its full argument in its own
   header; `deploy/docker/patch.upstream.status.md` is the index. The one to know
   about is `patch_hicache_rocm_staged_write_back.py`, because without it this
   base **kills the prefill scheduler** (exit −3, `Tensor match failed … device=rocm:0`)
   on the first request that reuses a prefix — startup, weight load and a
   single-shot request all succeed first, so it reads as a mid-run crash rather
   than a misconfiguration. `pool_host/mla.py` enables the staged write-back JIT
   on HIP while `DSAIndexerPoolHost`, in the same pool group, still gates it on
   CUDA: the group therefore moves the destination indices to the GPU while the
   MLA pool launches the JIT that needs them on the host. It is new in v0.5.16,
   which is why the MI355 image (v0.5.15.post1, no `pool_host/mla.py`) never hit
   it, and it is why `smoke.sh` sends a filler prompt long enough to force real
   page write-backs instead of trusting a one-line answer.
