# Brief — same-node KV transfer is unexpectedly slow (GLM-5.2 v0.5.21 1P1D P4D4 on MI355X)

Written 2026-10-22 from existing evidence only (no new runs). Every number below names its source; labels:
**[measured]** = run log / microbench / shim output, **[code-read]** = read in the cited source, **[hypothesis]** = untested.
Paths: `PK1` = `/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.v0521-g4tune-xfer.packup_20261020` (spur),
`PK2` = `/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.v0521-baseline-xnode-g3g4.packup_20261022` (spur),
`LEDGER` = `/home/yihou/dev/git/infera.glm52.view/glm52.v0521-cap-c44c48-23k_20261012/debug/poll.md` (spur),
`MC` = `/tmp/yihou/mooncake_faae8dd4/mooncake-transfer-engine` (workstation, read-only),
`SGL` = `/tmp/yihou/src.v0521/python/sglang/srt` (workstation; upstream 11b1fb21de tree as used for the image).

## 1. Problem statement

Same-node 1P1D P4D4 (prefill GPU0-3, decode GPU4-7, one MI355X node) moves the KV cache over Mooncake's **HIP
transport** (XGMI peer copies via IPC-mapped device memory). It is ~7x slower per request than cross-node RDMA on the
same engine config, and it costs throughput:

| quantity | same-node HIP | cross-node RDMA (135 -> 136) | source |
|---|---|---|---|
| sglang stage `transfer_kv` p50 / p90 (C56, A2) | 6.2-7.5 s / 19.7-22.6 s (3 nodes) | ~1.1 s / ~2.1 s | PK2 `README.md` headline, `results/summary.tsv` cols xfer_stage_* [measured] |
| per Mooncake `batch_transfer_sync` call (bytes / wall), 1 h of A1 | **2.62 GiB/s**, queue wait mean **5.43 s** (idxf1b, m2m-268) | **20.1 GiB/s**, wait 0.34 s (g4x1) | LEDGER L1153; PK2 `notes.md` §5 [measured] |
| KV bytes per request (`kv_transfer_total_mb`) | p50 4.8-5.4 GB, avg 7.2-7.9 GB | p50 4.5-4.9 GB, avg 6.8-7.4 GB | PK2 `results/summary.tsv` cols kvmb_* [measured] |
| AgentX score C56 (tok/s/chip, A2) | 21,289 mean of 3 (20,516-21,814) | 24,362 / 24,896 (G4); 25,376 (G3) | PK2 `README.md` [measured] |
| TTFT p50 | 13.8-15.3 s | ~4.5 s | PK2 `README.md` [measured] |

Caveats on the comparison (PK2 `notes.md` §5): n = 1 per point; different node pair; cross-node ISL mean 120-132k vs
136-141k same-node, not normalised. Same-node CONC 52 gave 23,367 with transfer_kv p50 4.3 s (PK2 §2).

**The core contradiction [measured]:** the same Mooncake call shape on an idle node runs at 57-63 GB/s, and at
>= 18 GB/s under synthetic load (§2.2), while production sees 2.62 GiB/s per call. The hipxfer shim, recomputed on
non-OVERFLOW batches (§2.3), says the call wall IS the GPU span of the copies (wall/span median 1.00 for batches > 1 GB)
and the peer copies run at **~4.9 GiB/s per copy median on the GPU**. The time is spent while the copies execute on the
GPU. Why peer copies run at ~5 GiB/s there is open.

## 2. What has been measured, and what is ruled out

### 2.1 Production sender behaviour (YIHOU-XFER log in the patched `conn.py`)
- [measured] One Mooncake call = 78 blocks (one per layer) x ~4.5 MiB = ~351 MiB; single calls up to 4.29 s (0.08 GiB/s)
  in g4xb5 (LEDGER L1049; L990 for blocks/call 76-78, avg block 4.6-5.4 MiB).
- [measured] With upstream sharding (`YIHOU_XFER_SHARD=dest`) every prefill DP uses ONE sender queue/thread; per-DP rate
  1-4 GiB/s per 5-min window while that thread is busy 50-100 %; queue wait mean up to 25.8 s (PK1 `notes.md` §2;
  `PK1/results/xfer_sum.g4xb5_g4xm4.txt`). The shim logs confirm one tid per prefill process (e.g. pid 673: 9810
  batches, all tid 2099; `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/g4h1/hipxfer.673.log`).
- [measured] `SHARD=mix` (hashed room -> 4 queues/DP): transfer_kv avg 9.98 -> 2.16 s, A2 +7.5 % (21,262 -> 22,854),
  but TPOT 13.3 -> 15.0 ms. **User decision 2026-10-20: mix NOT adopted** (P/D interfere, hampers analysis); `dest` is
  the baseline (PK1 `README.md`, `notes.md` §2). Do not re-propose mix as "the fix" without the user.
- [measured] Warm-up (first ~10 min) rates are ~0.2-0.3 GiB/s/DP in both modes (PK1 `notes.md` §2 caveat).
- [measured] GTT HiCache (PR #42553) did not shorten transfer: g4n1 transfer_kv p50 7.17 s
  (`/shared_nfs/yihou_glm52/g4tune_20261019/CHECKPOINT.g4tune.20261020-0610.md` L20).

### 2.2 Microbenchmarks (Mooncake HIP transport in the production image, 4 pairs g -> g+4 at once)
Scripts/logs: `/shared_nfs/yihou_glm52/g4tune_20261019/machine/` (v2-v6 exist ONLY there; PK1 `scripts/machine/` and
`results/machine/` hold mc_bw*, p2p_bw, run_sdma and xgmi_load v1). Summary in LEDGER L984-986, L1044-1052.

| test | geometry | result | source |
|---|---|---|---|
| v6 idle | **production: 78 x 4.5 MiB per call**, nbuf 1 or 78 | **57-63 GB/s** (6 ms/call) | `xgmi_load6.097.yihou.log` [measured] |
| v6 GEMM thread in the sender process (eager, separate stream) | production | **28-32 GB/s** (13-14 ms/call) | same [measured] |
| v6 GEMM in separate processes | production | ~61 GB/s | same [measured] |
| v4 per-phase loads in separate processes (GEMM / H2D 1 GiB / D2H 1 GiB / RCCL 256 MiB all-reduce GPU0-3) | 5825 x 360 KiB scattered (NOT production) | 42-53 GB/s each; **all four together 18-26 GB/s** | `xgmi_load4.097.yihou.log` [measured] |
| torch per-block copies (same layout) under GEMM | 5825 x 360 KiB | 11-12 -> ~0.95 GB/s | v2-v4 logs [measured] |
| mc_bw 10-19 (v3-021) | 5825 x 360 KiB | idle 47-51 GB/s; 1 x 2 GiB block ~9 GB/s; 64 KiB blocks ~13.6 GB/s | PK1 `notes.md` §4 [measured] |

Note: the ">= 18 GB/s under GEMM+H2D+D2H+RCCL" figure comes from v4, i.e. the 5825 x 360 KiB geometry, not the
production call shape. No microbench has combined production geometry with the full production load mix.

**Retracted / ruled out:**
- 10-19 `mc_bw2` "0.35-0.4 GB/s on GPU1-3 under sender GEMM, GPU0 unaffected" — NOT reproduced by v1-v6, retracted
  (CHECKPOINT L42; LEDGER L1050/L1052). PK1 `README.md` and `notes.md` §4 still quote it; treat those lines as superseded.
- Host-side enqueue cost: negligible (shim, §2.3) [measured].
- Time outside the copies' GPU span: not significant on clean batches (wall/span median 1.00-1.03, §2.3) [measured].
- `MC_HIP_STREAM_PRIORITY=high` "no change" (PK1 `notes.md` §4): **this env name does not occur anywhere in the
  Mooncake faae8dd4 tree** (`grep -rn HIP_STREAM_PRIORITY $MC/..` = 0 hits) and PR #4340 does not add it
  (`/tmp/yihou/mc_patches/mooncake-pr4340-e8062249.hip.yihou.diff` hunks touch only hip_transport.{h,cpp} register/relocate)
  [code-read]. The test was a no-op; stream priority was never tested. Streams are created with
  `hipStreamCreateWithFlags(..., hipStreamNonBlocking)`, default priority (`MC/src/transport/hip_transport/stream_pool.cpp:74`).
- `HSA_ENABLE_SDMA=1`: no change; `HSA_ENABLE_SDMA=0` under sender load: senders did not finish in 600 s (container
  stopped) — microbench only, never in production (PK1 `notes.md` §4; LEDGER L984-986) [measured].
- `MC_DISABLE_HIP_TRANSPORT` and `MOONCAKE_DISABLE_HIP_DMABUF` in the production env are read nowhere (no-ops)
  (PK1 `notes.md` §1, research workflow `PK1/analysis/research_wf.20261019.json`) [code-read].
- Chunked prefill 65536 is not a transfer fix: A1 = baseline, A2 crashed (`HSA_STATUS_ERROR_OUT_OF_RESOURCES`) (PK2 §3).

### 2.3 hipxfer LD_PRELOAD shim over a whole production run (g4h1, m2m-268, 2026-10-21)
- What it times [code-read, `PK2/scripts/mounts-src/hipxfer_trace.yihou.c`]: interposes `hipMemcpyAsync` (only kind
  `hipMemcpyDefault`, size >= 64 KiB), `hipEventRecord`, `hipEventSynchronize`; adds its own start/end events around each
  copy; one line per batch: `ts_ns tid dev n bytes wall_us enq_us gap_us sync_us gpu_span_us gpu_sum_us gpu_max_us streams`
  (header at L11-14). Prefill only, overlay `PK2/scripts/harness/config.yihou.xfertrace.sh`.
- **OVERFLOW lines [code-read + count]:** the shim times at most `MAXC = 1024` copies per batch
  (`hipxfer_trace.yihou.c:28`, `:164-168`); later copies are counted in `bytes` but not in `gpu_span/gpu_sum`, and the line
  is tagged `OVERFLOW`. In the raw logs (`/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/g4h1/hipxfer.{673,817,964,1118}.log`)
  14664 of 40641 batches are OVERFLOW. `streams` is capped at `MAXS = 16` (L29), so it under-counts.
- **[measured] Recomputed on the 25977 clean (non-OVERFLOW) batches, all 4 pids (team-lead, 2026-10-22):**

  | quantity | value |
  |---|---|
  | data moved | 12877 GiB |
  | wall | 22522 s -> 0.57 GiB/s |
  | GPU span | 14549 s -> 0.89 GiB/s |
  | per-copy GPU time sum | 25216 s |
  | host enqueue | 117 s |
  | sync wait | 19457 s |

  Batches > 1 GB (n = 2063): wall/GPU-span median **1.00**, p90 1.25; per-copy GPU rate median **4.89 GiB/s**.
  Batches 0.1-1 GB (n = 16956): wall/GPU-span median 1.03.
  => On clean batches the call wall is the GPU span of the copies: the time is spent while the copies execute on the GPU,
  at ~4.9 GiB/s per copy median.
- **SUPERSEDED:** the earlier whole-run figures (LEDGER L1162; PK2 `notes.md` §7; `tune.p4p4.yihou.md` §5) — 40641 batches,
  wall/GPU-span median **1.46** for batches > 1 GB ("~1/3 of wall outside the GPU span") and per-copy rate **6.35 GiB/s** —
  mixed in the OVERFLOW lines, where bytes counted every copy but the span timed only the first 1024. Do not quote them.
  The first-25-min figure (wall/span 1.02, LEDGER L1154) was also computed with OVERFLOW lines included.
- [measured, one line] An example clean line (pid 673, line 2): n 78, 351 MiB, wall 1215.5 ms, enq 0.8 ms, gap 0.1 ms,
  sync 633.4 ms, gpu_span 6.9 ms (~580 ms of wall in none of the interposed calls). Single sample, not representative of the
  clean-batch medians above.
- Shim run score 21,280 = inside the baseline band (PK2 §1), so the shim did not visibly perturb the run [measured, n = 1].

### 2.4 v0.5.19 same-node reference (2026-09-23, m2m-137, C40, 21,032 tok/s/chip, TTFT p50 3.60 s)
`/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.samenode-triton-idxshare.c40-full.packup_20260923/`
- [code-read of config] Both legs `MC_TE_FILTERS` = ionic rails; per-GPU map `{"0":"ionic_0",...,"3":"ionic_3"}` on both legs
  (`scripts/config.yihou.triton.sh` L60-100; `scripts/bench-harness/config.sh` L85-87).
- [measured there] RDMA between two DIFFERENT ionic devices on the same host silently fails (QPs connect, no completion);
  same-device loopback ~40 GB/s; so both legs must use the identical rail set (`config.yihou.triton.sh` L60-82, from that
  kit's `rounds/002-rdma-loopback/`). Decode GPUs 4-7 (NUMA1) then DMA through NUMA0 rails — accepted cost.
- [measured] Its logs show BOTH `Found 1 HCAs` AND `HIP transport installed for intra-node GPU P2P` per engine
  (`logs/key-excerpts.md` L30-37). **Which transport actually carried the KV in 09-23 is not established**
  (`tune.p4p4.yihou.md` L30: "inferred from config, not from logs"); the v0.5.19 image's Mooncake routing code was not read.
  Different C (40 vs 56), different engine; not a controlled comparison.

### 2.5 Not measured at all (no evidence either way)
Whether the production peer copies execute on SDMA engines or as blit kernels; per-engine SDMA occupancy; overlap of the
transfer copies with prefill kernels, HiCache kernels, OffloaderV2 H2D and RCCL on the same GPU.

## 3. Data path in code

### 3.1 sglang sender (prefill side) — production mounts `conn.yihou-xferfix2.py` over upstream `conn.py`
Upstream file `SGL/disaggregation/mooncake/conn.py` (3283 lines); patched copy
`/shared_nfs/yihou_glm52/v0521dbg_20261010/harness/mounts-v0521/src/conn.yihou-xferfix2.py` (3540 lines, md5 a04d6089…).
- Queues/threads [code-read, upstream L282-312]: `SGLANG_DISAGGREGATION_QUEUE_SIZE` FastQueues (production 4), each with a
  `ThreadPoolExecutor(pool_size // queue_size)`, one `transfer_worker` thread per queue (L2295).
- Sharding [code-read, upstream L2920-2928; patched L3172-3197]: `shard_idx = sum(decode session ports) % len(queues)`;
  with 1:1 DP affinity every chunk of a prefill rank lands in one queue -> one worker -> strictly serial calls.
- Call [code-read, upstream L783-788; patched L968-983]: `_transfer_data` -> `engine.batch_transfer_sync(session, src, dst,
  lens)`; all layers of a chunk merged into one call (`process_layers`, patched L1325-1336); blocking, pybind releases the GIL
  (PK2 `notes.md` §7).
- Logging [code-read, patched L89-208]: `YIHOU_XFER_LOG=1` -> per-shard per-minute `YIHOU-XFER shard=... wait_mean ...
  calls ... GiB ... wall ... GiB/s ... wall_max ... slowest=...` lines in `prefill-*.log`. `YIHOU_XFER_PAR=k` (L210-214,
  L1440-1466) splits one chunk's layers into k concurrent calls — exists; its earlier (10-10) results are in
  `/shared_nfs/yihou_glm52/v0521dbg_20261010/logs/watchdog2.yihou.md` (not reviewed for this brief).
- Stage boundaries [code-read, PK1 `notes.md` §1]: `prefill.py` L1054-1056 `send_kv_chunk(last_chunk)` +
  `set_prefill_transfer_queue_entry_time`; L1183-1197 poll Success -> `set_prefill_kv_transfer_finish_time` = stage
  `prefill_transfer_kv_cache`.

### 3.2 Mooncake HIP transport (faae8dd4 + PR #4340 port; #4340 does not touch the copy path, see §2.2)
- Install [code-read, `MC/src/transfer_engine_impl.cpp:400-413`]: under `USE_HIP` the HIP transport is always installed next to
  rdma/tcp. Production sets `MC_TE_FILTERS=yihou_no_such_rdma_device` -> `Found 0 HCAs` -> tcp + hip (LEDGER L1049; env in
  `PK2/results/dryruns/dryrun.g4h1-268.yihou.txt`).
- Routing [code-read, `MC/src/multi_transport.cpp:477-540`]: for a multi-protocol segment the transport owning the target
  buffer is chosen by fixed priority hip 4 > cxl 3 > rdma 2 > tcp 1; **`MC_DISABLE_HIP` set -> hip priority 0** (L489); hip is
  skipped for a different host (`isHipReachableTarget`, `MC/include/multi_transport_locality.h:74`). Route trace printed
  only with `MC_LOG_LEVEL=TRACE` (`MC/src/config.cpp:389-394`, `multi_transport.cpp:532-537`).
- Copy [code-read, `MC/src/transport/hip_transport/hip_transport.cpp`]: `submitTransfer` L553-621 = for each request
  `startAsyncTransfer` (L456-530: `relocateSharedMemoryAddress` L756-820 (RW spinlock, IPC open on first use), event from
  `EventPool`, stream from `StreamPool::getNextStream` round-robin, `hipMemcpyAsync(dst, src, len, hipMemcpyDefault, stream)`
  L505-511, `hipEventRecord` L520), then `synchronizePendingTransfers` L534-551 = `hipEventSynchronize` on each event in order.
  Synchronous per batch.
- Pools [code-read]: `MC_HIP_NUM_STREAMS` / `MC_HIP_NUM_EVENTS`, default 64 each (L36-41, L309-345); streams are per device,
  shared by all threads of the process, non-blocking, default priority (`stream_pool.cpp:38-90`); events
  `hipEventDisableTiming` (`event_pool.cpp:166`).
- Which hardware path a `hipMemcpyDefault` device -> peer-device copy takes (SDMA engine vs blit kernel) is decided by
  HIP/ROCr, not Mooncake — **not read, not measured**.

### 3.3 Other GPU copy traffic on the prefill GPUs (same process, same GPU)
Production prefill args (`PK2/results/dryruns/dryrun.g4h1-268.yihou.txt` L19): `--hicache-io-backend kernel
--hicache-mem-layout page_first --hicache-write-policy write_through --hicache-size 260 --hicache-ratio 1.5
--offload-mode cpu --offload-group-size 4 --offload-num-in-group 1 --offload-prefetch-step 1 --chunked-prefill-size 32768
--tp-size 4 --dp-size 4 --mem-fraction-static 0.90`.
- HiCache [code-read, `SGL/mem_cache/memory_pool_host.py` L458-527 backup (D2H), L560+ load (H2D)]: with io_backend `kernel`
  and layout `page_first` the copies are GPU kernels (`transfer_kv_all_layer_mla_lf_pf` or the JIT staged variant), i.e. they
  run on CUs, not as hipMemcpy. Host pools are GTT (`hipHostMalloc`, PR #42553; CHECKPOINT L7-10).
- OffloaderV2 cpu mode [code-read, upstream `SGL/utils/offloader.py` L358-367 `start_onload` on `alt_stream`
  (`get_stream("offload")`, L251), L424-430 `param.to("cuda", non_blocking=True)` from pinned host]: per-layer MoE weight H2D
  via torch copies on a side stream. Production mounts `offloader.yihou-attrs.py` over this file
  (`PK2/scripts/harness/mounts.yihou.offload-attrs-hipxfer.tsv`), so production line numbers differ.
- TP4 RCCL collectives on GPU0-3 (XGMI) during prefill forward.
- Which of these share copy engines / queues with the Mooncake peer copies: **[hypothesis]**, not measured.

## 4. Open questions and experiments (cheapest first)

Q0. ~~Is the "1/3 of wall outside GPU span" real?~~ **Answered (2026-10-22): no.** On non-OVERFLOW batches wall/span median is
  1.00 (> 1 GB) / 1.03 (0.1-1 GB); see §2.3. For the next shim run raise `MAXC` (e.g. 8192) and `MAXS` so large batches are
  timed completely; rebuild with the command in `PK2/REPRODUCE.md` §3.

Q1. **Why do peer copies run at ~5 GiB/s on the GPU** (4.89 GiB/s per copy median, clean batches > 1 GB) when the idle
  microbench reaches 57-63 GB/s? Candidates, all [hypothesis]: SDMA vs blit, copy-engine contention — (a) peer copies executed as blit kernels
  that compete with prefill / HiCache kernels for CUs; (b) SDMA engine shared with OffloaderV2 H2D (and other copies);
  (c) 64-stream pool -> more hardware queues than the device serves concurrently; (d) GPU busy with compute.
  - Cheapest decisive measurement: a short `rocprofv3` trace (kernel + memory-copy tracing) of ONE prefill rank during the
    A1 profiling phase of a normal same-node round: shows whether the 4.5 MiB peer copies appear as copy-engine ops or as
    blit kernels, their duration, and what overlaps them. Check first that rocprofv3 is in the image and that attaching it
    does not need a relaunch; keep the window to a few minutes (trace size).
  - Microbench extension (no production run): v6 production geometry (78 x 4.5 MiB) + in-process loads that mimic
    production one at a time — HiCache kernel-backend D2H (`transfer_kv_all_layer_mla_lf_pf` on a GTT buffer), OffloaderV2-style
    pinned H2D on a side stream, RCCL all-reduce, GEMM — then all together. Base on
    `/shared_nfs/yihou_glm52/g4tune_20261019/machine/xgmi_load6.yihou.py` (+ v4 for load generators). Existing data say
    in-process GEMM alone halves the rate (57-63 -> 28-32 GB/s) and separate-process loads barely matter.
  - Knob scans, one at a time, in the microbench first: `MC_HIP_NUM_STREAMS` (1, 4, 16, 64) [knob exists, code-read];
    copy-engine environment knobs of HIP/ROCr — **verify the exact variable names and semantics in the ROCm 7.0 HIP/ROCr
    source or docs before use**; `HSA_ENABLE_SDMA=0` hung under load in the microbench (§2.2), so do not try it in production first.

Q2. **Would same-node RDMA loopback beat HIP?** (the v0.5.19 reference used ionic RDMA by config; transport used then is
  not established, §2.4)
  - One-variable same-node round: both legs `MC_TE_FILTERS=ionic_0,ionic_1,ionic_2,ionic_3`, both legs per-GPU map
    `{"0":"ionic_0",...,"3":"ionic_3"}` (identical rail per DP pair; different-device RDMA on one host does not work, §2.4),
    `PD_DP_RANK_AFFINITY=1`, plus `MC_DISABLE_HIP=1` so routing prefers rdma (`multi_transport.cpp:489`) [code-read; not run].
    Verify the route with `MC_LOG_LEVEL=TRACE` on a smoke request (`chosen=rdma`) and YIHOU-XFER per-call rate.
    The live audit currently requires `MC_DISABLE_HIP` absent (PK2 `notes.md` §4) — the audit must be adjusted for this round.
    Precede with a same-node microbench: v6 geometry over rdma loopback (`MC_TE_FILTERS=ionic_k`, MC_DISABLE_HIP=1).
  - This is also listed as the first control in `tune.p4p4.yihou.md` §5.

Q3. **How much of transfer_kv is queueing vs the call?** YIHOU-XFER already separates queue wait (5.43 s mean) from call wall;
  per-call is 2.62 GiB/s. With `dest` there is one serial worker per DP. A faster call shortens the queue; parallel calls
  (`YIHOU_XFER_PAR`, `SHARD=mix`) were the other axis, and mix was rejected by the user (§2.1). Ask the user before
  re-testing any sharding/parallel variant.

Q4. **Is it per-node?** Same-node rounds ran on 097, 250, 268, 054, v3-017/021/032; per-call rates were similar where logged.
  No `collect_env` on 097/250/v3-0xx (PK1 `environment.md`); 268 has kernel 6.8.0-107 / amdgpu 6.14.14 (PK2 `environment.md`).

## 5. How to reproduce a same-node round, and tools

- Harness: `H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness`, runs `R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs`,
  workspace `W=/shared_nfs/yihou_glm52/g4tune_20261019`, tools `N=/shared_nfs/yihou_glm52/v0521cap2_20261013`.
- Baseline env: `PK2/scripts/envs/env.idxf1b-268.yihou.sh`; shim round: `env.g4h1-268.yihou.sh` with
  `CONFIG=$H/config.yihou.xfertrace.sh`. Full ordered commands: `PK2/REPRODUCE.md` §1 (node prep), §3 (same-node round:
  dryrun + token-diff vs baseline, `launch_after_prep.yihou.sh` or `stop_then_chain.yihou.sh`, chain_cap2 = up -> census ->
  smoke -> audit -> A1 -> flush -> A2), §5 (analysis). Stop a round: `bash $W/stop_round.yihou.sh <key> <round>` (docker stop only).
- One round = A1 + A2 x 3600 s plus bring-up, ~2.5-3 h. Baseline band is 20,516-21,814 (±3 %): a single-run delta inside it is
  not a result (PK2 `notes.md` §1). `SGLANG_SIMULATE_ACC_LEN=3.61` waives correctness by construction.
- Tools:
  - `python3 -I $H/tools/xfer_summary.yihou.py $R/<round>/server-logs/prefill-0.log <start-iso> [<end-iso>]` — per-call
    Mooncake stats from YIHOU-XFER lines (needs `YIHOU_XFER_LOG=1`, the baseline default via the xferfix2 overlay).
  - `PK1/scripts/session/xfer_sum.yihou.py` — 5-min-window per-DP rates/busy/wait (produced `PK1/results/xfer_sum.g4xb5_g4xm4.txt`).
  - `python3 -I $N/hist.yihou.py <label> <bench>/aiperf_artifacts/server_metrics_export.csv` — sglang stage histograms
    (copies: `PK1/scripts/tools/hist.yihou.py`, `PK2/scripts/tools/hist.yihou.py`).
  - hipxfer shim: source `PK2/scripts/mounts-src/hipxfer_trace.yihou.c` (= `$H/mounts-v0521/src/hipxfer_trace.yihou.c`), built lib
    `$H/mounts-v0521/lib/libhipxfer.yihou.so`; local build/test files `/tmp/yihou/g4tune/xfertrace/` (workstation:
    `t_hipxfer.yihou.c`, `t_local.yihou.c`, `fakemc.yihou.c`). Mind the MAXC/MAXS caps (§2.3).
  - Microbenches: `/shared_nfs/yihou_glm52/g4tune_20261019/machine/xgmi_load{,2,3,4,5,6}.yihou.py` + `*.097.yihou.log`,
    `mc_bw{,2,3,4}.yihou.py`, `run_sdma.yihou.sh` (shows the docker run line used: image, `--device /dev/kfd /dev/dri`,
    `MC_TE_FILTERS=yihou_no_such_rdma_device`, `MC_ENABLE_DEST_DEVICE_AFFINITY=1`, `MC_GID_INDEX=1`, `YX_IP=<node data IP>`).

## 6. Environment and access

- Cluster: spur (Crusoe, MI355X x 8 per node). Login `ssh spur006` (fallback spur007/005); the login host evicts a user above
  ~4 GB and has 1 CPU — only tiny cat/grep there; never a recursive grep over /shared_nfs (hung spur006 once, PK1 `notes.md` §5).
- **Nodes: request through the machine pool (`~/.claude/skills/machine-pool`); never sbatch/salloc yourself.** Slurm nodes are
  reached with `spur exec <job> -- ...`; `spur exec` may report "transport error" although the command ran — check before
  re-launching (PK2 `notes.md` §8). OAI holds of 10-21 (268 job 227779, 054 job 228279) may have ended; ask the pool.
  135/136 (cross-node pair) are plain-ssh nodes.
- Image `infera-sglang:v0521-yihou-1010-mc4340`, ID `9d755dd6a7bf…` on slurm nodes (`e8e2838f4a65…` after docker load on
  135/136); tar `/shared_nfs/yihou_glm52/v0521cap_20261012/images/infera-sglang.v0521-yihou-1010-mc4340.9d755dd6.yihou.tar.zst`.
  sglang 0.5.21.dev upstream `11b1fb21de` + mounted overlays; Mooncake `faae8dd4` + PR #4340 port (PK1 `environment.md`).
- Model `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`. Placement: prefill GPU0-3 TP4/DP4, decode GPU4-7 TP4/DP4, EAGLE MTP
  5/1/6 on decode, OffloaderV2 G4K1 step 1, HiCache GTT 260 GB/rank, C56.
- Rules that apply to anyone continuing: work only inside a `yihou` directory; never delete outside one; no recursive delete
  with a variable target; `docker stop`, never `docker rm`, of containers you did not start; English in files.

## 7. References (absolute paths)

spur:
- `/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.v0521-baseline-xnode-g3g4.packup_20261022/` — `README.md`, `notes.md` §5, §7,
  `REPRODUCE.md`, `environment.md`, `results/summary.tsv`, `results/rounds/g4h1/hipxfer/`, `results/dryruns/`,
  `scripts/mounts-src/hipxfer_trace.yihou.c`, `scripts/harness/config.yihou.xfertrace.sh`, `scripts/tools/xfer_summary.yihou.py`,
  `results/xnode/tpot_analysis.20261021.md`
- `/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.v0521-g4tune-xfer.packup_20261020/` — `README.md`, `notes.md` §1-4,
  `environment.md`, `results/xfer_sum.g4xb5_g4xm4.txt`, `results/machine/`, `scripts/machine/`, `patches/` (0001/0002 conn diffs),
  `analysis/research_wf.20261019.json`
- `/home/yihou/dev/git/infera.glm52.pd/yihou/tune.p4p4.yihou.md` §4-5; `/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.v0521-CONCLUSIONS.yihou.md`
- `/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.samenode-triton-idxshare.c40-full.packup_20260923/` — `README.md`,
  `scripts/config.yihou.triton.sh`, `logs/key-excerpts.md`, `environment.md`
- `/home/yihou/dev/git/infera.glm52.view/glm52.v0521-cap-c44c48-23k_20261012/debug/poll.md` L962, L984-990, L1001, L1010,
  L1044-1052, L1143, L1153-1154, L1162
- `/shared_nfs/yihou_glm52/g4tune_20261019/CHECKPOINT.g4tune.20261020-0610.md`; `/shared_nfs/yihou_glm52/g4tune_20261019/machine/`
- `/shared_nfs/yihou_glm52/v0521dbg_20261010/harness/mounts-v0521/src/conn.yihou-xferfix2.py`;
  `/shared_nfs/yihou_glm52/v0521dbg_20261010/harness/tools/xfer_summary.yihou.py`
- `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/g4h1/` (raw `hipxfer.*.log`, `server-logs/prefill-0.log` 16 MB, decode log 35 MB)

workstation (read-only):
- `/tmp/yihou/mooncake_faae8dd4/mooncake-transfer-engine/src/transport/hip_transport/{hip_transport,stream_pool,event_pool}.cpp`,
  `src/multi_transport.cpp`, `src/transfer_engine_impl.cpp`, `src/config.cpp`, `include/multi_transport_locality.h`
- `/tmp/yihou/mc_patches/mooncake-pr4340-e8062249.hip.yihou.diff`
- `/tmp/yihou/src.v0521/python/sglang/srt/disaggregation/mooncake/conn.py`, `.../mem_cache/memory_pool_host.py`, `.../utils/offloader.py`
- `/tmp/yihou/g4tune/xfertrace/` (shim source + local tests); `/tmp/yihou/g4tune/xnode/tpot_analysis.20261021.md`
