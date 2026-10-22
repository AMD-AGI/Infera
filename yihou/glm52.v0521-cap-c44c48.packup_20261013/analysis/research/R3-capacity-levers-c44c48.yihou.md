# R3: capacity and scheduling levers for C44/C48 above 23k tok/s/chip (Track A design)

Date: 2026-10-12, 08:10-08:40 UTC. Read-only research. No GPU work was done. The only host action was one
`/proc/meminfo` + `free -g` read on 102 through the spur shim.

Tags:
- [M] first-hand: I read the code line, log line or command output myself; path:line or grep given.
- [S] second-hand: a ledger, README or R2 statement.
- [I] inference.

Abbreviations: src = `/shared_nfs/yihou_glm52/v0521cap_20261012/src/image-9d755dd6/sglang/srt`,
H = `/shared_nfs/yihou_glm52/v0521dbg_20261010/harness`, RUNS = `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs`.

## 0. Bottom line

1. **At C44 the limiter is time-to-first-token (TTFT), not capacity.**
   - At fixed CONC, the score is set by the closed-loop rate λ = CONC / (idle gap ≈ 16 s + TTFT + decode time).
   - X1, C44 warm: 21,773/chip, TTFT 9.0 s, TPOT 11.7 ms. That is only +0.4 % over C40 warm (21,681) [S, R2 §4].
   - At C44, every 1 s of TTFT costs about 2.5 % of the score [I, §3 model].
   - The largest TTFT component at C44 is the prefill→decode KV send ("inflight"): 5.25 s, versus 2.95 s at C40 [S, R2 §5].
2. **Neither KV pool binds at C44.** Decode prealloc was never > 0, retractions were 0, the per-DP decode pool reached at most 0.87, and per-DP running hit the cap of 16 in under 0.1 % of samples [S, R2 §2-3]. Raising device-KV capacity therefore does not raise C44 throughput by itself. It is insurance for C48 tails.
3. **Prefill caches are under pressure at C44, but it is unproven that capacity binds.**
   - GPU hit fell from 0.92 to 0.87. Host-pool hit rose from 0.02 to 0.06.
   - The 450 GB host pool was 100 % full on all DPs by about minute 45 [M X1 TSV; S R2].
   - Round k1 (host pool 360) is the discriminator for host size.
4. **450 GB host pool per rank is already close to the safe RAM ceiling.** The real pinned footprint is 553 GB per rank, because the DSA indexer host pool comes on top of `--hicache-size`. Maximum safe is about 500 GB per rank; 540 is aggressive (§1).
5. **Path to above 23k (model, §3):**
   - C44 needs TTFT ≲ 7.5 s.
   - C48 reaches about 23-24k if TTFT stays ≲ 10-11 s.
   - So the top levers are on the KV-send path (xferfix mounts, copy-stream priority), then CONC 48 with running-cap and decode-capacity insurance, then prefill GPU KV.

## 1. Prefill HiCache host-memory budget [required check]

**Semantics.**
- `--hicache-size` is **GB (1e9) per rank** for the MLA KV host pool only.
  - [M] `src/mem_cache/host_memory.py:116-118`: "set --hicache-size explicitly (GB per rank)".
  - [M] e2g `prefill-0.log`: "Allocating kv hierarchical KV host pool: 10016064 tokens, 450.00 GB host memory" on each of DP0-3. 10,016,064 × 44,928 B (576×78) = 450.0e9.
- The **DSA indexer host pool is allocated separately, on top of it**, at the same token count.
  - [M] e2g prefill log, 05:43:55: "Allocating 103.13 GB host memory for DSA indexer (layout=page_first)" per rank. That is 10,296 B/token (132×78).
  - [M] Code: `src/mem_cache/pool_host/dsa.py:196-219`.
- So pinned bytes per rank = 1.2292 × S × 1e9. At S = 450 that is 553.1 GB per rank and **2,212.5 GB for the 4 ranks** [M/I].

**Budget check in code.**
- Per-rank budget = (psutil available − 10 GiB) / ranks-per-host.
  - [M] `pool_host/base.py:29` (`HICACHE_HOST_MEMORY_RESERVE_BYTES = 10 GiB`), `:91-108`; `host_memory.py:108-126`.
- No docker `--memory` limit is set [M]: `grep -- --memory dryrun.e2g-102.yihou.txt` returns nothing. The cgroup headroom is therefore the host's.
- `--shm-size 32g` is a tmpfs ceiling, not a reservation [M]: `engine.sh:115`. Host Shmem is 20 MB now.

**Registration chunk.**
- `_cuda_host_register` splits a buffer into chunks of `floor(CHUNK_GB·2^30 / granularity) × granularity` [M]: `pool_host/common.py:140-185`.
- The env default is 256 [M]: `environ.py:770`. The harness sets 512 [M]: `config.yihou.E2-target.sh:12`.
- More than one chunk leads to the H1 fault. JIT kernels only translate the base pointer, so slots past chunk 1 cause a MAF [S]: phase-2 `notes.md` item 4.
- Constraint: **S × 1e9 / 2^30 < CHUNK_GB**. With 512 this gives S ≤ 549. At S ≥ 550, set `HICACHE_HOST_REGISTER_CHUNK_GB=1024`. The indexer buffer (≤ 130 GB) is far below either value.

**Node RAM** [M, 102 at 08:19 UTC, node idle with k1 waiting for VRAM drain]:
- MemTotal 2,885,471,972 kB = 2,751.8 GiB; MemAvailable 2,694 GiB.
- Page cache 480 GiB (reclaimable). No swap. HugePages 0.

| S (GB/rank) | per-buffer GiB (< CHUNK_GB?) | pinned, 4 ranks (GB) | RAM left for everything else (GiB) | host tokens/DP | verdict |
|---|---|---|---|---|---|
| 360 (k1) | 335 (ok) | 1,770 | 1,103 | 8.01 M | safe |
| 450 (e2g) | 419 (ok) | 2,212 | 691 | 10.02 M | proven (many hours) |
| **500** | 466 (ok) | 2,458 | **462** | 11.13 M (+11 %) | **max recommended** [I] |
| 540 | 503 (ok, tight) | 2,655 | 279 | 12.02 M (+20 %) | aggressive |
| ≥ 550 | ≥ 512 → needs CHUNK_GB=1024 | ≥ 2,704 | ≤ 233 | — | not recommended |

- The "left" column must hold all non-HiCache memory: the decode leg (4 schedulers, controller, tokenizer), the prefill-process RSS, router, client, and kernel memory.
- That RSS was never measured [gap]. Next round: sample `/proc/meminfo` and per-container RSS every 60 s (add to the metrics sampler).
- All pinned memory is unswappable (no swap), so overshoot means the OOM killer, not slowdown [I].

**Is the 450 GB host pool binding?**
- X1 (C44 warm) [M], TSV `x1-e2j44/metrics.tsv`: `hicache_host_used_tokens` reached 10,016,064 (= total) on all 4 DPs. R2 has it full on all DPs by t = 2,662-3,082 s [S].
- e2g A2 (C40): 0.88 → 1.00 over the hour [S, R2 §3]. e1c (154 GB): "host pool 100 % full" [S, ledger R025]. e2i R1: 88 → 94 % [S, notes §3].
- A full pool is normal LRU behaviour. Whether more host capacity buys hits is **open**:
  - Combined hit (GPU + host) was 0.947 at C40 (e2g A2) versus 0.934 at C44 (X1) [S, R2 §4].
  - The extra 1.3 points of misses at C44 may be capacity misses or compulsory misses (new sessions).
  - Host hits are only 2-6 % of tokens.
  - **k1 (360 GB, C44) versus X1 (450 GB, C44) is the direct sensitivity test.** If k1 loses ≤ 0.5 % and its GPU+host hit is within 0.5 points, then > 450 is not worth the RAM risk [I].

## 2. Decode capacity and headroom [required check]

**e2g decode DP0** [M] (`strings RUNS/e2g/server-logs/decode-0.a1.log`; DP1-3 within 0.1 GB):

| stage | value |
|---|---|
| weights | target 119.82 GB + NextN 6.92 GB |
| main KV pool | 2,762,176 tok, 142.07 GB → "Memory pool end. avail 15.81 GB" |
| draft KV pool | 1.82 GB → avail 13.99 GB |
| target-verify graphs bs [1..8,10,12,14,16] | mem usage 4.10 GB, 94.9 s → avail 11.77 GB |
| draft-decode graphs | 0.46 GB → 11.31 GB |
| draft-extend graphs | 0.42 GB → **10.89 GB at serve start** |

**Graph batch size is not a lever.**
- Captured graph bs stops at 16 because max_running 64 / attn_dp 4 = 16 per DP. That overrides `--cuda-graph-max-bs-decode 64`.
  - [M] The log line above, plus `max_running_requests=16` in the "max_total_num_tokens=…" line.
  - [S] Ledger R042 cont.: `memory_hook.py:193-198`.
- `cuda_graph_max_bs 64` is therefore not binding and not a lever. Capacity is unchanged by graph bs [S, report L2].

**Runtime transient memory.**
- The DSA MQA-logits scratch, the main non-graph transient, is row-chunked against a free-memory budget [M]:
  - `src/layers/attention/mqa_logits_utils.py:20-24,71-91`: 0.3 × free memory, ROCm cap 2^31−1 B.
  - `dsa_indexer_kpool.py:84-120`.
  - Lower free memory means more chunks, not OOM [I].
- No decode OOM was ever logged at 0.95 across about 12 full-hour runs (e1c-e2j, X1) [S, ledger "OOM 0" lines 43/316/912/955/990].

**0.96 / 0.97.**
- Capacity: +54.7k tokens per 0.01 (≈ 2.0 % of the pool) [S, report §3.1]. That gives ≈ 2.817 M (0.96) and ≈ 2.871 M (0.97).
- Post-capture free memory: ≈ 10.89 − 2.94 = **7.9 GB (0.96)** and ≈ **5.2 GB (0.97)** [I, from the v0519 pre-capture Δ: 11.25 → 8.31 → 5.55].
- Verdict: 0.96 is plausibly safe [I]. 0.97 leaves about 2.5× the logits cap; plausible but riskier, especially combined with more graph buckets.

**Does decode capacity or decode step time bind?**
- C40: decode-bound, busy 0.93-0.96. Step ≈ 23-26 ms + 2.5-3.5 ms per request on the busiest DP. DP lockstep pads to the max DP, with busiest/mean 1.46-1.60 [S, R041/R042].
- C44 (X1) [S, R2]:
  - Decode is less loaded than at C40 (mean node running 14.2 vs 14.9; TPOT 11.69 vs 12.40 ms).
  - Pool used per DP: p99 0.73, max 0.87.
  - Prealloc 0 %, retractions 0.
- The extra lanes sit in the prefill/transfer path: decode transfer residence 4.5 → 8.2 s.
- e2b4/e2c (C44, fresh) show the same: TTFT 8.0-9.0 s versus 7.2-7.6 s at C40 fresh [S, R2 §4].
- So at C44 neither capacity nor decode step time binds. **TTFT binds.**

**max_running per DP at C48.**
- The cap is 16 per DP [M]. Decode extra transfer slots = 2 × 16 = 32 per DP [M]: `src/arg_groups/pd_disaggregation_hook.py:136-148`, server_args `disaggregation_decode_extra_slots: 32`.
- X1: running per DP p50/p90/max 3/6/16; running ≥ 12 in 0.2-0.9 % of samples, = 16 in 0-0.09 % [S, R2 §2].
- Model at C48: running ≈ 15-18 per node, i.e. 4-4.5 per DP mean (§3). With 1.5× imbalance and burst tails, the cap is reached rarely but more often than at C44 [I].
- Raising to 24 per DP costs about +4 graph buckets (≈ +40 s capture, ≤ +1.4 GB) [S, R042 cont.]. Extra slots auto-scale to 48.

## 3. Throughput model (calibrated on e2g A2 / X1) [I]

- λ = C / (g + TTFT + D)
  - g ≈ 16 s: AgentX idle gap per request. From λ·e2e: e2g A2 gives 16.3 s, X1 gives 15.8 s [S, R2 §4].
  - D = OSL × TPOT, with OSL ≈ 1,000 tok.
  - TPOT = (26 + 2.9 × 1.5 × R/4) / 3.61 ms, where R = λ·D is node running.
  - Score = λ × 148k / 8.
- Calibration:
  - C40 at TTFT 5.3 → model 22.6k vs 22.1k actual (+2 %).
  - C44 at TTFT 9.0 → model 22.4k at ISL 148k ≈ 21.7k at X1's ISL 143k, vs 21,773 actual.

| CONC | TTFT 5.3 s | 7 s | 9 s | 12 s |
|---|---|---|---|---|
| 40 | 22.6k | 21.7k | 20.6k | 19.2k |
| 44 | 24.5k | 23.5k | 22.4k | 20.9k |
| 48 | 26.2k | 25.2k | 24.1k | 22.5k |

- Reading: at C44, the 23k bar needs TTFT ≲ 7.5 s, i.e. about −1.5 s versus X1. C48 at the X1 TTFT (9 s) would give about 23.4-24.1k. C48 fails the bar if TTFT grows to ≳ 12 s.
- TTFT grew C40 → C44 from 5.3 to 9.0 s (+70 % for +10 % lanes), mostly as transfer residence. So **C48 is only worth running after the KV-send path is improved**.
- Workload noise: ISL mean varies 141-149k between runs, worth ±3 % of the score by itself [S, R2 §4]. Compare ISL-normalized λ as well as tok/s.

## 4. Ranked candidate changes

Settings go through `YIHOU_OVERRIDES`, which takes space-separated KEY=VALUE with no spaces inside a value [M]: `H/run.v2.yihou.sh:19-26`. Multi-token engine args (`*_EXTRA_ARGS_ADD`) need a new wrapper config file, in the same pattern as `config.yihou.xferfix.sh`. Harness default chunk 32768 is divided by attn_dp → 8192 per DP [M]: `src/arg_groups/parallel_hook.py:306-314`.

| # | candidate change | expected effect | risk | evidence | how to set |
|---|---|---|---|---|---|
| 1 | **KV-send path fix**: xferfix mounts (`YIHOU_NONFINAL_EVENT=launch`, `YIHOU_XFER_SHARD=room`). Removes the R036 over-wait; room sharding breaks the single-worker head-of-line blocking (R035). | TTFT −1.5…−2.5 s at C44 → **+4…+6 %** [I, model] | low-med: ran a full hour PASS, MAF 0 (e2f). e2f's low score was the fresh-stack class, not shown causal (R041/R042). Includes regbase [M: `conn.yihou-xferfix.py` has 7 regbase + 6 XFER_SHARD hits]. | [S] e2f C40 fresh: TTFT mean **5.57 s (best of all C40, fresh or warm)** vs A1 fresh 7.61; [S] R2: KV send is the largest C44 TTFT phase | `CONFIG=$H/config.yihou.xferfix.sh`, `YIHOU_OVERRIDES="XFERFIX_BASE_CONFIG=$H/config.yihou.mc4340.sh MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh YIHOU_NONFINAL_EVENT=launch YIHOU_XFER_SHARD=room YIHOU_XFER_LOG=1 KV_CKSUM=0 …"` (as `env.e2f-102.yihou.sh`) |
| 2 | **Mooncake copy-stream priority** (`MC_HIP_STREAM_PRIORITY=high` on prefill) | Peer copies no longer queue behind prefill kernels → lower KV-send residence; additive to #1 [I] | med: not production-tested; image `…-mc4340prio` (sha256:aacc9aeb) exists **on 102 only** | [S] R043 microbench: copy behind a 200 ms kernel 200.6 → 11.0 ms | `IMAGE=infera-sglang:v0521-yihou-1010-mc4340prio` + its image ID (new config file; `config.yihou.mc4340.sh:38` asserts the mc4340 tag), `PREFILL_EXTRA_ENV_BASE="MC_LOG_LEVEL=INFO MC_HIP_STREAM_PRIORITY=high"` in that file. Env-only alternative (R043 arm h, 13.4 ms): `GPU_MAX_HW_QUEUES=16 MC_HIP_NUM_STREAMS=8` |
| 3 | **CONC 48** (after #1/#2) | +7…+8 % over C44 at equal TTFT [I, model]; neutral if TTFT grows +3 s | med: TTFT growth, decode tails | §3; X1 | `CONC=48` (config + `agentx_bench.sh` read the same variable) |
| 4 | **Decode max_running 64 → 96** (16 → 24 per DP) | Removes the rare cap hit at C48 (insurance, ≤ +1 %) [I] | low: +4 graph buckets (≈ +40 s capture, ≤ 1.4 GB); extra_slots auto 48 | [M] cap 16/DP, graph list to 16; [S] R2 cap reached ≤ 0.09 % at C44 | `DECODE_MAX_RUNNING=96` (`bench-harness/config.sh:66` → `engine.sh:264`); keep `DECODE_GRAPH_MAX_BS=64` (≥ 24) |
| 5 | **Prefill device KV: mem 0.85 → 0.88 (0.90)** | +168k (+280k) tok/DP = +7.4 % (+12.3 %) GPU cache; recovers part of the C44 GPU-hit drop 0.92 → 0.87; fewer host load-backs in the TTFT path [I] | low-med: prefill free memory 41.5 GB → ~33 (~27) GB; runtime peak unmeasured; logits scratch adapts [M] | [M] prefill avail 41.49 GB at 0.85, cell 55,224 B (116.79 GiB / 2,270,656); [S] R2 hit rates | `PREFILL_MEM_FRACTION=0.88` |
| 6 | **Decode mem 0.95 → 0.96** | +54.7k tok/DP (+2 %); no C44 gain; C48 tail insurance (worst-DP usage 0.87 at C44) | low-med: post-capture free ≈ 7.9 GB (≈ 6.5 GB with #4) | §2 | `DECODE_MEM_FRACTION=0.96` |
| 7 | **Prefill chunk 8192 → 16384 per DP** (+ `--max-prefill-tokens 32768`) | Fewer chunk iterations for long uncached prompts → shorter prefill queue (2.26 s at C44) [I] | low-med; more prefill activation | [S] e2e (65536 harness) fresh TTFT 6.94 vs fresh 7.2-7.6: weak; R040: no score effect, inconclusive | `PREFILL_CHUNK_SIZE=65536`; max_prefill_tokens via `PREFILL_EXTRA_ARGS_ADD="--max-prefill-tokens 32768"` in a wrapper config |
| 8 | **HiCache host 450 → 500 GB** | +11 % host tokens; worth at most the 1.3-point combined-hit gap [I] | med: RAM headroom 691 → 462 GiB with decode/prefill RSS unmeasured; OOM-kill risk | §1; k1 decides | `PREFILL_HICACHE_SIZE=500` (CHUNK_GB 512 still OK up to 549) |
| 9 | **DP load balance** (decode lockstep pads to the busiest DP, 1.46-1.6× mean) | Potentially −10 % decode step at matched load [S, R042] | **high**: `PD_DP_RANK_AFFINITY=0` means cross-rank HIP-IPC transfers, correctness untested (needs `KV_CKSUM=1` first); lowering `KV_PREFILL_OVERLAP_WEIGHT` (20) costs cache hit | [S] R042 lever 3 | `PD_DP_RANK_AFFINITY=0` / `KV_PREFILL_OVERLAP_WEIGHT=…` (`bench-harness/launch.sh:202-221`) — not for the next round |
| — | Not levers (closed) | — | — | decode HiCache = no-op in PD; graph bs list (e2j: no gain); MTP off (loses spec decode); page_size forced 64; FP4 KV unreachable [S, report §3.6]. `schedule_conservativeness` 0.3 (DP-scaled, `parallel_hook.py:300-303`) only matters if admission binds, which it does not. HiCache prefetch policy needs a storage backend (none) [M server_args]. Write policy write_through → selective would reduce host hits. Decode transfer cap (Track B) moves the wait, does not remove it [S, R2 §0.3]. | — |

## 5. Proposed next rounds (two amd-oai nodes; warm-stack protocol: A1 fresh unscored → flush both legs → A2 scored)

Second-node prerequisites [M/I]:
- Load `/shared_nfs/yihou_glm52/v0521cap_20261012/images/infera-sglang.v0521-yihou-1010-mc4340.9d755dd6.yihou.tar.zst` (23.5 GB).
- `config.yihou.mc4340.sh:23-28` rejects nodes other than 102/136, so a **new** config file is needed, e.g. `config.yihou.mc4340-oai2.yihou.sh`.
  - It copies that file, adds the node's `EXPECTED_IMAGE_ID`, and should match sha256:9d755dd6… if `docker load` preserves the config digest [I; verify].
- Write a new `topology.yihou.<key>.tsv` (auto-written by `config.yihou.v0521-base.sh:57-62`) and a new env file with that node's YIHOU_NODE, IP and JOB.

| round | node | config (deltas vs e2g) | hypotheses tested at once | read-outs |
|---|---|---|---|---|
| **K2: control** | the node with the e2g history (102 if k1 finishes there; else oai2) | **e2g exactly, CONC 44.** Add a host meminfo/RSS sampler plus the X1 metrics sampler. | (a) C44 vs C40, single variable (vs e2g A2 22,114); (b) replicate X1 (21,773, e2j stack had graph-bs/host-variant extras); (c) RAM headroom of the real stack (sizes the HiCache-500 decision); (d) with k1: host 360 vs 450 at C44 | tok/s/chip, ISL-normalized λ, TTFT/transfer residence, GPU+host hit, host-pool full time, MemAvailable min, per-DP #run=16 frequency |
| **Q1: transfer + prefill KV** | the other node | e2g + **xferfix (launch, room)** + **PREFILL_MEM_FRACTION 0.88** + **DECODE_MAX_RUNNING 96**, CONC 44 | H1: KV-send fix cuts TTFT at C44 (prefill inflight residence, TTFT mean); H2: +7 % prefill GPU KV recovers GPU hit (≥ 0.90); H3: max_running 96 is free (step time at matched load unchanged, no OOM); all three are separable by telemetry, not by score | same as K2 + `YIHOU_XFER_LOG` summaries |
| **R1: C48 push** (after K2/Q1) | 102 (has mc4340prio) | Q1's winners + **mc4340prio image with `MC_HIP_STREAM_PRIORITY=high`** + **DECODE_MEM_FRACTION 0.96**, **CONC 48** | H4: copy priority further cuts KV-send; H5: C48 > C44 when TTFT is held; H6: decode 0.96 stays OOM-free at C48 tails (worst-DP usage, retractions, prealloc>0) | as above + decode free-token minimum per DP |

- If k1 shows host 360 costs > 1 % versus X1, add `PREFILL_HICACHE_SIZE=500` to R1, but only if K2's MemAvailable minimum leaves ≥ 250 GiB after subtracting +246 GB of extra pinned memory [I].
- If Q1's TTFT gain is < 1 s, R1 stays at C44 and swaps CONC 48 for chunk 16384 per DP (#7).

## 6. Side question (user, 2026-10-12): what got us to 22k, and what did v0.5.21 contribute?

**Lineage** [S unless marked]:

| step | run | config | tok/s/chip | state |
|---|---|---|---|---|
| v0519 cross-node | t2f | tilelang DSA, index_share false | 19,799 | fresh |
| v0519 same-node | prior run | tilelang (co-location −5.6 %) | ~18,700 (89 % window) | fresh |
| **v0519 same-node** | 0923 packup | **triton DSA both legs + index_share TRUE** (+12.5 % together), HiCache ratio 1.5 (154 GB/rank), decode 0.85, EAGLE 5/1/6, HIP-IPC intra-node, sim-acc 3.61 | **21,032** | fresh |
| v0.5.21, same config but index_share false | e1c | relaxed (154 GB, decode 0.85) | 21,049 | fresh |
| v0.5.21 target, C44 | e2b4 / e2c | HiCache 450 (+CHUNK_GB 512), decode 0.95, fused indexer off / on | 20,347 / 20,524 | fresh |
| v0.5.21 target, C40 | e2g A1 / e2e / e2f / e2i R1 / e2j R1 | target | 19.3-20.4k | fresh |
| v0.5.21 target, index_share TRUE | e2d | — | 21,383 | fresh (debug agent; the only fast fresh run) |
| **v0.5.21 target, C40** | **e2g A2** | target | **22,114** (replicates 21,921 / 21,696 / 21,681) | **warm (2nd run after flush)** |

Server args sources: [M] `glm52.samenode-triton-idxshare.c40-full.packup_20260923/results/server-info.29001.json` and `…29257.json` (v0519: prefill mem 0.85, hicache_size 0/ratio 1.5, decode mem 0.85, version 0.5.19.dev20260917); e2g logs (§2).

**Attribution** [I, from these numbers; no single-variable A/B exists for most steps]:

- **Largest single factor in the 22,114 headline: the warm-stack protocol.**
  - The same target stack scores 19.3-20.4k on its first run and 21.7-22.1k on the second, i.e. +8-13 %.
  - This is a decode-step penalty in fresh processes (mechanism open), not a config optimization.
  - The v0519 21,032 was a fresh run. **No warm v0519 run exists**, so 22,114 vs 21,032 is not like for like.
- **v0.5.21 itself: about 0 at equal conditions.**
  - e1c (v0.5.21, fresh, v0519-like relaxed config, index_share false) = 21,049 vs v0519 fresh 21,032 (+0.1 %), without v0519's index_share TRUE.
  - The new v0.5.21 fused DSA indexer: e2b4 → e2c +0.9 % at C44 (one sample each, not significant), at a cost of −0.4 % KV capacity.
- **Capacity config (HiCache 450 GB + decode mem 0.95):**
  - Fresh-vs-fresh it did not help: target fresh 19.3-20.4k vs relaxed e1c fresh 21,049 (both 1-sample-per-config, time-of-day confounded).
  - It is what the current recipe runs on, and it is not shown to be the source of the gain.
- **Engineering that made the v0.5.21 target config run at all** (stability/correctness, ~0 throughput):
  - H1 HiCache host-register chunk (CHUNK_GB 512);
  - Mooncake #4340 port;
  - DECODE_PID_OFFSET (PID collision under spur);
  - kv-events non-blocking publisher;
  - absorb-bmm config mount;
  - port plan;
  - index_share false (MAF at C44 with TRUE).
- **Tried, no measurable gain:** xferfix (TTFT best, score not; C40 decode-bound), chunk 65536, cuda-graph bs 1..16 + host-variant (e2j), NUMA pinning, index_share TRUE at C40.
- **The real throughput wins in the whole lineage were in v0519:** tilelang → triton DSA (+ index_share), about +12.5 % same-node.
