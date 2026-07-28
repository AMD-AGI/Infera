# 07 — PD mooncake RDMA + DP-attention, high-concurrency sweep (GLM-5.2-MXFP4, sglang)

**Ran:** 2026-07-27 · **Status:** ✅ PASS (all 6 concurrency points, zero failures)

## Goal

Turn **DP-attention ON** for the GLM-5.2-MXFP4 mooncake-RDMA PD path (03 was pure TP8, no DPA) and
stress it across a full concurrency ladder **conc = 64 / 128 / 256 / 512 / 1024 / 2048** at 1k/1k,
tuning the capacity-limiting knobs (context-length, chunked-prefill, KV pool, max-running-requests)
the way the DSv4 R4 high-conc reference did.

**Success criteria:** correct output + every concurrency point completes all requests (no
KVTransferError / retract / OOM / crash).

## Result — all 6 points PASS

| conc | completed | out tok/s | out/GPU | median TTFT | p99 TTFT | median TPOT | median ITL |
|-----:|:---------:|----------:|--------:|------------:|---------:|------------:|-----------:|
| 64   | 256/256   | 2188  | 274  | 1396 ms | 6423 ms  | 27.3 ms | 27.3 ms |
| 128  | 512/512   | 3485  | 436  | 1660 ms | 12096 ms | 32.6 ms | 32.1 ms |
| 256  | 1024/1024 | 6293  | 787  | 2402 ms | 6656 ms  | 36.7 ms | 36.9 ms |
| 512  | 2048/2048 | 10164 | 1271 | 2727 ms | 10618 ms | 45.1 ms | 45.2 ms |
| 1024 | 4096/4096 | **12855** | **1607** | 3532 ms | 18952 ms | 65.7 ms | 62.5 ms |
| 2048 | 4096/4096 | 11970 | 1496 | 17546 ms | 40350 ms | 129.4 ms | 84.5 ms |

- **Correctness:** 4/4 temp=0 probe (Paris / Beijing / 4 / Jupiter, with chain-of-thought) via router.
- **Transport:** real mooncake **RDMA** (8× `rdma_context.cpp HIP dmabuf disabled`, 0 TCP fallback).
- **Zero real errors:** 0 KVTransferError, **0 retractions** (peak KV pool usage only **0.15**), 0 OOM.
- Throughput scales ~6× from conc=64→1024 (the DP-attention high-conc payoff), **peaks at conc=1024**
  (12855 out tok/s), then dips slightly at 2048 as request count (4096) far exceeds steady KV capacity
  → longer queueing/TTFT, but **still 4096/4096 complete**. See `notes.md` §saturation.

Topology: prefill = chi2878 (10.2.122.3), decode = chi2879 (10.2.122.10). Both legs **symmetric
DP8**: `--dp-size 8 --enable-dp-attention --ep-size 8` (+ prefill-delayer on prefill), TP8, 8 ionic NICs.

## Why DP-attention, and why symmetric on both legs

- 03 (pure TP8) satisfies conc=64 but, like DSv4's no-DP R4, **falls off a cliff at conc≥256**
  (attention replicated across all TP ranks wastes compute at high batch). The DSv4 reference proved
  the fix is EP + **DP-attention**: each rank owns a shard of the batch, so attention scales.
- Both PD legs **must run the same `dp_size × tp_size` layout** — mooncake transfers the KV tensors
  shard-for-shard, so an asymmetric (one leg DP, one leg TP-only) config would mismatch the KV layout.
  Hence symmetric DP8 on prefill **and** decode.

## Capacity knobs (the parameters the task called out)

| knob | value | why |
|------|-------|-----|
| `--context-length` | 32768 (was 400000) | 1k/1k needs ≤2k/req; the huge ctx only ate KV headroom. |
| `--chunked-prefill-size` | 65536 (= ISL 8192 × TP 8) | DSv4 DP recipe: one chunk spans the DP group. |
| `--max-running-requests` | 2048 | scheduler cap sized for the top of the ladder. |
| `--cuda-graph-max-bs` | 128 | capture ≤128; larger batches replay eager (fine). |
| `--mem-fraction-static` | prefill 0.88 / decode 0.85 | DP prefill wants LOWER memfrac (HSA-OOM guard). |
| KV pool (result) | prefill 3.26M / decode 3.10M tokens | ~1550 × 2k-token reqs; peak usage only 0.15. |
| prefill-delayer | on, 5000 ms | R4 high-conc lever: batches incoming prefills (prefill leg only). |

## How to reproduce

See `REPRODUCE.md`. TL;DR: `pd-unified` image, both legs via `pd_leg_dpa.sh` with `DPA=1`, router,
then `sweep_dpa.sh` over the conc list. Builds directly on 03's container/libionic/transport setup.

## Folder map
- `REPRODUCE.md` — step-by-step (builds on 03)
- `scripts/pd_leg_dpa.sh` — DPA PD leg launcher (symmetric dp8+ep8, GATHERV, prefill-delayer)
- `scripts/up_dpa.sh` — orchestrator (containers + libionic + both DPA legs)
- `scripts/sweep_dpa.sh` — the conc-ladder bench client
- `scripts/probe.py` — temp=0 correctness probe
- `results/dpa_c*.jsonl` — 6 raw bench_serving results · `sweep_summary.csv` · `sweep_table.txt`
- `logs/bench_c*.log` — per-conc client logs · `{prefill,decode}_leg.trimmed.log` — server legs (trimmed)
- `notes.md` — DPA rationale, KV sizing, the conc=2048 saturation analysis, the "5257 retract" false alarm
