# GLM-5.2 v0.5.21 1P1D P4D4 — which leg's KV capacity binds at C48? (synthetic expert reduction on ONE leg)

**Ran:** 2026-10-13 06:28 → 20:03 UTC (5 rounds, 4 nodes). **Packed:** 2026-10-14. **Author:** yihou (with Claude).
**Status: DONE for the directive** (both one-leg arms measured; the decode arm needed a rerun after an HBM-ECC node).
Single runs, different nodes, synthetic (non-faithful) models on the reduced leg — read the caveats below.

## Goal (user, 2026-10-13 ~05:15 UTC; verbatim excerpt of part 1 — full text in `spec/mission.md` line 83 and `spec/directives.md`)

> 大hi-cache 大conc的我我不理解为啥没有提升，因为之之前强行8 expert开大kv cache是可以到26k, 你去并行/串行做两个实验a.单独让decode使用8 expert腾出更多的kv-cache. b. 让prefill单独开expert, 获得更大的kv-cache观察下那边有提升，看看谁是瓶颈

Plus two follow-ups the same day (both quotes second-hand: from the ledger `poll.md` and a config header comment, see `spec/directives.md`): **x8pc** (~15:00, "跑": prefill-only expert8 but prefill KV capped back to the 256-expert
size, to separate "fewer experts" from "more KV") and **x128p** (~16:00, "prefill侧expert砍半": prefill with 128 of 256
routed experts). Success criterion: identify the leg whose KV capacity limits the C48 score (mission target > 23k tok/s/chip).

## Headline (AgentX 3600 s, C48; score = `request_metrics.throughput.per_gpu.total_tput_tps`, tok/s/chip)

All rows: q1 recipe (e2g + decode max_running 96 + prefill mem 0.88, decode mem 0.95, HiCache 450 GB/rank write_through,
EAGLE MTP 5/1/6, `SGLANG_SIMULATE_ACC_LEN=3.61`, triton DSA, image `infera-sglang:v0521-yihou-1010-mc4340`). Only the
reduced leg changes. KV columns = `max_total_num_tokens` per DP rank, read from /metrics.

| round (node) | change vs q1 | prefill KV | decode KV | A1 fresh | **A2 warm (scored)** | vs q1 | TTFT mean / p95 s | GPU hit | CPU hit |
|---|---|---|---|---|---|---|---|---|---|
| q1 A3 (137) **reference** | — (real 256 experts both legs) | 2,433,088 | 2,762,176 | — | **22,306** (A3) | — | 11.97 / 38.19 | 0.839 | 0.088 |
| x8pc (206) | prefill 8 experts, KV **capped** to 2,433,088 | 2,433,088 | 2,762,176 | 21,591 | **22,489** | +0.8 % | 12.28 / 43.83 | 0.847 | 0.080 |
| x8d2 (206) | decode 8 experts (incl. MTP draft) | 2,433,088 | **3,670,016** (+32.9 %) | 22,511 | **23,591** | +5.8 % | 10.43 / 33.44 | 0.857 | 0.076 |
| x128p (085) | prefill 128 experts (truncated) | **3,318,400** (+36.4 %) | 2,762,176 | 24,316 | **25,297** | +13.4 % | 6.23 / 19.47 | 0.944 | 0.005 |
| x8p (274) | prefill 8 experts | **3,670,016** (+50.8 %) | 2,762,176 | 25,347 | **25,905** | +16.1 % | 5.45 / 17.02 | 0.950 | 0.002 |
| x8d (026) **INVALID** | decode 8 experts | 2,433,088 | 3,670,016 | 20,230 (flagged) | crashed | — | 14.03 / 46.28 (A1) | 0.893 | 0.041 |
| x8d2 A3 (206) | decode 8 experts, **C52** | 2,433,088 | 3,670,016 | — | 21,713 (A3) | — | 16.12 / 52.11 | 0.800 | 0.118 |

x8d: 026 GPU3 (KFD node 5, a prefill GPU) had 13,198,148 correctable + 2 deferred HBM ECC errors; A2 died 08:39 with
"Memory access fault by GPU node-5 … DRAM ECC failure / SRAM ECC failure". A1 ran on the same GPU → kept, flagged, not used.
x128p: 180 `InvalidInferenceResultError` (empty content) in **each** of A1 and A2 (others: 0-5); cause not investigated.

### What was measured (observations; mechanism not proven)

1. C48 A2 score orders with **prefill device KV**: 2.43 M → 22.3 k (256 exp) / 22.5 k (8 exp); 3.32 M → 25.3 k; 3.67 M → 25.9 k.
   With prefill KV held at 2.43 M, 8 prefill experts score like 256 (x8pc 22,489 vs q1 22,306).
2. In the same order: prefill GPU hit 0.84 → 0.95, prefill CPU (HiCache) hit 0.088 → 0.002, TTFT mean 12 s → 5-6 s,
   prompt tokens actually computed (not a cache hit, `server_metrics.tokens.prompt_by_source.computed`) 54.9 M (q1) /
   55.5 M (x8pc) → 42.8 M (x128p) / 41.4 M (x8p) per run while total prompt tokens rose 1.50 G → 1.68-1.72 G.
3. /metrics (`analysis/metrics_digest.tsv`, warm runs): prefill `token_usage` p90 per DP is 0.34-0.54 with the 2.43 M
   prefill pool (q1 A3, x8d2 A2, x8pc A2; max 0.92-1.00) vs 0.19-0.22 (x8p A2; max ≤ 0.75) and 0.22-0.28 (x128p A2;
   max ≤ 0.92). Prefill `num_queue_reqs` mean 1.44-1.67 vs 0.88-0.90. Decode retractions 0 in every run.
4. Decode KV +32.9 % (x8d2) gave +5.8 % vs q1 (+4.9 % vs x8pc on the same node 206); decode `token_usage` max fell from
   0.77-0.99 (q1) to 0.63-0.72; TTFT barely moved (10.4 vs 12.0 s).
5. C52 on the bigger-decode stack (x8d2 A3, 21,713) is below its C48 (23,591), as for q1 (C52 21,418 < C48 22,306).

**Reading [inferred, suspended]:** at C48 the prefill-side device KV (radix cache reuse) is the binding capacity on this
recipe; decode KV is secondary. Not proven: one run per arm, nodes differ (137/206/085/274), and the reduced leg also
changes MoE compute and weights. No faithful (256-expert) configuration reached > 23 k.

## Folder map

- `spec/` — `mission.md` (verbatim ledger mission incl. the 2026-10-13 updates), `directives.md` (the directives for these 5 rounds).
- `REPRODUCE.md` — node hold → checks → image → JIT seed → env → dryrun + tokdiff → chain via `spur exec` → score.
- `environment.md` + `environment/collect_env.{085,274}.20261014.txt` — HW/driver/RDMA/image; gaps listed.
- `scripts/harness/` — harness core (`bench-harness/`), round env files, configs, chain drivers, mounts TSVs + regbase sources,
  `tools/audit_live.v3.yihou.py`; `scripts/session/` — local ssh shim, metrics sampler, summary tools, expertn builder;
  `scripts/dryrun/` — rendered dryruns of all 5 rounds + the q1-137 reference; `scripts/alloc/hold_inf.yihou.sh`.
- `patches/` — expert8 (patch, mounted files, `.orig` = image files, diff vs image, config key-diff) and expertn (128) — see `patches/README.md`.
- `results/` — `results.csv` (all runs, all metrics), per-round `status.tsv`/`audit.txt`/`census.txt`/chain logs, per-run
  AgentX JSON + plots + gzipped benchmark/runner logs, gzipped /metrics TSVs, rendered tokdiffs; `results/README.md`.
- `analysis/` — `metrics_digest.tsv` (per-DP KV/queue digest), research notes copied from the session, ledger copies.
- `notes.md` — gotchas, wrong turns, caveats (what / why / how / context).
- `logs/LOG_INDEX.md` — absolute paths + sizes of server logs and aiperf artifacts NOT copied (17-38 MB / up to 1.4 GB).

Sibling packups (same parent dir): `glm52.v0521-cap-c44c48.packup_20261013` (q1 recipe, C44/C48/C52, Track B),
`glm52.v0521-decode-transfer-kvcap.packup_20261013` (decode in-flight KV cap), `yihou-expert8-pd.packup_20260924` (origin of
the expert8 patch), baseline kit `/home/yihou/dev/git/infera.glm52.view/glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012/`.
Other rounds of the same session (tb56/tb56n runshare at C56/C64, pc48* prefill_complete, hc50, pf88/90/92) are not in this packup.
