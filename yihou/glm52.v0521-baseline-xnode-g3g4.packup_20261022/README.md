# GLM-5.2 v0.5.21 1P1D P4D4 — new baseline (index_share + idlefix), same-node knobs, and CROSS-NODE 1P1D (2026-10-21)

**Written:** 2026-10-22. **Runs:** 2026-10-21 01:57Z – 19:03Z. **Author:** yihou (with Claude).
**Score** = AgentX 3600 s `request_metrics.throughput.per_gpu.total_tput_tps` (tok/s/chip, 8 GPUs). A1 = fresh stack, A2 = warm
stack after `/flush_cache` on both legs (the scored point); A3/A4 = further warm benches on the same live stack.
All runs use EAGLE MTP 5/1/6 with `SGLANG_SIMULATE_ACC_LEN=3.61`: **correctness is waived by construction** — none of these
numbers is an accuracy result. Every row is n = 1 unless stated. Previous packup: `glm52.v0521-idxshare-maf-fix.packup_20261021`.

## What this packup covers

1. **New baseline** = 10-20 rule (PR #42553 GTT HiCache 260 GB/rank, triton DSA, decode custom all-reduce ON, decode
   tokenizer/detokenizer 4/4, `index_share_for_mtp_iteration=true`) + the idle-batch fix (idlefix, upstream PR #45483),
   OffloaderV2 G4K1 step 1, prefill mem 0.90, decode mem 0.95, decode max_running 96, C56. Replicated on 3 nodes, 0 faults.
2. **One-variable same-node rounds** vs that baseline: hipxfer timing shim, chunk 65536, offload-prefetch-step 2, CONC 52.
3. **Cross-node 1P1D P4D4** (prefill crsuse2-m2m-135 GPU0-3 -> decode crsuse2-m2m-136 GPU0-3, KV over Mooncake RDMA
   rail i <-> rail i): G4 replicate + conc sweep, G3 + conc sweep, RDMA rail probes, decode-TPOT analysis.
4. Upstream PRs opened: sgl-project/sglang **#45483** (idlefix) and **#45484** ([AMD] offloader keeps `is_shuffled`).

## Headline (A2 unless noted; full table `results/summary.tsv`)

| setup | C | score | TTFT p50/p90 s | TPOT ms | transfer_kv stage p50/p90 s |
|---|---|---|---|---|---|
| same-node baseline (097 / 268 / 054) | 56 | 21814 / 21538 / 20516, **mean 21289** | 13.8-15.3 / 45-49 | 12.2-12.7 | 6.2-7.5 / 19.7-22.6 |
| same-node + hipxfer shim (268) | 56 | 21280 | 16.69 / 48.38 | 12.0 | 8.66 / 23.85 |
| same-node prefetch step 2 (054) | 56 | 21342 | 14.96 / 46.73 | 12.3 | 7.35 / 22.57 |
| same-node chunk 65536 (054) | 56 | A1 21092; A2 crashed (prefill HSA_STATUS_ERROR_OUT_OF_RESOURCES) | | | |
| **same-node CONC 52** (268) | 52 | **23367** (A1 23048) | 8.85 / 25.74 | 12.7 | 4.27 / 14.23 |
| **cross-node G4** (g4x1, g4x2) | 56 | A1 24976 / 24398, A2 24362 / 24896 | ~4.5 / 27-29 | 15.6-15.8 | ~1.1 / ~2.1 |
| cross-node G4 (g4x2, same stack) | 60 / 64 | 23056 / 22670 | 6.26 / 42.1 ; 8.45 / 53.1 | 16.0 / 16.1 | 1.22 / 2.22 ; 1.32 / 2.61 |
| **cross-node G3** (g3x1) | 56 | **A1 25524, A2 25376** (best so far) | 4.53 / 27.22 | 15.4 | 1.09 / 2.09 |
| cross-node G3 (same stack) | 60 / 64 | 23905 / 22860 | 5.56 / 37.3 ; 8.29 / 44.4 | 15.6 / 16.1 | 1.23 / 2.33 ; 1.29 / 2.49 |

Measured, not interpreted further (see `notes.md`):
- Cross-node C56 is +14-20 % vs the same-node baseline mean; its ISL mean is 4-7 % lower (127-132k vs 136-141k) — not normalised.
- Per Mooncake call (prefill YIHOU-XFER log, 1 h of A1): cross-node RDMA 20.1 GiB/s, queue wait 0.34 s; same-node HIP 2.62 GiB/s, 5.43 s.
- Cross-node decode TPOT 15.7 vs 12.5 ms: at equal decode load the per-step cost is the same; cross-node decodes more requests
  at once (4-DP running 23 vs 15) — `results/xnode/tpot_analysis.20261021.md`.
- Raising CONC above 56 cross-node lowers the score for both G4 and G3; transfer grows only 1.1 -> 1.3 s, prefill queue and GPU
  hit move.
- Same-node HIP copies (hipxfer shim, whole g4h1 run): copies run ~6.35 GiB/s median on the GPU and ~1/3 of the call wall is
  outside the copies' GPU span. Causes not measured.

## Navigation

| path | what |
|---|---|
| `REPRODUCE.md` | ordered commands: node prep, rail probe, same-node and cross-node rounds, conc sweep, analysis |
| `environment.md` | nodes, image, driver, RDMA, model, image IDs per node |
| `notes.md` | what/why/how of every finding, the traps hit, open questions |
| `patches/` | idlefix + offloader upstream PR patches/bodies, hipxfer shim source (in scripts/mounts-src) |
| `scripts/harness/` (+ `config-chain/` = the base overlays and bench-harness config/engine) | the 10-21 harness pieces (xnode config/driver/chain, conc sweep, hand-over, shim overlay) verbatim |
| `scripts/envs/`, `results/dryruns/` | env file + rendered dryrun of every round (tokdiff evidence) |
| `results/summary.tsv` | one row per bench: score, qps, ISL, TTFT, TPOT, hits, transfer histograms |
| `results/rounds/<round>/` | status, census, audit server-info, chain/pipeline/conc logs, agentx JSON, server_metrics_export.csv.gz, hipxfer logs |
| `results/xnode/` | RDMA rail probes (per rail, both directions), TPOT analysis report |
| `analysis/` | the TPOT matched-load binning script |
| `env/` | collect_env.sh output for 135 and 136 |

Not included (ask before adding): server logs (`R/<round>/server-logs/*.log`, GBs), 5 s metrics sampler series
(`/shared_nfs/yihou_glm52/v0521cap2_20261013/runs/<round>/metrics.A*.tsv`), Mooncake perf dumps of the rail probes.
