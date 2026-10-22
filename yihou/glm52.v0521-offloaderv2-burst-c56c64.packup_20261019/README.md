# GLM-5.2-MXFP4 v0.5.21 same-node P4D4 — OffloaderV2 at C56 / C64 on preemptible burst nodes (2026-10-15 13:30 → 2026-10-19 00:17 UTC)

Follow-up to `../glm52.v0521-offloaderv2-highconc.packup_20261015` (C48 ratio sweep + first C56 points). This kit packs every round run after that kit's last update (10-15 13:30 UTC). **Status: COMPLETE** — the user's 4-item queue (G2 C56, G3 C64, G2 C64, G4 C64) each has an official scored A2.

## Question (user)

- 10-15 07:50: "我们是要测offload在高conc下能不能拿到更多的吞吐，baseline早就有了" — offload at C56/C64; **no new no-offload runs** (user's decision).
- 10-15: every new round's decode runs `--tokenizer-worker-num 4 --detokenizer-worker-num 4` (overlay `config.yihou.tokw2.sh`).
- 10-16: burst nodes are preempted (20 min – hours) → **short rounds**: `chain_cap2 C C` = A1 fresh + A2 scored at one concurrency; a lost round is rerun under a new name.

## Result — offload by concurrency, lowest to highest (score = `request_metrics.throughput.per_gpu.total_tput_tps`, n = 1 each)

A1 = fresh stack, A2 = warm stack (scored). Prefill KV per rank: none 2,541,376 · G8 n/a here · G4 2,938,752 · G3 3,092,288 · G2 3,378,496 tokens.

| conc | offload (prefill experts in host memory) | round | node | decode tok/detok | A1 | **A2** | A2 TTFT mean/p95 (s) | A2 TPOT (ms) | A2 GPU / CPU hit |
|---|---|---|---|---|---|---|---|---|---|
| 48 | none (ref, pf90) | pf90 | 089 | 1/1 | 21,859 | **23,127** | 9.30 / 28.76 | 13.1 | 0.860 / 0.071 |
| 48 | G8K1 (1/8) | ofs0 | 089 | 1/1 | 22,380 | **22,825** | 9.75 / 30.51 | 12.9 | 0.887 / 0.049 |
| 48 | G4K1 (1/4) | ofg4b | 136 | 1/1 | 22,506 | **23,771** | 9.71 / 28.77 | 12.1 | 0.925 / 0.018 |
| 48 | G3K1 (1/3) | ofg3b | v3-020 | 1/1 | 22,484 | **23,200** | 10.78 / 29.35 | 12.0 | 0.936 / 0.010 |
| 48 | G2K1 (1/2) | ofg2 | 288 | 1/1 | 21,157 | (aborted) | — | — | — |
| 56 | none | — | — | — | — | — | no like-for-like run (tb56n A2 18,016 used HiCache ~154 GB / mem 0.88) | | |
| 56 | G4K1 | ofg4b A3 | 136 | 1/1 | — | **21,744** (3rd use of stack) | 20.04 / 51.99 | 12.4 | 0.857 / 0.067 |
| 56 | G3K1 | ofg3h2 | v3-010 | 4/4 | 20,559 | **21,310** | 20.27 / 58.47 | 13.2 | 0.882 / 0.048 |
| 56 | G2K1 | ofg2h21 | v3-006 | 4/4 | 19,650 | **21,780** | 21.13 / 54.37 | 13.2 | 0.918 / 0.019 |
| 56 | G2K1 (A1 only) | ofg2h5 / ofg2h17 | v3-008 / v3-006 | 4/4 | 21,766 / 19,746 | (preempted) | — | — | — |
| 64 | none | — | — | — | — | — | no like-for-like run (tb56n A3 15,108, same caveat) | | |
| 64 | G4K1 | ofg4h5 | v3-006 | 4/4 | 18,192 | **18,535** | 35.10 / 80.49 | 13.5 | 0.737 / 0.165 |
| 64 | G3K1 | ofg3h14 | v3-006 | 4/4 | 17,669 | **18,333** | 35.57 / 88.68 | 13.2 | 0.773 / 0.131 |
| 64 | G2K1 | ofg2h22 | v3-006 | 4/4 | 17,605 | **18,392** | 37.26 / 81.33 | 12.1 | 0.837 / 0.073 |
| 64 | G4K1 UNOFFICIAL | ofg4h4 | v3-006 | 4/4 | 16,905 | 17,693 | rescored after dropping one NUL line (see notes) | | |

Rows above the C56 G3 line come from the earlier kits; full columns in `results/results.csv`.

**What the measurements show (no mechanism claimed):**
- C64 A2 scores for G2/G3/G4 are within 1.1 % of each other (18,333 – 18,535), all on the same node v3-006.
- C64 is ~15 % below C56 for G2 on the same node (18,392 vs 21,780). Every C56 point is below every C48 A2 point.
- With rising concurrency the GPU prefix-cache hit falls and the CPU hit rises (G4 C64 A2: 0.737 / 0.165).
- **Open:** whether offload adds throughput at C56/C64 vs no offload. There is no like-for-like no-offload run at C56/C64; the user chose not to run one.
- **Open:** node effect. G2 C56 A1 on v3-006 was 19,746 and 19,650 (two allocations), on v3-008 21,766. Cause unknown; a same-session A/B of the two nodes would answer it.

## What happened (details in `notes.md`)

- Most rounds were lost to preemption (26 lost rounds, listed in `results/rounds/`). Of the 11 `amd-spur` holds that started, 10 ended at 20:00–20:03 and one at 31:30 (partition PreemptExemptTime=1200), so no `amd-spur` round ever reached a bench. All scored points came from `amd-spur-v2`; v3-006 alone held 19.5 h (10-18 05:46 → 10-19 01:51) and produced the four queued A2 points.
- `/shared_nfs` (420 T, cluster-wide) hit its quota twice (10-17 morning, 10-18 18:18–19:50). The second time killed the ofg4h4 driver and corrupted one line in each bench's `profile_export.jsonl` → no official ofg4h4 score; rerun as ofg4h5.
- spur controller outages (10-17 16:53–18:12, plus short ones) broke `spur exec` and sacct; retries via another login host worked.

## Navigate

`REPRODUCE.md` · `environment.md` · `notes.md` · `results/results.csv` · `results/<round>-c<C>-<A>/` (bench dirs) · `results/rounds/<round>/` (status, chain/pipeline/cont/stop logs, bench console logs, every lost round too) · `results/dryruns/` (+ tokdiffs) · `results/unofficial/` · `results/pyspy/` · `results/metrics/` · `logs/server/<round>/` · `analysis/ledger/poll.md` · `scripts/`

Rebuild: `bash scripts/tools/packup_copy_burst.yihou.sh` (copy-only).

**Not copied (>4 MB, left in place):** each bench's `aiperf_artifacts/profile_export.jsonl` (25–29 MB) and `server_metrics_export.json` (~1.6 GB) under `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/<round>/agentx-c<C>-<A>/aiperf_artifacts/`, including the partial `ofg2h17/agentx-c56-A2` (4,334 requests, no JSON).
