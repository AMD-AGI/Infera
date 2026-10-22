# GLM-5.2 PD with a DCP4 decode leg, cross-node 135 -> 136 (2026-10-22)

**Goal.** Take our best cross-node 1P1D config (g3x1, 10-21: A1 25524 / A2 25376 tok/s/chip at C56) and change ONLY the
decode leg to the DCP4 recipe of session sglang-pd-pp-dcp-fa
(`yihou/glm52.dcp4-pd.packup_20261022/RUN_BEST_DCP.md`): decode TP4 EP4 DP1 (no DP attention) + `--dcp-size 4`.
Then find out where DCP's TPOT/throughput goes and improve it.

**Result so far (one complete bench, dcpx6 A1, C56):** score **19691** tok/s/chip (g3x1: 25524). Client TPOT mean
15.6 ms (p90 19.1); g3x1 15.9 ms. TTFT p50 19.8 s (g3x1 4.3 s). KV transfer latency p50 21.3 s (g3x1 1.0 s).
Per-request decode is faster (decode_stats 08:34-08:49: 100.1 tok/s/req vs 72.7), but only ~13-17 requests decode at once
vs ~24, so total output is lower. dcpx6 A2 was still running when this kit was made (status in `results/rounds/dcpx6`).

| round | change vs previous | outcome |
|---|---|---|
| dcpx1 | g3x1 + decode -> DCP4 r5b (mem 0.85, max_running 64, idx false), image v0522 + overlay 93034eba91 on both legs | stopped during bring-up (NCCL_DEBUG=INFO spam from our base) |
| dcpx2 | dcpx1 without decode NCCL_DEBUG | stopped by user mid-A1 (no score); ITL last 60 s ~21 ms |
| dcpx3 | + index_share true, max_running 128, graph bs 128, decode mem 0.92, Mooncake dest-local-rail image | decode abort `HSA_STATUS_ERROR_OUT_OF_RESOURCES` 7 min into profiling |
| dcpx4 | decode mem 0.90 | same abort 12 min into profiling |
| dcpx5 | decode mem 0.85 | stopped before profiling (135 lent out) |
| dcps1 | dcpx5 on one node (136, HIP IPC) | stopped in warm-up (135 back) |
| dcpx6 | = dcpx5 (rerun) | A1 complete 19691; A2 running at pack-up time |

## Navigate
- `REPRODUCE.md` — image build, env, dryrun, launch.
- `environment.md` — nodes, images, versions.
- `notes.md` — findings, traps, open questions (what / why / how / context).
- `results/summary.tsv` — one row per complete bench (dcpx6 A1 + g3x1 A1-A4 reference) with transfer histograms.
- `results/rounds/<round>/` — status, census, audit (server-info json), metrics sampler tsv (gz), decode crash tails.
- `results/dryruns/`, `scripts/envs/` — exact launch lines and env files of every round.
- `results/rdma/` — ionic hw-counter snapshots.
- `imgbuild/` — Dockerfile + Mooncake patches of the `-destpin` image.
- `scripts/harness/`, `scripts/tools/` — harness pieces and analysis tools used.
