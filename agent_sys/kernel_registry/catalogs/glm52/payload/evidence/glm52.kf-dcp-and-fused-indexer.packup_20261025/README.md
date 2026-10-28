> Redacted historical snapshot. Measurements and correctness statements refer to the original run, not these published bytes. Revalidate the published source and adapters on the target stack before reuse.

# GLM-5.2 decode KernelForge, round 2: DCP4 kernels and the gfx950 fused indexer (2026-10-23 12:50Z .. 10-25 03:xxZ)

Follows `../glm52.kf-decode-kernels-e2e.packup_20261023/` (DPA: three KF kernels, TPOT 14.7-15.3 -> 11.2-11.3 ms;
that overlay was also run by session historical-e2e-session on contributor's 4+4 C80 base: +10.4 %, TPOT 19.8 -> 15.4 ms,
their pack-up `registry/glm52.llrepro-kf-ar-topk.packup_20261024/` in historical-e2e-workspace).

All e2e rounds: cluster cluster nodes, one node per round (prefill GPU0-3, decode GPU4-7, HIP IPC), image
`historical-sglang-image-a` + sglang overlay 93034eba91, AgentX C56 3600 s, simulated acceptance 3.61
(speed only, **no correctness claim**), one variable per comparison (dryrun token diffs in `cluster/kf_e2e_20261023/`).

## Results (`results/summary.tsv`)

| comparison | rounds | TPOT mean ms | score tok/s/chip | verdict |
|---|---|---|---|---|
| DCP base vs DCP KF sparse+indexer | kda1 vs kdc1 (both compute-node-05) | 19.44 / 19.26 vs 21.33 / 21.13 | 27.06k vs 25.6k | KF **slower** (-5 %) |
| DCP KF indexer only | kde1 (compute-node-05) | 21.51 (A1, recomputed from raw records) | n/a | the indexer is the regression |
| DCP KF sparse only | kdg1 (compute-node-07) | 18.92 / 18.82 | 27.3-27.4k | -2.5 % TPOT, within node spread (other node) |
| DPA KF vs DPA KF + fused indexer | kfb3 (192) vs kfc1 (046) | 11.17 / 11.29 vs 11.25 / 11.25 | 25.5-25.6k vs 25.2-25.4k | no measurable change |

Kernel level (v0522 image, task drivers, CUDA graph): DCP sparse 0.0635 -> 0.0385 ms weighted (1.45-1.91x by rows,
never slower), DCP indexer 0.332 -> 0.141 ms (2.0-2.7x) on SYNTHETIC inputs; fused indexer 0.1036 -> 0.0762 ms
(iter 4, 1748c11; KF-measured iter 10 8dcc5a1 = 1.40x, not run e2e), all exact on the driver inputs.

## Why

- **DCP indexer regression, found on REAL inputs** (`results/dcp_indexer_realdata.kdf1.txt`; inputs captured from a live
  DCP decode, kdf1): in 4/24 calls the KF top-k selector's candidate count exceeded its 8,192 cap and rows took a slow
  exact fallback: 1.8-2.5 ms vs 0.5 ms production. Synthetic driver data never reached the cap. Also NOT exact on real
  data: 1-3 rows per call differ from production in 11/24 calls (cause not established).
- **Fused indexer gain invisible**: ~0.027 ms per call x 21 indexer layers = ~0.57 ms per ~40 ms decode iteration
  (~1.4 %, ~0.16 ms TPOT) - below the ~3 % run/node spread. The whole fused path is ~2 ms of the iteration.
- **Why rows <= 48 never reached the KF indexer** (question of historical-e2e-session; code read): `Indexer.forward_cuda`
  serves DECODE / TARGET_VERIFY with 1 <= rows <= 48 through `Gfx950FusedIndexer` (gfx950_fused_indexer.py
  MAX_ROWS = 48, 4 own HIP kernels) and returns before `_get_topk_paged`. That path is what `tasks/fused_indexer` targets.

## Fused indexer: kept as a valid result (user, 10-25)
The fused-indexer optimization is a real, exact kernel win (1.36x in the v0522 image, KF best 1.40x 8dcc5a1) on the
path that serves every decode / TARGET_VERIFY call with <= 48 rows on gfx950. It is worth keeping and carrying into
future overlays (use 8dcc5a1 after an exactness re-check). Its e2e effect at DPA C56 is below run noise; that limits
what this setup can show, not the kernel. Next candidate to integrate: `candidates/fused_indexer.best.iter10.8dcc5a1.py`.

## Status at pack-up
Phase paused (user, 10-25 ~05Z): DCP work stopped; the post-KF decode profile was dropped on purpose (no suitable
machine and no validated method, so its result would not be trustworthy). All machines returned to the pool manager:
our containers on compute-node-07 and compute-node-02 were stopped and removed, and historical-container on compute-node-11 was stopped. KF campaigns ended: dcp_sparse 1.691x (cabd158), dcp_indexer 2.389x (09f754e) - both stopped by 3
KF orchestration infrastructure errors; fused_indexer 1.40x (8dcc5a1), cut by the compute-cluster job end.

## Navigate
- `notes.md` - integration method additions, traps, open questions. `REPRODUCE.md` - commands. `environment.md`.
- `overlay/` - current overlay (`MD5SUMS`): adds DCP hooks (`INFERA_KERNEL_DCP_SPARSE/INDEXER`), fused indexer hook
  (`INFERA_KERNEL_FUSED_INDEXER`), DCP input capture (`INFERA_KERNEL_DUMP_DCP_IDX`).
- `tasks/{dcp_sparse,dcp_indexer,fused_indexer}/` - KF task definitions (driver, program, campaign.env, pristine kernel).
- `candidates/` - best kernels. `results/campaigns/<task>/` - KF iteration history + git log + forge_result.json.
- `cluster/` - harness copy, env files, dryruns, per-round status/audit/census/AgentX JSON, kernel-level logs
  (`dcpk.log`, `fik.log`, `dcpreal.log`). Real-input dumps (1.6 GB) stay at `REDACTED_LOCAL_PATH`.
