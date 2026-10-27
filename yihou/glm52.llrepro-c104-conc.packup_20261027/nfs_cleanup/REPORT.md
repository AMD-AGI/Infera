# /shared_nfs/yihou_glm52 disk audit (2026-10-27, analysis only, nothing deleted)

Method: one read-only `find -printf` walk per top-level dir from crsuse2-m2m-135 (nice 19, ionice idle),
03:07-03:24Z. Raw listings: `du_audit.yihou/list/<dir>.tsv` (type, size, 512B-blocks, mtime, owner, path).
Aggregations: `agg1.txt` (sizes, >1 GB files, owners), `agg2.txt` (by category), `agg3.txt` (cleanup
candidates with full path lists, protected paths and runs touched in the last 24 h excluded).
Sizes are allocated GiB. 137 dirs could not be read (root-owned `xdg/comgr` mode 700, all inside cache
dirs, mostly dcp_pd_20261021/cache/v3-0xx), so their sizes are not counted.

**Total measured: 938.9 GB** = yihou 756.2 + root 182.7. **No files owned by any other human user**
(owners seen: yihou, root). The root-owned files are in our own dirs, written by our containers
(client-home, triton, aiter-jit, uv-python, traces). Their parent dirs are root-owned 755, so user yihou
**cannot** delete them. Removing them needs root, e.g. a throwaway container (user's call).

## 1. Size by directory

| GB | files | newest mtime | dir |
|---:|---:|---|---|
| 312.0 | 556k | 10-25 | v0521dbg_20261010 |
| 174.9 |  | 10-25 | . runs (160 run dirs; ~3.2 GB each, mostly server_metrics_export.json) |
| 137.0 |  | 10-24 | . cache (per-node caches 046..268, ~12 GB each) |
| 284.2 | 957k | 10-23 | llying_dcp_20261022 |
| 261.6 |  | 10-23 | . work (S1-219 .. S3-210, 23 stage dirs; 11.4-12.9 GB each; ~11.3 GB of each is cache/agentx) |
| 22.5 |  | 10-22 | . images (1 docker tar) |
| 192.9 | 1.17M | 10-23 | dcp_pd_20261021 |
| 134.7 |  | 10-23 | . pd (cache 97.0, images 22.5, runs 14.5) |
| 45.9 |  | 10-22 | . cache (v3-045 22.4, v3-060 15.5, agentx-hf 3.4, ...) |
| 6.6 / 4.4 |  | 10-22 | . bench (traces 5.9) / mixrun |
| 62.1 | 4 | 10-08 | img274 (one 62 GB docker tar) |
| 54.4 | 46k | 10-27 | llrepro_20261023 (ACTIVE): work/runs 42.1, work/cache 11.4 |
| 22.2 | 5.6k | 10-13 | v0521cap_20261012 (images: 21.9 GB tar.zst) |
| 7.9 | 43k | 10-24 | agentx44_20261009 (cache/102 6.4) |
| 1.5 / 1.3 | | 10-24 / 09-18 | v0521cap2_20261013 / agentx_c72 |
| < 0.3 each | | | idxshare_dbg, kvcap, g4tune, setupws_test, kf_e2e, dcpx |

## 2. Big files (> 1 GB): 228 files, 672.8 GB

| what | count x size | total GB | where |
|---|---|---:|---|
| aiperf dataset mmap `dataset.dat` (two datasets, hash 78a713.. and f2d4b5..; copies of the same cache) | 46 x 6.22 | 286 | every `cache/<id>/client-home/.cache/aiperf/dataset_mmap/`, `llying_dcp/work/S*/cache/agentx/dataset_mmap/`, `llrepro/work/cache/agentx/` (active), `llrepro/work/runs/llr21-c80/agentx/tmp/aiperf_mmap_*` |
| docker image tars | 4 | 129 | img274/v0519-yihou-0917-nextnfix-hicache.yihou.tar (62.1, 10-08); llying_dcp/images/infera-sglang_v0522-d395c3b3-1020-mc4340.tar (22.5, 10-22, has .sha256); dcp_pd/pd/images/infera-sglang_v0522-1020-yihou-dcp-mc4340.tar (22.5, 10-21); v0521cap/images/infera-sglang.v0521-yihou-1010-mc4340.9d755dd6.yihou.tar.zst (21.9, 10-12) |
| aiperf `server_metrics_export.json` | ~130 x 1.4-3.1 | ~190 | `*/aiperf_artifacts/` in v0521dbg/runs, llrepro/work/runs, dcp_pd, llying |
| HF hub blob 3d0b21d7.. (dataset `semianalysisai/cc-traces-weka-062126`) | 46 x 1.72 | 79 | every `.../agentx/hf/hub/blobs/3d/`; a matching ~1.7 GB `hf/datasets` arrow copy sits next to each |
| KV dump `kvpool.L10.pt` (root) | 1 x 1.16 | 1.2 | v0521dbg/runs/kdf1/dcpdump |

No core dumps or model weight copies (.safetensors) were found.

## 3. Classification (by category; from agg2/agg3)

| category | GB | class | notes |
|---|---:|---|---|
| aiperf dataset_mmap copies | 282.2 (yihou 150.5, root 131.7) | (a) regenerable | aiperf rebuilds it at run start. Active copy `llrepro/work/cache/agentx` excluded |
| other content of stale cache dirs (HF hub + datasets arrow, aiperf uv-cache + venv, aiter JIT per image id, triton, uv-python, comgr) | 259.6 (yihou 221.7, root 37.9) | (a) regenerable | HF needs a re-download; aiter JIT needs a recompile on the next cold start |
| server_metrics_export.json | 194.3 | (b) raw data, gzip it | summaries already exist next to it (profile_export_aiperf.json/csv, server_metrics_export.csv). gzip -1 on a 256 MB prefix: 268 MB -> 12.1 MB (22x) |
| profile_export_aiperf_timeslices.{csv,json} | 28.0 | (b) gzip | csv prefix 134 MB -> 11.7 MB (11x) |
| profile_export.jsonl | 6.4 | (b) gzip | ratio not measured |
| logs >= 50 MB | 1.3 (7.9 for all logs) | (b) gzip | |
| docker image tars | 129.0 | (c) user's call | see notes below |
| torch traces (.trace.json.gz, already compressed) | 7.1 (root 6.6) | (c) | dcp_pd/bench/traces (letrit 2.8, ...), dcp_pd/mixrun/traces, llrepro/trace_llr7r.yihou (0.5; keep, the REPORT is built on it) |
| KV dump .pt | 1.5 (root) | (c) | v0521dbg/runs/kdf1/dcpdump |
| everything else (run results, src/overlays, harness, patches, env, scripts, ledgers, pack-ups) | 13.7 | (b) keep | |
| protected active paths | 15.2 | KEEP | see below |

Image tar facts (for the user's call):
- `infera-sglang:v0522-1020-yihou-dcp-mc4340` (dcp_pd tar) and `v0521-yihou-1010-mc4340` (v0521cap tar.zst) are both loaded
  on 135 and on 136 (docker images). The tars are still the only copy usable on a fresh node, unless a registry has them (not checked).
- The llying `v0522-d395c3b3` tar and the 62 GB v0519 tar (mode 600, uncompressed) are not loaded on 135/136.
- The two v0522 tars differ in size (24,163,640,832 vs 24,162,576,896 B), so they are different images.
- An earlier cleanup (10-14, `v0521cap2_20261013/cleanup_exec/`) listed img274, v0521cap/images,
  v0521dbg/cache/{102,136}, agentx44/cache/102 and runs pf88/pf90/pf92/pf88b/pc64/wts48 as PROTECTED. Reconfirm those with the user.
- zstd ratio on these tars: not measured.

Active / referenced (never delete):
- llrepro_20261023: kit/, *.site.env, work/cache/agentx, work/cache/InferenceX, prep_next.yihou/aiter_retune/{merged,ptpc},
  mxdraft.yihou/patched, plus runs touched in the last 24 h (llr24..llr36-c96, excluded from all candidate totals).
- kf_e2e_20261023/kf/e2e/overlay.
- Referenced by llrepro scripts: v0521cap2_20261013/hist.yihou.py; dcp_pd_20261021/cache/InferenceX
  (prep_llr.yihou.sh copies it when the local copy is missing); dcp_pd_20261021/src/* (sglang overlays, incl. 93034eba91 from CLAUDE.md);
  dcp_pd_20261021/pd/harness; v0521dbg_20261010/harness; agentx44_20261009/cache/InferenceX
  (v0521dbg/cache/InferenceX is a symlink to it).
- `llrepro_20261023/xferlat.yihou/extract.yihou.py` streams `server_metrics_export.json`. After a gzip it needs a gzip.open/zcat path.
- At audit time the only running container was yihou-llr-etcd, with no /shared_nfs mounts. Recheck mounts right before any deletion.

## 4. Proposed cleanup (safest first)

Exact per-path lists with sizes are in `agg3.txt` (sections A1/A2/B*/C*).

| # | action | est. freed GB | deletable as yihou? |
|---|---|---:|---|
| 1 | Delete yihou-owned aiperf `dataset_mmap/<hash>/` and `aiperf_mmap_*/` copies (all except llrepro/work/cache/agentx): the 23 `llying_dcp_20261022/work/S*/cache/agentx/dataset_mmap/78a713..` + `llrepro_20261023/work/runs/llr21-c80/agentx/tmp/aiperf_mmap_5745e66a4c25` | 150.5 | yes |
| 2 | Delete yihou-owned regenerable content of stale cache dirs: `llying_dcp_20261022/work/S*/cache/agentx/{hf,aiperf}` (23 x ~5.0), `v0521dbg_20261010/cache/<node>/{agentx/hf,agentx/aiperf,aiter}` (12 nodes, ~5.9 each), `dcp_pd_20261021/pd/cache/<node>/` (8 x ~5.8), `dcp_pd_20261021/cache/{v3-045,v3-060,v3-075,046,054,151,268b,agentx-hf}` (keep `cache/InferenceX`), `agentx44_20261009/cache/102*` (keep `cache/InferenceX`) | up to 221.7 | yes (yihou part) |
| 3 | gzip (or zstd) in place: `server_metrics_export.json` (194.3 -> ~9), `profile_export_aiperf_timeslices.*` (28.0 -> ~2.5), `profile_export.jsonl` (6.4), logs >= 50 MB (1.3). Skip runs touched in the last 24 h. Update xferlat/extract.yihou.py first | ~210 (ratio extrapolated from 2 samples) | yes (3.6 GB of it is root-owned) |
| 4 | Root-owned caches: dataset_mmap under `*/cache/<id>/client-home/` (131.7) + triton/aiter-jit/uv-python/comgr (37.9) | 169.6 | no, needs root (e.g. a container), user's call |
| 5 | Image tars: delete or recompress the v0519 tar (62.1) and the llying v0522-d395c3b3 tar (22.5). Keep the dcp and v0521 tars unless a registry copy exists | 22.5-129 | yes, user's call |
| 6 | Traces / dumps: dcp_pd bench+mixrun traces (6.6, root), kdf1 dcpdump (1.5, root) | 8.1 | no (root), user's call |

Steps 1-3 (yihou-owned, regenerable or compressible) free about 580 GB. Step 4 frees another ~170 GB but needs root.
Keep: all patches, site/env files, scripts, ledgers, summaries, pack-up material, src/overlay/harness trees, run result
summaries (13.7 GB) and the protected paths (15.2 GB).

Others' files: none. The only non-yihou owner is root (182.7 GB, all inside our dirs, container-written).
