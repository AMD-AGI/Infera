# GLM-5.2-MXFP4 on sglang (ROCm / MI355X) — reproduction kits

**Ran:** 2026-07-27
**Author:** ndl (yihou workspace)
**Engine:** sglang 0.5.15.post1 via `rocm/infera:sglang-v0.1.0-rc6`

Seven self-contained experiment packups bringing up **GLM-5.2-MXFP4** on the sglang engine
across MI355X (gfx950) nodes: single-node, PD disaggregation (mori / mooncake), MTP speculative
decoding (single-node + PD), and a DP-attention high-concurrency sweep (conc 64→2048). Each
subfolder is an independent reproduction kit (README + REPRODUCE + scripts + results + notes + logs).

## Mission

From `mission.glm5.2.md`: run GLM-5.2-MXFP4 on sglang — (1) single-node mix, (2) PD mooncake,
(3) PD mori — each **correct (coherent) output + pass conc=64**. Plus follow-ups the user added
live: single-node MTP, and PD-disaggregated MTP.

## Results at a glance

| # | Experiment | Transport | MTP | Image | Correctness | conc=64 | Throughput | Verdict |
|---|-----------|-----------|-----|-------|-------------|---------|-----------|---------|
| 01 | single-node mix | colocated | no | rc6 | 4/4 | 256/256 | 4621 tok/s | ✅ PASS |
| 02 | PD mori | MoRI RDMA | no | rc6 | 4/4 | 256/256 | 5167 tok/s | ✅ PASS |
| 03 | PD mooncake | mooncake **RDMA** | no | pd-unified | 4/4 | 256/256 | 5147 tok/s | ✅ PASS |
| 04 | single-node MTP | colocated | EAGLE 5-draft | rc6 | 4/4 | single-stream | 219 tok/s (~2.7× vs no-MTP) | ✅ PASS |
| 05 | PD MTP | MoRI RDMA | EAGLE 3-draft (decode) | rc6 | 4/4 | 256/256 | **7444 tok/s** (+44% vs 02) | ✅ PASS |
| 06 | PD mooncake MTP | mooncake **RDMA** | EAGLE 3-draft (decode) | pd-unified | 4/4 | 256/256 | 5302 tok/s (+3% vs 03) | ✅ PASS |
| 07 | PD mooncake **DPA** sweep | mooncake **RDMA** | no (DP-attn) | pd-unified | 4/4 | 64→2048 all full | 12855 out tok/s peak @1024 | ✅ PASS |

Machine-readable: `results_summary.csv`. **All 7 experiments PASS.**

### 07 concurrency sweep (DP-attention, 1k/1k, mooncake RDMA)

| conc | completed | out tok/s | out/GPU | median TTFT | median TPOT |
|-----:|:---------:|----------:|--------:|------------:|------------:|
| 64   | 256/256   | 2188  | 274  | 1396 ms | 27.3 ms |
| 128  | 512/512   | 3485  | 436  | 1660 ms | 32.6 ms |
| 256  | 1024/1024 | 6293  | 787  | 2402 ms | 36.7 ms |
| 512  | 2048/2048 | 10164 | 1271 | 2727 ms | 45.1 ms |
| 1024 | 4096/4096 | **12855** | **1607** | 3532 ms | 65.7 ms |
| 2048 | 4096/4096 | 11970 | 1496 | 17546 ms | 129.4 ms |

DP-attention (`--dp8 --enable-dp-attention --ep-size 8`, symmetric on both PD legs) gives the ~6×
throughput scaling from conc=64→1024 that pure TP8 (03) cannot. Peaks at conc=1024; conc=2048 is past
the single-decode-leg saturation knee (longer TTFT) but still completes all requests. Zero retracts /
KVTransferError / OOM. See `07_pd_mooncake_dpa_sweep/`.

Two images are used: **rc6** (`rocm/infera:sglang-v0.1.0-rc6`) for 01/02/04/05, and **pd-unified**
(`infera/engine-sglang:pd-unified`, Infera PR #19) for the mooncake experiments 03/06 — PR #19's
rebuild is what makes mooncake cross-node **RDMA** work (see 03/notes.md).

## The two GLM-5.2-on-rc6 fixes everything shares

Every experiment needs the **base DSA-ROCm recipe** (else the DSA topk tries a CUDA-only kernel
that won't build on gfx950): env `SGLANG_OPT_USE_TILELANG_INDEXER=1 SGLANG_OPT_USE_TOPK_V2=0
SGLANG_OPT_USE_JIT_NORM=0 SGLANG_USE_AITER=1 SGLANG_ROCM_FUSED_DECODE_MLA=0` + flags
`--nsa-prefill-backend tilelang --nsa-decode-backend tilelang --kv-cache-dtype fp8_e4m3`.

The **MTP experiments (04, 05)** additionally need two rc6-specific fixes (both discovered here):
1. a **1-line patch** to `deepseek_nextn.py` (quark submodule-exclude match for `eh_proj`);
2. env **`SGLANG_DSA_ENABLE_MTP_PRECOMPUTE_METADATA=0`** (skip a gfx950-incompatible CUDA JIT kernel).
See `04_single_node_mtp/patches/` + `notes.md`.

The **DPA sweep (07)** adds the DSv4 R4 high-concurrency recipe on top of the GLM DSA recipe:
`--dp8 --enable-dp-attention --ep-size 8 + SGLANG_DP_USE_GATHERV=1 + --enable-prefill-delayer`
(symmetric on both PD legs), plus capacity tuning (`--context-length 32768`,
`--chunked-prefill-size 65536`, `--max-running-requests 2048`). See `07_pd_mooncake_dpa_sweep/notes.md`.

## Folder map

- `environment.md` — shared HW/SW/model/secrets for all 7 (read this first).
- `results_summary.csv` — all numbers in one table.
- `01_single_node_mix/` … `07_pd_mooncake_dpa_sweep/` — one reproduction kit each.
- Each subfolder: `README.md` (what/result), `REPRODUCE.md` (exact steps), `scripts/`,
  `results/`, `notes.md`, `logs/` (+ `patches/` for 04/05).

## Notes on scope & honesty

- **03 (mooncake) now PASSES over real RDMA** on the `pd-unified` image (PR #19). An earlier attempt
  on the rc6 image only worked over TCP and failed conc=64 (50/256) — that failed run is preserved
  under `03_pd_mooncake/tcp_fail_appendix/` for the record, with the root cause (HIP-IPC mis-select)
  and the PR #19 fix in `03_pd_mooncake/notes.md`.
- **Image distribution caveat:** `pd-unified` is a local build (not on a public registry). We moved
  it node→node by streaming `docker save | ssh <dst> docker load` (NFS `docker save` of the 78 GB
  image is very slow). See `03_pd_mooncake/REPRODUCE.md` §0.
- The **reference programs** we diffed against (read-only, NOT copied here): a colleague's prior
  GLM-5.2 work at `/mnt/vast/jiejing/crusoe_glm_52/` (image v0.1.1 — why 04/05 needed rc6-specific
  re-fixes), and the PR #19 test packup `sglang_unified_pd_test.packup_20260727` (the pd-unified
  method, validated on DSv4 — 03/06 adapt it to GLM-5.2).
