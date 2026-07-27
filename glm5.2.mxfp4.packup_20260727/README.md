# GLM-5.2-MXFP4 on sglang (ROCm / MI355X) — reproduction kits

**Ran:** 2026-07-27
**Author:** ndl (yihou workspace)
**Engine:** sglang 0.5.15.post1 via `rocm/infera:sglang-v0.1.0-rc6`

Five self-contained experiment packups bringing up **GLM-5.2-MXFP4** on the sglang engine
across MI355X (gfx950) nodes: single-node, PD disaggregation (mori / mooncake), and MTP
speculative decoding (single-node + PD). Each subfolder is an independent reproduction kit
(README + REPRODUCE + scripts + results + notes + logs).

## Mission

From `mission.glm5.2.md`: run GLM-5.2-MXFP4 on sglang — (1) single-node mix, (2) PD mooncake,
(3) PD mori — each **correct (coherent) output + pass conc=64**. Plus follow-ups the user added
live: single-node MTP, and PD-disaggregated MTP.

## Results at a glance

| # | Experiment | Transport | MTP | Correctness | conc=64 | Throughput | Verdict |
|---|-----------|-----------|-----|-------------|---------|-----------|---------|
| 01 | single-node mix | colocated | no | 4/4 | 256/256 | 4621 tok/s | ✅ PASS |
| 02 | PD mori | MoRI RDMA | no | 4/4 | 256/256 | 5167 tok/s | ✅ PASS |
| 03 | PD mooncake | mooncake TCP | no | 4/4 | **50/256** | 886 tok/s | ❌ conc FAIL (TCP-bound; RDMA = driver dead-end) |
| 04 | single-node MTP | colocated | EAGLE 5-draft | 4/4 | single-stream | 219 tok/s (~2.7× vs no-MTP) | ✅ PASS |
| 05 | PD MTP | MoRI RDMA | EAGLE 3-draft (decode) | 4/4 | 256/256 | **7444 tok/s** (+44% vs 02) | ✅ PASS |

Machine-readable: `results_summary.csv`.

## The two GLM-5.2-on-rc6 fixes everything shares

Every experiment needs the **base DSA-ROCm recipe** (else the DSA topk tries a CUDA-only kernel
that won't build on gfx950): env `SGLANG_OPT_USE_TILELANG_INDEXER=1 SGLANG_OPT_USE_TOPK_V2=0
SGLANG_OPT_USE_JIT_NORM=0 SGLANG_USE_AITER=1 SGLANG_ROCM_FUSED_DECODE_MLA=0` + flags
`--nsa-prefill-backend tilelang --nsa-decode-backend tilelang --kv-cache-dtype fp8_e4m3`.

The **MTP experiments (04, 05)** additionally need two rc6-specific fixes (both discovered here):
1. a **1-line patch** to `deepseek_nextn.py` (quark submodule-exclude match for `eh_proj`);
2. env **`SGLANG_DSA_ENABLE_MTP_PRECOMPUTE_METADATA=0`** (skip a gfx950-incompatible CUDA JIT kernel).
See `04_single_node_mtp/patches/` + `notes.md`.

## Folder map

- `environment.md` — shared HW/SW/model/secrets for all 5 (read this first).
- `results_summary.csv` — all numbers in one table.
- `01_single_node_mix/` … `05_pd_mtp/` — one reproduction kit each.
- Each subfolder: `README.md` (what/result), `REPRODUCE.md` (exact steps), `scripts/`,
  `results/`, `notes.md`, `logs/` (+ `patches/` for 04/05).

## Notes on scope & honesty

- **03 (mooncake)** is kept as a **documented failure**: correct output over TCP but conc=64 fails
  (50/256). The intended RDMA path is a known driver dead-end on this stack (per jiejing's prior
  work on kernel-136); we went straight to TCP and did NOT independently re-test mooncake RDMA on
  our nodes — see `03_pd_mooncake/notes.md` for the honest boundary.
- The **reference program** we diffed against throughout is a colleague's prior GLM-5.2 work at
  `/mnt/vast/jiejing/crusoe_glm_52/` (read-only; NOT copied here). It used image v0.1.1; we used
  rc6, which is why 04/05 needed rc6-specific re-fixes.
