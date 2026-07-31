# DPA + MTP + PD reproduction/debug — working process

Target: get GLM-5.2-MXFP4 running on mooncake PD with **both** DP-attention (dp8+ep8, symmetric
both legs) **and** MTP (EAGLE spec-dec on decode leg). User reports this combo "doesn't work".
Pass = coherent probe 4/4 + conc=64 completes with 0 errors.

Nodes: prefill=chi2835 (10.2.122.78, has pd-unified img), decode=chi2867 (10.2.122.44, img streamed in).
Image: infera/engine-sglang:pd-unified (05967248b58f). Reference: exp07 (DPA-only) + exp06 (MTP-only).

## Prior facts (from packup memory)
- exp07 DPA-only: PASS conc 64..2048, pure recipe = GLM DSA + R4 DP args (dp8/ep8/gatherv/delayer).
- exp06 MTP-only: PASS conc=64, EAGLE steps=3 draft=4 reserved=256, decode gmu=0.80, needs nextn
  eh_proj 1-line patch (pd-unified line 360). No precompute-env needed on pd-unified.
- DPA+MTP NEVER run together. Suspected interaction: reserved-decode-tokens * DP-batch-shard vs
  DP KV pool; MTP draft/verify under SGLANG_DP_USE_GATHERV.

## Round 1 — MVP: merged leg, ctx 32768, MTP=1, conc=1 probe then conc=8
Hypothesis: the two recipes conflict at launch or first decode. MVP small-ctx isolates it cheaply
before any caseA full run.
Change: merged pd_leg_dpa_mtp.sh = exp07 dpa skeleton + exp06 MTP block. decode gmu=0.80.
Command: up_dpamtp.sh (MTP=1 CTX=32768).
Result: PENDING (image transfer chi2835->chi2867 in progress).

## SIDE-QUEST: aiter as DSA main backend (user: tilelang too slow)

### Round A1 — aiter DSA backend, stock CSV → CRASH
Change: single-node chi2835, `--dsa-prefill/decode-backend aiter` (else exp01 recipe), ctx 32768.
Result: 8-GPU **Memory access fault (Write to read-only page)** during warmup forward.
py-spy stack: `flydsl_hgemm → aiter_dsv3_router_gemm (rocm_linear_utils.py:14) → deepseek_v2.py:522`.
NOT a deadlock (stack advanced load→forward); crash in aiter FlyDSL GEMM.

### Root-cause analysis (source read, not guess)
- `aiter_dsv3_router_gemm` → `tgemm.mm` → `gemm_a16w16` → `get_GEMM_A16W16_config` reads
  `bf16_tuned_gemm.csv`.
- Untuned shape (e.g. M=48) → default → `torch` (SAFE).
- BUT csv has **5 pre-tuned rows with libtype=flydsl** (all N=6144, K=2048/3072, M∈{1,4,8,16}).
  Those return a FlyDSL kernel that **faults on gfx950**. That's the landmine, not the untuned path.
- No global disable env (only per-kernel FLYDSL_* knobs).

### Round A2 — scrub the 5 flydsl rows from CSV, mount, relaunch (IN PROGRESS)
Hypothesis: removing the 5 flydsl rows → those shapes fall through to safe torch/asm default → no crash.
Change (single): mount bf16_tuned_gemm.noflydsl.csv (113→108 rows, 0 flydsl) over image csv.
Files: /mnt/vast/c_huggingface/glm52_dsa_test/{bf16_tuned_gemm.orig.csv,.noflydsl.csv},
       /tmp/launch_aiter_scrub.sh. AITER_LOG_TUNED_CONFIG=1 to trace picks.
Result: PENDING (watch_aiter2.sh polling health/fault).

### Round A3/A4/A5 — flydsl scrub 无效 → 真凶是 aiter DSA kernel (DONE 2026-07-29)
A2 mount 错文件(runtime 读合并产物 /tmp/aiter_configs/)。A3 scrub glm5 model_config 仍崩(合并
glob 所有 model → 728 条 flydsl)。A4 用 env AITER_CONFIG_GEMM_BF16 绕过合并、flydsl_picks=0 → **仍崩**
→ FlyDSL 排除。A5 aiter+eager(禁 cuda-graph)→ **仍崩 @ warmup forward** → cuda-graph 排除。
**最终:aiter DSA attention kernel 本身在 gfx950/GLM-5.2 GPU fault,tilelang 是唯一正确路径。**
完整实验表 + 交接见 backend.md §已验证 + §交接。容器已清理,VRAM 回 baseline。
用户决定:重开 agent 仔细调试(profile / 试 flashmla 等其他 DSA_CHOICES)。
