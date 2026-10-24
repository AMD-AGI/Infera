# Reproducing llying's 4+4 C80 (PP4 prefill + DPA4 decode) on 135/136, then our options one at a time (2026-10-23..24)

User order: (1) reproduce llying's 37,310, (2) add our e2e options one at a time, (3) add the KF decode kernels.

**Results (AgentX C80, 3600 s, one run each, sim acceptance 3.61 - no correctness claim; per-GPU over 8 GPUs):**

| run | change | score | qps | ISL mean | TTFT p50/p90 s | TPOT ms | KV xfer p50/p90 s | verdict |
|---|---|---|---|---|---|---|---|---|
| llying (10-22, 138->140) | his kit v6-pp4, image v0522-fdbafb60 | 37,310 | 2.553 | 115.9k | 3.75 / 10.83 | ~20.3 (ITL) | - | target |
| **llr1** | same kit + same image recipe, 135 -> 136 GPU4-7, 8 rails | **37,392** | 2.557 | 116.0k | 3.54 / 10.14 | 20.5 | 1.65 / 3.98 | reproduced (+0.2 %) |
| llr2 | + decode mem 0.92 -> 0.95 | 35,997 | **2.619** | 109.1k | 3.05 / 6.51 | 19.7 | 1.53 / 3.57 | ambiguous: qps +2.4 %, TTFT p90 -36 %, but mean ISL -6 % lowered the score; NOT a bad point (see notes) |
| **llr3** | llr1 + index_share_for_mtp_iteration true (both legs) | **37,743** | 2.574 | 116.4k | 3.68 / 9.76 | **19.8** | 1.70 / 4.14 | kept (+0.9 %, TPOT -3.4 %, same mix, no fault) |
| llr4 | llr3 + decode detokenizer 4 | 37,601 | 2.568 | 116.2k | 3.73 / 10.68 | 19.8 | 1.79 / 4.38 | neutral, not kept |
| llr5 | llr3 + KF decode kernels | running at pack-up | | | | | | |

Our own stack for comparison (same nodes, GPU0-3, our image/overlay/harness, `results/ppda3/summary.tsv`): ppda1 34,288;
ppda3 = ppda1 + router R2 36,543 (TTFT p90 22.95 -> 10.04 s); ppda2 = ppda1 + KF 40,780 (TPOT 20.0 -> 14.2 ms).

What this establishes: on our nodes llying's recipe gives his number, and the 8 % gap of our own stack (ppda1) was almost all
router R2 (decode-rank choice by demand). Our transfer patches (DCP pack fix, hashed shard) do not apply to a DPA decode
(the KV transfer stage is already equal), so step 2 on this base was: decode mem, index_share, detokenizer workers.

## Layout
- `kit/sglang_2p1d_glm5.2/` - the exact kit used: Infera `06796e13` examples/sglang_2p1d_glm5.2 + llying `config/kit-v6-pp4.diff`
  + `patches/deepseek_v2.py` built from the image + 3 yihou edits (AgentX client without `--rm`; decode hooks
  `DECODE_ENV_EXTRA` / `DECODE_MOUNT_EXTRA`, empty by default). Diffs vs llying: `notes.md`.
- `scripts/` - `llr*.site.env` (one per run), `run_llr.yihou.sh` (wait idle, remove only our exited `yihou-llr-*`
  containers, up.sh, agentx.sh, save logs, docker stop), `prep_llr.yihou.sh`, `kit_kfhook.yihou.sh`, `llsum.yihou.py`, run logs.
- `image/` - `build_ll.yihou.sh` (Infera fdbafb60 `deploy/docker/Dockerfile.sglang`, default ARGs) + build log + image ID /
  sglang version (0.5.22.dev20261018+g8dee3f6bdb).
- `results/llr/<run>/` - agentx json, runtime.env, engine-config.env, workers.json, aiperf server metrics (gz), decode /
  prefill key log lines; `results/llr/summary.txt`; `results/ppda3/` (our stack + R2).
