# InferenceX PR sweep (321 PRs, all states) — deltas vs earlier reports (2026-10-21)

Raw per-PR reads: /var/tmp/yihou/research_glm52_dcp/sweep2/prN.md

## SGLang with DCP > 1
- Only #4064 (open; sweep 38612691278 failed: `lse` UnboundLocalError, forward_mla_rocm.py:969, image v0.5.20 predates #43735/#44322).
- #2275 (closed): GLM-5.2 FP8 H200 Dynamo+SGLang PD, PCP8 prefill + decode TP8/DCP4|DCP8 (flashmla_kv) recipes on an explore
  branch; both failed at c2 ("TP8/DCP4 and TP8/DCP8 decode failed in SGLang attn_mqa_for_dcp_decode"); shipped without DCP.
- No InferenceX run has produced an SGLang + DCP + GLM-5.2 data point.

## Other-engine PD + DCP references
- #3582 (merged) Kimi-K3 MI355X vLLM PD, DCP8 on P and D: 1P/1D DCP8 c48 10,882 tok/s/GPU (16 GPUs). Env: keep DCP collectives
  on graph-compatible RCCL (`VLLM_USE_DIRECT_DCP_*=0`), `VLLM_ROCM_AITER_MLA_DCP_VERIFY=asm`.
- #4311 (open) Kimi-K3 MI355X ATOM PD DCP8 P and D.
- #4045 (open) GLM-5.2 MI355X ATOM PD: prefill PP4xTP1 (layers 20/20/20/18) + decode TP4/DCP4 (1P1D same node), Mooncake;
  run 39028510364 success: 1P1D c56 31,067 / c72 38,487 / c96 42,884 tok/s/GPU (PR body).

## PP (new beyond #4300, Qwen #2812/#3231/#3840/#4333, Kimi PP2 list)
- #4162 (closed) Qwen3.5 GB300 3P PP4 x 1D DEP8 c704: 138,945 tok/s/GPU.
- #4289 (open) Muse-Spark H100 SGLang agg TP8xPP2 with spec (`SGLANG_ENABLE_PP_SPEC=1`, non-overlap scheduler).
- #2196 (closed) DSv4 H100 disagg TP8xPP2 hung (dynamo + PP + NIXL stall).

## GLM-5.2 PD on MI355X with SGLang (no DCP) reference
- #2679 (closed, no results): 1P1D TP8/TP8, MoRI, tilelang DSA, HiCache write_through_selective, MTP2.
