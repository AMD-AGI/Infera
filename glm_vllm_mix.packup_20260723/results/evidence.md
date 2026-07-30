# Evidence — GLM-5.1-FP8 vLLM single-node PD-mix

Run: 2026-07-23, chi2879 card0-3, `infera/engine-vllm:test-local` (e91a6d7d3a91), TP4.

## temp=0 factual probes (via /v1/chat/completions, temperature=0, thinking OFF) — VERBATIM

| Prompt | Reply (verbatim) | Expect | Verdict |
|--------|------------------|--------|---------|
| `The capital of France is` | `Paris` | Paris | ✅ PASS |
| `The capital of China is` | `The capital of China is Beijing.` | Beijing | ✅ PASS |
| `2+2=` | `2 + 2 = 4` | 4 | ✅ PASS |

**VERDICT: ALL PASS** — facts correct, coherent, non-empty. Probes used
`chat_template_kwargs={"enable_thinking": false}`, max_tokens 200.

## Final working launch command (verbatim, plain vllm serve — NO kv-transfer)

```
HIP_VISIBLE_DEVICES=0,1,2,3 VLLM_USE_V1=1 VLLM_ROCM_USE_AITER=1 \
AITER_BF16_FP8_MOE_BOUND=0 PYTHONHASHSEED=0 VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1 \
vllm serve /mnt/vast/xiaobo/models/GLM-5.1-FP8 --served-model-name GLM-5.1-FP8 \
  --host 0.0.0.0 --port 8000 --tensor-parallel-size 4 --trust-remote-code \
  --kv-cache-dtype fp8 --reasoning-parser glm45 --no-enable-prefix-caching \
  --gpu-memory-utilization 0.85 --max-model-len 9472 --max-num-batched-tokens 8192 \
  --distributed-executor-backend mp
```

## Bring-up

health 200, "Application startup complete". CUDA graphs captured in 181s (8.29 GiB).
No OOM at gpu-util 0.85. Ready ~6-7 min. Image e91a6d7d3a91 (with the moriio pagelen
fix) ran GLM cleanly; the fix is a no-op here (single-node mix has no KV-transfer path).

## Source
- Raw bring-up report: `agent_result_raw.md` (this dir).
- Full server log: `../logs/glm_vllm_mix.log.gz`.
