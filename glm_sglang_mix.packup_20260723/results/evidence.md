# Evidence — GLM-5.1-FP8 SGLang single-node mix

Run: 2026-07-23, chi2866 card4-7, `lmsysorg/sglang:v0.5.14-rocm720-mi35x`, TP4.

## temp=0 factual probes (via /v1/chat/completions, temperature=0) — VERBATIM

| Prompt | Reply (verbatim) | Expect | Verdict |
|--------|------------------|--------|---------|
| `The capital of France is` | `Paris` | Paris | ✅ PASS |
| `The capital of China is` | `The capital of China is Beijing.` | Beijing | ✅ PASS |
| `2+2=` | `2 + 2 = 4` | 4 | ✅ PASS |

**VERDICT: ALL PASS** — coherent, correct facts (not "is is is" garbage, not
off-topic). The `glm45` reasoning parser split `reasoning_content` cleanly.

## Final working launch command (verbatim)

```
export HIP_VISIBLE_DEVICES=4,5,6,7
export ROCM_VISIBLE_DEVICES=4,5,6,7
export SGLANG_USE_AITER=1
python3 -m sglang.launch_server \
  --model-path /mnt/vast/xiaobo/models/GLM-5.1-FP8 \
  --tp-size 4 --trust-remote-code \
  --host 0.0.0.0 --port 30000 \
  --mem-fraction-static 0.85 \
  --reasoning-parser glm45
```

## SGLang auto-config for GlmMoeDsa (what it chose, left untouched)

```
attention_backend=dsa ; page_size=64 ; kv_cache_dtype=bfloat16
dsa_prefill_backend=tilelang ; dsa_decode_backend=tilelang ; dsa_topk_backend=sgl-kernel
decode CUDA-graph=full (52 batch sizes) ; prefill CUDA-graph=disabled
max_total_num_tokens=651712 ; context_len=202752
```

## Bring-up timing

~15 min total: weight load (142 fp8 shards) → silent tilelang JIT + aiter GEMM
tuning (~8-10 min, CPU-side, looks like a stall) → CUDA-graph capture 52/52
(~3.5 min) → `/health` 200.

## Source

- Raw bring-up report: `agent_result_raw.md` (this dir).
- Full server log: `../logs/glm_sglang_mix.log.gz`.
