# GLM-5.1-FP8 on vLLM — single-node PD-mix (chi2879) — PASS

Date: 2026-07-23  Node: chi2879 (8x MI355X gfx950)  Cards used: 0-3 (HIP_VISIBLE_DEVICES=0,1,2,3)
Image: infera/engine-vllm:test-local (e91a6d7d3a91, contains MoRIIO page-len fix; no-op in single-node mix)

## 1. Bring-up: SUCCESS
Server came up cleanly. health=200, "Application startup complete", CUDA graphs captured in 181s (8.29 GiB),
no errors/OOM. Model load ~70s (142 shards), total ready ~6-7 min from launch. No kv-transfer-config (single-node mix needs none).

Final working command (inside container, plain `vllm serve`):
  vllm serve /mnt/vast/xiaobo/models/GLM-5.1-FP8 --served-model-name GLM-5.1-FP8 \
    --host 0.0.0.0 --port 8000 --tensor-parallel-size 4 --trust-remote-code \
    --kv-cache-dtype fp8 --reasoning-parser glm45 --no-enable-prefix-caching \
    --gpu-memory-utilization 0.85 --max-model-len 9472 --max-num-batched-tokens 8192 \
    --distributed-executor-backend mp
Env: VLLM_USE_V1=1 VLLM_ROCM_USE_AITER=1 AITER_BF16_FP8_MOE_BOUND=0 PYTHONHASHSEED=0 VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1

## 2. temp=0 correctness (urllib, /v1/chat/completions, max_tokens=200, chat_template_kwargs={enable_thinking:false})
- "The capital of France is" -> CONTENT: 'Paris'                          PASS
- "The capital of China is"  -> CONTENT: 'The capital of China is **Beijing**.'  PASS
- "2+2="                     -> CONTENT: '2 + 2 = 4'                      PASS
All facts correct, coherent, non-empty. OVERALL: PASS.

## 3. Notes / deviations
- Flags: used the known-good GLM recipe verbatim; no flag changes needed. `--reasoning-parser glm45` accepted.
  Did NOT use --moe-backend aiter (not in the plain-serve recipe; aiter already active via VLLM_ROCM_USE_AITER=1 env).
- No OOM at gpu-util 0.85 (no need to drop to 0.80). No timeouts; compile/CG window was silent-but-normal.
- `vllm serve --help` fails in-image without GPU (device inference at parse time) - cosmetic, not a blocker.
- Port 8000 host-publish collided with pre-existing glm_pd container; dropped `-p`, probed via docker exec localhost.
- Image e91a6d7d3a91 ran GLM cleanly. MoRIIO fix is a no-op here (no KV transfer path exercised).

## 4. Teardown: CLEAN
Container glm_vllm_mix_c_hf removed. Cards 0-3 back to ~298MB idle (all 8 free). No KFD PIDs. Image kept.
Pre-existing glm_pd + pd_etcd containers left untouched (they were idle/not holding GPU before I started).
