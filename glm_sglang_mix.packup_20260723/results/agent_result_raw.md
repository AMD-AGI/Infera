# GLM-5.1-FP8 on SGLang — single-node PD-mix (chi2866, card4-7)

Date: 2026-07-23. Node: chi2866, 8x MI355X (gfx950). Used ONLY card4-7 (HIP_VISIBLE_DEVICES=4,5,6,7).
Foreign containers (titan/zirui/primus-*) on card0-3 left untouched.

## Result: SUCCESS — came up + temp=0 factual probes ALL PASS.

## Image
lmsysorg/sglang:v0.5.14-rocm720-mi35x  (sglang 0.5.14). This image DOES implement
GlmMoeDsaForCausalLM (python/sglang/srt/models/glm4_moe.py:1477, subclass of
DeepseekV2ForCausalLM -> uses the DeepSeek MLA+DSA path). No infera sglang image was
present on the node; this lmsys image is the correct/newest one.

## Container
docker run -d --name glm_sglang_c_hf --privileged --ipc host --shm-size 64g \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  -v /mnt/vast:/mnt/vast --entrypoint "" \
  lmsysorg/sglang:v0.5.14-rocm720-mi35x sleep infinity
(single-node mix -> no RDMA/ionic/MoRIIO/Mooncake needed.)

## Final working launch command (inside container)
export HIP_VISIBLE_DEVICES=4,5,6,7
export ROCM_VISIBLE_DEVICES=4,5,6,7
export SGLANG_USE_AITER=1
python3 -m sglang.launch_server \
  --model-path /mnt/vast/xiaobo/models/GLM-5.1-FP8 \
  --tp-size 4 --trust-remote-code \
  --host 0.0.0.0 --port 30000 \
  --mem-fraction-static 0.85 \
  --reasoning-parser glm45

## Auto-config sglang chose for GlmMoeDsa (do NOT override — this is why the run is minimal)
attention_backend=dsa ; page_size=64 (aiter preshuffle paged-MQA available) ;
kv_cache_dtype=bfloat16 ; dsa_prefill_backend=tilelang ; dsa_decode_backend=tilelang ;
dsa_topk_backend=sgl-kernel ; decode CUDA-graph=full (52 batch sizes), prefill CG=disabled.
max_total_num_tokens=651712, context_len=202752.

## Flags changed vs the reference recipe (sglang_single_r4_20260707_080726, which was DSv4)
- DROPPED --attention-backend dsv4 (that was DSv4-model specific). GlmMoeDsa auto-selects
  `dsa`; forcing dsv4/page-size 256 would fight the DSA auto-config.
- DROPPED --page-size 256 / --swa-full-tokens-ratio / --disable-radix-cache /
  --disable-shared-experts-fusion / --cuda-graph-max-bs 128 and the big R4 SGLANG_OPT_*
  env block (perf-tuning for DSv4 throughput; not needed for a correctness bring-up).
- ADDED --reasoning-parser glm45 (GLM reasoning) ; kept SGLANG_USE_AITER=1 ; tp 8->4.

## Bring-up timing
~15 min total: weight load (142 fp8 shards) -> tokenizer/tilelang eager JIT + aiter GEMM
tuning (silent, CPU-side, ~8-10 min, looked like a stall but py-spy showed forward
progress) -> CUDA-graph capture 52/52 (~3.5 min) -> ready.

## temp=0 factual probes (via /v1/chat/completions, temp=0)  [VERBATIM]
- "The capital of France is" -> CONTENT: 'Paris'                          PASS
- "The capital of China is"  -> CONTENT: 'The capital of China is Beijing.'  PASS
- "2+2="                     -> CONTENT: '2 + 2 = 4'                       PASS
VERDICT: ALL PASS (coherent, correct facts; glm45 parser split reasoning_content cleanly).

## Teardown
docker rm -f glm_sglang_c_hf ; confirm card4-7 VRAM back to ~283MB. Foreign cards/containers untouched.
