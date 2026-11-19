# Baseline: InferenceX dsv41flash-fp4-mi355x-vllm-agentic-dspark (benchmarks/single_node/srt-slurm-recipes/dsv41flash/
# vllm/mi355x-fp4-mtp/agentic.yaml) base args + its TP4 c4/c8/c64 override, as one infera-registered aggregated worker.
GPUS=${GPUS:-0,1,2,3}
PORT=${PORT:-30000}
WORKER_ENV=(
    VLLM_ROCM_USE_AITER=1
    VLLM_ROCM_USE_AITER_MOE=1
    VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=NONE
    AITER_TRITON_LOG_LEVEL=ERROR
    VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=True
    VLLM_USE_BREAKABLE_CUDAGRAPH=1
    OMP_NUM_THREADS=1
    VLLM_USE_RUST_FRONTEND=${VLLM_USE_RUST_FRONTEND:-1}
    PYTHONUNBUFFERED=1
    VLLM_ENGINE_READY_TIMEOUT_S=7200
    # Stability settings from Infera dev/pd_opt/vllm_glm52_agentx (GLM-5.2 P4D4 on vLLM 0.34), not performance knobs:
    # uvicorn's 5 s keep-alive races the Rust router's 90 s idle pool (502 on a reused connection, U24); long agentic
    # prompts overrun the 30 s chat-template render timeout (U25); faulthandler attributes host-side crashes (U21).
    VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120
    VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT=600
    PYTHONFAULTHANDLER=1
)
VLLM_ARGS=(
    --model "$MODEL"
    --served-model-name "$SERVED"
    --language-model-only
    --tokenizer-mode deepseek_v41
    --tool-call-parser deepseek_v41
    --enable-auto-tool-choice
    --reasoning-parser deepseek_v41
    --moe-backend aiter
    --gpu-memory-utilization 0.9
    --attention-config '{"indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}'
    --block-size 128
    --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}'
    --max-model-len 1048576
    --monokernel
    --disable-uvicorn-access-log
    --tensor-parallel-size 4
    --engram-config '{"cpu_offload":false}'
    --max-cudagraph-capture-size 1024
    --max-num-batched-tokens 16384
)
