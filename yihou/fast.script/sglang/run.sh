#!/usr/bin/env bash
set -euo pipefail

ENGINE=sglang
DEFAULT_IMAGE=lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261111
DEFAULT_GPUS=0,1,2,3
DEFAULT_PORT=31841

engine_configure() {
    local arg option
    local rank_batch_size=$(( (MAX_CONCURRENCY + GPU_COUNT - 1) / GPU_COUNT ))
    local max_running_requests=$(( rank_batch_size * GPU_COUNT ))
    local -a entrypoint=(python3 -m sglang.launch_server)
    local -a speculative_args=()
    local -a suite_options=(
        --config --model --model-path --served-model-name --context-length --host --port
        --disaggregation-mode --disaggregation-transfer-backend
        --disaggregation-decode-enable-radix-cache
        --tp --dp --ep --pp --tensor-parallel-size --data-parallel-size
        --expert-parallel-size --pipeline-parallel-size --nnodes --node-rank
        --base-gpu-id --gpu-id-step --dist-init-addr --load-balance-method
        --speculative-algorithm --speculative-dspark-block-size
    )

    for arg in "${ENGINE_EXTRA_ARGS[@]}"; do
        [[ "$arg" == --* ]] || continue
        for option in "${suite_options[@]}"; do
            # argparse also accepts unambiguous option prefixes.
            if [[ "$option" == "${arg%%=*}"* ]]; then
                printf 'This option conflicts with the SGLang fake-decode endpoint (%s).\n' "$arg" >&2
                return 2
            fi
        done
    done

    case "${SGLANG_DSPARK:-0}" in
        0) ;;
        1)
            entrypoint=(python3 /suite/sglang/fake_dspark.py)
            speculative_args=(--speculative-algorithm DSPARK --speculative-dspark-block-size 5)
            ;;
        *)
            printf 'SGLANG_DSPARK must be 0 or 1.\n' >&2
            return 2
            ;;
    esac

    ENGINE_CMD=(
        "${entrypoint[@]}"
        --model-path "$MODEL_IN_CONTAINER"
        --served-model-name "$SERVED_MODEL"
        --trust-remote-code
        --host 127.0.0.1
        --port "$PORT"
        --context-length "$CONTEXT_LENGTH"
        --disaggregation-mode decode
        --disaggregation-transfer-backend fake
        --tensor-parallel-size "$GPU_COUNT"
        --data-parallel-size "$GPU_COUNT"
        --expert-parallel-size "$GPU_COUNT"
        --enable-dp-attention
        --enable-dp-lm-head
        --enable-dp-attention-local-control-broadcast
        --load-balance-method follow_bootstrap_room
        --mem-fraction-static 0.85
        --swa-prefix-tails 0
        --max-running-requests "$max_running_requests"
        --cuda-graph-max-bs-decode "$rank_batch_size"
        --cuda-graph-backend-prefill disabled
        --enforce-shared-experts-fusion
        --fp8-gemm-backend aiter
        --reasoning-parser auto
        --tool-call-parser auto
        --tokenizer-backend fastokens
        --tokenizer-worker-num 2
        --image-processor-backend pil
        --watchdog-timeout 3600
        --enable-metrics
        "${speculative_args[@]}"
        "${ENGINE_EXTRA_ARGS[@]}"
    )
    ENGINE_ENV=(
        SGLANG_TIMEOUT_KEEP_ALIVE=900
        SGLANG_DEFAULT_THINKING=1
        SGLANG_DSV41_REASONING_EFFORT=high
        SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
        SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=shared
        SGLANG_USE_AITER=1
        SGLANG_USE_ROCM700A=0
        TRITON_HIP_USE_ASYNC_COPY=0
        ROCM_QUICK_REDUCE_QUANTIZATION=INT8
        SGLANG_OPT_HIP_FLYDSL_MXFP8=1
        SGLANG_HIP_FP4_INDEXER_ROWGROUP=1
        TORCH_BLAS_PREFER_HIPBLASLT=1
        HSA_NO_SCRATCH_RECLAIM=0
        GPU_MAX_HW_QUEUES=5
        SGLANG_SHARED_EXPERT_TP1=1
        SGLANG_DP_SHARED_EXPERT_LOCAL=1
        SGLANG_DP_USE_GATHERV=1
        SGLANG_DP_USE_REDUCE_SCATTER=1
        SGLANG_AITER_CAPTURE_COPY_IN=1
        SGLANG_HIP_DSV41_INDEX_K_PAGE8=1
        SGLANG_CUDA_COREDUMP_BEFORE_CRASH=0
        SGLANG_PYSPY_DUMP_BEFORE_CRASH=0
    )
    if [[ -n ${HSA_ENABLE_SDMA:-} ]]; then
        ENGINE_ENV+=("HSA_ENABLE_SDMA=$HSA_ENABLE_SDMA")
    fi
    if [[ ${SGLANG_DSPARK:-0} == 1 ]]; then
        ENGINE_ENV+=(
            SGLANG_RAGGED_VERIFY_MODE=static
            "SGLANG_SIMULATE_ACC_LEN=${SGLANG_SIMULATE_ACC_LEN:-3.51}"
            SGLANG_SIMULATE_ACC_METHOD=match-expected
            SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
        )
    fi
}

# shellcheck source=../common/run.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/../common/run.sh" "$@"
