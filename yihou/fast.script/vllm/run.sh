#!/usr/bin/env bash
set -euo pipefail

ENGINE=vllm
DEFAULT_IMAGE=
DEFAULT_GPUS=0,1
DEFAULT_PORT=6310

engine_configure() {
    local tp=${TP_SIZE:-1} dp=${DP_SIZE:-2}
    local accept_length=${VLLM_DSPARK_ACCEPT_LENGTH:-3.51}
    local value option

    if [[ ! $tp =~ ^[1-9][0-9]?$ || ! $dp =~ ^[1-9][0-9]?$ ]] || (( tp * dp != GPU_COUNT )); then
        printf 'vllm: TP_SIZE × DP_SIZE must equal the number of GPUs (%s)\n' "$GPU_COUNT" >&2
        return 2
    fi
    if [[ ! $accept_length =~ ^([1-5](\.[0-9]+)?|6(\.0+)?)$ ]]; then
        printf 'vllm: VLLM_DSPARK_ACCEPT_LENGTH must be in [1, 6]\n' >&2
        return 2
    fi
    for value in "${ENGINE_EXTRA_ARGS[@]}"; do
        option=${value%%=*}
        case ${option//_/-} in
            --kv-transfer-config|--kv-transfer-config.*|--kv-offloading-size|\
            --enable-prefix-caching|--disable-hybrid-kv-cache-manager|\
            --device-ids|--distributed-executor-backend|--config|\
            --tensor-parallel-size|-tp|--data-parallel-size|-dp|\
            --pipeline-parallel-size|-pp|--data-parallel-size-local|-dpl|\
            --data-parallel-rank|-dpn|--data-parallel-start-rank|-dpr|\
            --data-parallel-address|-dpa|--data-parallel-*-lb|\
            --model|--served-model-name|--max-model-len|\
            --host|--port|--uds|--grpc|--headless)
                printf 'vllm: option %s is managed by the decode-only runner\n' "$option" >&2
                return 2
                ;;
        esac
    done

    local kv_config='{"kv_connector":"CanonicalDecodeBenchConnector","kv_connector_module_path":"canonical_decode_bench","kv_role":"kv_both","kv_connector_extra_config":{"fill_mean":0.015,"fill_std":0.0}}'
    local spec_config
    printf -v spec_config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"synthetic","enable_adaptive_verification":false,"synthetic_acceptance_length":%s}' "$accept_length"

    ENGINE_ENV=(
        PYTHONPATH=/suite/vllm
        PYTHONHASHSEED=0
        PYTHONFAULTHANDLER=1
        OMP_NUM_THREADS=1
        AITER_TRITON_LOG_LEVEL=ERROR
        VLLM_ROCM_USE_AITER=1
        VLLM_ROCM_USE_AITER_MOE=1
        VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=NONE
        VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=True
        VLLM_USE_BREAKABLE_CUDAGRAPH=1
        VLLM_USE_RUST_FRONTEND=0
        "VLLM_ENGINE_READY_TIMEOUT_S=$STARTUP_TIMEOUT"
    )
    ENGINE_CMD=(
        vllm serve "$MODEL_IN_CONTAINER"
        --served-model-name "$SERVED_MODEL"
        --host 127.0.0.1 --port "$PORT" --api-server-count 1
        --trust-remote-code --language-model-only --seed 1234
        --tensor-parallel-size "$tp" --data-parallel-size "$dp"
        --moe-backend aiter --gpu-memory-utilization 0.95
        --attention-config '{"indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}'
        --block-size 128 --kv-cache-dtype fp8_e4m3
        --max-model-len "$CONTEXT_LENGTH" --max-num-batched-tokens 16384
        --engram-config '{"cpu_offload":true}'
        --max-cudagraph-capture-size 1728
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
        --no-enable-prefix-caching --kv-transfer-config "$kv_config"
        --speculative-config "$spec_config" --enable-logging-iteration-details
        --enable-prompt-tokens-details --disable-uvicorn-access-log
        "${ENGINE_EXTRA_ARGS[@]}"
    )
}

source "$(dirname -- "${BASH_SOURCE[0]}")/../common/run.sh" "$@"
