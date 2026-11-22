#!/usr/bin/env bash
# Same-image reference for the 1P1D example: InferenceX's top MI355X vLLM point (dsv41flash/vllm/mi355x-fp4-mtp/
# agentic.yaml override_dp2_umbp_c384, 427.5k tok/s/GPU on nightly-60bd6dd2) replayed verbatim on the base nightly
# of the Infera image: one `vllm serve` DP2 x TP1 on GPUs 0-1, UMBP standalone server (449 GB DRAM, 4 KiB pages),
# vllm-router 0.1.14 consistent_hash, golden acceptance 3.51, one formal AgentX point. Runs on the node (srun).
#   CONC (384), FAST (0), GPUS (0,1), UMBP_GB (449), RUN
set -uo pipefail
source "$(dirname "$0")/env.sh"
CONC=${CONC:-384} FAST=${FAST:-0} GPUS=${GPUS:-0,1} UMBP_GB=${UMBP_GB:-449}
RUN=${RUN:-agg-ix-dp2-umbp-c$CONC-$(date -u +%m%dT%H%M)}
R=$W/logs/$RUN
mkdir -p "$R" && chmod 777 "$R"
say() { echo "$(ts) $RUN: $*" | tee -a "$W/logs/summary.log"; }
CTR=${CTR_PREFIX}agg
UMBP_DIR=/dev/shm/${CTR_PREFIX}umbp
PORT=30000
cleanup() {
    docker logs "$CTR" > "$R/docker-agg.log" 2>&1
    docker rm -f "$CTR" "${CTR_PREFIX}agentx-c$CONC" >/dev/null 2>&1
    rm -rf "$UMBP_DIR"
}
trap cleanup EXIT
say "start on $(hostname -s) job ${SLURM_JOB_ID:-?}, amdgpu $(cat /sys/module/amdgpu/version), $BASE_TAG, agg DP2 GPUs $GPUS, UMBP ${UMBP_GB} GB, c$CONC FAST=$FAST"
pull_image "$BASE_IMAGE" > "$R/pull.log" 2>&1 || { say "image pull failed"; exit 1; }
docker rm -f "$CTR" >/dev/null 2>&1
rm -rf "$UMBP_DIR" && mkdir -p "$UMBP_DIR"
docker run -d --name "$CTR" --network host --ipc host --shm-size 64g \
    --device /dev/kfd --device /dev/dri --device /dev/infiniband $(host_gids) \
    --cap-add SYS_PTRACE --cap-add IPC_LOCK --security-opt seccomp=unconfined \
    --ulimit memlock=-1:-1 --ulimit nofile=1048576:1048576 \
    -v /perf_apps:/perf_apps -v "$UMBP_DIR:/umbp-host/umbp" -v "$NODE_CACHE:/root/.cache" -e HF_HUB_OFFLINE=1 \
    --entrypoint "" "$BASE_IMAGE" sleep infinity >/dev/null || { say "docker run failed"; exit 1; }

# vllm-router, as InferenceX's srt-slurm-recipes/configs/vllm-router.sh installs it.
docker exec "$CTR" python3 -m pip install --quiet --break-system-packages "vllm-router==0.1.14" > "$R/router-install.log" 2>&1 \
    || { say "vllm-router install failed: $(tail -2 "$R/router-install.log" | tr '\n' ' ')"; exit 1; }

# UMBP standalone server; the binary ships inside the MORI package and loads its libraries from there.
SOCK=unix:///umbp-host/umbp/standalone.grpc.sock
docker exec -d "$CTR" bash -c "P=\$(python3 -c 'import mori, os; print(os.path.dirname(mori.__file__))') && \
    UMBP_STANDALONE_ADDRESS=$SOCK UMBP_DRAM_CAPACITY=$((UMBP_GB * 1000000000)) UMBP_DRAM_USE_HUGEPAGES=0 \
    UMBP_SSD_ENABLED=0 MORI_UMBP_LOG_LEVEL=INFO LD_LIBRARY_PATH=\$P:\$LD_LIBRARY_PATH \
    exec \$P/umbp_standalone_server $SOCK > $R/umbp.log 2>&1"
t0=$SECONDS
until grep -q '\[StandaloneServer\] listening' "$R/umbp.log" 2>/dev/null; do
    ((SECONDS - t0 > 1800)) && { say "UMBP server not ready: $(tail -3 "$R/umbp.log" | tr '\n' ' ')"; exit 1; }
    docker exec "$CTR" pgrep -f umbp_standalone_server >/dev/null || { say "UMBP server died: $(tail -3 "$R/umbp.log" | tr '\n' ' ')"; exit 1; }
    sleep 10
done
say "UMBP server listening after $((SECONDS - t0)) s"

SPEC='{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"synthetic","synthetic_acceptance_length":3.51,"enable_adaptive_verification":false}'
KVT="{\"kv_connector\":\"UMBPStoreConnector\",\"kv_role\":\"kv_both\",\"kv_load_failure_policy\":\"recompute\",\"kv_connector_extra_config\":{\"mode\":\"standalone\",\"endpoint\":\"$SOCK\"}}"
cat > "$R/serve.sh" <<EOF
export HIP_VISIBLE_DEVICES=$GPUS CUDA_VISIBLE_DEVICES=$GPUS
export VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=NONE AITER_TRITON_LOG_LEVEL=ERROR
export VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=True VLLM_USE_BREAKABLE_CUDAGRAPH=1 OMP_NUM_THREADS=1 VLLM_USE_RUST_FRONTEND=1
export PYTHONUNBUFFERED=1 VLLM_ENGINE_READY_TIMEOUT_S=3600 UMBP_STANDALONE_ADDRESS=$SOCK
exec vllm serve $MODEL --host 0.0.0.0 --port $PORT \\
    --served-model-name $SERVED --language-model-only --tokenizer-mode deepseek_v41 \\
    --tool-call-parser deepseek_v41 --enable-auto-tool-choice --reasoning-parser deepseek_v41 \\
    --moe-backend aiter --gpu-memory-utilization 0.9 \\
    --attention-config '{"indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}' --block-size 128 \\
    --speculative-config '$SPEC' --max-model-len 1048576 --disable-uvicorn-access-log \\
    --tensor-parallel-size 1 --data-parallel-size 2 --engram-config '{"cpu_offload":true}' \\
    --max-cudagraph-capture-size 1024 --max-num-batched-tokens 16384 --prefill-schedule-interval 11 \\
    --kv-cache-dtype fp8_e4m3 --kv-transfer-config '$KVT'
EOF
docker exec -d "$CTR" bash -c "bash $R/serve.sh > $R/agg.log 2>&1"
t0=$SECONDS
until curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null; do
    ((SECONDS - t0 > 3600)) && { say "engine not ready in 3600 s"; exit 1; }
    if ! docker exec "$CTR" pgrep -f "vllm serve" >/dev/null; then
        say "engine died: $(grep -aE 'Error|error:|Traceback' "$R/agg.log" | tail -3 | tr '\n' ' ' | head -c 600)"; exit 1
    fi
    sleep 20
done
say "engine ready in $((SECONDS - t0)) s; $(grep -aoE 'GPU KV cache size: [0-9,]+ tokens' "$R/agg.log" | head -2 | tr '\n' ' ')$(grep -aoE 'synthetic_acceptance_length[^,}]*' "$R/agg.log" | head -1)"

docker exec -d "$CTR" bash -c "vllm-router --policy consistent_hash --prometheus-host 127.0.0.1 --prometheus-port 18000 \
    --request-timeout-secs 14400 --disable-retries --intra-node-data-parallel-size 2 --worker-startup-timeout-secs 3600 \
    --worker-urls http://127.0.0.1:$PORT --host 0.0.0.0 --port $ROUTER_PORT > $R/router.log 2>&1"
for _ in $(seq 90); do curl -sf -m 5 "http://127.0.0.1:$ROUTER_PORT/health" >/dev/null && break; sleep 2; done
curl -s -m 10 "http://127.0.0.1:$ROUTER_PORT/workers" > "$R/workers.json" 2>&1
curl -s -m 300 "http://127.0.0.1:$ROUTER_PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? Answer in one word.\"}],\"max_tokens\":512}" > "$R/smoke.json"
say "router workers: $(head -c 300 "$R/workers.json" | tr '\n' ' ') | smoke: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); c=d["choices"][0]; print(c.get("finish_reason"), repr((c["message"].get("content") or "")[:40]))' "$R/smoke.json" 2>&1 | head -c 200)"

left=$(squeue -h -j "${SLURM_JOB_ID:?}" -o %L | awk -F'[-:]' '{n=NF; s=$n+60*$(n-1)+(n>2?3600*$(n-2):0)+(n>3?86400*$(n-3):0); print int(s)}')
say "AgentX $( ((FAST)) && echo fast || echo formal) c$CONC starting ($((left / 60)) min left in the job)"
CONC=$CONC FAST=$FAST MODE=agg AGG_GPUS=2 DP_ATTENTION=true SIM_ACCEPT_LEN=3.51 KV_OFFLOADING=dram TOTAL_CPU_DRAM_GB=$UMBP_GB \
    METRICS_URLS="http://127.0.0.1:18000/metrics,http://127.0.0.1:$PORT/metrics" \
    timeout $((left - 240)) bash "$W/scripts/agentx.sh" "$R/agentx-c$CONC"
rc=$?
say "AgentX c$CONC rc=$rc: $(grep -aE 'Saved aggregated|ERROR' "$R/agentx-c$CONC/runner.log" | tail -2 | tr -s ' ' | tr '\n' ' ' | head -c 400)"
j=$(ls "$R"/agentx-c$CONC/agentx_conc*.json 2>/dev/null | head -1)
[ -n "$j" ] && say "result: $(python3 /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/scripts/ax_summary.py "$j" 2>&1 | tail -2 | tr '\n' ' ' | head -c 600)"
say "UMBP: $(grep -aiE 'hit|evict|capacity' "$R/umbp.log" | tail -2 | tr '\n' ' ' | head -c 300)"
say "end"
