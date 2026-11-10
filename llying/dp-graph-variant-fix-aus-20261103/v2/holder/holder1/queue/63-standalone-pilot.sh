#!/usr/bin/env bash
# REPORT-20261110 section 6: does a non-PD server (no Mooncake, no RDMA registration) also segfault in HIPStream::stream() on an amdgpu 6.19.14 node? Production decode args minus PD, KOPT_MEGATP=0, 50-prompt probe.
# without PD: the production decode server args (TP8 / DP8 attention, EAGLE 3/1/4, KF overlay, kopt MoE row + absorb
# configs) minus the disaggregation flags, graphs up to bs 64. KOPT_MEGATP=1 first, then 0 (same overlay-kopt otherwise);
# each: wait for /health, the 200-prompt GSM8K probe all in flight (verify_suite's prompts), probe_stats, server log.
# Output: bench/out/megatp_srv/{on,off}.{server.log,probe.jsonl,stats.txt}
PKG=/perf_apps/liyingli/bench_agentx/opt-aus-20261103/pkg
K=$PKG/kit
KC=/perf_apps/liyingli/bench_agentx/kopt-20261109/configs
C=/sgl-workspace/aiter/aiter/configs/model_configs
G=/sgl-workspace/aiter/aiter/ops/triton/configs/gfx950/triton/gemm
KCJ=FUSED-FP8-BMM-ROPE-CAT-AND-CACHE-MLA-N=512-K=192.json
VCJ=BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT-N=256-K=512.json
OUT=/perf_apps/liyingli/bench_agentx/kopt-20261109/bench/out/standalone_pilot
NAME=llying-kopt-srv
PORT=27399
mkdir -p $OUT
prompts=$OUT/gsm8k_probe_prompts.jsonl
[ -s $prompts ] || gunzip -c $PKG/data/gsm8k_probe_prompts.jsonl.gz > $prompts
cleanup() { docker rm -f $NAME >/dev/null 2>&1; }
trap cleanup EXIT
trap 'cleanup; exit 143' TERM
ARGS=(--model-path /perf_apps/data/models/GLM-5.2-MXFP4 --served-model-name glm5.2-mxfp4 --trust-remote-code
      --host 127.0.0.1 --port $PORT --ep-size 1 --kv-cache-dtype fp8_e4m3 --dsa-prefill-backend triton
      --dsa-decode-backend triton --chunked-prefill-size 32768 --max-running-requests 256 --enable-fused-qk-norm-rope
      --enable-aiter-allreduce-fusion --reasoning-parser glm45 --tool-call-parser glm47 --watchdog-timeout 3600
      --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
      --tp-size 8 --dp-size 8 --enable-dp-attention --mem-fraction-static 0.9 --cuda-graph-max-bs-decode 64
      --stream-interval 30 --json-model-override-args '{"index_share_for_mtp_iteration":true}')
run() {  # tag KOPT_MEGATP
    local tag=$1 mega=$2
    echo "$(date -u +%T) $tag: start (KOPT_MEGATP=$mega)"
    docker rm -f $NAME >/dev/null 2>&1
    docker run -d --name $NAME --network host --ipc host --shm-size 64g --device /dev/kfd --device /dev/dri \
      --group-add video --group-add render --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
      --ulimit memlock=-1:-1 --ulimit nofile=65536:65536 -v /perf_apps:/perf_apps \
      -v $K/overlay-kopt:/yihou_overlay:ro \
      -v $K/aiter_csv/yihou_glm52_dpa_bf16_tuned_gemm.csv:$C/yihou_glm52_dpa_bf16_tuned_gemm.csv:ro \
      -v $K/aiter_csv/yihou_glm52_ptpc_bf16_tuned_gemm.csv:$C/yihou_glm52_ptpc_bf16_tuned_gemm.csv:ro \
      -v $KC/glm5_fp4_tuned_fmoe.csv:$C/glm5_fp4_tuned_fmoe.csv:ro \
      -v $KC/$KCJ:$G/fused_fp8_bmm_rope_cat_and_cache_mla/$KCJ:ro \
      -v $KC/$VCJ:$G/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant/$VCJ:ro \
      -e PYTHONPATH=/yihou_overlay -e YIHOU_KF_SPARSE=1 -e YIHOU_KF_INDEXER=1 -e YIHOU_KF_MOE=1 \
      -e SGLANG_OPT_USE_TOPK_V2=false -e SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=8 -e SGLANG_GLM_NEXTN_MOE_PTPC=1 \
      -e KOPT_MEGATP=$mega -e KOPT_MEGATP_COMM=bf16 -e HSA_ENABLE_IPC_MODE_LEGACY=1 \
      --entrypoint python3 infera-sglang:v0522-fdbafb60 -m sglang.launch_server "${ARGS[@]}" >/dev/null || return 1
    local ok=0
    for i in $(seq 1 180); do
        if curl -sf -m 5 http://127.0.0.1:$PORT/health >/dev/null; then ok=1; break; fi
        [ "$(docker inspect -f '{{.State.Running}}' $NAME 2>/dev/null)" = true ] || break
        sleep 20
    done
    docker logs $NAME > $OUT/$tag.server.log 2>&1
    if [ $ok != 1 ]; then
        echo "$(date -u +%T) $tag: server did not come up"; grep -a "kopt-megatp\|Error\|error\|Traceback" $OUT/$tag.server.log | tail -20
        return 1
    fi
    echo "$(date -u +%T) $tag: healthy"
    timeout 3600 python3 $PKG/scripts/gsm8k_probe.py $prompts http://127.0.0.1:$PORT glm5.2-mxfp4 $OUT/$tag.probe.jsonl \
        --limit 50 --concurrency 50 > $OUT/$tag.probe.log 2>&1
    python3 $PKG/scripts/probe_stats.py $OUT/$tag.probe.jsonl --review $OUT/$tag.review.txt > $OUT/$tag.stats.txt 2>&1
    docker logs $NAME > $OUT/$tag.server.log 2>&1
    echo "$(date -u +%T) $tag: probe $(cat $OUT/$tag.stats.txt | tail -1)"
    grep -a "kopt-megatp" $OUT/$tag.server.log | sort | uniq -c | sort -rn | head -12
    grep -a "Decode batch" $OUT/$tag.server.log | tail -3 | cut -c1-260
    docker rm -f $NAME >/dev/null 2>&1
}

run off 0
echo "$(date -u +%T) done"
