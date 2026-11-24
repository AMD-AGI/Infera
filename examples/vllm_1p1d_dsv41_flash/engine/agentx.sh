#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: one InferenceX AgentX point against the router, through InferenceX's OWN client
#       (inferencex-e2e/benchmarks/srt_agentic.sh: agentx-harness, AIPerf replay of the
#       semianalysisai/cc-traces-weka-062126 coding traces, aggregation and validation).
#         agentx.sh CONC          formal: 3600 s measured, 10 warmup requests per replay lane
#         FAST=1 agentx.sh CONC   agentx-fast (one warmup per lane, 1200 s) — debugging only
# why : the agentic workload (multi-turn sessions, 100k+ token contexts, prefix reuse) is what this
#       deployment is tuned for; a fixed-length sweep says little about it. Using InferenceX's client
#       keeps the numbers comparable with its published MI355X / B300 points.
# how : decode must run golden acceptance first (golden.sh golden): AgentX scores throughput at the
#       committed acceptance length, not at what the draft happens to reach on this trace.
#       Needs INFERENCEX_DIR (a SemiAnalysisAI/InferenceX checkout) and AGENTX_HF_HOME (a Hugging Face
#       cache that holds, or can download, the trace dataset). The tok/s/GPU denominator is
#       PREFILL_GPUS + DECODE_GPUS.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/../common.sh"
CONC="${1:?usage: agentx.sh CONC}"
FAST="${FAST:-0}"
require_env PREFILL_IP; require_env INFERENCEX_DIR "a SemiAnalysisAI/InferenceX checkout"
require_env AGENTX_HF_HOME "Hugging Face cache for the trace dataset"
require_env PREFILL_GPUS; require_env DECODE_GPUS
grep -q "'rejection_sample_method': 'synthetic'" "$LOG_DIR/decode.log" 2>/dev/null \
  || warn "decode is NOT on golden acceptance — this point is not comparable with InferenceX (golden.sh golden)"
ngpu(){ echo $(( $(tr -cd , <<<"$1" | wc -c) + 1 )); }
OUT="$LOG_DIR/agentx-c$CONC"; mkdir -p "$OUT" "$AGENTX_HF_HOME/hub" && chmod 777 "$OUT"
RUNTIME="${AGENTX_RUNTIME_DIR:-/tmp/agentx-runtime}"; mkdir -p "$RUNTIME"
dram_gb="${UMBP_GB:-449}"
cat > "$OUT/runtime.env" <<EOF
INFMAX_CONTAINER_WORKSPACE=$INFERENCEX_DIR/inferencex-e2e
MODEL=$MODEL
TOKENIZER=${TOKENIZER:-$MODEL}
SERVED_MODEL_NAME=$SERVED
MODEL_PREFIX=dsv41flash
FRAMEWORK=vllm
PRECISION=fp4
IMAGE=$INFERA_IMAGE
RUNNER_TYPE=mi355x
CONC=$CONC
DURATION=${DURATION:-3600}
AIPERF_EXPERIMENTAL_FAST=$FAST
AIPERF_WARMUP_REQUESTS_PER_LANE=${WARMUP_PER_LANE:-10}
RESULT_DIR=$OUT/result
RESULT_FILENAME=agentx_conc$CONC
AGENTIC_OUTPUT_DIR=$OUT
EVAL_ONLY=false
IS_AGENTIC=1
SCENARIO_TYPE=agentic-coding
ENABLE_AGENTX_POWER=0
REQUIRE_POWER=0
IS_MULTINODE=true
DISAGG=true
PREFILL_NUM_WORKERS=1
PREFILL_TP=$(ngpu "$PREFILL_GPUS")
PREFILL_EP=1
PREFILL_DP_ATTN=true
DECODE_NUM_WORKERS=1
DECODE_TP=$(ngpu "$DECODE_GPUS")
DECODE_EP=1
DECODE_DP_ATTN=true
PREFILL_HARDWARE=mi355x
DECODE_HARDWARE=mi355x
KV_P2P_TRANSFER=mooncake
SPEC_DECODING=mtp
SIMULATE_ACC_LEN=${GOLDEN_AL:-3.51}
$(if [ "${UMBP:-1}" = 1 ]; then printf 'KV_OFFLOADING=dram\nKV_OFFLOAD_BACKEND=umbp\nKV_OFFLOAD_BACKEND_METADATA={"name":"umbp"}\nTOTAL_CPU_DRAM_GB=%s\n' "$dram_gb"
  else printf 'KV_OFFLOADING=none\nTOTAL_CPU_DRAM_GB=%s\n' "$(awk '/MemTotal/ {print int($2 / 1048576)}' /proc/meminfo)"; fi)
AIPERF_SERVER_URL=http://$PREFILL_IP:$ROUTER_PORT
AIPERF_SERVER_METRICS_URLS=http://$PREFILL_IP:$ROUTER_PORT/metrics,http://$PREFILL_IP:$PREFILL_PORT/metrics,http://${DECODE_IP:-$PREFILL_IP}:$DECODE_PORT/metrics
AIPERF_REQUIRED_SERVER_METRIC_PREFIX=vllm:
AIPERF_FAILED_REQUEST_THRESHOLD=0.10
AIPERF_LIVE_FAILED_REQUEST_THRESHOLD=0.10
AIPERF_TRACE_IDLE_GAP_CAP_SECONDS=300
AGENTIC_WARMUP_GRACE_PERIOD=${WARMUP_GRACE:-1800}
AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS=3600
AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING=0
AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=false
AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=true
AIPERF_HTTP_TCP_USER_TIMEOUT=900000
AIPERF_DRAIN_TIMEOUT_SECONDS=1800
AIPERF_RUNTIME_DIR=$RUNTIME
HF_HOME=$AGENTX_HF_HOME
HF_HUB_CACHE=$AGENTX_HF_HOME/hub
MPLCONFIGDIR=$RUNTIME/mpl
XDG_CACHE_HOME=$RUNTIME/xdg
PYTHONFAULTHANDLER=1
EOF
log "AgentX $([ "$FAST" = 1 ] && echo fast || echo formal) c$CONC -> $OUT (formal: ~1.3 h incl. warmup)"
docker rm -f "${CTR}-agentx" >/dev/null 2>&1
docker run --rm --name "${CTR}-agentx" --network host --ipc host --shm-size 32g \
  -v "$INFERENCEX_DIR:$INFERENCEX_DIR" -v "$AGENTX_HF_HOME:$AGENTX_HF_HOME" -v "$RUNTIME:$RUNTIME" \
  -v "$LOG_DIR:$LOG_DIR" -v "$MODEL_MOUNT:$MODEL_MOUNT:ro" --env-file "$OUT/runtime.env" \
  --entrypoint bash "${CLIENT_IMAGE:-$INFERA_IMAGE}" -c 'cd "$INFMAX_CONTAINER_WORKSPACE" && bash benchmarks/srt_agentic.sh' \
  > "$OUT/runner.log" 2>&1
rc=$?
echo "agentx rc=$rc" >> "$OUT/runner.log"
j=$(ls "$OUT"/agentx_conc*.json 2>/dev/null | head -1)
[ -n "$j" ] || { tail -20 "$OUT/runner.log"; die "AgentX c$CONC produced no result (rc=$rc): $OUT/runner.log"; }
python3 - "$j" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
m = d.get("request_metrics", {}); L = m.get("latency", {}); T = m.get("throughput", {})
g = lambda x, *ks: (g(x.get(ks[0]), *ks[1:]) if isinstance(x, dict) and ks else (x if not ks else None))
f = lambda v, s="{:,.0f}": s.format(v) if isinstance(v, (int, float)) else "?"
print(f"AgentX c{d.get('conc')} on {d.get('num_gpus')} GPUs: {d.get('num_requests_successful')}/{d.get('num_requests_total')} requests ok")
print(f"  total tok/s/GPU {f(g(T, 'per_gpu', 'total_tput_tps'))}  output tok/s/GPU {f(g(T, 'per_gpu', 'output_tput_tps'))}")
print(f"  interactivity p50/p90 {f(g(L, 'intvty', 'p50'), '{:.1f}')}/{f(g(L, 'intvty', 'p90'), '{:.1f}')} tok/s/user"
      f"  TTFT p50/p90 {f(g(L, 'ttft', 'p50'), '{:.2f}')}/{f(g(L, 'ttft', 'p90'), '{:.2f}')} s")
print(f"  server prefix-cache hit rate {f(g(d.get('server_metrics') or {}, 'cache', 'overall_cache_hit_rate'), '{:.3f}')}"
      f"  (trace's theoretical {f(g(m, 'cache', 'theoretical_cache_hit_rate'), '{:.3f}')})")
PY
exit $rc
