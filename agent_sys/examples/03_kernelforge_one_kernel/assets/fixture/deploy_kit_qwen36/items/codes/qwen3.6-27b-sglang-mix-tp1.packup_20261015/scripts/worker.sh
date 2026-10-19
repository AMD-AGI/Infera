#!/usr/bin/env bash
# The engine worker. RUNS INSIDE THE ENGINE CONTAINER; deploy.sh stages it into
# the work root and starts it with `docker exec -d`. Every value arrives as an
# environment variable from deploy.sh; nothing here is site-specific.
#
# From the infera repository's examples/sglang_mix_glm5.3/engine/worker.sh.
# Differences, each a fact about Qwen3.6-27B (Qwen3_5ForConditionalGeneration:
# dense, hybrid linear/full attention) rather than a tuning choice:
#   * no --dsa-* flags: the model has no DSA path (E2E_DSA_ARGS=none);
#   * no --ep-size / --moe-runner-backend: the model is dense;
#   * bf16 KV (engine default), not fp8;
#   * --reasoning-parser qwen3 (E2E_PARSER_ARGS) instead of glm45/glm47;
#   * --enable-metrics so later stages can read sglang:* series.
# MIX mode is selected by OMITTING --disaggregation-mode.
#
# `exec` at the end, so the engine's own /proc/<pid>/cmdline is the argv (that
# is what check_deploy_serves and graph_ceiling.py read the ceiling from).
set -u
: "${MY_IP:?}" "${MODEL:?}" "${SERVED:?}" "${PORT:?}" "${ETCD_PORT:?}" "${TP:?}" "${GPUS:?}"
: "${CTX:?}" "${GMU:?}" "${GRAPH_MAX_BS:?}" "${KV_PUB_PORT:?}" "${KV_SNAP_PORT:?}" "${LOG:?}"
: "${DSA_ARGS:=none}" "${PARSER_ARGS:=none}" "${EXTRA_ARGS:=}" "${EXTRA_ENV:=}"

export SGLANG_USE_AITER=1
export SAFETENSORS_FAST_GPU=1 HIP_FORCE_DEV_KERNARG=1 HSA_NO_SCRATCH_RECLAIM=1
export NCCL_IGNORE_CPU_AFFINITY=1
export PYTHONHASHSEED=0            # stable block hashes -> stable kv-aware keys
export SGLANG_HOST_IP="$MY_IP" HOST_IP="$MY_IP"
export INFERA_SGLANG_READY_TIMEOUT="${READY_TIMEOUT:-3600}"
NIC=$(ip -o -4 addr show 2>/dev/null | awk -v ip="$MY_IP" '$4 ~ ("^" ip "/") {print $2; exit}')
[ -n "$NIC" ] && export SGLANG_LOCAL_IP_NIC="$NIC" GLOO_SOCKET_IFNAME="$NIC"

# E2E_KIT_ENGINE_EXTRA_ENV: exported into THIS process only. The router is a
# separate process started by router.sh and never sees it.
for kv in $EXTRA_ENV; do
  case "$kv" in *=*) export "$kv" ;; *) echo "[worker] ignoring EXTRA_ENV item without '=': $kv" >&2 ;; esac
done

[ "$DSA_ARGS" = none ] && DSA_ARGS=""
[ "$PARSER_ARGS" = none ] && PARSER_ARGS=""
read -r -a DSA_A <<< "$DSA_ARGS"
read -r -a PARSER_A <<< "$PARSER_ARGS"
read -r -a EXTRA_A <<< "$EXTRA_ARGS"

ARGV=(python3 -m infera.engine.sglang
  --model-path "$MODEL" --served-model-name "$SERVED"
  --tp-size "$TP" --trust-remote-code
  --host "$MY_IP" --port "$PORT"
  --context-length "$CTX" --mem-fraction-static "$GMU"
  --watchdog-timeout 3600 --enable-metrics
  "${DSA_A[@]}" "${PARSER_A[@]}"
  --cuda-graph-backend-decode full --cuda-graph-backend-prefill disabled
  --cuda-graph-max-bs-decode "$GRAPH_MAX_BS"
  --advertise-host "$MY_IP" --etcd-endpoint "$MY_IP:$ETCD_PORT"
  --discovery-backend etcd --request-transport http --kv-event-transport zmq
  --kv-events-bind "tcp://$MY_IP:$KV_PUB_PORT"
  --kv-snapshot-host "$MY_IP" --kv-snapshot-port "$KV_SNAP_PORT"
  "${EXTRA_A[@]}")

mkdir -p "$(dirname "$LOG")"
{ echo "[worker] $(date -u +%FT%TZ) HIP_VISIBLE_DEVICES=$GPUS extra_env='$EXTRA_ENV'"
  printf '[worker] argv:'; printf ' %q' "${ARGV[@]}"; echo; } > "$LOG"
HIP_VISIBLE_DEVICES="$GPUS" exec "${ARGV[@]}" >> "$LOG" 2>&1
