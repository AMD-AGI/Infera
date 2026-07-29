#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# SGLang DECODE leg (decode node): same GLM-5.2-FP8 + MTP + DP-attention recipe as
# the prefill leg on its own 8 GPUs, receiving KV over Mooncake RDMA.
# REQUIRED: ETCD_ENDPOINT=<prefill-node-ip>:2379 bash infera_3_sglang_decode.sh
#
# The two legs must agree on model, TP/DP and KV dtype — a flag that drifts between
# them surfaces as a KV shape mismatch twenty minutes into the load.
set -euo pipefail

HOST_IP="${HOST_IP:-${POD_IP:-$(ip -o -4 route get 1.1.1.1 | awk '{print $7}')}}"
# etcd runs on the prefill node, so there is no sane local default.
: "${ETCD_ENDPOINT:?set ETCD_ENDPOINT=<prefill-node-ip>:2379 (the node running infera_0_etcd.sh)}"
MODEL="${MODEL:-/wekafs/models/GLM-5.2-FP8}"
PORT="${PORT:-31001}"
TP="${TP:-8}"; DP="${DP:-$TP}"
MEM_FRAC="${MEM_FRAC:-0.85}"
# Must match the prefill leg. See the note in infera_2_sglang_prefill.sh.
CHUNK="${CHUNK:-131072}"

# Off by default on this leg even when the prefill leg publishes: infera appends
# --disaggregation-decode-enable-radix-cache to a mooncake decode worker whenever
# kv events are on (infera/engine/sglang/args.py:257-263), and SGLang rejects that
# flag together with speculative decoding. Safe to set KV_EVENTS=1 only while
# MTP=0. With it off, decode routing falls back to load-only, which is fine —
# prefix reuse is a prefill-side win.
KV_EVENT_ARGS=(--no-enable-kv-events --kv-events off)
if [[ "${KV_EVENTS:-0}" == "1" ]]; then
    KV_EVENT_ARGS=(--enable-kv-events --kv-events on --kv-event-transport zmq)
fi
# Must match the prefill leg's rail. See the note in infera_2_sglang_prefill.sh.
IB_DEVICE="${IB_DEVICE-rdma0}"
# Must match the prefill leg's MTP setting, including the IndexShare override that the
# MTP branch adds. See the note in infera_2_sglang_prefill.sh.
MTP="${MTP:-0}"
LOG="${LOG:-$(dirname "$0")/infera_3_sglang_decode.log}"

MTP_ARGS=()
if [[ "$MTP" != "0" ]]; then
    MTP_ARGS=(--speculative-algorithm EAGLE --speculative-num-steps 3
              --speculative-eagle-topk 1 --speculative-num-draft-tokens 4)
    # Mandatory with MTP on this leg — this is the leg that deadlocks without it.
    # See the note in infera_2_sglang_prefill.sh.
    if [[ "${EXTRA_ARGS:-}" != *json-model-override-args* ]]; then
        MTP_ARGS+=(--json-model-override-args '{"index_share_for_mtp_iteration":false}')
    fi
fi

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
export SGLANG_HOST_IP="$HOST_IP" HOST_IP
export SGLANG_DSA_TRITON_PREFILL=1 SAFETENSORS_FAST_GPU=1
export HSA_NO_SCRATCH_RECLAIM=1
# Decides the DSA page size (64 vs 1) — must match the prefill leg or the KV
# hand-off is garbage. See the note in infera_2_sglang_prefill.sh.
export SGLANG_USE_AITER="${SGLANG_USE_AITER:-1}"
export MC_GID_INDEX="${MC_GID_INDEX:-3}"
# See the note in infera_2_sglang_prefill.sh: HIP IPC cannot reach the other node.
if [[ "$(strings "$(python3 -c 'import mooncake.engine as e; print(e.__file__)')" \
        | grep -c MC_ENABLE_HIP_TRANSPORT || true)" == "0" ]]; then
    echo "[decode] mooncake still forces HIP IPC — run: bash $(dirname "$0")/patch_mooncake_hip.sh" >&2
    exit 1
fi
# Both sglang patches must be present on both legs. See the note in
# infera_2_sglang_prefill.sh for what each one prevents.
SGLANG_ROOT="$(python3 -c 'import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec("sglang").origin).parents[2])')"
if ! grep -q wait_event "$SGLANG_ROOT/python/sglang/srt/disaggregation/mooncake/conn.py"; then
    echo "[decode] sglang mooncake transport has no KV wait-event barrier — run: bash $(dirname "$0")/patch_sglang.sh" >&2
    exit 1
fi
if [[ "$MTP" != "0" ]] \
   && ! grep -q q_fp8_mqa "$SGLANG_ROOT/python/sglang/srt/layers/attention/dsa/dsa_indexer.py"; then
    echo "[decode] MTP=$MTP needs the DSA padded-row patch — run: bash $(dirname "$0")/patch_sglang.sh" >&2
    exit 1
fi
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT="${SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT:-3600}"
export INFERA_ENGINE_READY_TIMEOUT="${INFERA_ENGINE_READY_TIMEOUT:-5400}"

# Kill the sglang child too: infera spawns it as a separate process that outlives
# the wrapper, and it keeps both the port and its VRAM.
pkill -f "(infera.engine.sglang|sglang.launch_server) .*--port ${PORT}( |\$)" 2>/dev/null || true
sleep 5

nohup python3 -m infera.engine.sglang \
    --model-path "$MODEL" --host 0.0.0.0 --port "$PORT" --advertise-host "$HOST_IP" \
    --etcd-endpoint "$ETCD_ENDPOINT" --discovery-backend etcd \
    --request-transport http "${KV_EVENT_ARGS[@]}" \
    --tp-size "$TP" --dp-size "$DP" --enable-dp-attention \
    --trust-remote-code --kv-cache-dtype fp8_e4m3 \
    --reasoning-parser glm45 --tool-call-parser glm47 \
    --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
    --mem-fraction-static "$MEM_FRAC" --chunked-prefill-size "$CHUNK" \
    --watchdog-timeout 1200 --disable-custom-all-reduce \
    "${MTP_ARGS[@]}" ${EXTRA_ARGS:-} \
    --weight-loader-prefetch-checkpoints \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 32}' \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mooncake ${IB_DEVICE:+--disaggregation-ib-device "$IB_DEVICE"} \
    > "$LOG" 2>&1 &

echo "[decode] started on ${HOST_IP}:${PORT} (${IB_DEVICE}, MTP=${MTP}) -> etcd ${ETCD_ENDPOINT} — logs -> $LOG"
tail -f "$LOG"
