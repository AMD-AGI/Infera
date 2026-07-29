#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# SGLang PREFILL leg (prefill node): GLM-5.2-FP8 + MTP + DP-attention on 8 GPUs,
# KV handed to the decode leg over Mooncake RDMA.
# Override: ETCD_ENDPOINT=... bash infera_2_sglang_prefill.sh
#
# Cold start is ~20 min: 704 GiB of weights, read twice because MTP extracts the
# checkpoint's own nextn layer as the EAGLE draft model. Do not kill a slow launch.
set -euo pipefail

HOST_IP="${HOST_IP:-${POD_IP:-$(ip -o -4 route get 1.1.1.1 | awk '{print $7}')}}"
ETCD_ENDPOINT="${ETCD_ENDPOINT:-${HOST_IP}:2379}"
MODEL="${MODEL:-/wekafs/models/GLM-5.2-FP8}"
PORT="${PORT:-30001}"
TP="${TP:-8}"; DP="${DP:-$TP}"
MEM_FRAC="${MEM_FRAC:-0.85}"
# SGLang divides this by dp_size under DP attention, so the per-rank chunk is
# CHUNK/8. At the 131072 default that is 16384, which splits a 58k prompt into
# four prefill chunks — see REPORT.md section 3.1.
CHUNK="${CHUNK:-131072}"

# Publishing kv events is what makes --router-policy kv-aware able to see which
# DP rank already holds a prefix. One ZMQ port per DP rank is opened on
# advertise-host and the router subscribes to it; that is management traffic, it
# does not touch the Mooncake RoCE path.
KV_EVENT_ARGS=(--no-enable-kv-events --kv-events off)
if [[ "${KV_EVENTS:-0}" == "1" ]]; then
    KV_EVENT_ARGS=(--enable-kv-events --kv-events on --kv-event-transport zmq)
fi
# Pin one rail on both legs. Mooncake's auto-discovery (IB_DEVICE="") gives each GPU
# its NUMA-local HCA, which is faster in aggregate but only works if the two legs land
# on rails that can route to each other — here every rail is its own /31 to a leaf
# switch, so a mismatched pair never gets the RoCE QP to RTR and times out under load.
# One shared rail measures 229 Gb/s here, ~50x the KV rate a single prefill leg can
# produce, so it is not the bottleneck at 1P1D.
IB_DEVICE="${IB_DEVICE-rdma0}"
# MTP=1 enables EAGLE speculative decoding. Both legs must agree — the nextn layer
# adds a KV layer to the hand-off. Verified working at MTP=1 with the two additions
# below; see REPORT.zh.md 1.2/1.3 for why each is mandatory rather than optional.
MTP="${MTP:-0}"
LOG="${LOG:-$(dirname "$0")/infera_2_sglang_prefill.log}"

MTP_ARGS=()
if [[ "$MTP" != "0" ]]; then
    MTP_ARGS=(--speculative-algorithm EAGLE --speculative-num-steps 3
              --speculative-eagle-topk 1 --speculative-num-draft-tokens 4)
    # Turning GLM-5.2's IndexShare off is required, not a tuning choice. With it on,
    # dsa_topk_indices arrives None on a PREBUILT batch, which makes can_cuda_graph a
    # per-rank decision (eagle_worker_v2.py:511-517): idle ranks keep the draft graph
    # while the one rank with work falls to eager. The two paths disagree on both the
    # collective count and the DP padding mode, and the decode leg deadlocks in PD
    # warmup forever. It lives in the checkpoint's config.json, so it can only be
    # turned off by overriding hf_config.
    if [[ "${EXTRA_ARGS:-}" != *json-model-override-args* ]]; then
        MTP_ARGS+=(--json-model-override-args '{"index_share_for_mtp_iteration":false}')
    fi
fi

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
export SGLANG_HOST_IP="$HOST_IP" HOST_IP
export SGLANG_DSA_TRITON_PREFILL=1 SAFETENSORS_FAST_GPU=1
# gfx942 firmware: distributed init FATALs without this.
export HSA_NO_SCRATCH_RECLAIM=1
# The image already sets this, but pin it: it decides whether the DSA indexer takes
# aiter's preshuffle paged-MQA path, which sets page_size 64 vs 1. A leg that loses
# it pages its KV cache differently from its peer, and PD then hands over garbage.
export SGLANG_USE_AITER="${SGLANG_USE_AITER:-1}"
# Mooncake dials this GID. Broadcom bnxt_re puts the routable RoCEv2 IPv4 entry at
# index 3, not the ionic fleet's 1 — see preflight_rdma.sh.
export MC_GID_INDEX="${MC_GID_INDEX:-3}"
# Stock Mooncake installs a HIP IPC transport and prefers it over RDMA, so the
# cross-node KV hand-off dies in hipIpcOpenMemHandle. patch_mooncake_hip.sh gates
# it; without that gate this launch is 20 minutes of loading followed by a failure
# on the first request, so refuse to start. MC_ENABLE_HIP_TRANSPORT stays unset.
if [[ "$(strings "$(python3 -c 'import mooncake.engine as e; print(e.__file__)')" \
        | grep -c MC_ENABLE_HIP_TRANSPORT || true)" == "0" ]]; then
    echo "[prefill] mooncake still forces HIP IPC — run: bash $(dirname "$0")/patch_mooncake_hip.sh" >&2
    exit 1
fi
# Same deal for the two sglang patches. The first one is the nastier of the two: without
# it every prefill chunk but the last is RDMA-read while the forward is still writing
# those pages, and nothing errors — prompts above the per-rank chunk size (CHUNK/8) just
# come back partially wrong. See REPORT.md section 3.1.
SGLANG_ROOT="$(python3 -c 'import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec("sglang").origin).parents[2])')"
if ! grep -q wait_event "$SGLANG_ROOT/python/sglang/srt/disaggregation/mooncake/conn.py"; then
    echo "[prefill] sglang mooncake transport has no KV wait-event barrier — run: bash $(dirname "$0")/patch_sglang.sh" >&2
    exit 1
fi
# The second only matters with MTP on: the HIP DSA indexer feeds top-k the DP-padded row
# count, which asserts as soon as two DP ranks hold a near-but-unequal request count. The
# 8-way concurrent PD warmup is enough to produce that. Inert with MTP off.
if [[ "$MTP" != "0" ]] \
   && ! grep -q q_fp8_mqa "$SGLANG_ROOT/python/sglang/srt/layers/attention/dsa/dsa_indexer.py"; then
    echo "[prefill] MTP=$MTP needs the DSA padded-row patch — run: bash $(dirname "$0")/patch_sglang.sh" >&2
    exit 1
fi
# Both legs load 704 GiB off the same filesystem; SGLang's 300 s bootstrap wait and
# Infera's 1800 s readiness wait both expire long before that.
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
    --disaggregation-mode prefill --disaggregation-bootstrap-port 8998 \
    --disaggregation-transfer-backend mooncake ${IB_DEVICE:+--disaggregation-ib-device "$IB_DEVICE"} \
    > "$LOG" 2>&1 &

echo "[prefill] started on ${HOST_IP}:${PORT} (bootstrap :8998, ${IB_DEVICE}, MTP=${MTP}) — logs -> $LOG"
tail -f "$LOG"
