#!/bin/bash
# Shared configuration for the Kimi agentic benchmark. Source this from the launch
# scripts. Every value is overridable from the environment; the defaults below are
# placeholders — set at least PREFILL_IP / DECODE_IP and MODEL for your cluster.
set -uo pipefail

# --- Cluster addresses ---------------------------------------------------------
# Address each engine advertises and the router/benchmark connect to. For a
# single-node run, point both at the same host.
export PREFILL_IP=${PREFILL_IP:-127.0.0.1}      # node hosting the prefill engine + router
export DECODE_IP=${DECODE_IP:-127.0.0.1}        # node hosting the decode engine
export ETCD_EP=${ETCD_EP:-${PREFILL_IP}:2379}   # etcd endpoint used for PD discovery
export ROUTER=${ROUTER:-http://${PREFILL_IP}:8100}

# --- Model ---------------------------------------------------------------------
export MODEL=${MODEL:-/models/Kimi-K2.6-MXFP4}  # local path to the model weights
export SERVED=${SERVED:-kimi2.6-mxfp4}          # served-model-name

# --- Optional: KV cache offload (kvd L3) ---------------------------------------
# Only used when the prefill engine is launched with KVD=1. KVD_L3 must be a
# single NVMe mount (the hipFile GPU-direct path is supported on a single NVMe).
export KVD_SOCK=${KVD_SOCK:-/tmp/kvd-pd.sock}
export KVD_L3=${KVD_L3:-/mnt/nvme/kvd-l3}
