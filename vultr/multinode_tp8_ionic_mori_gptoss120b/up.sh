#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Bring up a 2-node 1P1D gpt-oss-120b stack over MoRI:
#   prefill node ($PREFILL_NODE / $PREFILL_IP): etcd + infera.server + 1x prefill TP=8
#   decode  node ($DECODE_NODE / $DECODE_IP): 1x decode TP=8
set -euo pipefail
PREFILL_NODE="${PREFILL_NODE:?set PREFILL_NODE}";  PREFILL_IP="${PREFILL_IP:?set PREFILL_IP}"
DECODE_NODE="${DECODE_NODE:?set DECODE_NODE}";    DECODE_IP="${DECODE_IP:?set DECODE_IP}"
ETCD_ENDPOINT="${ETCD_ENDPOINT:-${PREFILL_IP}:12379}"
INFERA_KVD_L3="${INFERA_KVD_L3:-0}"
MODEL_PATH="${MODEL_PATH:-/PATH/TO/gpt-oss-120b}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
NATS_SERVER="${NATS_SERVER:-nats://${PREFILL_IP}:4222}"
# Dir holding these scripts. Derived from location; must be a path that
# resolves identically on every node (shared storage). Override SD if the
# repo lives at a different path on the remote nodes.
SD="${SD:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
step(){ printf "\n=========== %s ===========\n" "$*"; }

step "0/5 nats broker on ${PREFILL_NODE}"
ssh "$PREFILL_NODE" "docker rm -f pd-mori-nats 2>/dev/null; docker run -d --name pd-mori-nats --network=host --restart=unless-stopped nats:latest -p 4222 >/dev/null && echo '[nats] up on ${PREFILL_IP}:4222'"
step "1/5 etcd on ${PREFILL_NODE}"
ssh "$PREFILL_NODE" "ETCD_HOST_IP=${PREFILL_IP} bash ${SD}/launch_etcd.sh"
step "2/5 infera.server on ${PREFILL_NODE}"
ssh "$PREFILL_NODE" "ETCD_HOST_IP=${PREFILL_IP} ROUTER_POLICY=round-robin TOKENIZER_PATH=${TOKENIZER_PATH} NATS_SERVER=${NATS_SERVER} bash ${SD}/launch_server.sh"
KVD_L3_DIR="${KVD_L3_DIR:-/mnt/kvd-l3}"
step "3/5 prefill engine on ${PREFILL_NODE} (TP=8, kvd_l3=${INFERA_KVD_L3})"
ssh "$PREFILL_NODE" "ROLE=prefill DATA_PLANE_IP=${PREFILL_IP} ETCD_ENDPOINT=${ETCD_ENDPOINT} MODEL_PATH=${MODEL_PATH} NATS_SERVER=${NATS_SERVER} INFERA_KVD_L3=${INFERA_KVD_L3} KVD_L3_DIR=${KVD_L3_DIR} bash ${SD}/launch_engine.sh"
step "4/5 decode engine on ${DECODE_NODE} (TP=8, kvd_l3=${INFERA_KVD_L3})"
ssh "$DECODE_NODE" "ROLE=decode DATA_PLANE_IP=${DECODE_IP} ETCD_ENDPOINT=${ETCD_ENDPOINT} MODEL_PATH=${MODEL_PATH} NATS_SERVER=${NATS_SERVER} INFERA_KVD_L3=${INFERA_KVD_L3} KVD_L3_DIR=${KVD_L3_DIR} bash ${SD}/launch_engine.sh"
cat <<MSG

Both engines kicked off (cold start ~8-12 min, parallel). Watch:
  ssh ${PREFILL_NODE} 'docker exec pd_mori_sgl tail -f ${SD}/logs/pd_prefill_*'
  ssh ${DECODE_NODE}  'docker exec pd_mori_sgl tail -f ${SD}/logs/pd_decode_*'
Workers:  curl -s http://${PREFILL_IP}:8000/v1/workers | python3 -m json.tool
Smoke:    curl -s http://${PREFILL_IP}:8000/v1/chat/completions -H 'Content-Type: application/json' \\
            -d '{"model":"/PATH/TO/gpt-oss-120b","messages":[{"role":"user","content":"1+1?"}],"max_tokens":16,"temperature":0}'
MSG
