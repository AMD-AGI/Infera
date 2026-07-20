#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: one-shot sglang mix bring-up: container -> etcd -> router -> worker -> smoke.
# why : single entry for external users. how: orchestrates common.sh + engine.sh; caller = user.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/../../../common.sh"
require_env INFERA_MODEL; require_file "$INFERA_MODEL"
CTR="${CTR:-dsv4_sgl_mix}"; NODE_IP="${NODE_IP:-127.0.0.1}"; export CTR NODE_IP
start_container "$CTR"
start_etcd repro-etcd "$NODE_IP"
start_router "$CTR" "$NODE_IP"
bash "$DIR/engine.sh"
smoke "http://$NODE_IP:$ROUTER_PORT"
log "sglang mix ready on http://$NODE_IP:$ROUTER_PORT (bench: bash $DIR/bench.sh <conc>)"
