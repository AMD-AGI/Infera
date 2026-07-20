#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: tear down the vllm mix case (reap worker, remove containers). why: free VRAM before
# the next run (relaunch OOMs otherwise). how: reap then docker rm; caller = user.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/../../../common.sh"
CTR="${CTR:-dsv4_vllm_mix}"
reap "$CTR"
docker rm -f "$CTR" repro-etcd >/dev/null 2>&1 || true
log "vllm mix torn down"
