#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# One AgentX point with the 4P4D agentx.sh. It takes the worker counts and
# /metrics endpoints from the router, so throughput per GPU is over 12 GPUs.
#
# Usage: bash engine/agentx.sh   (DURATION, WARMUP_PER_LANE, PREPARE_ONLY as in 4P4D)
set -euo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../env.sh
source "$KIT/env.sh"
exec bash "$BASE_KIT/engine/agentx/agentx.sh"
