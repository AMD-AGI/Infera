#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# One AgentX point with the 4P4D agentx.sh; its metadata reports the prefill
# as TP1 x PP4, so throughput per GPU is still over 8 GPUs.
#
# Usage: bash engine/agentx.sh   (DURATION, WARMUP_PER_LANE, PREPARE_ONLY as in 4P4D)
set -euo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../env.sh
source "$KIT/env.sh"
exec bash "$BASE_KIT/engine/agentx/agentx.sh"
