#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Correctness checks through the router (smoke.py): PD registration,
# completion, chat, streaming, and a ~66k-token needle prompt sent twice, the
# second time from the prefix cache. Needs real MTP acceptance (MTP_AL empty).
# Output: <run>/smoke/.
#
# Usage: bash engine/smoke.sh
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "$KIT/engine/common.sh"
load_run
require_real_acceptance
python3 "$KIT/engine/smoke.py" "$ROUTER_URL" "$MODEL" "$RUN_DIR/smoke"
