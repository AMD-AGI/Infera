#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Correctness checks through the router (smoke.py): worker registration,
# completion, chat, streaming, and a ~66k-token needle prompt sent twice, the
# second time from the prefix cache. Run it on an engine started with real MTP
# acceptance (SIM_ACCEPT_LEN empty); forced acceptance produces wrong text.
# Output: <run>/smoke/.
#
# Usage: bash engine/smoke.sh
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "$KIT/engine/common.sh"
RUN_DIR="$(run_dir)"
python3 "$KIT/engine/smoke.py" "$ROUTER_URL" "$SERVED_MODEL" "$RUN_DIR/smoke"
