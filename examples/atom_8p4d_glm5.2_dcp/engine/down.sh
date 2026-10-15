#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# The 4P4D down.sh on the head node; prefill1 goes down on the second node in
# parallel. Waits for VRAM to drain on both.
#
# Usage: bash engine/down.sh
set -euo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../env.sh
source "$KIT/env.sh"
exec bash "$BASE_KIT/engine/down.sh"
