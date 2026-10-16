#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# The 4P4D GSM8K run; it needs real MTP acceptance.
#
# Usage: bash engine/eval.sh     (EVAL_LIMIT, EVAL_CONCURRENCY as in 4P4D)
set -euo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../env.sh
source "$KIT/env.sh"
exec bash "$BASE_KIT/engine/eval.sh"
