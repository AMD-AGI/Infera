#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Bring up 4P4D with the 4P4D up.sh: etcd, the PP4 prefill and the decode,
# then the router.
#
# Usage: bash engine/up.sh            (RUN_ID=<name> to name the run directory)
set -euo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../env.sh
source "$KIT/env.sh"
exec bash "$BASE_KIT/engine/up.sh"
