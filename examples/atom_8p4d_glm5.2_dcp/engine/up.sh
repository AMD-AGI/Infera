#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Bring up 8P4D from the head node with the 4P4D up.sh: etcd, prefill0 and
# decode here, prefill1 on the second node (engine/prefill1.sh), then the
# router, which must list two prefills and one decode.
#
# Usage: bash engine/up.sh            (RUN_ID=<name> to name the run directory)
set -euo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../env.sh
source "$KIT/env.sh"
bash "$KIT/engine/prefill1.sh" check
exec bash "$BASE_KIT/engine/up.sh"
