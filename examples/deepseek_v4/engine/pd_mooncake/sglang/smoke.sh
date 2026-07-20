#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: wait for all PD workers ready, then smoke a completion through the router.
# why : prove infera.server paired prefill+decode and KV hand-off works. caller = user / up.sh.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/../../../common.sh"
require_env PREFILL_IP; require_env INFERA_MODEL
URL="http://$PREFILL_IP:$ROUTER_PORT"
wait_health "$URL/health" 200 || die "router health never came up"
smoke "$URL"
