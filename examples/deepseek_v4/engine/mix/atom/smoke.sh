#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: curl a smoke completion through the router. why: prove worker registered + serves.
# how: thin wrapper over common.sh smoke(); caller = user / run.sh.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/../../../common.sh"
NODE_IP="${NODE_IP:-127.0.0.1}"; smoke "http://$NODE_IP:$ROUTER_PORT"
