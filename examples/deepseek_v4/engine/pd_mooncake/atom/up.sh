#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: bring up atom 1P1D PD (mooncake) across prefill+decode nodes. why: single entry; atom PD
# is 1p1d-only. how: delegates to shared _pd_1p1d.sh over the shared-fs kit; caller = user.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CTR="${CTR:-dsv4_pd_atom}"; export CTR
source "$DIR/../_pd_1p1d.sh"
pd_1p1d_up engine/pd_mooncake/atom
