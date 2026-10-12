#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Runner listeners live for weeks and keep the scheduler env they started with; a stale
# SPUR_CONTROLLER_ADDR makes every submission fail auth. Export the host's current values.
set -euo pipefail

for f in /etc/environment /etc/profile.d/spur.sh; do
  [ -r "$f" ] || continue
  sed -nE 's/^[[:space:]]*(export[[:space:]]+)?(SPUR_[A-Z_]+)=["'\'']?([^"'\'']*)["'\'']?[[:space:]]*$/\2=\3/p' "$f"
done | awk -F= '!seen[$1]++' | tee -a "${GITHUB_ENV:?not running under GitHub Actions}"
