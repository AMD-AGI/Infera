#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Derive the Linux x86_64 wheel from the pure one: the same files plus the
# router at infera/_bin/infera-router, where launch_rust looks before PATH.
# Repacking the built wheel rather than building a second one keeps the Python
# side of the two wheels identical.
#   .github/scripts/bundle_router_wheel.sh <py3-none-any.whl> <infera-router> <out-dir>
#
# Needs `wheel` and `auditwheel>=6.8` importable by `python`.
set -euo pipefail

pure="${1:?usage: bundle_router_wheel.sh <pure-wheel> <binary> <out-dir>}"
binary="${2:?usage: bundle_router_wheel.sh <pure-wheel> <binary> <out-dir>}"
out="${3:?usage: bundle_router_wheel.sh <pure-wheel> <binary> <out-dir>}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

python -m wheel unpack -d "$work/tree" "$pure"
tree=("$work"/tree/*)
install -D -m 0755 "$binary" "${tree[0]}/infera/_bin/infera-router"
# auditwheel only accepts a platlib wheel, and it retags linux -> manylinux.
sed -i -e 's/^Root-Is-Purelib: true$/Root-Is-Purelib: false/' \
  -e 's/^Tag: py3-none-any$/Tag: py3-none-linux_x86_64/' "${tree[0]}"/*.dist-info/WHEEL
python -m wheel pack -d "$work" "${tree[0]}"

# Both the gate and the tagger: it fails unless the binary needs nothing past
# manylinux_2_28, and --patcher none forbids it to graft libraries in, so the
# wheel carries exactly the binary the router job built.
python -m auditwheel repair --plat manylinux_2_28_x86_64 --only-plat --patcher none \
  -w "$out" "$work"/*-py3-none-linux_x86_64.whl
