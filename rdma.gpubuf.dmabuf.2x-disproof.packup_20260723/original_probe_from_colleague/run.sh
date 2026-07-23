#!/usr/bin/env bash
# One-shot runner: prints the ionic driver/firmware versions, then runs the probe.
# Run this INSIDE a ROCm container that can see the ionic NICs (>= a few GiB free VRAM).
#   bash run.sh                # default registers 4 GiB
#   PROBE_GIB=2 bash run.sh    # smaller, if VRAM is tight
#   PROBE_DEV=ionic_3 bash run.sh   # force a specific NIC (default: first ionic*)
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "===== ENV (confirm this is the NEWER ionic driver) ====="
echo "driver ionic      : $(cat /sys/module/ionic/version 2>/dev/null || modinfo ionic 2>/dev/null | sed -n 's/^version:[[:space:]]*//p')"
echo "driver ionic_rdma : $(cat /sys/module/ionic_rdma/version 2>/dev/null || modinfo ionic_rdma 2>/dev/null | sed -n 's/^version:[[:space:]]*//p')"
echo "firmware fw_ver   : $(cat /sys/class/infiniband/ionic_0/fw_ver 2>/dev/null)"
echo "libionic userspace: $(dpkg -l 2>/dev/null | awk '/libionic1/{print $3}')"
echo "kernel            : $(uname -r)"
echo

echo "===== RUN probe (PROBE_GIB=${PROBE_GIB:-4}) ====="
python3 "$DIR/ionic_vram_test.py"
