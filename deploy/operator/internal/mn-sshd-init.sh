#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# NOTE: For Hyperloom testing use only — not part of the standard Infera
# serving path.
#
# Hyperloom multi-node control-plane SSH bootstrap (idle-pod mode).
#
# The inference_optimizer multi-node backend deploys worker pods with
# an idle command (`tail -f /dev/null`) and drives sglang/vllm (re)starts over
# SSH, mirroring the RayJob "long-lived pod + restart-server" pattern but
# without Ray. This script makes a freshly-started idle pod reachable:
#
#   1. Install the operator-provided public key from $MN_SSH_AUTHORIZED_KEY
#      (runtime injection keeps the key decoupled from the image layer).
#   2. Generate host keys if missing.
#   3. Launch sshd on $MN_SSH_PORT (default 2222) in the background.
#
# Port 2222 (not 22) is deliberate: the pod may run with
# `dnsPolicy: ClusterFirstWithHostNet` and be promoted to hostNetwork, where
# binding :22 would collide with the node's own sshd. A dedicated port avoids
# that clash on every cluster.
#
# Idempotent: safe to re-run (re-writes authorized_keys, skips existing host
# keys, and `sshd` simply fails to re-bind if already listening).
#
# Usage (worker entryPoint set by the workload builder):
#   /usr/local/bin/mn-sshd-init.sh && tail -f /dev/null
#
set -euo pipefail

MN_SSH_PORT="${MN_SSH_PORT:-2222}"

mkdir -p /root/.ssh /run/sshd
chmod 700 /root/.ssh

# 1. Install the operator-provided public key (runtime, not baked into image).
if [ -n "${MN_SSH_AUTHORIZED_KEY:-}" ]; then
  printf '%s\n' "${MN_SSH_AUTHORIZED_KEY}" > /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
  echo "[mn-sshd-init] authorized_keys installed for root"
else
  echo "[mn-sshd-init] WARN: MN_SSH_AUTHORIZED_KEY unset; key auth will fail" >&2
fi

# 2. Host keys (no-op if already present).
ssh-keygen -A

# 3. sshd config drop-in: key-only root login on the dedicated port. Ubuntu's
#    stock sshd_config carries `Include /etc/ssh/sshd_config.d/*.conf`; if it
#    is missing (non-standard base), append the include so our drop-in wins.
mkdir -p /etc/ssh/sshd_config.d
cat > /etc/ssh/sshd_config.d/hyperloom-mn.conf <<EOF
Port ${MN_SSH_PORT}
PermitRootLogin prohibit-password
PubkeyAuthentication yes
PasswordAuthentication no
EOF
if ! grep -qE '^\s*Include\s+/etc/ssh/sshd_config\.d/\*\.conf' /etc/ssh/sshd_config 2>/dev/null; then
  echo "Include /etc/ssh/sshd_config.d/*.conf" >> /etc/ssh/sshd_config
fi

# 4. Launch sshd detached; the caller (entryPoint) keeps pid1 alive via tail.
/usr/sbin/sshd -e
echo "[mn-sshd-init] sshd listening on :${MN_SSH_PORT}"
