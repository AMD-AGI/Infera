#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# NOTE: For Hyperloom testing use only — not part of the standard Infera
# serving path.
#
# Idle worker entrypoint for Hyperloom multi-node mode.
#
# Starts the SSH control plane (mn-sshd-init.sh) and then blocks forever so the
# pod stays alive as an "allocated but idle" node. The inference_optimizer
# backend then SSHes in to (re)launch sglang/vllm with whatever flags a
# given benchmark round needs — mirroring the RayJob restart-server pattern.
#
# CRITICAL: this script intentionally IGNORES all positional arguments ("$@").
# The orchestrator dispatcher unconditionally appends the sglang multi-node
# flags (`--nnodes N --node-rank $LWS_WORKER_INDEX --dist-init-addr <leader>:5000`)
# to a multinode-role worker's command. In idle mode those flags must be inert —
# we inject the real --node-rank/--dist-init-addr ourselves over SSH at launch
# time — so letting them land as ignored args (rather than corrupting a
# `tail -f` cmd) keeps the pod healthy with zero orchestrator-side change.
#
set -uo pipefail

/usr/local/bin/mn-sshd-init.sh || echo "[mn-idle] sshd init failed (continuing idle)" >&2

# Block forever; the SSH-launched sglang/vllm runs as a separate detached
# process, so pid1 just needs to stay alive.
exec tail -f /dev/null
