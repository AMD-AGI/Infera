#!/bin/bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Inject host libionic provider before exec'ing the worker command.
#
# Background: On AMD MI355X + Pensando ionic hosts, the userspace
# libionic.so ABI is tied to the host kernel module's ABI. The
# container's libionic (shipped by the SGLang base image) is often
# from a different release train than the host's ionic_rdma kernel
# module — when this happens libibverbs prints
#     "Driver ionic does not support the kernel ABI of N (supports M to M)"
# and ibv_get_device_list() returns ZERO devices, which silently
# downgrades Mooncake (and any other RDMA user) to TCP.
#
# This script fixes it by copying the host's libionic provider over
# the container's existing one and re-running ldconfig, so the same
# in-container libibverbs.so dlopen()s a provider that speaks the
# host kernel's ABI.
#
# Wiring (docker run / k8s): bind-mount the host's libionic.so
# at /host-libionic/libionic.so inside the container — Docker
# resolves the host-side symlink chain at mount time so the version
# is never hardcoded in the manifest.
#
# When /host-libionic/libionic.so is absent (NVIDIA hosts, non-ionic
# RDMA fabrics, or boxes where the container's libionic already
# matches), the script is a no-op and the original command runs
# unchanged.

set -e

SRC=/host-libionic/libionic.so
if [ -e "$SRC" ]; then
    # Follow the symlink chain: libionic.so -> libionic.so.1 -> libionic.so.1.x.y.z
    TGT=$(readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1 2>/dev/null || true)
    if [ -n "$TGT" ] && [ -f "$TGT" ]; then
        if ! cmp -s "$SRC" "$TGT"; then
            cp -f "$SRC" "$TGT"
            ldconfig 2>/dev/null || true
            echo "infera-inject-host-ionic: replaced $TGT with host build" >&2
        fi
    else
        echo "infera-inject-host-ionic: container has no libionic.so.1; skipping" >&2
    fi
fi

exec "$@"
