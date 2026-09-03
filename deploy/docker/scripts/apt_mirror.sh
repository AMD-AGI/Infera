#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
# Point apt at a mirror this fleet can reach. A no-op unless APT_MIRROR or
# APT_SECURITY_MIRROR is set, so a default build is byte-identical to before.
#
# WHY it exists: several layers of every engine image apt-install build
# dependencies -- Mooncake's dependencies.sh alone pulls yaml-cpp, grpc and
# protobuf -- so a fleet whose nodes cannot route to archive.ubuntu.com dies in
# the middle of a 40-minute build with a cmake "Could not find yaml-cpp", an
# error that names neither apt nor the network. Seen on the MI300X fleet, where
# the hosts reach repo.radeon.com, pypi and github but time out on
# archive.ubuntu.com over both http and https.
#
# WHY only the switch is in the tree: a mirror hostname is a property of the
# cluster, not of the image, so its value rides in as a build arg (from
# INFERA_E2E_BUILD_ARGS), the same channel MOONCAKE_HIP_DMABUF uses. The
# alternative a site is otherwise pushed into -- pre-building a base image and
# overriding *_BASE_IMAGE -- throws away the base's @sha256 pin, and has to be
# repeated on every node because docker images are per-node.
#
# WHERE it is wired in: Dockerfile.vllm, Dockerfile.atom and
# Dockerfile.sglang.gfx942 -- the images a gfx942 run builds. Dockerfile.sglang
# is gfx950-only (see tests/e2e/harness/images.py) and that fleet reaches
# archive.ubuntu.com, so wiring it there would only add a layer that never fires.
set -euo pipefail

MIRROR="${APT_MIRROR:-}"
SECURITY="${APT_SECURITY_MIRROR:-}"
if [ -z "$MIRROR$SECURITY" ]; then
    echo "[apt] no mirror requested — sources left untouched"
    exit 0
fi

# Both layouts: the single sources.list of <=24.04 bases and the deb822
# *.sources of >=24.04. Rewriting only the first would turn this into a silent
# no-op the day a base bumps, and the build would fail exactly as it does now.
shopt -s nullglob
files=(/etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources)
if [ "${#files[@]}" -eq 0 ]; then
    echo "[apt] FATAL: no apt sources files to rewrite; is this an apt-based image?" >&2
    exit 1
fi

for f in "${files[@]}"; do
    [ -f "$f" ] || continue
    # The host part is matched loosely so a base that already points at a
    # regional mirror (xx.archive.ubuntu.com) is redirected too.
    if [ -n "$MIRROR" ]; then
        sed -i "s|https\?://\([a-z0-9-]*\.\)*archive\.ubuntu\.com/ubuntu/\?|$MIRROR|g" "$f"
    fi
    if [ -n "$SECURITY" ]; then
        sed -i "s|https\?://security\.ubuntu\.com/ubuntu/\?|$SECURITY|g" "$f"
    fi
done

echo "[apt] sources now:"
# `|| true` because this only prints: under `pipefail` a grep that matches
# nothing would abort the script from a display statement, which would be a
# baffling way for a build to end.
grep -rhE '^(deb |URIs:)' "${files[@]}" 2>/dev/null | sort -u | sed 's/^/[apt]   /' || true

# Rebuild the index here rather than let the first apt-install three layers down
# do it: a mirror that does not answer is a one-character build-arg mistake, and
# this is the only place that can say so in those words.
apt-get update -qq

# ...but only if we check. `apt-get update` exits 0 with a mere "W: Failed to
# fetch" for a source that does not resolve at all (measured: a deb line for
# nonexistent.invalid still gives rc=0), so an unreachable mirror would sail
# through here and fail as the original cmake error a layer later. apt names each
# downloaded index after its host, so the index is the thing to assert on.
_fetched() {
    local url="$1" label="$2" host
    host=$(printf '%s' "$url" | sed -E 's|^[a-z]+://||; s|/.*||')
    [ -n "$host" ] || return 0
    # Only assert on a host the sources still mention. A base is free to route
    # jammy-security through archive.ubuntu.com rather than security.ubuntu.com
    # (the SGLang one does), in which case APT_MIRROR already rewrote those lines
    # and APT_SECURITY_MIRROR matched nothing — no index to fetch, nothing wrong.
    if ! grep -rhE '^(deb |URIs:)' "${files[@]}" 2>/dev/null | grep -qF "$host"; then
        echo "[apt] $label mirror unused: no source refers to $host"
        return 0
    fi
    if ! ls /var/lib/apt/lists/ 2>/dev/null | grep -q "^${host}_"; then
        echo "[apt] FATAL: sources point at '$host' but no index came back —" >&2
        echo "[apt]        the mirror is unreachable from this builder, or the URL is wrong." >&2
        return 1
    fi
    echo "[apt] verified: index fetched from $host"
}
[ -z "$MIRROR" ] || _fetched "$MIRROR" archive
[ -z "$SECURITY" ] || _fetched "$SECURITY" security
