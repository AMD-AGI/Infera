#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Apply the two out-of-tree sglang patches this recipe needs. Run once per node,
# BEFORE the engine legs — the containerised equivalent of what the image build would
# do, for clusters where we only get a running container and cannot rebuild it.
#
#   bash patch_sglang.sh            # apply both, verify
#   bash patch_sglang.sh --status   # report only
#
# 1. sglang_disagg/mooncake_early_send_wait_event.diff — REQUIRED FOR CORRECTNESS.
#    Without it every prefill chunk except the last is RDMA-read while the forward is
#    still writing those pages, so the decode leg gets half-written KV. It does not
#    crash: prompts longer than one chunk (>16k per DP rank here) come back partially
#    wrong. See §3.1 of REPORT.md.
# 2. sglang_dsa/dsa_indexer_hip_dp_padded_rows.diff — REQUIRED BEFORE ENABLING MTP.
#    The HIP indexer feeds top-k the DP-padded row count while `lengths` is sized to
#    the real one, so DP-attention + MTP dies on
#        RuntimeError: Expected lengths.size(0) == B to be true, but got false.
#    as soon as two DP ranks hold a near-but-unequal number of requests. With MTP off
#    it is inert; with MTP on, the 8-way concurrent PD warmup alone can trigger it.
#
# Both are applied with `patch` rather than `git apply`: the DSA diff was cut against
# 0.5.15.post1 and needs one line of fuzz on v0.5.16, which git apply cannot do. The
# resulting edit was verified line by line against v0.5.16 — see the note in
# deploy/docker/patches/sglang_dsa/README.md.
set -euo pipefail

# find_spec rather than `import sglang`: importing it pulls in aiter, which costs ten
# seconds and a screenful of output before we have done anything.
SGLANG_ROOT="${SGLANG_ROOT:-$(python3 -c 'import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec("sglang").origin).parents[2])' 2>/dev/null || echo /sgl-workspace/sglang)}"
REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
PATCH_DIR="${PATCH_DIR:-$REPO/deploy/docker/patches}"

# name : diff path : file to probe : marker that only exists once patched
PATCHES=(
    "pd-kv-race:sglang_disagg/mooncake_early_send_wait_event.diff:python/sglang/srt/disaggregation/mooncake/conn.py:wait_event"
    "dsa-dp-rows:sglang_dsa/dsa_indexer_hip_dp_padded_rows.diff:python/sglang/srt/layers/attention/dsa/dsa_indexer.py:q_fp8_mqa"
)

is_applied() { grep -q "$2" "$SGLANG_ROOT/$1" 2>/dev/null; }

status() {
    printf 'sglang    : %s (%s)\n' "$SGLANG_ROOT" \
        "$(git -C "$SGLANG_ROOT" rev-parse --short HEAD 2>/dev/null || echo 'not a git tree')"
    local entry name file marker
    for entry in "${PATCHES[@]}"; do
        IFS=: read -r name _ file marker <<<"$entry"
        printf '%-12s: %s\n' "$name" \
            "$(is_applied "$file" "$marker" && echo present || echo ABSENT)"
    done
}

case "${1:-}" in
    --status) status; exit 0 ;;
    "") ;;
    *) echo "usage: bash $0 [--status]" >&2; exit 2 ;;
esac

[[ -d "$SGLANG_ROOT/python/sglang" ]] \
    || { echo "[sgl-patch] $SGLANG_ROOT is not an sglang source tree" >&2; exit 1; }

for entry in "${PATCHES[@]}"; do
    IFS=: read -r name diff file marker <<<"$entry"
    diff="$PATCH_DIR/$diff"
    [[ -f "$diff" ]] || { echo "[sgl-patch] $name: patch not found: $diff" >&2; exit 1; }

    if is_applied "$file" "$marker"; then
        echo "[sgl-patch] $name: already applied — skipping"
        continue
    fi
    # --reverse --dry-run exits 0 only if the patch is already in the tree, which
    # catches the case where the marker moved but the change is present.
    if patch -d "$SGLANG_ROOT" -p1 -R --dry-run --force <"$diff" >/dev/null 2>&1; then
        echo "[sgl-patch] $name: already applied (marker drifted) — skipping"
        continue
    fi
    if ! patch -d "$SGLANG_ROOT" -p1 --forward <"$diff"; then
        echo "[sgl-patch] $name: FAILED to apply — sglang drifted too far from the" \
             "version this diff was cut against. Re-cut it, do not force." >&2
        exit 1
    fi
    is_applied "$file" "$marker" \
        || { echo "[sgl-patch] $name: applied but marker '$marker' is missing" >&2; exit 1; }
    echo "[sgl-patch] $name: applied"
done

# Compile every file the diffs touch: a bad patch usually shows up as a SyntaxError
# here rather than twenty minutes later on the first request. py_compile rather than
# import because importing sglang drags in aiter and prints a screenful.
for entry in "${PATCHES[@]}"; do
    IFS=: read -r name diff _ _ <<<"$entry"
    while read -r file; do
        python3 -m py_compile "$SGLANG_ROOT/$file" \
            || { echo "[sgl-patch] $name: $file does not compile after patching" >&2; exit 1; }
    done < <(awk '/^\+\+\+ b\// { sub(/^\+\+\+ b\//, ""); print $1 }' "$PATCH_DIR/$diff")
done

echo "[sgl-patch] OK"
status
