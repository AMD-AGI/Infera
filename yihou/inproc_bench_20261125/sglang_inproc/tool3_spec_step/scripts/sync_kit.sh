#!/usr/bin/env bash
# Copy this kit to a host directory the node's docker daemon can read, with the provenance run_decode.sh records
# (git HEAD + uncommitted diff of the kit, since the copy has no .git). Usage: sync_kit.sh [host:]/data/yihou/.../kit_t3
# Staging files are overwritten in place under /var/tmp/yihou/t3_sync (nothing is deleted).
set -euo pipefail
DEST=${1:?usage: sync_kit.sh [host:]/path/with/yihou/kit}
case "$DEST" in *yihou*) ;; *) echo "destination must be under a yihou dir" >&2; exit 2;; esac
KIT_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
STAGE=/var/tmp/yihou/t3_sync
mkdir -p "$STAGE"
git -C "$KIT_ROOT" rev-parse HEAD > "$STAGE/GIT_HEAD" 2>/dev/null || echo unknown > "$STAGE/GIT_HEAD"
{ git -C "$KIT_ROOT" diff --binary -- . ; git -C "$KIT_ROOT" ls-files --others --exclude-standard -- . | sed 's/^/untracked: /'; } > "$STAGE/CODE_DIFF" 2>/dev/null || true
rsync -a --exclude research --exclude __pycache__ "$KIT_ROOT/" "$DEST/"
rsync -a "$STAGE/GIT_HEAD" "$STAGE/CODE_DIFF" "$DEST/"
echo "synced $KIT_ROOT -> $DEST"
