#!/bin/bash
# Stage the kf P/D + calib kit onto /perf_apps (docker on openai nodes cannot read HOME NFS). Run on a16-19.
set -euo pipefail
WS=$(cd "$(dirname "$0")/.." && pwd)
ST=/perf_apps/yihou/dsv41_kf_20261120/pd
MUSE_STAGE=/home/yihou/dev/git/infera.muse/yihou/muse_agentx_opt_20261027/research/calib_profiler/stage
mkdir -p "$ST/logs" "$ST/calib/stage/wheels" "$ST/gem"
rsync -a --exclude logs --exclude hf-cache --exclude gem/ "$WS/pd/" "$ST/"
rsync -a "$WS/pd/gem/hip_gem_alloc.c" "$ST/gem/"
rsync -a --exclude muse_orig --exclude __pycache__ "$WS/scripts/calib/" "$ST/calib/" --exclude stage
[ -f "$ST/calib/stage/rocprof-sdk-src.tar.gz" ] || cp "$MUSE_STAGE/rocprof-sdk-src.tar.gz" "$ST/calib/stage/"
cp -n "$MUSE_STAGE"/wheels/*.whl "$ST/calib/stage/wheels/" 2>/dev/null || true
(cd "$ST/calib/stage" && sha256sum rocprof-sdk-src.tar.gz > SHA256.kf && diff <(cut -d' ' -f1 SHA256.kf) <(cut -d' ' -f1 "$MUSE_STAGE/SHA256"))
chmod -R go-w "$ST" && chmod 1777 "$ST/logs" "$ST/gem"
echo "staged to $ST"
