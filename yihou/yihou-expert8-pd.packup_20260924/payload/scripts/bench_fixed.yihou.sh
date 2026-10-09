#!/usr/bin/env bash
# Run exact-length synthetic load in a retained client container; preserve exit status.
set -euo pipefail
W=/home/yihou/dev/git/infera.glm52.pd/bench/glm5p2_pd/results/yihou-expert8-pd
source "$W/scripts/config.yihou.expert8.sh"
OUT="${1:?usage: bench_fixed.yihou.sh OUTPUT_DIR [client arguments]}"
shift
[[ "$OUT" == "$W/"* ]] || { printf 'output outside workspace\n' >&2; exit 2; }
mkdir -p "$(dirname "$OUT")" "$W/cache/client"
NAME="yihou-expert8-client-$(date -u +%Y%m%dT%H%M%S)"
args=(docker run --name "$NAME" --network host -v "$W:$W" -w "$W"
    --tmpfs "$W/tmp/client:rw,exec,size=2g" -e "TMPDIR=$W/tmp/client"
    -e "XDG_CACHE_HOME=$W/cache/client" -e PYTHONDONTWRITEBYTECODE=1
    "$IMAGE" python3 -u "$W/scripts/bench_fixed.yihou.py" --out "$OUT" "$@")
printf -v remote '%q ' "${args[@]}"
set +e
ssh -o BatchMode=yes -o ClearAllForwardings=yes "$CONTROL_NODE" "$remote" >"${OUT}.log" 2>&1
rc=$?
set -e
printf '%s\n' "$rc" >"${OUT}.exit-code"
printf '%s\n' "$NAME" >"${OUT}.container"
exit "$rc"
