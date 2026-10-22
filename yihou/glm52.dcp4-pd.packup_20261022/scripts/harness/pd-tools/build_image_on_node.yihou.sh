#!/usr/bin/env bash
# Build + verify infera-sglang:v0522-1020-yihou-dcp-mc4340 on THIS node (run via spur exec <job> on the node),
# then record node<TAB>image<TAB>id in PD/harness/image_ids.dcp.tsv (read by config.yihou.dcp.sh).
# CPU only; needs network (base pull, Mooncake clone, libionic, cargo). ~3-5 min with the base pulled.
set -euo pipefail
PD=/shared_nfs/yihou_glm52/dcp_pd_20261021/pd
SRC=$PD/build/infera-src.yihou            # branch yihou.dcp.v0522.20261021 (80458804)
IMG=infera-sglang:v0522-1020-yihou-dcp-mc4340
BASE=lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261020
BASE_DIGEST=sha256:3a75c91365cc4f4e1194bf5d64b92f2f1ef134d8b9045da165fba28a8dada464
node=$(hostname -s); ts=$(date -u +%Y%m%dT%H%M%SZ); log=$PD/logs/build.$node.$ts.log
echo "node=$node log=$log"
git -C "$SRC" log --oneline -1 | tee "$log"
docker pull "$BASE@$BASE_DIGEST" >>"$log" 2>&1 && docker tag "$BASE@$BASE_DIGEST" "$BASE"
(cd "$SRC" && docker build -f deploy/docker/Dockerfile.sglang.v0522-1020-mc4340.yihou \
   --build-arg BUILD_MOONCAKE=1 --build-arg REQUIRE_MOONCAKE_CROSS_HOST_ROUTING=1 --progress=plain -t "$IMG" .) >>"$log" 2>&1
docker run --rm --network none --name "yihou-dcp-pdrun-verify-$node" -v "$PD/tools:/tools:ro" --entrypoint bash "$IMG" \
   -c 'bash /tools/verify/verify_v0521.sh; a=$?; bash /tools/verify/verify_mc4340.sh; b=$?; exit $((a|b))' >>"$log" 2>&1 ||
   { echo "VERIFY FAILED (see $log)"; exit 1; }
id=$(docker image inspect --format '{{.Id}}' "$IMG")
printf '%s\t%s\t%s\n' "$node" "$IMG" "$id" >> "$PD/harness/image_ids.dcp.tsv"
echo "OK $node $IMG $id"
