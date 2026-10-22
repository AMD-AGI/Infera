#!/bin/bash
# build_on_node.yihou.sh: run ON the build node (spur exec <job> bash <this>). Builds
# infera-sglang:v0522-d395c3b3-1020-mc4340 from the prepared context and docker-saves it to NFS;
# also extracts the ab7a898b77 overlay if it is not there yet. Creates files only under yihou paths.
set -euo pipefail
LL=/shared_nfs/yihou_glm52/llying_dcp_20261022
SRCROOT=/shared_nfs/yihou_glm52/dcp_pd_20261021/src
IN=$LL/build/inputs                     # llimage-ctx.*.tar.gz + sglang-src-ab7a898b77.tar.gz copied here
CTX=$LL/build/ctx-d395c3b3-mc4340
TAG=infera-sglang:v0522-d395c3b3-1020-mc4340
BASE=lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261020
OUT=$LL/images/infera-sglang_v0522-d395c3b3-1020-mc4340.tar
LOG=$LL/build/build.$(hostname -s).$(date -u +%Y%m%dT%H%M%SZ).log
mkdir -p $LL/images $LL/build
exec > >(tee -a "$LOG") 2>&1
echo "# $(date -u +%FT%TZ) host=$(hostname) tag=$TAG"

# (a) overlay
if [ ! -e $SRCROOT/sglang-ab7a898b77 ]; then
  mkdir -p $SRCROOT/sglang-ab7a898b77
  tar -xzf $IN/sglang-src-ab7a898b77.tar.gz -C $SRCROOT/sglang-ab7a898b77
fi
ls $SRCROOT/sglang-ab7a898b77/python/sglang/srt/models/deepseek_v2.py

# (b) image
if [ ! -e $CTX ]; then
  mkdir -p $CTX
  tar -xzf $IN/llimage-ctx.d395c3b3-mc4340.tar.gz -C $CTX --strip-components=1
fi
docker image inspect $BASE >/dev/null 2>&1 || docker pull $BASE
docker build --network host --progress plain \
  -f $CTX/deploy/docker/Dockerfile.sglang.v0522-d395c3b3-1020-mc4340.yihou -t $TAG $CTX
docker image inspect $TAG --format '{{.Id}} {{.Created}}'

# artifact checks in a throwaway container named under the guarded prefix
docker run --rm --init --name yihou-dcp-pd-151-imgcheck --entrypoint bash $TAG -c '
  set -e
  so=$(python3 -c "import mooncake.engine as e; print(e.__file__)")
  for s in MC_ENABLE_DEST_LOCAL_RAIL MC_IB_TIMEOUT MC_DISABLE_HIP "extends past its allocation"; do
    strings "$so" | grep -q "$s" && echo "OK $s" || { echo "MISSING $s"; exit 1; }; done
  python3 -c "from mooncake.engine import TransferEngine; import infera, sglang; print(\"import ok\", sglang.__version__)"
  /usr/local/bin/infera-router --help >/dev/null && echo "router ok"
  n=$(python3 -m infera.server --help 2>&1 | grep -cE -- "--session-affinity |--pd-prefill-guard-release|--decode-input-demand"); echo "router flags in infera.server --help: $n/3"; /usr/local/bin/infera-router --help 2>&1 | grep -cE "decode.input.demand|prefill.guard.release|session.affinity" || true'

docker save -o $OUT.partial $TAG
mv $OUT.partial $OUT
sha256sum $OUT | tee $OUT.sha256
echo "# DONE $(date -u +%FT%TZ)"
