#!/usr/bin/env bash
# prep_llr (task llrepro 2026-10-23, ON 135): install llying's kit (Infera 06796e13 examples/sglang_2p1d_glm5.2 +
# config/kit-v6-pp4.diff) under the yihou NFS dir, build patches/deepseek_v2.py from the fdbafb60 image exactly as llying
# did (image file + his 2-hunk diff), drop --rm from the AgentX client (we keep stopped containers until checked),
# pre-seed InferenceX 918524ff.
set -euo pipefail
D=/shared_nfs/yihou_glm52/llrepro_20261023; K=$D/kit/sglang_2p1d_glm5.2
[[ -d $K ]] || { mkdir -p $D/kit; tar -xzf $D/kit1.tgz -C $D/kit; }
mkdir -p $K/patches
docker run --rm --entrypoint cat infera-sglang:v0522-fdbafb60 /sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py > $K/patches/deepseek_v2.py.orig
cp $K/patches/deepseek_v2.py.orig $K/patches/deepseek_v2.py
patch --no-backup-if-mismatch $K/patches/deepseek_v2.py < $D/ds.patch
grep -n "config.hidden_size == 7168\|                    config.hidden_size," $K/patches/deepseek_v2.py
sed -i 's/^docker run --rm --name "\$CTR-agentx"/docker run --name "$CTR-agentx"/' $K/engine/agentx/agentx.sh
grep -n 'docker run --name "$CTR-agentx"' $K/engine/agentx/agentx.sh
mkdir -p $D/work/cache
[[ -d $D/work/cache/InferenceX/.git ]] || cp -a /shared_nfs/yihou_glm52/dcp_pd_20261021/cache/InferenceX $D/work/cache/InferenceX
git -C $D/work/cache/InferenceX rev-parse HEAD
ssh -o BatchMode=yes crsuse2-m2m-136 "ls $K/engine/worker.sh && docker image inspect infera-sglang:v0522-fdbafb60 --format {{.Id}}"
docker image inspect infera-sglang:v0522-fdbafb60 --format {{.Id}}
echo PREP_LLR_DONE
