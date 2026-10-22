#!/bin/bash
# Inspect image: sglang install, commit, versions; then overlay check.
IMG=lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261020
WS=/shared_nfs/yihou_glm52/dcp_pd_20261021
SRC=$WS/src/sglang-4792dc0e52/python/sglang
docker image inspect --format '{{.Id}} {{.Created}}' $IMG
docker run --rm --network none --entrypoint bash $IMG -c '
cd /sgl-workspace/sglang 2>/dev/null && git log -1 --format="%H %cd %s" 2>&1 | head -2
pip show sglang 2>/dev/null | grep -E "Version|Location|Editable"
python3 -c "import sglang,os;print(\"sglang\",sglang.__file__)"
pip list 2>/dev/null | grep -i -E "aiter|triton|torch |^torch|flydsl|mooncake|sgl"
cd /sgl-workspace/sglang && git log --oneline -1 --grep=43735 ; git log --oneline | grep -c . ; git merge-base --is-ancestor 8bf145d119 HEAD && echo has43735 || echo no43735
ls /sgl-workspace
'
echo ==== overlay
docker run --rm --network none -v $SRC:/sgl-workspace/sglang/python/sglang:ro --entrypoint bash $IMG -c '
python3 -c "import sglang,os;print(\"sglang\",sglang.__file__)"
python3 -c "import sglang.srt.layers.attention.dsa_backend as m; print(\"dsa ok\", m.__file__)" 2>&1 | tail -3
python3 -c "import sglang.srt.entrypoints.http_server" 2>&1 | tail -3
python3 -m sglang.launch_server --help 2>&1 | grep -E "dcp-size|dsa-prefill" | head
'
