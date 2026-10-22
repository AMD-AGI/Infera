IMG=lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261020
SRC=/shared_nfs/yihou_glm52/dcp_pd_20261021/src/sglang-4792dc0e52/python/sglang
ls -la $SRC | head -5
docker run --rm --network none --device /dev/kfd --device /dev/dri -v $SRC:/sgl-workspace/sglang/python/sglang:ro --entrypoint bash $IMG -c '
ls -la /sgl-workspace/sglang/python/sglang | head -5
cd /tmp && python3 -c "import sglang;print(sglang.__file__, sglang.__path__)"
python3 -c "import sglang.srt.layers.attention.dsa_backend as m; print(\"dsa ok\", m.__file__)" 2>&1 | tail -3
python3 -c "import sglang.srt.entrypoints.http_server; print(\"http ok\")" 2>&1 | tail -3
grep -n "is_hip" /sgl-workspace/sglang/python/sglang/srt/arg_groups/model_overrides/deepseek_v2.py | head -3
'
