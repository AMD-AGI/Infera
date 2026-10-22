V=/shared_nfs/yihou_glm52/dcp_pd_20261021/mixrun/venv-sgleval
docker run --rm --name yihou-dcp-sgleval-setup --network host -v /shared_nfs/yihou_glm52/dcp_pd_20261021:/shared_nfs/yihou_glm52/dcp_pd_20261021 -w /tmp --entrypoint bash lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261020 -c "
/usr/bin/python3 -m venv $V 2>&1 | tail -2 || python3 -m venv --without-pip $V
$V/bin/python -m pip install -q sgl-eval==0.1.3 2>&1 | tail -3
$V/bin/sgl-eval --help | head -30
$V/bin/sgl-eval run --help | head -80
"
