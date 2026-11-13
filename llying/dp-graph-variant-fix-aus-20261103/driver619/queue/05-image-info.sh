#!/bin/bash
# DRIVER619-20261113: copy out of the production image what the GEM KV integration has to patch (SGLang custom mem
# pool hook) and what decides Mooncake's HIP registration path (KFD pin anchor present or not).
OUT=$K/out/image-info
mkdir -p "$OUT"
srun --jobid=$JOB --overlap -N1 -n1 -w $NODE616 bash -c "
docker image inspect infera-sglang:v0522-fdbafb60 >/dev/null 2>&1 || docker load -i /perf_apps/liyingli/images/infera-sglang-v0522-fdbafb60.tar >/dev/null
docker run --rm -v $OUT:/out --entrypoint bash infera-sglang:v0522-fdbafb60 -c '
S=\$(python3 -c \"import sglang,os;print(os.path.dirname(sglang.__file__))\")
echo sglang \$S > /out/info.txt
for f in srt/disaggregation/mooncake/utils.py srt/mem_cache/utils.py srt/environ.py srt/disaggregation/mooncake/conn.py; do
    [ -e \$S/\$f ] && mkdir -p /out/sglang/\$(dirname \$f) && cp \$S/\$f /out/sglang/\$f
done
M=\$(python3 -c \"import mooncake,os;print(os.path.dirname(mooncake.__file__))\")
echo mooncake \$M >> /out/info.txt
ls -la \$M >> /out/info.txt
for so in \$M/*.so; do echo \"== \$so\"; strings \$so | grep -E \"KFD pin probe|MC_HIP_KFD_PIN_FIX|HIP dmabuf disabled\" | sort -u | head; done >> /out/info.txt
python3 -c \"import torch;print(torch.__version__, torch.version.hip)\" >> /out/info.txt
cat /opt/rocm/.info/version >> /out/info.txt 2>/dev/null
command -v gcc hipcc >> /out/info.txt
ls /usr/include/infiniband/verbs.h /opt/rocm/include/hsa/hsa_ext_amd.h >> /out/info.txt 2>&1
'
docker image inspect -f 'entrypoint {{json .Config.Entrypoint}} cmd {{json .Config.Cmd}}' infera-sglang:v0522-fdbafb60 >> $OUT/info.txt 2>&1
"
cat "$OUT/info.txt"
