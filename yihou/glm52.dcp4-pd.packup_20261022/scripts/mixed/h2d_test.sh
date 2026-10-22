#!/bin/bash
# h2d_test.sh [CPUSET]: torch H2D copy smoke on all 8 GPUs (threads, like the weight loader)
CS=${1:-}
args=(); [[ -n $CS ]] && args=(--cpuset-cpus $CS)
timeout 300 docker run --rm --name yihou-dcp-h2dtest "${args[@]}" --device /dev/kfd --device /dev/dri --group-add video --ipc host -w /tmp --entrypoint python3 lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261020 -c '
import torch, time, concurrent.futures as cf
t=time.time()
print("devices", torch.cuda.device_count(), flush=True)
def one(i):
    d=torch.device("cuda",i%torch.cuda.device_count())
    x=torch.randn(256,1024,1024)  # 1 GiB fp32 pageable
    y=torch.empty_like(x,device=d); y.copy_(x); torch.cuda.synchronize(d)
    return i, float(y[0,0,0].item()==x[0,0,0].item())
with cf.ThreadPoolExecutor(16) as ex:
    for r in ex.map(one, range(32)): pass
print("OK 32x1GiB H2D in", round(time.time()-t,1),"s", flush=True)
'
echo "rc=$?"
