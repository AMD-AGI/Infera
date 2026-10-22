#!/bin/bash
# bigalloc_test.sh GPU: allocate 4 GiB chunks up to 200 GiB on one GPU, H2D-copy (pageable) into each, print progress
G=${1:-7}
timeout 400 docker run --rm --name yihou-dcp-bigalloc --device /dev/kfd --device /dev/dri --group-add video --ipc host -e HIP_VISIBLE_DEVICES=$G -w /tmp --entrypoint python3 lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261020 -c '
import torch, time
x=torch.ones(1024,1024,1024,dtype=torch.float32)  # 4 GiB pageable
keep=[]; t=time.time()
for i in range(50):
    y=torch.empty_like(x,device="cuda"); y.copy_(x); torch.cuda.synchronize()
    keep.append(y)
    print("GiB", 4*(i+1), "ok", round(time.time()-t,1), "s", flush=True)
'
echo rc=$?
