#!/bin/bash
# mmap_copy_test.sh: H2D copy of safetensors tensors read via mmap from NFS (file-backed pageable) vs cloned (anonymous).
timeout 240 docker run --rm --name yihou-dcp-mmaptest --device /dev/kfd --device /dev/dri --group-add video --ipc host -e HIP_VISIBLE_DEVICES=7 \
  -v /shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4:/m:ro -w /tmp --entrypoint python3 lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261020 -c '
import torch, time, glob, sys
from safetensors import safe_open
files=sorted(glob.glob("/m/*.safetensors"))
f=files[len(files)//2]
print("file", f, flush=True)
with safe_open(f, framework="pt") as s:
    keys=list(s.keys())
    big=sorted(keys, key=lambda k: -s.get_slice(k).get_shape()[0])[:3]
    for mode in ("clone","mmap"):
        for k in big:
            t=s.get_tensor(k)
            if mode=="clone": t=t.clone()
            t0=time.time(); y=t.to("cuda"); torch.cuda.synchronize()
            print(mode, k, tuple(t.shape), t.dtype, t.numel()*t.element_size()>>20, "MiB", round(time.time()-t0,3), "s", flush=True)
print("DONE", flush=True)
'
echo rc=$?
