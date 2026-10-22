#!/bin/bash
# threaded_copy_test.sh NSHARDS [MODE=mmap|clone] [EXTRA docker -e ...]: loader-like threaded H2D copies (ThreadPoolExecutor default
# workers) of safetensors tensors mmap'd from NFS into preallocated GPU tensors. Prints progress; timeout = hang.
N=${1:-8}; MODE=${2:-mmap}; shift 2
timeout 300 docker run --rm --name yihou-dcp-thrtest-${G:-7} --device /dev/kfd --device /dev/dri --group-add video --ipc host -e HIP_VISIBLE_DEVICES=${G:-7} "$@" \
  -v /shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4:/m:ro -w /tmp --entrypoint python3 lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261020 -c '
import torch, time, glob, sys, concurrent.futures as cf
from safetensors import safe_open
N=int(sys.argv[1]); mode=sys.argv[2]
files=sorted(glob.glob("/m/*.safetensors"))[100:100+N]
torch.cuda.set_device(0)
ex=cf.ThreadPoolExecutor(); print("workers", ex._max_workers, flush=True)
futs=[]; n=0; t0=time.time(); handles=[]
def cp(dst, src):
    torch.cuda.set_device(0); dst.copy_(src); return src.numel()*src.element_size()
for f in files:
    s=safe_open(f, framework="pt"); handles.append(s)
    for k in s.keys():
        t=s.get_tensor(k)
        if mode=="clone": t=t.clone()
        d=torch.empty(t.shape, dtype=t.dtype, device="cuda")
        futs.append(ex.submit(cp, d, t)); n+=1
print("submitted", n, "tensors from", N, "shards in", round(time.time()-t0,1), "s", flush=True)
done=0; tot=0
for fu in cf.as_completed(futs):
    tot+=fu.result(); done+=1
    if done % 500 == 0: print("done", done, tot>>20, "MiB", round(time.time()-t0,1), "s", flush=True)
torch.cuda.synchronize(); print("ALL DONE", done, tot>>20, "MiB", round(time.time()-t0,1), "s", flush=True)
' $N $MODE
echo rc=$?
