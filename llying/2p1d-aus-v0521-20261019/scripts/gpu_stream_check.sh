#!/usr/bin/env bash
# Per-GPU check that a torch HIP stream can be created and its handle read (the b1 prefill-0 schedulers on n02-25
# segfaulted in c10::hip::HIPStream::stream() when reading schedule_stream.cuda_stream). One container per GPU.
# Usage (on a node): gpu_stream_check.sh [IMAGE]
IMG="${1:-infera-sglang:v0521-llying-destpin-aus1019}"
T='import torch
torch.cuda.set_device(0)
x = torch.ones(1024, device="cuda")
s = torch.cuda.Stream(priority=0)
print("ok", hex(s.cuda_stream), torch.cuda.get_device_properties(0).name, float(x.sum()), flush=True)'
for g in 0 1 2 3 4 5 6 7; do
    out="$(timeout 120 docker run --rm --network none --device /dev/kfd --device /dev/dri --group-add video \
        --group-add render --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=$g --entrypoint python3 "$IMG" \
        -c "$T" 2>&1 | grep -v '<unknown>' | tail -1)"
    echo "$(hostname -s) HIP GPU$g: ${out:-<no output, crashed>}"
done
