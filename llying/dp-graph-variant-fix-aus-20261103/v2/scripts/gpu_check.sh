#!/usr/bin/env bash
# One small torch kernel per GPU in the v2 image (11-07: n02-25 prefill rank 1 segfaulted in its first fill kernel twice).
# Usage: gpu_check.sh [GPU...]   (default 0-7)
IMAGE=${IMAGE:-infera-sglang:v0522-main-44a294f8}
for i in "${@:-0 1 2 3 4 5 6 7}"; do
    for g in $i; do
        r="$(docker run --rm --network host --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined \
            -e HIP_VISIBLE_DEVICES="$g" "$IMAGE" \
            python3 -c 'import torch; x = torch.ones(1 << 20, device="cuda"); torch.cuda.synchronize(); print("ok", int(x.sum().item()))' 2>&1 | tail -1)"
        echo "GPU $g: rc=$? $r"
    done
done
