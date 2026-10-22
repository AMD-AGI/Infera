# Task g4tune 2026-10-19: raw GPU->GPU copy bandwidth (torch peer copy) for prefill GPUs 0-3 -> decode GPUs 4-7.
import torch, time
def bw(src, dst, mb, reps=5, blocks=1):
    n = mb * 1024 * 1024
    a = torch.empty(n, dtype=torch.uint8, device=f"cuda:{src}")
    b = torch.empty(n, dtype=torch.uint8, device=f"cuda:{dst}")
    step = n // blocks
    for _ in range(2):
        b.copy_(a)
    torch.cuda.synchronize(src); torch.cuda.synchronize(dst)
    t = time.time()
    for _ in range(reps):
        for i in range(blocks):
            b[i*step:(i+1)*step].copy_(a[i*step:(i+1)*step], non_blocking=True)
    torch.cuda.synchronize(src); torch.cuda.synchronize(dst)
    return reps * n / (time.time() - t) / 1e9
for s, d in [(0, 4), (1, 5), (2, 6), (3, 7), (0, 1)]:
    print(f"GPU{s}->GPU{d} 4GiB 1 block: {bw(s, d, 4096):.1f} GB/s ; 4GiB in 10000 blocks: {bw(s, d, 4096, reps=2, blocks=10000):.1f} GB/s", flush=True)
