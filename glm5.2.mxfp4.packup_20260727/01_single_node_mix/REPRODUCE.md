# Reproduction kit — 01 single-node mix

Goal: coherent GLM-5.2-MXFP4 on one node + conc=64 pass. Est. time ~10 min (~5 min cold start).

## 0. Prerequisites

- **Machine:** one 8× MI355X node (we used chi2879 = 10.2.122.10). Reachable via jump host
  `root@149.28.124.225` then `ssh chi2879`.
- **Secrets:** cluster SSH only (see ../environment.md). rc6 image is public, no registry login.
- **External deps (absolute paths):**
  - Model: `/mnt/vast/xiaobo/models/GLM-5.2-MXFP4` (shared VAST mount).
- **Image:** `rocm/infera:sglang-v0.1.0-rc6` (digest in ../environment.md).

## 1. Launch the server (on the node)

Copy `scripts/launch.sh` to the node and run it (or run inline). It `docker run`s the rc6 image
with the DSA-ROCm envs and `sglang.launch_server`, TP8, max-running-requests 64.

    # from the node (chi2879):
    bash launch.sh
    # cold start ~5 min (282 shards warm in NFS) — weights load, JIT, cudagraph capture.
    # wait for: docker logs glm52-mix-p1  ->  "The server is fired up and ready to roll!"

Poll readiness:

    curl -sf -o /dev/null -w '%{http_code}\n' -m5 http://127.0.0.1:30000/health   # want 200

## 2. Verify correctness (temp=0 probe) — DO NOT skip

A 200/health says nothing about output correctness. Run the factual probe:

    docker cp probe.py glm52-mix-p1:/tmp/probe.py
    docker exec glm52-mix-p1 python3 /tmp/probe.py http://127.0.0.1:30000 glm5.2-mxfp4
    # expect: 4/4 correct (France->Paris, China->Beijing, 2+2->4, largest planet->Jupiter)

## 3. Run the conc=64 stress test (1k/1k)

    NAME=glm52-mix-p1 bash bench.sh
    # or inline: docker exec glm52-mix-p1 python3 -m sglang.bench_serving --backend sglang-oai \
    #   --base-url http://127.0.0.1:30000 --model /mnt/vast/xiaobo/models/GLM-5.2-MXFP4 \
    #   --tokenizer /mnt/vast/xiaobo/models/GLM-5.2-MXFP4 --dataset-name random \
    #   --random-input-len 1024 --random-output-len 1024 --random-range-ratio 1.0 \
    #   --max-concurrency 64 --num-prompts 256 --warmup-requests 64 --request-rate inf

## Expected output

- Probe: `4/4 correct`.
- Bench: `Successful requests: 256`, total throughput ~4600 tok/s, median TPOT ~22 ms.
  See `results/bench_conc64.txt`.

## If it doesn't reproduce

See `notes.md`. Most common: mistaking the ~5-min silent JIT/weight-load window for a hang
(it's not — watch VRAM climb to ~267 GB/card). Ensure the 3 DSA envs are set or the DSA topk
tries a CUDA kernel that won't build on gfx950.
