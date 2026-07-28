# Reproduction kit — 03 PD mooncake RDMA (pd-unified image)

Goal: coherent GLM-5.2 over mooncake RDMA across 2 nodes + conc=64 pass. Est. ~20 min (image
distribution + ~5 min cold start).

## 0. Prerequisites

- **Machines:** 2× 8-MI355X, same 10.2.122.x RoCE subnet.
  - prefill = chi2878 (10.2.122.3), decode = chi2879 (10.2.122.10). NIC `enp193s0f1np1`.
  - Kernel needs `CONFIG_PCI_P2PDMA` + `ib_peer_mem` loaded (both true on chi2878/chi2879 — the
    peermem KV path needs them). Verify: `lsmod | grep ib_peer_mem`.
- **Image:** `infera/engine-sglang:pd-unified` (PR #19 build). It is a LOCAL build — NOT on a public
  registry. Get it onto both nodes:
  - If a node already has it (chi2798/chi2878 did), great.
  - To copy node→node WITHOUT the slow NFS `docker save` (78 GB over NFS is very slow): stream it —
    `ssh <src> 'docker save infera/engine-sglang:pd-unified' | ssh <dst> 'docker load'` (direct pipe
    over the data-plane; ~5 min). Or `docker build` it on each node from the Infera repo
    (`deploy/docker/Dockerfile.sglang`, branch of PR #19).
- **Model:** `/mnt/vast/xiaobo/models/GLM-5.2-MXFP4`.
- **Host libionic:** injected into the container (else libibverbs enumerates 0 RDMA devices). `up.sh`
  does this; verify `ibv_devinfo | grep -c PORT_ACTIVE` → 8.
- **Fabric pre-check:** `SERVER=chi2878 CLIENT=chi2879 DEV=ionic_0 GIDIDX=1 DUR=5 bash
  <cluster_tools>/rail_test.sh` → ~335 Gb/s.

## 1. Stage the leg script + bring up both legs

    scp scripts/pd_leg.sh <jump>:/mnt/vast/c_huggingface/glm52_p2b/pd_leg.sh
    CONC=64 DMABUF=0 bash scripts/up.sh
    # up.sh: recreate pd_uni container + inject libionic on both nodes, copy pd_leg.sh in, launch
    #   prefill (chi2878 :30000) + decode (chi2879 :30001), mooncake, dmabuf OFF, all-8-ionic.
    # NO MC_FORCE_TCP — the pd-unified image defaults MC_DISABLE_HIP_TRANSPORT=1 → real RDMA.
    # cold start ~5 min. Wait for BOTH legs to log "The server is fired up and ready to roll!".

Confirm the RDMA path (not TCP):

    ssh chi2878 'grep "rdma_context.cpp.*HIP dmabuf disabled" \
       /mnt/vast/c_huggingface/glm52_p2b/pd_prefill_30000.log | head'
    # want 8 lines (one per NIC) = mooncake using the RDMA rdma_context path, dmabuf off.

## 2. Start the router (sglang_router mini-LB, in-container)

    ssh chi2878 'docker exec -d pd_uni bash -lc "pkill -9 -f sglang_router; sleep 3; \
       python3 -m sglang_router.launch_router --pd-disaggregation \
       --prefill http://10.2.122.3:30000 8998 --decode http://10.2.122.10:30001 \
       --host 0.0.0.0 --port 8002 > /tmp/router.log 2>&1"'
    sleep 16
    ssh chi2878 'curl -s http://10.2.122.3:8002/workers'   # expect prefill + decode, is_healthy true

## 3. Verify correctness

    ssh chi2878 'docker cp /mnt/vast/c_huggingface/glm52_probe.py pd_uni:/tmp/probe.py \
       && docker exec pd_uni python3 /tmp/probe.py http://10.2.122.3:8002 glm5.2-mxfp4'
    # expect 4/4 correct. (Use probe.py / urllib — inline curl JSON gets mangled through nested ssh.)

## 4. Run conc=64 (1k/1k)

    ssh chi2878 'docker exec pd_uni bash -lc "python3 -m sglang.bench_serving --backend sglang-oai \
       --base-url http://10.2.122.3:8002 --model glm5.2-mxfp4 \
       --tokenizer /mnt/vast/xiaobo/models/GLM-5.2-MXFP4 --dataset-name random \
       --random-input-len 1024 --random-output-len 1024 --random-range-ratio 1.0 \
       --max-concurrency 64 --num-prompts 256 --warmup-requests 32 --request-rate inf"'

## Expected output

- 8× "HIP dmabuf disabled" in the prefill log (RDMA confirmed), probe 4/4, bench 256/256, total
  ~5150 tok/s, median TTFT ~543 ms, TPOT ~21 ms, zero transfer errors. See `results/bench_conc64.txt`.

## If it doesn't reproduce

See `notes.md`. Do NOT set `MC_FORCE_TCP` (that's the slow fallback). If you see
`hipIpcOpenMemHandle` errors, you're not on the pd-unified image (HIP transport wasn't gated off).
Router false-circuit-break after a failed round: fully `pkill -9 -f sglang_router` before relaunch.
