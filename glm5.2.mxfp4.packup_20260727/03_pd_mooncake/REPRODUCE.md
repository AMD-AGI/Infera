# Reproduction kit — 03 PD mooncake (TCP: correct output; conc=64: fails)

Goal: reproduce (a) correct GLM-5.2 output over mooncake TCP, and (b) the conc=64 failure.
Est. ~15 min (~5 min cold start).

## 0. Prerequisites

Same as 02_pd_mori (same 2 nodes chi2878/chi2879, same model, same image, same libionic mount,
same NATS/etcd/router). The only differences are the transport backend + MC_* envs (in `engine.sh`).

- `scripts/engine.sh` staged to a shared path (we used `/mnt/vast/c_huggingface/glm52_p2/engine.sh`).
- **MODE selector** in `engine.sh`:
  - `MODE=tcp` (default) → `MC_FORCE_TCP=1 MC_DISABLE_HIP_TRANSPORT=1 MC_GID_INDEX=1
    MC_IB_GID_INDEX=1 MOONCAKE_DISABLE_HIP_DMABUF=1`. **This is the only path that yields correct
    output on this stack.**
  - `MODE=rdma` → attempts RDMA (see notes.md — expected to hit a driver segfault wall; we did not
    run this on our nodes, only carried the option).

## 1. Bring up the stack (TCP mode)

    scp scripts/engine.sh <jump>:/mnt/vast/c_huggingface/glm52_p2/engine.sh
    MODE=tcp CONC=64 bash scripts/up.sh
    # NATS(both) + etcd + router(http,kv-aware) + prefill leg + decode leg. Cold start ~5 min.
    # decode log will show "TcpTransport: listen on port ..." confirming TCP transport.

## 2. Wait for readiness

    curl -sf -o /dev/null -w '%{http_code}\n' http://10.2.122.3:8100/health   # 200
    # both legs: grep "fired up and ready to roll" in the leg logs.

## 3. Verify correctness (this PASSES)

    ssh chi2878 'docker cp /mnt/vast/c_huggingface/glm52_probe.py glm52-router-p2:/tmp/probe.py \
      && docker exec glm52-router-p2 python3 /tmp/probe.py http://10.2.122.3:8100 glm5.2-mxfp4'
    # expect 4/4 correct — output IS coherent over mooncake TCP.

## 4. Run conc=64 (this FAILS — the point of keeping this kit)

    ssh chi2878 'docker exec glm52-router-p2 bash -lc "python3 -m sglang.bench_serving \
      --backend sglang-oai --base-url http://10.2.122.3:8100 --model glm5.2-mxfp4 \
      --tokenizer /mnt/vast/xiaobo/models/GLM-5.2-MXFP4 --dataset-name random \
      --random-input-len 1024 --random-output-len 1024 --random-range-ratio 1.0 \
      --max-concurrency 64 --num-prompts 256 --warmup-requests 32 --request-rate inf"'

## Expected output

- Probe: 4/4 correct.
- Bench: **only ~50/256 successful**, median TTFT ~50 s, total ~886 tok/s. The prefill log floods
  with `KVTransferError(...): Decode instance could be dead, remote mooncake session <ip:port> is
  not alive` — the TCP KV-transfer sessions drop under concurrent load. See
  `results/bench_conc64_FAIL.txt` and `logs/prefill.log`.

## If you want the pass instead

Use **02_pd_mori** (MoRI RDMA) — same workload passes conc=64 at 5167 tok/s. mooncake RDMA on this
stack is a driver dead-end (notes.md).
