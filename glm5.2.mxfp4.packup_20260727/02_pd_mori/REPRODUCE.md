# Reproduction kit — 02 PD mori (1P1D over MoRI RDMA)

Goal: coherent GLM-5.2 across 2 nodes over MoRI RDMA + conc=64 pass. Est. ~15 min (~5 min cold start).

## 0. Prerequisites

- **Machines:** 2× 8-MI355X nodes, same 10.2.122.x RoCE subnet.
  - prefill = chi2878 (10.2.122.3), decode = chi2879 (10.2.122.10). Data-plane NIC `enp193s0f1np1`.
  - Orchestrate from a host that can `ssh` both (we drive via jump host `root@149.28.124.225`).
- **Secrets:** cluster SSH only. rc6 image public.
- **External deps (absolute paths):**
  - Model: `/mnt/vast/xiaobo/models/GLM-5.2-MXFP4`.
  - Host libionic (injected into container for RDMA visibility): the node's
    `/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so` (engine.sh mounts it — do not skip).
  - `scripts/engine.sh` must live on a shared-fs path identical on both nodes; we staged it to
    `/mnt/vast/c_huggingface/glm52_p3/engine.sh`. Adjust `KIT` in `up.sh` if different.
- **Image:** `rocm/infera:sglang-v0.1.0-rc6`.
- **Fabric pre-check (do once):** confirm the rail + GID before trusting transfer:
  `SERVER=chi2878 CLIENT=chi2879 DEV=ionic_0 GIDIDX=1 DUR=5 bash <cluster_tools>/rail_test.sh`
  → expect ~335 Gb/s. All 8 ionic NICs on both nodes carry the routable RoCE-v2 GID at index 1.

## 1. Stage engine.sh + bring up the whole stack

Edit `scripts/pd_env.sh` if your node IPs differ, then from the orchestrating host:

    # stage the leg launcher onto the shared fs (both nodes read it):
    scp scripts/engine.sh <jump>:/mnt/vast/c_huggingface/glm52_p3/engine.sh
    # bring up NATS(both) + etcd + router(prefill) + prefill leg + decode leg:
    CONC=64 bash scripts/up.sh

`up.sh` starts, in order: NATS broker on both nodes → etcd on prefill → `infera.server` router
(`:8100`, **`--request-transport http --router-policy kv-aware`**) on prefill → prefill leg
(chi2878) → decode leg (chi2879). Legs cold-start ~5 min.

## 2. Wait for readiness + PD pairing

    # router health + both workers registered:
    curl -sf -o /dev/null -w '%{http_code}\n' http://10.2.122.3:8100/health          # 200
    ssh chi2878 'docker exec repro-etcd-p3 etcdctl --endpoints http://10.2.122.3:2379 \
       get --prefix /infera/workers/'      # expect 2 workers: one prefill, one decode

## 3. Verify correctness through the router

    ssh chi2878 'docker cp /mnt/vast/c_huggingface/glm52_probe.py glm52-router-p3:/tmp/probe.py \
      && docker exec glm52-router-p3 python3 /tmp/probe.py http://10.2.122.3:8100 glm5.2-mxfp4'
    # expect 4/4 correct

## 4. Run the conc=64 stress test (1k/1k)

    ssh chi2878 'docker exec glm52-router-p3 bash -lc "python3 -m sglang.bench_serving \
      --backend sglang-oai --base-url http://10.2.122.3:8100 --model glm5.2-mxfp4 \
      --tokenizer /mnt/vast/xiaobo/models/GLM-5.2-MXFP4 --dataset-name random \
      --random-input-len 1024 --random-output-len 1024 --random-range-ratio 1.0 \
      --max-concurrency 64 --num-prompts 256 --warmup-requests 32 --request-rate inf"'

## Expected output

- Probe 4/4; bench `Successful requests: 256`, total ~5100 tok/s, median TTFT ~535 ms,
  TPOT ~21 ms (stable — PD isolates decode from prefill). See `results/bench_conc64.txt`.

## If it doesn't reproduce

See `notes.md` — the two router-flag bugs (do NOT pass `--kv-events off` to the router; use
`--request-transport http` not `nats`), the NATS-broker-per-node requirement, the libionic mount,
and the between-runs RDMA reset ritual (drain VRAM to idle before relaunch).
