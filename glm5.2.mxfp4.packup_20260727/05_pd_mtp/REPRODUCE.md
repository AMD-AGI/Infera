# Reproduction kit — 05 PD MTP (1P1D MoRI RDMA + decode-leg spec-dec)

Goal: PD disaggregation with MTP on decode, conc=64 pass at ~7400 tok/s. Est. ~15 min (~6 min cold
start — prefill node may be cold in NFS).

## 0. Prerequisites

- **Machines:** 2× 8-MI355X, same RoCE subnet.
  - prefill = chi2832 (10.2.122.79, NO MTP), decode = chi2879 (10.2.122.10, MTP). NIC `enp193s0f1np1`.
  - (We used chi2832 for prefill because chi2878 was taken mid-mission — any 8-MI355X node on the
    same subnet with routable GID at idx1 works; check the per-NIC GID layout, see 02/notes.)
- **Model:** `/mnt/vast/xiaobo/models/GLM-5.2-MXFP4`.
- **Image:** `rocm/infera:sglang-v0.1.0-rc6`. **Patch:** `patches/deepseek_nextn.rc6patch.py`
  staged to `/mnt/vast/c_huggingface/glm52_nextn_patch/deepseek_nextn.py` (decode leg mounts it).
- **Fabric pre-check:** `SERVER=chi2832 CLIENT=chi2879 DEV=ionic_0 GIDIDX=1 DUR=5 bash
  <cluster_tools>/rail_test.sh` → ~335 Gb/s.
- `scripts/engine.sh` staged to shared path (we used `/mnt/vast/c_huggingface/glm52_p3b/engine.sh`).

## 1. Bring up the stack

    scp patches/deepseek_nextn.rc6patch.py \
        <jump>:/mnt/vast/c_huggingface/glm52_nextn_patch/deepseek_nextn.py
    scp scripts/engine.sh <jump>:/mnt/vast/c_huggingface/glm52_p3b/engine.sh
    CONC=64 bash scripts/up.sh
    # up.sh: NATS(both) + etcd + router(http,kv-aware) on chi2832,
    #        prefill leg chi2832 (MTP=0), decode leg chi2879 (MTP=1).
    # decode leg: mounts nextn patch + SGLANG_DSA_ENABLE_MTP_PRECOMPUTE_METADATA=0 + EAGLE flags
    #   (steps 3, eagle-topk 1, num-draft-tokens 4, num-reserved-decode-tokens 256, mem-frac 0.80).

## 2. Wait for readiness + pairing

    curl -sf -o /dev/null -w '%{http_code}\n' http://10.2.122.79:8100/health   # 200
    ssh chi2832 'docker exec repro-etcd-p3b etcdctl --endpoints http://10.2.122.79:2379 \
       get --prefix /infera/workers/'   # 2 workers: prefill + decode
    # NOTE: chi2832 prefill can take ~6 min if weights are cold in NFS on that node — watch
    # its logfile "waiting for SGLang HTTP ..." + VRAM climbing; not a hang.

## 3. Verify correctness

    ssh chi2832 'docker cp /mnt/vast/c_huggingface/glm52_probe.py glm52-router-p3b:/tmp/probe.py \
      && docker exec glm52-router-p3b python3 /tmp/probe.py http://10.2.122.79:8100 glm5.2-mxfp4'
    # expect 4/4 correct

## 4. Confirm decode-leg spec-dec is active

    ssh chi2879 'grep "accept len" /mnt/vast/c_huggingface/glm52_p3b/pdmtp_decode.log | tail'
    # expect accept len ~2.75-2.88 (steps=3), accept rate ~0.58-0.62

## 5. Run conc=64 (1k/1k)

    ssh chi2832 'docker exec glm52-router-p3b bash -lc "python3 -m sglang.bench_serving \
      --backend sglang-oai --base-url http://10.2.122.79:8100 --model glm5.2-mxfp4 \
      --tokenizer /mnt/vast/xiaobo/models/GLM-5.2-MXFP4 --dataset-name random \
      --random-input-len 1024 --random-output-len 1024 --random-range-ratio 1.0 \
      --max-concurrency 64 --num-prompts 256 --warmup-requests 32 --request-rate inf"'

## Expected output

- Probe 4/4; bench `Successful requests: 256` (0 fail — no high-conc crash), total ~7400 tok/s,
  median TPOT ~12 ms (1.7× faster than no-MTP PD's 20.9), median TTFT ~412 ms. See
  `results/bench_conc64.txt`.

## If it doesn't reproduce

See `notes.md`. Same MTP failure modes as 04 (shape crash → patch missing; hang → precompute env
missing). PD-specific: if decode crashes at high conc with a draft-extend KV-pool error, keep
steps=3 / reserved 256 / mem-frac 0.80 (don't raise steps to 5 on the PD decode leg).
