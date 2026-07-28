# Reproduction kit — 06 PD mooncake RDMA + MTP (pd-unified)

Goal: mooncake RDMA PD with MTP on decode, conc=64 pass. Est. ~20 min. Builds on 03.

## 0. Prerequisites

Same as 03 (pd-unified image on both nodes, model, libionic, fabric pre-check). Additionally:
- **The pd-unified nextn patch** (`patches/deepseek_nextn.unified_patch.py`) staged to
  `/mnt/vast/c_huggingface/glm52_nextn_patch_unified/deepseek_nextn.py` — the **decode** container
  bind-mounts it. (This is DISTINCT from the rc6 patch in 04/05 — use THIS one for pd-unified.)

## 1. Bring up prefill leg (no MTP) — same as 03

    scp scripts/pd_leg_mtp.sh <jump>:/mnt/vast/c_huggingface/glm52_p2b/pd_leg_mtp.sh
    scp patches/deepseek_nextn.unified_patch.py \
        <jump>:/mnt/vast/c_huggingface/glm52_nextn_patch_unified/deepseek_nextn.py
    # prefill container + libionic + prefill leg (reuse 03/up.sh's prep, then):
    ssh chi2878 'docker exec -d pd_uni env ROLE=prefill MY_IP=10.2.122.3 \
       MODEL=/mnt/vast/xiaobo/models/GLM-5.2-MXFP4 SERVED=glm5.2-mxfp4 PORT=30000 DMABUF=0 MTP=0 \
       MAX_RUNNING=64 bash /pd_leg_mtp.sh'

## 2. Bring up decode leg WITH MTP — recreate the container with the patch mounted

    ssh chi2879 'docker rm -f pd_uni; docker run -d --name pd_uni --network=host --ipc=host \
       --shm-size=32G --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
       --group-add video --group-add render --cap-add=SYS_PTRACE --cap-add=IPC_LOCK \
       --security-opt seccomp=unconfined --ulimit memlock=-1:-1 -v /mnt/vast:/mnt/vast \
       -v /mnt/vast/c_huggingface/glm52_nextn_patch_unified/deepseek_nextn.py:\
/sgl-workspace/sglang/python/sglang/srt/models/deepseek_nextn.py:ro \
       --entrypoint "" infera/engine-sglang:pd-unified sleep infinity'
    # inject libionic (same as 03), then:
    ssh chi2879 'docker cp /mnt/vast/c_huggingface/glm52_p2b/pd_leg_mtp.sh pd_uni:/pd_leg_mtp.sh
       docker exec -d pd_uni env ROLE=decode MY_IP=10.2.122.10 \
       MODEL=/mnt/vast/xiaobo/models/GLM-5.2-MXFP4 SERVED=glm5.2-mxfp4 PORT=30001 DMABUF=0 MTP=1 \
       MAX_RUNNING=64 bash /pd_leg_mtp.sh'
    # decode cold start ~5 min. Confirm draft head loads WITHOUT the shape crash:
    ssh chi2879 'grep -E "Load weight end.*NextN|tensor a.*must match" \
       /mnt/vast/c_huggingface/glm52_p2b/pd_decode_30001.log'
    # want "Load weight end ... type=DeepseekV3ForCausalLMNextN quant=quark" (NO tensor mismatch).

## 3. Router + correctness (same as 03)

    ssh chi2878 'docker exec -d pd_uni bash -lc "pkill -9 -f sglang_router; sleep 3; \
       python3 -m sglang_router.launch_router --pd-disaggregation \
       --prefill http://10.2.122.3:30000 8998 --decode http://10.2.122.10:30001 \
       --host 0.0.0.0 --port 8002 > /tmp/router.log 2>&1"'
    sleep 16
    ssh chi2878 'docker exec pd_uni python3 /tmp/probe.py http://10.2.122.3:8002 glm5.2-mxfp4'  # 4/4

## 4. Confirm spec-dec active + run conc=64

    ssh chi2879 'grep "accept len" /mnt/vast/c_huggingface/glm52_p2b/pd_decode_30001.log | tail'
    # accept len ~2.65-2.90, rate ~0.55-0.63
    ssh chi2878 'docker exec pd_uni bash -lc "python3 -m sglang.bench_serving --backend sglang-oai \
       --base-url http://10.2.122.3:8002 --model glm5.2-mxfp4 \
       --tokenizer /mnt/vast/xiaobo/models/GLM-5.2-MXFP4 --dataset-name random \
       --random-input-len 1024 --random-output-len 1024 --random-range-ratio 1.0 \
       --max-concurrency 64 --num-prompts 256 --warmup-requests 32 --request-rate inf"'

## Expected output

- Draft head loads (no crash), probe 4/4, accept len ~2.7, bench 256/256, total ~5300 tok/s,
  median TPOT ~19 ms (faster than no-MTP's 20.9). See `results/bench_conc64.txt`.

## If it doesn't reproduce

See `notes.md`. Shape crash `3072 vs 6144` → the pd-unified patch isn't mounted (or you mounted the
rc6 one — wrong file). Keep EAGLE steps=3 (not 5) on the PD decode leg for KV-pool stability.
