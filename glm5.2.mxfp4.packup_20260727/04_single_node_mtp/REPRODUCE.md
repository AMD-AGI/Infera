# Reproduction kit — 04 single-node MTP

Goal: GLM-5.2 with EAGLE spec-dec on one node, coherent + real draft acceptance. Est. ~10 min.

## 0. Prerequisites

- **Machine:** one 8× MI355X node (chi2879). SSH via jump host.
- **Model:** `/mnt/vast/xiaobo/models/GLM-5.2-MXFP4`.
- **Image:** `rocm/infera:sglang-v0.1.0-rc6`.
- **The patch (in this packup):** `patches/deepseek_nextn.rc6patch.py`. Stage it to a path the
  launch mounts — we used `/mnt/vast/c_huggingface/glm52_nextn_patch/deepseek_nextn.py`. `launch.sh`
  bind-mounts it over the image's `.../srt/models/deepseek_nextn.py`.

## 1. Stage the patch + launch

    scp patches/deepseek_nextn.rc6patch.py \
        <jump>:/mnt/vast/c_huggingface/glm52_nextn_patch/deepseek_nextn.py
    # from the node:
    bash launch.sh
    # launch.sh = rc6 image + base DSA envs + these two MTP-critical additions:
    #   -v <patch>:/sgl-workspace/sglang/python/sglang/srt/models/deepseek_nextn.py:ro
    #   -e SGLANG_DSA_ENABLE_MTP_PRECOMPUTE_METADATA=0
    # + EAGLE flags: --speculative-algorithm EAGLE --speculative-num-steps 5
    #   --speculative-eagle-topk 1 --speculative-num-draft-tokens 6
    # cold start ~5 min: base weights load + the nextn DRAFT head loads (type=DeepseekV3ForCausalLMNextN).

Confirm the draft head loaded without the shape crash:

    docker logs glm52-mtp 2>&1 | grep -E 'Load weight end.*NextN|tensor a.*must match'
    # want: "Load weight end ... type=DeepseekV3ForCausalLMNextN quant=quark"  (NO tensor-mismatch)

## 2. Verify correctness (temp=0 probe)

    docker cp probe.py glm52-mtp:/tmp/probe.py
    docker exec glm52-mtp python3 /tmp/probe.py http://127.0.0.1:30000 glm5.2-mxfp4
    # expect 4/4 correct (and it must NOT hang — if it hangs, the precompute env is missing).

## 3. Confirm spec-dec is actually accepting draft tokens

Drive a longer generation, then read the decode batch stats:

    docker exec glm52-mtp bash -lc 'curl -s http://127.0.0.1:30000/v1/chat/completions \
      -H "Content-Type: application/json" -d "{\"model\":\"glm5.2-mxfp4\",\"messages\":\
      [{\"role\":\"user\",\"content\":\"Write a short paragraph about computing history.\"}],\
      \"max_tokens\":200,\"temperature\":0}" -o /dev/null'
    docker logs glm52-mtp 2>&1 | grep 'accept len' | tail
    # expect: accept len ~3.5-4.8 (of 6), accept rate ~0.5-0.77, gen throughput up to ~219 tok/s.

## Expected output

- Draft head loads (no shape crash), probe 4/4, accept len 3.5–4.8, ~219 tok/s single-stream
  (~2.7× the ~80 tok/s no-MTP baseline). See `results/single_stream.txt`.

## If it doesn't reproduce

See `notes.md`. Two failure modes:
- **Shape crash `3072 vs 6144`** at draft load → the patch isn't applied / not mounted. Check the
  mount and that the file has `ckpt_prefix = f"model.layers.{N}.eh_proj"`.
- **Probe hangs** (server ready, single short req works, load hangs) → `SGLANG_DSA_ENABLE_MTP_
  PRECOMPUTE_METADATA=0` is missing → decode retries an un-compilable CUDA kernel every request.
- **`'DeepseekV3ForCausalLMNextN' is not a registered model`** → you mounted the reference's
  v0.1.1 `deepseek_nextn.py` instead of the rc6 patch here. Use THIS packup's patch.
