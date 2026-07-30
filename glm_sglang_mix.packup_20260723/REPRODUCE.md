# Reproduction kit — GLM-5.1-FP8 on SGLang single-node PD-mix

Goal: bring GLM-5.1-FP8 up on SGLang (single-node, TP4) and prove temp=0
correctness. Estimated time: ~15 min bring-up (incl. an ~8-10 min silent
JIT/tuning window) + seconds per probe.

All commands run from your workstation via `ssh chiXXXX` (ProxyJump
preconfigured). Single-node mix needs **no RDMA / second node / MoRIIO /
Mooncake**.

## 0. Prerequisites

- **Machine:** one node with ≥4 free MI355X (gfx950) GPUs. This run used
  **chi2866**, card4-7 (card0-3 held foreign `titan` training — untouched).
  Confirm the cards are free (~283 MB/card idle):
  `ssh chi2866 'rocm-smi --csv --showmeminfo vram | grep -E "card[4-7]"'`
- **GLM weights (absolute path, not in repo):**
  `/mnt/vast/xiaobo/models/GLM-5.1-FP8` (`GlmMoeDsaForCausalLM`, fp8 dynamic, 78
  layers, DSA lightning indexer index_head_dim=128). On the shared VAST mount,
  same path in-container. Confirm: `ssh chi2866 'ls /mnt/vast/xiaobo/models/GLM-5.1-FP8'`.
- **Image:** `lmsysorg/sglang:v0.5.14-rocm720-mi35x` (sglang 0.5.14; implements
  `GlmMoeDsaForCausalLM` at `python/sglang/srt/models/glm4_moe.py`, a
  `DeepseekV2ForCausalLM` subclass → DeepSeek MLA+DSA path). Present on the node;
  else pull it.
- **Secrets:** cluster SSH via configured ProxyJump. No registry/etcd/API creds
  needed (single-node, image already on node).

## 1. Stage scripts on the shared VAST dir

    STAGE=/mnt/vast/c_huggingface/glm_sglang_mix
    ssh chi2866 "mkdir -p $STAGE"
    for f in ./scripts/*; do scp "$f" chi2866:$STAGE/; done

## 2. Start the sglang container (privileged, card4-7, VAST mounted)

    ssh chi2866 'docker run -d --name glm_sglang_c_hf --privileged --ipc host \
      --shm-size 64g --device /dev/kfd --device /dev/dri \
      --group-add video --group-add render \
      -v /mnt/vast:/mnt/vast --entrypoint "" \
      lmsysorg/sglang:v0.5.14-rocm720-mi35x sleep infinity'
    # (single-node mix -> no ionic injection / RDMA needed.)

## 3. Launch the server (inside the container)

    ssh chi2866 "docker exec -e HIP_VISIBLE_DEVICES=4,5,6,7 glm_sglang_c_hf \
      bash $STAGE/glm_sglang_mix.sh"
    # poll readiness (~15 min; an ~8-10 min SILENT tilelang-JIT + aiter-GEMM-tuning
    # window looks like a stall but is forward progress — DO NOT kill):
    for i in $(seq 1 60); do
      h=$(ssh chi2866 'docker exec glm_sglang_c_hf curl -s -o /dev/null -m3 -w "%{http_code}" http://127.0.0.1:30000/health')
      echo "health=$h"; [ "$h" = 200 ] && break; sleep 20
    done

## 4. Correctness probe (temp=0 — the ACTUAL test)

    ssh chi2866 "docker exec glm_sglang_c_hf python3 $STAGE/glm_probe.py http://127.0.0.1:30000 GLM-5.1-FP8"

### Expected output

    [PASS] 'The capital of France is' -> 'Paris'
    [PASS] 'The capital of China is' -> 'The capital of China is Beijing.'
    [PASS] '2+2=' -> '2 + 2 = 4'
    VERDICT: ALL PASS

Tie-back to success criterion: **GLM-5.1-FP8 on SGLang single-node mix produces
correct, coherent output at temp=0** — MET.

## 5. Teardown

    ssh chi2866 'docker rm -f glm_sglang_c_hf'
    # confirm card4-7 VRAM back to ~283 MB; leave foreign titan/zirui/primus untouched.
    ssh chi2866 'rocm-smi --csv --showmeminfo vram | grep -E "card[4-7]"'

## If it doesn't reproduce

See `notes.md`. Top gotchas: (1) the ~8-10 min **silent** JIT/GEMM-tuning window
is NOT a hang — `py-spy dump` shows forward progress; don't kill it. (2) Do NOT
add the DSv4 flags (`--attention-backend dsv4`, `--page-size 256`) — GlmMoeDsa
auto-selects the DSA path and the DSv4 flags fight it. (3) `--trust-remote-code`
is required (custom GLM code). (4) use the urllib probe, not nested `curl -d`.
