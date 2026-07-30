# Reproduction kit — GLM-5.1-FP8 on vLLM single-node PD-mix

Goal: bring GLM-5.1-FP8 up on vLLM (single-node, TP4) and prove temp=0
correctness. Estimated time: ~6-7 min bring-up + seconds per probe.

All commands run from your workstation via `ssh chiXXXX` (ProxyJump
preconfigured). Single-node mix needs **no RDMA / second node / MoRIIO / Mooncake
/ kv-transfer-config**.

## 0. Prerequisites

- **Machine:** one node with ≥4 free MI355X (gfx950) GPUs. This run used
  **chi2879**, card0-3 (node was fully free). Confirm free (~283 MB/card):
  `ssh chi2879 'rocm-smi --csv --showmeminfo vram | grep -E "card[0-3]"'`.
- **GLM weights (absolute path, not in repo):**
  `/mnt/vast/xiaobo/models/GLM-5.1-FP8` (`GlmMoeDsaForCausalLM`, fp8 dynamic, 78
  layers, ships `chat_template.jinja`). Confirm:
  `ssh chi2879 'ls /mnt/vast/xiaobo/models/GLM-5.1-FP8'`.
- **Image:** `infera/engine-vllm:test-local` (id e91a6d7d3a91, 36.3GB) — built from
  this repo branch (vLLM v0.25.1 base + the moriio pagelen fix). Present on chi2879.
  If absent, build it: `docker build -f deploy/docker/Dockerfile.vllm -t
  infera/engine-vllm:test-local .` (see the repo). Single-node mix does not need the
  fix, but any vLLM-0.25.1 GLM-capable image works.
- **Secrets:** cluster SSH via configured ProxyJump. No registry/etcd/API creds
  (single-node, image on node).

## 1. Stage scripts on the shared VAST dir

    STAGE=/mnt/vast/c_huggingface/glm_vllm_mix
    ssh chi2879 "mkdir -p $STAGE"
    for f in ./scripts/*; do scp "$f" chi2879:$STAGE/; done

## 2. Launch container + server (docker run + vllm serve; card0-3)

    ssh chi2879 "bash $STAGE/glm_vllm_mix.sh"
    # poll readiness (~6-7 min; CUDA-graph/torch.compile silent window is normal):
    for i in $(seq 1 40); do
      h=$(ssh chi2879 'docker exec glm_vllm_mix_c_hf curl -s -o /dev/null -m3 -w "%{http_code}" http://127.0.0.1:8000/health' 2>/dev/null)
      echo "health=$h"; [ "$h" = 200 ] && break; sleep 20
    done

Note: if host port 8000 collides with a pre-existing container, the script's
container is host-internal; probe via `docker exec` on localhost (step 3 already
does this).

## 3. Correctness probe (temp=0 — the ACTUAL test)

    scp ./scripts/glm_probe.py chi2879:$STAGE/
    ssh chi2879 "docker exec glm_vllm_mix_c_hf python3 $STAGE/glm_probe.py http://127.0.0.1:8000 GLM-5.1-FP8"

### Expected output

    [PASS] 'The capital of France is' -> 'Paris'
    [PASS] 'The capital of China is' -> 'The capital of China is Beijing.'
    [PASS] '2+2=' -> '2 + 2 = 4'
    VERDICT: ALL PASS

GLM is a THINKING model — the probe sends `chat_template_kwargs={"enable_thinking":
false}` + max_tokens 200 so the direct answer isn't crowded out by reasoning
preamble (with thinking ON + small max_tokens the content looks empty/truncated,
which is NOT a correctness failure).

Tie-back to success criterion: **GLM-5.1-FP8 on vLLM single-node mix produces
correct, coherent output at temp=0** — MET.

## 4. Teardown

    ssh chi2879 'docker rm -f glm_vllm_mix_c_hf'
    ssh chi2879 'rocm-smi --csv --showmeminfo vram | grep -E "card[0-3]"'  # ~283MB

## If it doesn't reproduce

See `notes.md`. Top gotchas: (1) GLM is a thinking model — disable thinking in the
probe or content looks empty. (2) single-node mix uses NO kv-transfer-config —
that's the pd_disag path. (3) OOM at gpu-util 0.85 → drop to 0.80. (4) use the
urllib probe, not nested `curl -d`.
