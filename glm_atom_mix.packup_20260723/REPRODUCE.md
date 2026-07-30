# Reproduction kit — GLM-5.1-FP8 on ATOM single-node

Goal: bring GLM-5.1-FP8 up on ATOM (single-node, TP4) and prove temp=0
correctness. Estimated time: a few min bring-up (aiter MoE compile + cudagraph
capture) + seconds per probe.

All commands run from your workstation via `ssh chiXXXX` (ProxyJump
preconfigured). Single-node needs **no RDMA / second node / MoRIIO / Mooncake**.

## 0. Prerequisites

- **Machine:** one node with ≥4 free MI355X (gfx950) GPUs. This run used
  **chi2866**, card4-7 (card0-3 held foreign `titan` training — untouched).
  Confirm free (~283 MB/card): `ssh chi2866 'rocm-smi --csv --showmeminfo vram | grep -E "card[4-7]"'`.
- **GLM weights (absolute path, not in repo):**
  `/mnt/vast/xiaobo/models/GLM-5.1-FP8` (`GlmMoeDsaForCausalLM`, model_type
  `glm_moe_dsa`, fp8 dynamic, 78 layers, 142 shards, ships `chat_template.jinja`).
  Confirm: `ssh chi2866 'ls /mnt/vast/xiaobo/models/GLM-5.1-FP8'`.
- **Image:** `infera/engine-atom:kimi` (id c7505e171e31, 45.1GB) — **already
  loaded on chi2866**. A staged tar exists at
  `/mnt/vast/c_huggingface/dsv4_repro_atom_img.tar`.
  ⚠️ **DO NOT `docker load` / `docker pull` it** — its 45GB unpack briefly filled
  the node `/` to 100% and hung ssh. Only load it if the image is truly absent AND
  the node has ≫50GB free.
- **Secrets:** cluster SSH via configured ProxyJump. No registry/etcd/API creds
  (single-node, image already on node).

## 1. Stage scripts on the shared VAST dir

    STAGE=/mnt/vast/c_huggingface/glm_atom_mix
    ssh chi2866 "mkdir -p $STAGE"
    for f in ./scripts/*; do scp "$f" chi2866:$STAGE/; done

## 2. Launch container + server (runs docker run; card4-7)

    ssh chi2866 "bash $STAGE/glm_atom_mix.sh"
    # poll readiness (aiter MoE compile + cudagraph capture [1..512], a few min):
    for i in $(seq 1 40); do
      h=$(ssh chi2866 'curl -s -o /dev/null -m3 -w "%{http_code}" http://127.0.0.1:8000/health' 2>/dev/null)
      echo "health=$h"; [ "$h" = 200 ] && break; sleep 15
    done
    # if /health isn't wired, wait for the log line instead:
    ssh chi2866 'docker logs glm_atom_c_hf 2>&1 | grep -m1 "ready to accept requests"'

## 3. Correctness probe (temp=0 — the ACTUAL test)

    scp ./scripts/glm_probe.py chi2866:$STAGE/
    ssh chi2866 "docker exec glm_atom_c_hf python3 $STAGE/glm_probe.py http://127.0.0.1:8000 GLM-5.1-FP8"

### Expected output

    [PASS] 'The capital of France is' -> 'Paris'
    [PASS] 'The capital of China is' -> 'The capital of China is Beijing.'
    [PASS] '2+2=' -> '2 + 2 = 4'
    VERDICT: ALL PASS

GLM is a THINKING model — the probe sends `chat_template_kwargs={"enable_thinking":
false}` + max_tokens 200 so the direct answer isn't crowded out by reasoning
preamble (with thinking ON + small max_tokens the content looks empty/truncated,
which is NOT a correctness failure).

Tie-back to success criterion: **GLM-5.1-FP8 on ATOM single-node produces correct,
coherent output at temp=0** — MET.

## 4. Teardown

    ssh chi2866 "bash $STAGE/down.sh"
    # removes ONLY our container; confirms card4-7 ~283MB + disk not worse.
    # NEVER rmi the atom image; NEVER prune; NEVER scancel.

## If it doesn't reproduce

See `notes.md`. Top gotchas: (1) do NOT re-load the atom image (disk trap). (2)
GLM is a thinking model — disable thinking in the probe or it looks empty. (3) NO
`--method mtp` — GLM has no MTP draft weights and gfx950 plain decode is correct.
(4) use the urllib probe, not nested `curl -d`. (5) card4-7 only; never touch
card0-3 / foreign containers.
