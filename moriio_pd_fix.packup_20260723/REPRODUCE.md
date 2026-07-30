# Reproduction kit — vLLM v0.25.1 MoRIIO PD page-len fix

Goal: reproduce (a) the MoRIIO PD **garbage** on DeepSeek-V4 / GLM-5.1, and
(b) that `patch_moriio_pagelen.py` **fixes** it — correct, coherent output.
Estimated time: ~15 min bring-up per model (DSv4) / ~10 min CG capture (GLM) per
side, plus seconds per probe.

All commands run from your workstation via `ssh chiXXXX` (ProxyJump preconfigured).
Scripts live in `./scripts/`; stage them + the model probes + the patch to the
shared VAST dir and run there.

## 0. Prerequisites (arrange before you start)

- **Machines:** a producer node + a consumer node, 8×MI355X each. This run used
  P=**chi2879** (10.2.122.10) ↔ D=**chi2866** (10.2.122.47). Obtain via slurm hold
  (see the `slurm-cluster` skill / jump host). Ensure the GPUs you'll use are free
  (~298 MB/card). NOTE chi2866 is also the jump host — this run used its card4-7
  for decode (card0-3 held foreign training).
- **Secrets** (values NOT included — source them yourself):
  - Cluster SSH: ProxyJump in `~/.ssh/config` (direct `ssh chiXXXX`).
  - Docker image: on-node already; else `inferaimage` creds from team vault.
- **External deps (absolute paths, not in repo):**
  - DSv4: `/mnt/vast/d_huggingface/models/DeepSeek-V4-Pro-fixed` (the `-fixed` copy).
  - GLM:  `/mnt/vast/xiaobo/models/GLM-5.1-FP8`.
- **Image:** `inferaimage/infera:vllm-v0.25.1-20260721` (id 368cadb4d983; base
  digest in environment.md). This image was built BEFORE the fix, so §2b applies a
  runtime hot-patch. If you build a fresh image from repo `yihou.dev.vllm.image.update`
  @ 29021d3 **after** adding `deploy/docker/patches/vllm/patch_moriio_pagelen.py`, the
  Dockerfile patch loop bakes the fix in and you SKIP §2b.
- **Container:** a running privileged container named `glm_pd` on each node (the
  infera engine image, `/opt/infera`, `-v /mnt/vast:/mnt/vast`,
  `--device=/dev/infiniband`, libionic injected). This kit assumes it exists (it is
  how the infera-native launch runs). etcd runs as its own container `pd_etcd` on P.

## 1. Stage scripts + probes + patch on the shared VAST dir

    STAGE=/mnt/vast/c_huggingface/vllm_patch_verify
    for f in ./scripts/*; do scp "$f" chi2879:$STAGE/; done
    # (the probe .py files and patch_moriio_pagelen.py are inside ./scripts/)

Edit the `IP=` lines at the top of the prefill/decode scripts if your nodes differ
(prefill IP = P data-plane, decode IP = D data-plane, ROUTER/ETCD = P).

## 2a. RDMA teardown ritual (MANDATORY between every attempt)

Between runs, kill only YOUR engine procs and confirm release BEFORE relaunch:

    for n in chi2879 chi2866; do ssh $n 'docker exec glm_pd bash '"$STAGE"'/killworkers.sh'; done
    # confirm VRAM back to ~298 MB/card AND MoRIIO ports free:
    ssh chi2879 'rocm-smi --csv --showmeminfo vram | sed -n "2,5p"; ss -ltn | grep -cE ":3610|:3620"'
    ssh chi2866 'rocm-smi --csv --showmeminfo vram | sed -n "6,9p"; ss -ltn | grep -cE ":3610|:3620"'

Skipping this causes port/OOM phantom failures (see notes.md).

## 2b. Apply the fix (ONLY on the pre-fix image; skip if fix is baked into image)

    scp ./patches/patch_moriio_pagelen.py chi2879:$STAGE/
    for n in chi2879 chi2866; do ssh $n "docker exec glm_pd python3 $STAGE/patch_moriio_pagelen.py"; done
    # expect "moriio-pagelen: patched ..." (or "already patched" on re-run)

To reproduce the BUG first, SKIP this step (run §3-5 on the stock image → garbage),
then do §2a + §2b + §3-5 again → correct. That is the differential proof.

## 3. Launch etcd + both engines (restart BOTH together — never one side alone)

    # etcd on P (its own container; the engine image has no etcd binary):
    ssh chi2879 'docker start pd_etcd 2>/dev/null || true'   # or your etcd bring-up
    # engines — prefill on P, decode on D, run INSIDE the container:
    ssh chi2879 "docker exec glm_pd bash -lc 'bash $STAGE/dsv4_prefill.sh'"   # or glm_prefill.sh
    ssh chi2866 "docker exec glm_pd bash -lc 'bash $STAGE/dsv4_decode.sh'"    # or glm_decode.sh
    # poll readiness (both must be 200; ~15 min DSv4 / ~10 min GLM CG capture):
    for i in $(seq 1 45); do
      p=$(ssh chi2879 'docker exec glm_pd curl -s -o /dev/null -m3 -w "%{http_code}" http://10.2.122.10:30001/health')
      d=$(ssh chi2866 'docker exec glm_pd curl -s -o /dev/null -m3 -w "%{http_code}" http://10.2.122.47:30012/health')
      echo "prefill=$p decode=$d"; [ "$p" = 200 ] && [ "$d" = 200 ] && break; sleep 20
    done

## 4. Launch the router (from the HOST — the router script does `docker exec`)

    ssh chi2879 "bash $STAGE/dsv4_router.sh"     # or glm_router.sh
    # expect: {"status":"ok","active_workers":2}
    ssh chi2879 'docker exec glm_pd curl -s http://10.2.122.10:8000/health'

## 5. Correctness probe (temp=0 — the ACTUAL test)

    # DSv4:
    ssh chi2879 "docker exec glm_pd python3 $STAGE/pd_probe.py"
    # GLM:
    ssh chi2879 "docker exec glm_pd python3 $STAGE/glm_probe.py"
    # isolation (PD vs prefill-direct :30001) to prove prefill compute is fine:
    ssh chi2879 "docker exec glm_pd python3 $STAGE/isolate.py"

## Expected output

- **Without the fix (§2b skipped):** DSv4 → short factual prompts garble
  ("France"→"a good idea"); GLM → total garbage ("is is is"). Prefill-direct is
  correct in both → only the cross-engine transfer corrupts. (See results/evidence.md.)
- **With the fix:** DSv4 → France→Paris, China→Beijing; GLM → France→"Paris...",
  China→Beijing, 2+2→4, sky→Rayleigh. PD == prefill-direct. Deterministic-coherent.

Tie-back to success criterion: **cross-node MoRIIO PD produces correct, coherent
output at temp=0 for GLM-5.1-FP8** — MET.

## If it doesn't reproduce

See `notes.md`. Top gotchas: (1) the RDMA teardown ritual is mandatory — leftover
workers hold ports 36100-36203; (2) `vllm bench serve` never checks output
correctness — ALWAYS use the temp=0 probe; (3) routers run from the HOST (they
`docker exec`), engines run INSIDE the container; (4) `infera.server` rejects
`--trust-remote-code` (already removed from the router scripts here); (5) DSv4
requires the `-fixed` weights and fp8 KV (bf16 asserts out).
