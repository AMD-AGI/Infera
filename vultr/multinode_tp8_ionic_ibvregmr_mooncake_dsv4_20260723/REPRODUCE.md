# Reproduction kit — SGLang PD mooncake/mori cross-node (v0.5.15.post1 base)

Goal: reproduce a coherent cross-node 1P1D PD completion over mooncake AND mori
on the new base. Estimated time: ~40 min (of which ~20 min is DSv4 tp8 cuda-graph
cold start per leg; legs load in parallel).

## 0. Prerequisites (arrange before you start)

- **Machines:** two MI355X (gfx950) nodes on the SAME rail group. This run used
  `chi2879` (prefill, data-plane `10.2.122.10`) + `chi2865` (decode, `10.2.122.52`).
  Held via slurm; ssh reachable through the jump host `chi2866` (149.28.124.225).
  Both must have 8 ionic NICs `PORT_ACTIVE` and `ib_peer_mem` loaded. See
  `environment_nodes.txt`.
- **Secrets needed** (values NOT included — source them yourself):
  - Cluster SSH: through the jump host `root@149.28.124.225` (ProxyJump to chiXXXX).
  - Docker registry: not needed if you build the image locally (below).
- **External dependencies (absolute paths, not in this repo):**
  - Model: `/mnt/vast/d_huggingface/models/DeepSeek-V4-Pro-fixed` (DSv4-Pro fp8,
    64 shards, `model_type: deepseek_v4`; **stat a shard's byte size** to confirm
    it's real weights, not LFS stubs — this path has flipped to stubs before).
  - Shared FS `/mnt/vast` visible on both nodes (holds the model + the kit).
  - Host `libionic` injected into the container by the kit's `up.sh` (RDMA ABI).
- **Repo state:** branch `yihou.dev.sglang.mooncake` @ commit `2198bae`
  (the HIP-gate fix). The PD kit lives at `examples/deepseek_v4/`.
- **Image:** built below from `deploy/docker/Dockerfile.sglang`; base
  `lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x`. The Dockerfile now runs
  `patches/patch_mooncake_sglang.sh` (HIP-gate) at build time.

## 1. Build the HIP-gated sglang image (once, on any node with the base)

    # from the repo root, on a node that has the base image pulled
    docker build --network=host -f deploy/docker/Dockerfile.sglang \
      -t infera/engine-sglang:pd-fix .
    # expect the build log to show: [mc-hip-gate] DONE — HIP transport gated

## 2. Put the image on BOTH nodes

    # if built on the prefill node, stream it to the decode node:
    docker save infera/engine-sglang:pd-fix | ssh chi2865 'docker load'

## 3. Set the run environment (edit for your nodes/model)

See `scripts/pd_env.sh` — set KIT_DIR, INFERA_IMAGE, INFERA_MODEL(+MOUNT),
PREFILL/DECODE NODE+IP, ROUTER_PORT (8100 to avoid a squatted :8000), CONC.
`INFERA_IMAGE` must be the image from step 1/2.

## 4. Launch 1P1D — mooncake (run from the jump host, which can ssh both nodes)

    source scripts/pd_env.sh
    export BACKEND=mooncake CTR=dsv4_pd_sgl
    bash "$KIT_DIR/engine/pd_mooncake/sglang/up.sh"     # = scripts/run_mooncake.sh
    bash scripts/pd_poll.sh                              # wait for BOTH_READY
    bash scripts/pd_smoke.sh                             # expect: Paris / 1..10

## 5. Teardown, then launch 1P1D — mori (RDMA reset ritual between runs)

    bash scripts/pd_teardown.sh                          # frees GPU on both nodes
    export BACKEND=mori CTR=dsv4_pd_sgl_mori
    bash "$KIT_DIR/engine/pd_mori/sglang/up.sh"          # = scripts/run_mori.sh
    bash scripts/pd_poll_mori.sh                         # wait for BOTH_READY
    bash scripts/mori_smoke.sh                           # expect: Paris / 1..10

## Expected output

`pd_smoke.sh` / `mori_smoke.sh` print the two registered workers (one `prefill`,
one `decode`, both `active`) then:

    REPLY: Paris
    REPLY: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10

A coherent reply = KV transferred prefill→decode over RDMA = success.

## If it doesn't reproduce

See `notes.md`. Most likely: (a) unpatched mooncake → 500 `hipIpcOpenMemHandle
failed` (rebuild the image with the HIP-gate, step 1); (b) GPU memory not back to
~0.3 GB baseline before a re-run (run `pd_teardown.sh`, wait); (c) model path is
LFS stubs (stat the shard sizes); (d) wrong data-plane IP (must be the ionic-rail
`10.2.x` on `enp193s0f1np1`, not the public NIC).
