# Environment

## Nodes — two were used, and it matters

| | `smci355-ccs-aus-n01-33` | `smci355-ccs-aus-n01-21` |
|---|---|---|
| arms | PD run 1, DPA-off conc 24, MIX TP8 reference | DPA-off conc 1/8/16, hip A/B, repetition arm |
| data-plane IP | 10.235.192.136 | 10.235.192.130 |
| interface | `fenic` | `fenic` |
| image | earlier build of `Dockerfile.sglang` | 2026-09-03 rebuild |

**Measured node/build delta: 1.01× at conc 1, 1.05× at conc 8** — same PD
configuration on both. That is the noise floor every differential in this packup
is read against. It also means the rebuilt image reproduces the original
behaviourally, not just structurally.

Common to both:

| | |
|---|---|
| GPUs | 8 × AMD Instinct MI355X, **gfx950**, 288 GB HBM3E |
| driver | amdgpu **6.14.14** |
| kernel | 6.8.0-107-generic |
| CPU / RAM | 256 threads / 3023 GB |
| RDMA | 8 × `ionic` @ 400 Gb/s, all `PORT_ACTIVE`, `ib_peer_mem` **loaded** |
| registration mode | **A** — bare `ibv_reg_mr` + peer-mem; nothing pinned, KV pool full |
| `MC_GID_INDEX` | **1** (routable `192.168.x.x`; never the link-local `fe80::`) |
| rail used | **`ionic_0` only** — preflight's single-node rule pins one device on both legs |

## Software

| | |
|---|---|
| image | `infera/engine-sglang:v0518-glm53` |
| image ID (n01-21 rebuild) | `sha256:7489a5a0eb6178a58b084a0656d6b6c28351ae022e6c3ba445c27638c87d1b00` |
| built | 2026-09-03T04:29:53Z |
| base | `lmsysorg/sglang:v0.5.18-rocm720-mi35x` |
| sglang | **0.5.18** |
| torch | **2.9.1+rocm7.2.0.git7e1940d4** |
| etcd | `quay.io/coreos/etcd:v3.5.14` |
| repo | `AMD-AGI/Infera`, branch `yihou.dev.glm53.expr`, commit `46e79746` |
| kit | `examples/sglang_1p1d_glm5.3/` (wrappers) driving `examples/sglang_1p1d_glm5.2/` (engine) |

**The five patches in `patches/` are part of the environment**, not an optional
extra. Without them the shape does not run, or runs on the wrong topology.

## mooncake — the tree in the image is NOT the source of its binary

| | |
|---|---|
| `.so` compiled at | **`faae8dd4`** (2026-08-04) — from the build log |
| tree left in image | `01d1eb2a` (2026-07-01) — `/sgl-workspace/Mooncake` HEAD |
| build flags | `USE_HIP=ON`, `ENABLE_MULTI_PROTOCOL=ON` |

Some later build stage resets the checkout; which stage was not identified. **A
source read taken from the shipped tree is about a month-older codebase and can
be wrong about the running binary.** To read the real source, no network needed:

```bash
docker run --rm --entrypoint /bin/bash <image> -c \
  'cd /sgl-workspace/Mooncake && git cat-file -t faae8dd4a6309c3ecd47e0721a83b0250d686fa2 && \
   git show faae8dd4a6309c3ecd47e0721a83b0250d686fa2:mooncake-transfer-engine/src/multi_transport.cpp | sed -n "470,530p"'
```

## Model

`/perf_apps/data/models/GLM-5.3-MXFP4` — 408 GB, 282 shards, `glm_moe_dsa` /
`GlmMoeDsaForCausalLM`, Quark MXFP4 E2M1 per-group-32 with E8M0 scales.

**`/apps/data/models` is a symlink onto a separate NFS mount.** Bind the
realpath; binding the symlink's parent yields an empty directory whose failure
surfaces much later as `Unrecognized processing class`.

## Deployment shape

TP4 prefill (GPUs 0-3, `mem-fraction-static 0.70`) + TP4 decode (GPUs 4-7,
`0.85`), mooncake transfer backend, etcd discovery, infera kv-aware router.
MTP **off** on both legs. kvd and HiCache **off**. Context 262144,
`chunked-prefill-size 65536`, `kv-cache-dtype fp8_e4m3`.

Pool sizes observed: prefill `max_total_num_tokens=1930368`, decode `2396928`.

## Secrets required

None stored here. SSH to the node; docker registry login only if the image must
be pulled rather than built.

## Capture gaps

- No `collect_env.sh` snapshot at run time; reconstructed 2026-09-03.
- ROCm userspace version captured only via the torch build string.
- The build stage that resets the mooncake checkout was not identified.
