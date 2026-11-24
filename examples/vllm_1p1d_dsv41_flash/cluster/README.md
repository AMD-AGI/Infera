# Adapt to your cluster — change things HERE

Everything site-specific lives in one of the two wrappers in this directory (or
in environment variables of the same names). `common.sh` and `engine/*.sh` carry
the recipe and contain no addresses, paths, NIC names or GID indices.

## Which wrapper?

Run `IMAGE=<tag> bash preflight_rdma.sh` on each node. It picks by the amdgpu
version and whether a GPU peer-memory module is loaded:

| preflight says | wrapper | why |
|---|---|---|
| `MR_MODE=gem` (amdgpu ≥ 6.19, or no peer memory) | [`cluster.gem.sh`](cluster.gem.sh) | KFD charges an RDMA-pinned VRAM buffer twice; the KV cache goes in GEM BOs (charged once), registered by dma-buf |
| `MR_MODE=peermem` (amdgpu 6.16 with `ib_peer_mem`/`amdp2p`) | [`cluster.peermem.sh`](cluster.peermem.sh) | bare `ibv_reg_mr` through peer memory is charged once |

## Fields

| var | what |
|---|---|
| `PREFILL_NODE` / `DECODE_NODE` | hosts; equal for a single-node pair. Run the wrapper on the prefill node (e.g. under `srun`), or set `SSH_CMD` |
| `PREFILL_IP` / `DECODE_IP` | data-plane addresses the legs advertise to each other and the router |
| `PREFILL_GPUS` / `DECODE_GPUS` | each leg's GPUs; the number listed is its DP size (recipe: 2 + 2) |
| `INFERA_IMAGE` | the tag `build_image.sh` printed |
| `MODEL_MOUNT` / `MODEL` | weights directory (bind-mounted read-only) and the checkpoint under it |
| `LOG_DIR` | host directory for every log; shared storage if the legs are on two nodes |
| `RDMA_IB_DEVICES` | every `PORT_ACTIVE` RDMA device (from preflight) |
| `RDMA_GPU_NICS` | the NIC for each GPU index, position k = GPU k; default = `RDMA_IB_DEVICES` |
| `MC_GID_INDEX` | the routable RoCE v2 GID index (an IPv4-mapped GID, never `fe80::`); per node |
| `UMBP`, `UMBP_GB`, `UMBP_MODE` | host-DRAM KV tier on the prefill: on, 449 GB, `standalone` (one pool, both ranks) or `embedded` (one per rank) |
| `INFERENCEX_DIR`, `AGENTX_HF_HOME`, `AGENTX_RUNTIME_DIR` | AgentX client: InferenceX checkout, HF cache with the traces, node-local scratch |

Two values bite: **`MC_GID_INDEX` is per node** (an empty slot on one machine
shifts every later index), and **a down rail must not be listed** in
`RDMA_IB_DEVICES`. Preflight reads both off the node.

## What not to change

`engine/leg.sh` encodes settings that are coupled and fail plausibly rather than
loudly: `fp8_e4m3` and the block size must match on both legs (it is also the
transfer format, and `canonical_kv` is only true while they match); the decode
keeps prefix caching off; `MC_DISABLE_HIP=1` keeps a same-node pair on RDMA;
`prefill-schedule-interval` stays off the prefill leg.
