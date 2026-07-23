# Environment

The #1 reproducibility trap for PD/RDMA. Captured from the live nodes on
2026-07-23. Raw per-node snapshot: `environment_nodes.txt`.

## Hardware / fabric

| | chi2879 (prefill) | chi2865 (decode) |
|---|---|---|
| GPU arch | gfx950 (MI355X, 8×) | gfx950 (MI355X, 8×) |
| ROCm | 7.2.0 | 7.2.0 |
| Docker | 28.5.1 | 28.4.0 |
| Data-plane IP (`enp193s0f1np1`) | 10.2.122.10/23 | 10.2.122.52/23 |
| ionic NICs `PORT_ACTIVE` | 8 | 8 |
| `ib_peer_mem` loaded | yes | yes* |
| ephemeral port range | 32768–60999 | 32768–60999 |

\* chi2865's `lsmod` grep showed 0 at snapshot time but RDMA worked (ionic_rdma
pulls it); not load-bearing for the result.

- **Rail:** both nodes are in fabric group Y (leaves sf1–sf8), ALL ALIGNED — so
  same ionic index = same rail. Cross-node RDMA verified by the working KV
  transfer (mori used all 8 ionic NICs; mooncake used the rail via GID index 1).
- **GID index:** 1 (ionic RoCE-v2; index 0 is link-local and crashes).
- **Jump host:** chi2866 = `149.28.124.225` (slurm login + ssh ProxyJump to chiXXXX).

## Software

- **Base image:** `lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x`
  - inside: sglang 0.5.15.post1, torch 2.9.1+rocm7.2.0, Ubuntu 22.04.5, Python 3.10.12
  - **bundled Mooncake:** commit `01d1eb2a` (upstream #2682 — the regression source)
- **Built image:** `infera/engine-sglang:pd-final` (this packup's fix), built from
  `deploy/docker/Dockerfile.sglang` with `BUILD_MOONCAKE_GATE=1`.
  - post-build: mooncake `engine.so` contains `MC_ENABLE_HIP_TRANSPORT` gate (verified), `from mooncake.engine import TransferEngine` OK.
- **Repo:** branch `yihou.dev.sglang.mooncake` @ commit `2198bae` (HIP-gate fix).
  Draft PR #19. (Base-image upgrade itself landed earlier as PR #15 on `main`.)
- **PD kit:** `examples/deepseek_v4/engine/pd_{mooncake,mori}/sglang/`.

## Model

- `/mnt/vast/d_huggingface/models/DeepSeek-V4-Pro-fixed` — DSv4-Pro fp8
  (`model_type: deepseek_v4`), 64 safetensors shards (~13.8 GB each), tokenizer.json
  ~6.3 MB. **Real weights, not LFS stubs** (verified by stat at run time).
- Mounted read-only into the container; kit mounts `/mnt/vast` so symlink targets resolve.
