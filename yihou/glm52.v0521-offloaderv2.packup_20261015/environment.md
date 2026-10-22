# Environment

## Nodes

| node | job | data IP | rounds | env capture |
|---|---|---|---|---|
| `crsuse2-m2m-089` | oai 213048 (2026-10-13 19:27 → 10-14 19:27, ended at time limit) | 10.245.145.201 | pf90, cx90 (refs), ofs0c, ofs0 | `env/collect_env.089.20261014.txt` (10-14 01:57, before the rounds) |
| `crsuse2-m2m-252` | oai 214829 (10-14 18:02 → 10-15 18:02) | 10.245.149.156 | ofg4c, ofg4 | `env/collect_env.252.20261015.txt` (10-15 ~03:00, during ofg4c) |
| `crsuse2-m2m-266` | oai 215368 (10-14 17:56 → released by user ~10-15 02:50 for admin maintenance) | 10.245.157.132 | ofg3c, ofg3 (aborted) | none |
| `crsuse2-m2m-288` | oai 215369 (10-14 17:57 → 10-15 17:57) | 10.245.152.60 | ofg2c, ofg2 | `env/collect_env.288.20261015.txt` (10-15 ~03:00, during ofg2) |
| `crsuse2-m2m-136` | none (plain ssh) | 10.245.154.168 | ofs0/ofs0c prepared, never run (node taken by another user) | `env/collect_env.136.20261014.txt` |

Pre-run checks at allocation (10-14 18:15, `analysis/ledger/poll.md`) on 252/266/288:
- ROCR 0-7, 8 × MI355X;
- `amd-smi` ECC uncorrectable 0 (288: one GPU with 8 correctable);
- no containers, 0 KFD processes;
- host RAM 2751 GiB;
- docker root `/mnt/m2m_nobackup/docker`.

Hardware (captures):
- Ubuntu 24.04, kernel 6.8.0-107-generic;
- 2 × AMD EPYC 9575F; 2.7 TiB RAM;
- 8 × AMD Instinct MI355X (gfx950, 287.98 GiB VRAM); amdgpu driver 6.14.14;
- ionic_0..7 RoCE.

Same-node KV transfer uses GPU IPC (mooncake HIP transport). RDMA is not on the path.

## Software

| item | value |
|---|---|
| image | `infera-sglang:v0521-yihou-1010-mc4340`, ID `sha256:9d755dd6a7bf2f3705a45dcc7877b9ec3625efb8370ab3b144751799e31300d2` on 089/252/266/288 (tar `/shared_nfs/yihou_glm52/v0521cap_20261012/images/…9d755dd6.yihou.tar.zst`) |
| sglang | `0.5.21.dev20261008+g11b1fb21de`; source copy `/shared_nfs/yihou_glm52/v0521cap_20261012/src/image-9d755dd6/sglang` |
| OffloaderV2 | in the image: `sglang/srt/utils/offloader.py` (copy `patches/offloader.py.orig`) |
| patch | YIHOU-OFFLOAD-ATTRS, bind mount on the prefill leg only (`patches/`) |
| runtime mounts | `mounts.yihou.v2-regbase.tsv` (as in all v0521cap2 rounds) + `mounts.yihou.offload-attrs.tsv` on prefill |
| topology | same-node 1P1D P4D4: prefill GPU0-3, decode GPU4-7, TP4 DP4 DP-attention, fp8_e4m3 KV |
| prefill | mem 0.90, HiCache 450 GB write_through, chunked 8192 per DP rank, max_running 64 |
| decode | mem 0.95, max_running 96, 2,762,176 tok/rank, EAGLE 5/1/6; perf rounds `SGLANG_SIMULATE_ACC_LEN=3.61`, correctness rounds OFF |
| AgentX client | InferenceX `918524ff94045b3f091115f1051c22a8588edf2b`, scenario `inferencex-agentx-mvp`, 3600 s, warmup 10 |
| harness | not in git: `/shared_nfs/yihou_glm52/v0521dbg_20261010/harness` (verbatim copy `scripts/harness/`) |
| pack-up repo | `/home/yihou/dev/git/infera.glm52.pd` on spur (this folder is uncommitted) |

## Gaps

- No env capture for 266. The 089 capture predates these rounds.
- No ECC capture after the runs.
- No per-layer timing profile (assessment §E step 2). That H2D is hidden behind compute is not measured directly. Only the end-to-end scores are measured.

## Secrets

None. The copied env/JSON/log files were grepped for HF_TOKEN / hf_ / ghp_ / AKIA / private keys before packing (see `SHA256SUMS` step in the builder).
