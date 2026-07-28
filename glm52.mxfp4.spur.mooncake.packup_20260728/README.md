# GLM-5.2-MXFP4 sglang mooncake PD (mlx5+dmabuf) on crsuse spur — reproduction kit

**Ran:** 2026-07-28 (UTC) · **By:** yihou workspace (Claude) · **Engine:** sglang 0.5.15.post1

Reproduces, on the AMD **crsuse spur** cluster, GLM-5.2-MXFP4 sglang PD-disaggregation with
**mooncake RDMA + DP-attention + MTP** (correctness + conc=128 stress), then enables **infera kvd +
kv-aware routing**. Fuses two prior kits:
- **recipe** (GLM-5.2 DSA + DPA + MTP): `glm5.2.mxfp4.packup_20260727/{03,06,07}` (proven on *vultr*, ionic+peermem).
- **transport** (mlx5 + dmabuf, no peermem): `infera.yihou.dev/crsuse/multinode_tp8_mlx5_dmabuf_dsv4_dockerfile_20260727` (proven on *DSv4*).

## Task (user spec, verbal)

> `…/multinode_tp8_mlx5_dmabuf_dsv4_dockerfile_20260727` is the correct config for running PD with the
> mlx5 NIC on spur; `glm5.2.mxfp4.packup_20260727/{06_pd_mooncake_mtp, 07_pd_mooncake_dpa_sweep}` are
> how to run glm5.2 sglang + mooncake + dpa + mtp on the vultr cluster. Combine the two: reproduce
> glm5.2 mooncake + dpa + mtp correctness and a conc=128 stress on this crsuse spur cluster. Once
> clean, turn on infera's kvd and kv-aware and do a quick test.
> (Build → save the image as `infera.yihou.sglang.1.0` to NFS under /home/yihou.)

## Results at a glance (all PASS)

| Config | Transport | Feature | Correctness | conc=128 | Throughput / note |
|--------|-----------|---------|-------------|----------|-------------------|
| A | mooncake RDMA (mlx5+dmabuf) | DP-attention (dp8+ep8, both legs) | 4/4 | 512/512 | 7218 tok/s, TPOT 31.3ms |
| B | mooncake RDMA (mlx5+dmabuf) | MTP (EAGLE steps=3, decode) | 4/4 | 512/512 | 8990 tok/s, TPOT **19.2ms** (1.6× faster) |
| C | single-node mix | infera kvd + kv-aware routing | 4/4 | prefix-reuse | warm TTFT 1.87s→**0.31s**, cache_hits=31 |

- **Transport verified RDMA:** `installTransport type=rdma` on `mlx5_0` GID idx 3, 0 TCP, 0 KVTransferError.
- **DPA + MTP fused does NOT work** (DSA indexer topk crash under EAGLE draft-extend + DP) — run A or B.
- **kvd L3 write-back GPU-faults on gfx950** — kv-aware routing works (needs only kv-events); the
  live cache demo (Config C) uses the engine's native radix cache. kvd daemon + backend wiring healthy.

See `RESULTS.md` for the full numbers, `notes.md` for every trap (what/why/how/context).

## Image (user-requested name)

`infera.yihou.sglang.1.0` — built on-node from `infera.yihou.dev` `Dockerfile.sglang.dmabuf`
(mooncake rebuilt USE_HIP_DMABUF + HIP-gate). Saved to **`/home/yihou/infera.yihou.sglang.1.0.tar`**
(27 GB, NFS). Base `lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x` (sha256 40e940a0…).

## Reading order
1. `notes.md` — the traps (mlx5/dmabuf switch, DPA+MTP crash, kvd GPU-fault, node wedge). Most valuable.
2. `environment.md` — hardware/RDMA/image sha/git/model/secrets.
3. `REPRODUCE.md` — exact commands, build → PD (A/B) → kvd+kv-aware (C).

## Folder map
- `README.md` (this) · `RESULTS.md` (numbers) · `environment.md` · `REPRODUCE.md` · `notes.md`
- `scripts/` — `pd_leg_spur.sh` (fused PD leg: DPA/MTP/dmabuf switches), `infera_worker.sh` (kvd
  variant), `infera_worker_nokvd.sh` (kv-aware native-cache variant), `sweep_dpa.sh`, `probe.py`,
  `kv_test.py`, `build_dmabuf.sh`.
- `patches/` — `deepseek_nextn.unified_patch.py` (MTP eh_proj fix), `Dockerfile.sglang.dmabuf.copy`,
  `build_mooncake_dmabuf.sh.copy`, `mooncake_cpp/` (HIP-gate + auto-chunk diffs).
- `results/` — `dpa_c128.jsonl`, `mtp_c128.jsonl` (raw bench), `transport_evidence.txt`,
  `kv_aware_kvd_evidence.txt`, `results_summary.csv`.
- `agenticbench/` — Optimus-AgenticBench caseA run: `README.md`, `caseA_summary.txt`, trimmed log.
- `logs/` — trimmed (head+tail) server/router/kvd/build logs.

## Optimus-AgenticBench caseA (follow-on)
Ran the AgenticBench GLM-5.2 production workload Case A (74K/155K/235K input, 89% cache-hit design)
against a 262K-context GLM-5.2 server on node069. **PASS**: success 95.9% (1 transient large-prompt
write error, 0 server crashes), sustain-phase TTFT p50 750ms / TPOT p50 17ms, peak prefill 622K tok/s,
actual cache-hit 86.5% (97.2% of the 89% ideal). See `agenticbench/`.

## Dependencies / not-in-git
- Model `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` (~408 GiB, shared NFS).
- Image tar `/home/yihou/infera.yihou.sglang.1.0.tar` (27 GB, NFS — not in git).
- **No secrets/tokens required** (public base image, model on shared NFS, spur identity automatic).

## Held nodes (spur)
prefill/kv-aware = crsuse2-m2m-069, decode = crsuse2-m2m-321 (later kernel-wedged by a hicache
alloc — see notes ★7). Both held via `-q amd-burst-qos -N1 -G8`.
