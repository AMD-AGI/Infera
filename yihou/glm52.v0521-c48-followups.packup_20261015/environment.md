# Environment

| node | job | data IP | rounds | image ID | env capture |
|---|---|---|---|---|---|
| `crsuse2-m2m-089` | oai 213048 (10-13 19:27 → 10-14 19:27) | 10.245.145.201 | pf90 (ref), cx90, tw90 | `sha256:9d755dd6a7bf2f3705a45dcc7877b9ec3625efb8370ab3b144751799e31300d2` | `env/collect_env.089.20261014.txt` (01:57, before the rounds) |
| `crsuse2-m2m-136` | none (plain ssh) | 10.245.154.168 | wts48 (ref), wts64 | `sha256:9872bbdabfcb8f6676b04378d8ea75c922dfc8db6a8c5c87acd801e205b3b575` (136's own build) | `env/collect_env.136.20261014.txt` |

Hardware:
- Ubuntu 24.04, kernel 6.8.0-107-generic;
- 2 × AMD EPYC 9575F, 2.7 TiB RAM;
- 8 × MI355X (gfx950, 287.98 GiB), amdgpu 6.14.14.

| item | value |
|---|---|
| image | `infera-sglang:v0521-yihou-1010-mc4340`; sglang `0.5.21.dev20261008+g11b1fb21de` |
| topology | same-node 1P1D P4D4: prefill GPU0-3, decode GPU4-7, TP4 DP4 DP-attention, fp8_e4m3 KV; KV over GPU IPC |
| pf90 / cx90 / tw90 | prefill mem 0.90, HiCache 450 GB write_through; decode mem 0.95, max_running 96, EAGLE 5/1/6; sim 3.61 (cx90 OFF) |
| wts48 / wts64 | prefill mem 0.88, HiCache 450 GB write_through_selective (see 10-14 packup) |
| AgentX client | InferenceX `918524ff94045b3f091115f1051c22a8588edf2b`, `inferencex-agentx-mvp`, 3600 s, warmup 10 |
| harness | `/shared_nfs/yihou_glm52/v0521dbg_20261010/harness` (copy `scripts/harness/`) |

## Gaps

- wts48/wts64 server logs are node-local on 136 (`/mnt/m2m_nobackup/yihou_v0521dbg/...`). They were not reachable at packing, and 136 now belongs to another user.
- No post-run env/ECC capture on 089 (the job ended 10-14 19:27).

## Secrets

None (grepped; see builder step).
