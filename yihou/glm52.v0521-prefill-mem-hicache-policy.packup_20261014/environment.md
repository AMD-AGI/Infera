# Environment

## Nodes (captures `env/collect_env.<K>.20261014.txt`, 2026-10-14 01:57 UTC — before today's rounds)

| node | job | data IP | used by | image ID (live `IMAGE_IDS` in runtime.env) |
|---|---|---|---|---|
| `crsuse2-m2m-089` | oai 213048 | 10.245.145.201 | pf90, pc64, cx90 | `sha256:9d755dd6a7bf2f3705a45dcc7877b9ec3625efb8370ab3b144751799e31300d2` |
| `crsuse2-m2m-085` | oai 211169 (ended 05:07 NODE_FAIL at ~24 h RunTime; cause [inferred]) | 10.245.153.145 | pf92 | 9d755dd6… |
| `crsuse2-m2m-274` | oai 211171 (ended 03:10 NODE_FAIL) | 10.245.157.34 | pf88 | 9d755dd6… |
| `crsuse2-m2m-136` | none (plain ssh, NFS read-only) | 10.245.154.168 | wts48, wts64, pf93/pf94 (armed) | `sha256:9872bbdabfcb8f6676b04378d8ea75c922dfc8db6a8c5c87acd801e205b3b575` (136's own build) |
| `crsuse2-m2m-137` (reference q1, 10-12) | — | 10.245.153.247 | q1 A3 | see sibling packup cap-c44c48 |

Common hardware (089 capture; 085/274/136 same class, see their captures): Ubuntu 24.04.4, kernel 6.8.0-107-generic, 2 × AMD EPYC 9575F
(236 CPUs), 2.7 TiB RAM, 8 × AMD Instinct MI355X (gfx950), 287.98 GiB VRAM each, amdgpu driver 6.14.14; ionic_0..7 RoCE 400 Gb/s.
Same-node KV transfer goes over GPU IPC (mooncake HIP transport); RDMA not used.

## Software

| item | value |
|---|---|
| sglang | `0.5.21.dev20261008+g11b1fb21de` (live server-info of every bench, `results/*/service/*-server-info.json.gz`); source copy `/shared_nfs/yihou_glm52/v0521cap_20261012/src/image-9d755dd6/sglang` |
| runtime mounts | `mounts.yihou.v2-regbase.tsv` (kv_events_publisher noblock, forward_mla_rocm noabsorbcfg, mooncake conn regbase); pc64 adds the attndp hook (`mounts.yihou.v2-regbase-attndp.tsv`, `patches/`) |
| topology | same-node 1P1D P4D4: prefill GPU0-3, decode GPU4-7, TP4 DP4 DP-attention, fp8_e4m3 KV |
| prefill (live) | mem 0.88 / 0.90 / 0.92, max_running 64 (16 per DP rank in the log), HiCache 450 GB write_through (wts48: write_through_selective; pc64: 370 write_back), chunked 8192 per DP rank (log) |
| decode (live) | mem 0.95 (pc64 0.96), max_running 96, 2,762,176 tok/rank (pc64 2,816,896), EAGLE 5/1/6, `SGLANG_SIMULATE_ACC_LEN=3.61` |
| decode alloc policy | `early` everywhere except pc64 (`prefill_complete`) |
| AgentX client | InferenceX `918524ff94045b3f091115f1051c22a8588edf2b`, scenario `inferencex-agentx-mvp`, 3600 s, warmup 10 |
| harness | not in git: `/shared_nfs/yihou_glm52/v0521dbg_20261010/harness` (verbatim copy `scripts/harness/`) |
| pack-up repo | `/home/yihou/dev/git/infera.glm52.pd` on spur006 (this folder is uncommitted) |

## Gaps

- No env capture after the runs; 136's driver/kernel is in `env/collect_env.136.20261014.txt` (01:57).
- No ECC capture for 089/085/274 during today's rounds.
- 136 runtime server logs are node-local (`/mnt/m2m_nobackup/yihou_v0521dbg/...`), not reachable for packing.

## Secrets

None. Copied env/JSON/log files were grepped for HF_TOKEN / hf_ / ghp_ / AKIA / private keys: no match.
- pf92 on 085 has no dryrun of its own: `env.pf92-085` differs from `env.pf92-206` (dryrun `results/dryruns/dryrun.pf92-206.yihou.txt`,
  token diff vs q1 in `results/dryruns/tokdiff.q1-vs-pf92-206.txt`) only by the header comment and `YIHOU_NODE` / `YIHOU_IP` / `YIHOU_JOB`.
  Its live audit (`results/rounds/pf92/audit.txt`) is AUDIT PASS and live server-info shows mem 0.92 / 2,649,600 tok.
  All other rounds' audits: AUDIT PASS (pf90, pf88, wts48, pc64).
