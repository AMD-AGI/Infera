# Environment

Time: 2026-10-20 01:55Z (g4n1/g4n2 launched) → 2026-10-20 17:49Z (idxf1a A2 done); replicates idxf1b/idxf1c started 2026-10-21 01:57:14Z
(in progress when the kit was built, 2026-10-21 02:0xZ).

## Nodes

| node | pool / job | data IP | rounds |
|---|---|---|---|
| crsuse2-m2m-097 | OAI job 225299 (ended at MaxWall 10-20 ~18:4xZ; our containers stopped 18:30Z by stop_before_maxwall) | 10.245.144.36 | g4n1, g4n3, dbge1a, dbge2a-2d, dbge3a, dbge4a, dbge5a, dbge6a, idxf1a |
| crsuse2-m2m-250 | OAI job 225298 (MaxWall 10-20 13:42Z) | 10.245.158.155 | g4n1b, g4n2, g4n2b, g4n3b, g4n4, dbge1b, dbge2e, dbge3b, dbge4b, dbge5b |
| crsuse2-m2m-268 | OAI job 227779 (from 10-21 01:12Z) | 10.245.145.242 | idxf1b (running) |
| crsuse2-m2m-054 | OAI job 228279 (from 10-21 01:21Z) | 10.245.144.125 | idxf1c (running) |

Note: m2m-250 kills every job launched on it since 10-21 00:5xZ (ledger); dbge5b containers there could not be stopped after the job ended.
Cause not established.

- Hardware: 8 × MI355X per node, same-node P4D4 (prefill GPU0-3, decode GPU4-7). GTT `mem_info_gtt_total` 1477 GB/GPU, MemTotal 2.75 TiB (m2m-097, ledger).
- Kernel / amdgpu driver / firmware of these four nodes: **not captured** (no `collect_env` run). Closest capture: previous kit
  `../glm52.v0521-offloaderv2-burst-c56c64.packup_20261019/env/collect_env.v3-020.20261015.txt`.

## Software

- Image `infera-sglang:v0521-yihou-1010-mc4340`, ID `sha256:9d755dd6a7bf2f3705a45dcc7877b9ec3625efb8370ab3b144751799e31300d2`
  (verified at each launch: `results/launch/launch.<round>.yihou.log`).
- sglang in the image: HEAD `9f5d8712514c2d9dd29c76c6080774db3fd50aa2` (2 commits after upstream `11b1fb21de`; earlier kits record 11b1) plus
  uncommitted image-local modifications (`patches/0000-image-local-diff.vs-9f5d8712.diff`, 353 lines: dsa_backend GLM52_BUG2B page-table trim,
  dsa_indexer GLM52_P1V2/P1V3, speculative_hook, glm4_moe, prefill, pool_host/mla, serving_responses). Source of these facts:
  `analysis/NOTES.prep.20261020.md`; local tree `/tmp/yihou/idxshare/sgl-image-9f5d` on the workstation (worktree with the diff applied).
- `eager_runner.py` in the image: md5 `35d3c52b608e7a5b4e88f92e9fcbed82`; fixed file md5 `23d30505b4fab3dd8ee8d1d2880b1b44`.
- Upstream reference for the PR diff: `sgl-project/sglang` origin/main `93088901fb413e5a81eeb81b675fc21b4adc5b1f` (2026-10-20 18:50 -0700),
  `eager_runner.py` md5 `d0feca5d16e3d83c2ccd580535e50a26`. The idle-draft path was read at `16eb87c06f` (an ancestor of 93088901).
- Runtime overlays mounted in every round (not in the image): `mounts.yihou.v2-xferfix2-gtt.tsv` → kv_events_publisher noblock,
  forward_mla_rocm noabsorbcfg, mooncake conn xferfix2, prefill nonfinal-event, pool_host/common PR #42553 (all in `scripts/mounts-src/`).
- Engine knobs (from env files): OffloaderV2 `OFF_G=4 OFF_K=1 OFF_STEP=1`, `PREFILL_HICACHE_SIZE=260` (GB/rank), `PREFILL_MEM_FRACTION=0.90`,
  `DECODE_MEM_FRACTION=0.95`, `DECODE_MAX_RUNNING=96`, chunk 32768 (g4n2/g4n2b: 65536), `DECODE_SIMULATE_ACC_LEN=3.61`, `YIHOU_XFER_SHARD=dest`,
  decode tokenizer/detokenizer workers 4/4, `INDEX_SHARE_FOR_MTP=true` (g4n3/g4n3b false).
- Bench: AgentX (InferenceX) C56, 3600 s, warm-up 10/lane; A1 fresh, A2 scored.
- Harness: `/shared_nfs/yihou_glm52/v0521dbg_20261010/harness` (not in git; files used copied to `scripts/harness/`).

## Secrets

None in the kit. Needed: spur login (ssh `spur00{5,6,7}`), the `amd-oai` account / `amd-oai-qos` via the machine-pool manager.
