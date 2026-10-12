# Task: debug GLM-5.2 same-node 1P1D P4D4 on sglang v0.5.21 until AgentX passes

Goal (user, 2026-10-10): make GLM-5.2-MXFP4 same-node 1P1D P4D4 run successfully on an
infera image built on **`lmsysorg/sglang-rocm:v0.5.21-rocm720-mi35x-20261008`**. First relax
the configuration until a full AgentX run passes, then step back toward the target (triton
DSA, prefill HiCache 450 GB, decode mem-fraction 0.95, EAGLE MTP 5/1/6, AgentX full 3600 s at
CONC=44) one group at a time, localizing triggers — or report the best stable config.
**Each experiment should test several hypotheses at once.**

Task of record: `glm52.oai-1p1d-v0521-agentx-c44_20261009/mission.md` (re-injected every
10 min, "CURRENT DIRECTIVE" on top). Plan of record: `…/PLAN.md` (v2). Debug ledger:
`…/debug/working_process.md` (R001-R006+), poll ledger `…/debug/poll.md`.

## Workspaces

- Docs: `glm52.oai-1p1d-v0521-agentx-c44_20261009/` (+ `phase2-maf-debug/` for this phase).
- Runtime (phase 2): `/shared_nfs/yihou_glm52/v0521dbg_20261010/` (harness copy, runs, diag).
- Phase-1 runtime: `/shared_nfs/yihou_glm52/agentx44_20261009/` (image build clone
  `infera-src.yihou` @ `fb8ae9c9`, rounds c44a-c44e, research JSON in `logs/`).
- `/home/yihou/dev/git/infera.glm52.pd` is **READ-ONLY** (other tests run there).

## Key facts (first-hand)

- **H1 (FAULT 2 root cause candidate, code-level confirmed):** HiCache host pool >256 GiB is
  registered in chunks (`SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB`, default 256) but kernels
  address it from one `hipHostGetDevicePointer(base)` → slots past 6,118,144 tokens/rank fault
  at `--hicache-size 450`. Fix: `SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB=512` on prefill.
- **H2 (FAULT 1):** `index_share_for_mtp_iteration=true` forces sustained eager DP-padded
  draft → decode MAF. Keep **false** unless ≥2 clean long replicates prove otherwise.
- Fixes in place (mounts): non-blocking KV-metrics send (R001); `_absorb_bmm_config → None`
  (R003, hand config at M=256 aborts triton LLVM). Harness validator reads `attn_dp_size` (R002).
- Transport recipe: `MC_TE_FILTERS=<non-existent>` → 0 HCAs → KV over HIP IPC (XGMI). Never set
  `MC_DISABLE_HIP`. Mooncake engine.so must be `f22b1364…` (faae8dd4 rebuild).
- Nodes: 136-138 taken by other users (yaoc, limou) as of 03:10 UTC; 135 busy; 139-141 no
  usable dockerd. Use an **amd-oai** spur node: jobs 205476 / 202546 (holds use
  `sleep infinity` so TimeLimit raises work).

## Core principles

1. Measure, don't assume; rank evidence (first-hand vs second-hand). Suspend, don't conclude.
2. Work in containers; host actions are docker / ssh / spur shim only. serena/LSP for code.
3. Never delete a file whose path lacks `yihou`. Never disturb other users' containers/GPUs.
4. English for work, Chinese for reports.
5. Agent team: leader polls every 20 min (first sighting → record only); watchdog every 10 min
   while a service/bench runs. Ultracode: orchestrate substantive work with workflows.

## Other notable details

- RCCL same-node init race: retry decode only, never restart prefill.
- VRAM release after teardown can take 10-40 min: gate launches on all GPUs idle.
- Cold bring-up ~45 min; warm ~15 min. AgentX warmup is max_tokens=1 by design (decode idle).
- Logs contain binary bytes: `strings <log> | grep`.
- Pass config deltas via the CONFIG file so `agentx_bench.sh` matches `launch.sh`.
