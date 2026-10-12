# MISSION — GLM-5.2 same-node 1P1D P4D4 on sglang v0.5.21, debugged to a passing AgentX run

Re-injected every 10 minutes. This file is the task of record.

## CURRENT DIRECTIVE (user, 2026-10-10 ~02:55 UTC) — supersedes the node choice below

**Goal: debug until GLM-5.2 runs successfully on sglang v0.5.21** (same-node 1P1D P4D4).

1. **First relax the configuration as far as needed to get a successful run** on v0.5.21.
2. **Then step back toward the target config one group at a time, localizing the
   trigger(s)**, ending at roughly the current target (triton DSA, prefill HiCache 450,
   decode gmu 0.95, MTP, AgentX full CONC=44) — or, if that cannot be made stable, report
   the best-performing stable configuration.
3. **Machines: pick idle ones among crsuse2-m2m-136..138** (likely fewer problems than the
   oai node). Checked 02:52 UTC: 136 and 138 idle (8×283 MiB, no GPU containers); 137 busy
   (another user's vLLM). Plain ssh works.
4. Experience: `/home/yihou/dev/git/infera.glm52.pd/yihou/` holds many v0.5.19 successes.
5. On every bug: read sglang source (serena/LSP) + official docs / issues / PRs, combined.
   **Each experiment should test several hypotheses at once** to cut debug cost.
6. Ultracode is on: orchestrate substantive work with workflows; verify adversarially.

Debug state so far: `debug/working_process.md` R001–R006 (two fixes in place: non-blocking
KV metrics send; hand-built absorb-bmm config disabled. Open: GPU memory access faults
6–12 min into profiling — c44c decode, c44d prefill).

## Task (user, 2026-10-09)

On the **current oai machine** (`crsuse2-m2m-102`, `amd-oai` / `amd-oai-qos`), using an
infera image built on **`lmsysorg/sglang-rocm:v0.5.21-rocm720-mi35x-20261008`** (user
re-confirmed this base explicitly), with:

- **triton** DSA backends (prefill + decode)
- prefill **`--hicache-size 450`**
- **gmu = 0.95** (`--mem-fraction-static`; decode leg — see PLAN for prefill)
- plus already-stable optimizations from the reference packups

deliver:

1. **Same-node 1P1D P4D4 deployment that starts and serves** (prefill GPUs 0-3, decode 4-7,
   KV over XGMI/HIP IPC — no TCP, no NIC).
2. **AgentX full-config benchmark at CONC=44** (full mode: 3600 s, warmup 10/lane), passing.

The user rebased `infera.glm52.pd` (`dev/pd_opt/glm_5.2_agentx` → `28121cb4`) and
stash-applied; this may have broken the router. **Find and fix such errors ourselves.**
FOUND: unresolved conflict markers in `rust/router/src/{policy.rs,disagg.rs}` (PD DP-rank
affinity vs upstream's new `pick_targets`/sessions API). Fixed in our private clone.

## Experience to build on

1. `glm52.oai-samenode-1p1d-p4d4-xgmi.packup_20261009/` (ours, this tree): the only recipe
   proven on this node class. KEY: `MC_TE_FILTERS=<non-existent device>` → `Found 0 HCAs` →
   KV over HIP IPC. Never set `MC_DISABLE_HIP`. RCCL same-node race → retry decode only.
   VRAM can return slowly → gate on idle.
2. `/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.samenode-p4d4.agentx-c40.packup_20260922`
3. `/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.samenode-triton-idxshare.c40-full.packup_20260923`
   (triton + index_share=true, same-node full C40: 168,260 tok/s, 0/4157 errors, on 137).

## Rules (MUST, at all times)

- Re-inject this mission every 10 min.
- Agent team. Leader polls every 20 min; first sign of a problem → record only; intervene
  only if the next poll shows it unresolved. While service/bench run, a **watchdog**
  teammate checks every 10 min.
- Follow user CLAUDE.md; ask the user interactively before deviating.
- research → plan → sub-workspace → CLAUDE.md → work.
- Work in docker containers, not on the host. Use LSP/serena.
- **Never delete a file whose path lacks `yihou`.**
- English for work; Chinese only for reports to the user.
- CUDA graph capture can take ~30 min; watch the build/cache dir.
- Use the debug skills.
- Carried over: `infera.glm52.pd` is READ-ONLY (other tests run there) — change only our
  clone/workspace/containers. TCP fallback unacceptable. XGMI, not mlx5.

## Allocation

- Job **200276** on 102, ends **2026-10-09 11:17:38 UTC**.
- Job **202546** (`yihou-oai-go2`, 24 h, `-w crsuse2-m2m-102`, `afterany:200276`) queued
  to keep the node after that.

## UPDATE (user, 2026-10-10 ~03:45 UTC)

Use a teammate to obtain **two burst-QOS nodes** (amd-burst / amd-burst-qos) and watch them every
10 min; plan autonomously. **Before any new run, re-run the previously successful same-node oai
v0.5.19 experiment (Stage 1 r012/r013 recipe) on each node to prove the machine works.**
Team: alloc-watch (allocation + 10-min watch), sanity (v0519 acceptance run), harness-v2 (v0521
image staging + E1/E2 configs, done: dryruns match intended deltas), repro (E0 H1 micro-repro +
debug-agent smoke; running on 138 GPU 7).
- 05:26 UTC: burst nodes allocated — job 205515 crsuse2-m2m-227 (10.245.151.183), job 205548
  crsuse2-m2m-067 (10.245.157.102), 8 idle GPUs each. v0519 acceptance first on both, then E1/E2.

## UPDATE (user, 2026-10-11 ~05:00 UTC) — FLOOR + FALLBACKS (current, on top of the PERF GOAL)

1. **Floor: must exceed the "22k" of `/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.samenode-triton-idxshare.c40-full.packup_20260923`.**
   Fact check (leader, first-hand): that packup's results/agentx_conc40.json = 168,260 tok/s total =
   **21,032 tok/s/chip** (no 22k anywhere in the yihou packups); our best is 21,696 (R030). Keep 22k/chip as the goal.
2. User asked to confirm the triton backend: confirmed first-hand — server_args of BOTH legs in e1c/e2c/e2d/e2e/e2f
   show dsa_prefill_backend='triton', dsa_decode_backend='triton' (explicit args are honoured on HIP:
   arg_groups/overrides.py:668-676); dsa_topk_backend sgl-kernel, paged_mqa_logits auto — same as the packup.
3. If 22k cannot be reached on 102: run on the same machine as the triton run (crsuse2-m2m-137).
   05:00 check: 137 is busy (yaoc glm52-4p4d-dcp137 stack, up 25 min, VRAM 88-93 %).
4. If still no result for a long time: first reproduce the packup run exactly (v0519 image
   infera-sglang:v0519-yihou-0917-nextnfix-hicache, its REPRODUCE.md, on 137).

## UPDATE (user, 2026-10-10 ~15:55 UTC) — PERF GOAL (current)

Target config PASSED (e2c, R027; pack-up `infera.glm52.pd/yihou/glm52.samenode-v0521-oai102.agentx-c44.packup_20261010`).
New directive: **run C40**; user recalls v0519 best ≈ 22k tok/s/chip and asks why v0521 is lower; if
it dropped, find the cause. **Goal: exceed 22k tok/s/chip** (reference = the yihou `*triton*` pack-up,
`glm52.samenode-triton-idxshare.c40-full.packup_20260923`: documented 21,032/chip, C40, index_share TRUE).
