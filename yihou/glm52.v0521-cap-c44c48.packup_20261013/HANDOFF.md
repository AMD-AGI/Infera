# HANDOFF — GLM-5.2 v0.5.21 1P1D P4D4: capacity + concurrency (> 23k tok/s/chip) and decode in-flight KV cap

Written 2026-10-13 ~03:20 UTC on `crsuse2-slog-003` for a Claude Code session on another machine. Read this file
first, then the task of record. Everything referenced lives on NFS (`/home/yihou`, `/shared_nfs`) unless noted.

## 0. One-minute start (new machine)

```bash
# 1) restore the session (optional — this README is enough to continue without it)
mkdir -p ~/.claude && tar -C ~/.claude -I zstd -xf /home/yihou/claude-session-export.yihou.20261013.tar.zst
#    contains: transcript b77d319b-1818-43ee-a42e-fd71ec1bb336 (+ subagents/workflows), auto-memory, global CLAUDE.md,
#    skills. NOT contained: settings.json / .credentials.json (secrets) and plugins — configure on the new machine.
#    Restore notes: /home/yihou/claude-session-export.yihou.20261013.RESTORE.md
cd /home/yihou/dev/git/infera.glm52.view && claude --resume b77d319b-1818-43ee-a42e-fd71ec1bb336   # or start fresh
# 2) in the new session, first prompt:
#    "Read /home/yihou/dev/git/infera.glm52.view/HANDOFF.v0521cap.yihou.md and continue the task."
```
If the new machine's cwd differs, the Claude project key changes (cwd with '/' and '.' → '-'); rename
`~/.claude/projects/-home-yihou-dev-git-infera-glm52-view` accordingly, or just start a fresh session from this README.

## 1. The task (inverted pyramid)

- **Goal (user, 2026-10-12):** GLM-5.2-MXFP4, sglang v0.5.21 image `infera-sglang:v0521-yihou-1010-mc4340`, same-node
  1P1D P4D4 (prefill GPUs 0-3, decode 4-7, KV over XGMI/HIP IPC). Raise prefill + decode capacity, raise AgentX CONC
  (44 → 48), push **> 23,000 tok/s/chip** (AgentX full 3600 s, warmup 10/lane). Keep triton DSA, EAGLE MTP 5/1/6.
- **Track B (user):** decode rule "admit only if (requested + held KV) − running KV < 50 % of the pool"; patch + debug
  until **C40 runs with no regression** vs 22,114, then apply to the best config.
- **Last item (user):** test `index_share_for_mtp_iteration=true` at the best config.
- **Status:** best **22,306/chip** (C48, node 137). 23k not reached. Track B patch v2 works functionally; its scored C40
  run is still owed. **No node is held** (102 job 207495 TIMEOUT 10-12 20:01; 256 job 208970 ended 10-13 02:23).

Task of record (re-read every 10 min): `/home/yihou/dev/git/infera.glm52.view/glm52.v0521-cap-c44c48-23k_20261012/mission.md`
(all user directives, verbatim + English) and `…/PLAN.md` (v1-v5). Ledger `…/debug/working_process.md`, polls `…/debug/poll.md`,
research `…/research/` (R1 decode admission code, R2 KV occupancy, R3 levers, R5 upstream, each + `.verify`).
Project CLAUDE.md (auto-loaded in that cwd): `/home/yihou/dev/git/infera.glm52.view/CLAUDE.md`.

## 2. Results so far (first-hand; score = `request_metrics.throughput.per_gpu.total_tput_tps`)

| run | node | config delta vs e2g recipe (HiCache 450, decode mem 0.95, max_running 64) | CONC | tok/s/chip |
|---|---|---|---|---|
| e2g A2 (baseline) | 102 | — | 40 | 22,114 |
| **q1 A3** | 137 | decode max_running **96** + prefill mem **0.88** (3rd run on the stack) | **48** | **22,306** |
| q1 A2 / A4 | 137 | same | 44 / 52 | 21,862 / 21,418 |
| k1 A2 | 102 | HiCache 360 | 44 | 21,342 |
| pc40c A2 | 102 | runshare v1 | 40 | 21,271 (regression) |
| pc40d A1 | 102 | runshare v2 (fresh, unscored) | 40 | 20,680 |

Key findings: C48 is the sweet spot (score ≈ qps × mean ISL; more lanes cut ISL and add TTFT). The limiter is the
**prefill side** (Little's law, C44→C52: prefill queue 2.0→5.2 s, KV send 4.5→8.2 s; decode running flat ~12.5 s);
decode KV also tight at C48/C52 (max usage 0.99/0.98, 0 retractions). HiCache 360 < 450. Decode radix cache would cut
KV send but is blocked with EAGLE in v0.5.21.

## 3. Pack-ups (deliverables; read README → REPRODUCE → notes)

- Capacity/concurrency: `/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.v0521-cap-c44c48.packup_20261013/`
- Track B (decode in-flight KV cap): `/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.v0521-decode-transfer-kvcap.packup_20261013/`
- Baseline 22,114 kit (image recipe, warm-stack protocol): `/home/yihou/dev/git/infera.glm52.view/glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012/`
- Decode KV capacity report: `/home/yihou/dev/git/infera.glm52.view/decode.kvcache.report.md`
- `infera.glm52.pd` is **READ-ONLY** except our own `yihou/` pack-up dirs.

## 4. Where things live

| what | path |
|---|---|
| harness (env/config/chain per round, engine.sh) | `/shared_nfs/yihou_glm52/v0521dbg_20261010/harness/` (`RUNBOOK.v2.yihou.md`) |
| runs (JSON, server logs; q1 logs also on 137 `/mnt/m2m_nobackup/yihou_v0521dbg/runs/q1`) | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/<round>/` |
| tools (metrics sampler, long-prompt burst, Little's-law residence) | `/shared_nfs/yihou_glm52/v0521cap_20261012/tools/` |
| /metrics TSVs | `/shared_nfs/yihou_glm52/v0521cap_20261012/runs/` |
| image tarball (docker load; ID may change per node) | `/shared_nfs/yihou_glm52/v0521cap_20261012/images/infera-sglang.v0521-yihou-1010-mc4340.9d755dd6.yihou.tar.zst` |
| sglang source from the image (code reading) | `/shared_nfs/yihou_glm52/v0521cap_20261012/src/image-9d755dd6/sglang` |
| Track B patches | `$WS/harness/mounts-v0521/src/decode.yihou-runshare{,2}.py` + `config.yihou.runshare{,2}.sh` |
| model | `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` |
| hold script / spur ssh shim | `…packup_20261012/scripts/alloc/hold_inf.yihou.sh`; shim dir `/shared_nfs/yihou_glm52/kvcap_20261008/scripts/shimbin` |

## 5. Next steps (in order)

1. **Get a node:** `sbatch -A amd-oai -p amd-spur -q amd-oai-qos -N 1 --gres=gpu:8 -c 160 -t 24:00:00 -J yihou-oai-x hold_inf.yihou.sh`
   (amd-oai QOS cap node=8 — may pend `QOSGrpNodeLimit`). Do **not** use amd-burst-qos (cancelled at 20 min). 135-139 only
   with user approval and if idle (137 is used by llying since 10-13 01:29). Load the image tarball, add the node's image
   ID to `config.yihou.mc4340n.sh` (write a new file), seed JIT caches from `$WS/cache/102`.
2. **pc40d A2** (Track B v2 no-regression check, C40 warm protocol) — `env.pc40d-102.yihou.sh` retargeted to the new node;
   see the Track B pack-up REPRODUCE.
3. **q4**: q1 config + decode mem 0.96, C48 (`env.q4-137.yihou.sh`, `chain_q4.yihou.sh`; retarget node).
4. KV-send lever at C48 (copy-stream priority: env `GPU_MAX_HW_QUEUES=16 MC_HIP_NUM_STREAMS=8`, or the 102-only
   mc4340prio image), prefill cache (HiCache 500, prefill mem 0.90), then runshare v2 on the best config.
5. index_share TRUE (last, user).

## 6. Rules and gotchas (must follow)

- Communicate with the user in **Chinese**; work/notes/code in English. Measure, rank evidence (first-hand vs
  second-hand), suspend rather than conclude. Ask before deviating from the mission.
- **Never delete a file whose path lacks `yihou`. Never disturb other users' containers/GPUs** (check `docker ps` for
  foreign GPU containers before using any node).
- **Login hosts are memory-policed** (user: keep memory use small; prefer serial over parallel; no heavy workflows/subagents
  on the login host). Drivers on a lost login host die — run drivers where they survive.
- Score with the **warm-stack protocol**: run 1 (fresh, unscored) → `POST /flush_cache` decode then prefill → run 2 (scored).
  Further CONC levels can run as 3rd/4th runs on the same stack (CONC is client-side only).
- VRAM drain after a HiCache-450 stack: 28-35 min (harness gates on idle). RCCL init race: retry decode only.
- Logs contain binary bytes: `strings <log> | grep -a`. Never edit a running bash script in place (new file + mv).
- Pass config deltas via the CONFIG/env chain; token-diff the dryrun against the proven run (`tokdiff.yihou.sh`) before launching.
- spur nodes: no plain ssh — use the shim with `SPUR_SSH_JOBS=<node>:<job>`; `--pid=host` denied.

## 7. Session memory (also in the export; on NFS at `/home/yihou/.claude/projects/-home-yihou-dev-git-infera/memory/`)

- analysis-before-experiment — exhaust code reading and upstream search before spending GPU time.
- diff-against-proven-run-params — token-diff launch env/args vs the proven run before debugging.
- evidence-before-causal-claims — the user pushes back on causal claims from uncontrolled comparisons.
- idle-gpu-not-free — 0 % VRAM on a shared node can be a gap in another user's looping job.
Skills used (in the export, `~/.claude/skills/`): experiment-result-packup, iterative-debug-loop, spur-cluster-usage,
spur-interactive-debug, open-source-pr. Global rules: `~/.claude/CLAUDE.md` (in the export).
Crons used in this session (recreate if wanted): mission re-injection every 10 min; team poll at :07/:27/:47 recording
first sightings in `debug/poll.md`.
