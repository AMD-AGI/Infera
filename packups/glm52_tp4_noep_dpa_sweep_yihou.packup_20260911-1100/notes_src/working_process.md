# Working process — TP4 / EP1 / DPA-on internal decode sweep (C 4->24)

## 2026-09-11T07:5x-08:0xZ — research and setup (leader)
- Goal from user: rerun the **internal, server-free** method (`bench/profile_decode.py`) with 4 GPUs,
  **no EP**, **DPA enabled**, global concurrency sweep 4/8/16/20/24, on node 217.
- Verified in source that this needs parameter changes only:
  - `bench/profile_decode.py:44` `--ep-size` already defaults to 1; `server_cli` forces
    `--moe-a2a-backend none`, so EP1 is a plain TP-MoE configuration.
  - `bench/topology.py`: `--batch-size` is GLOBAL; with DPA `dp_size = tp_size = 4`, so
    `local_batch_size = C/4` and C must divide by 4 (all five points do). CUDA-graph bs is set from
    `local_batch_size` (1/2/4/5/6).
- Allocation: `scontrol show job 133750` -> RUNNING, node crsuse2-m2m-217, `ReqTRES=cpu=1,node=1`,
  ends 2026-09-11T15:54Z. Spur allocations are exclusive (user statement), so the node is ours.
- Node state before work: foreign container `s1b_cap45`
  (`aigmodelzicheng/instellavl-mi355:07292026`, started 2026-09-10T21:37:07Z) held all 8 GPUs,
  8 KFD processes at 36-60 GB VRAM each with active memory traffic. Asked the user; user chose to
  stop it. `docker stop -t 30 s1b_cap45` reported "did not receive an exit event" but `docker ps`
  then showed no running containers, and a throwaway ROCm container reported VRAM 0% on all GPUs.
  Nothing was deleted; the stopped container and its image remain on disk.
- Pinned image `sha256:b9a83742f631...` absent on 217. Started
  `zstd -dc /shared_nfs/yihou/playground/glm52_mix_repro_20260908/image/rocm-llm-bench.tar.zst |
  docker load` in background (shell b4ns8jfsy). Docker root is `/var/lib/docker` on `/`
  (123G total, ~90G free) — free space must be watched.
- Created workspace `sweeps/tp4_noep_dpa_yihou_20260911-0800/` with `scripts/`, `iterations/`,
  `results/`. Backed up previous CLAUDE.md as
  `CLAUDE.reverse-fake-server-sweep.20260911-0800.md.bak` and wrote a new one for this task.
- Wrote `scripts/run_tp4_noep_dpa_sweep_yihou.sh` (wraps the unchanged `scripts/run_decode.sh`) and
  `scripts/collect_noep_sweep_yihou.py` (asserts ep_size=1, tp_size=4, dp_size=4, dpa=True).
- Scheduled 10-minute mission re-injection (cron 27f82468) and 20-minute team poll (cron f6d06d80).

## Open items
- Image load duration and resulting free disk on 217 not yet confirmed.
- EP1 + DPA has not been run before with this harness; MoE under DP attention with `ep_size=1` is the
  one untested combination. Treat a first failure as information, not as a config error to tune away.

## 2026-09-11T08:08:43Z — image load finished (leader)
- Shell b4ns8jfsy exit 0. `Loaded image: rocm-llm-bench:latest`, digest
  sha256:b9a83742f631d7321c6b75aceb031a8b347c70f97c62ef8c62370c686853de7d, 91.7GB virtual size.
- Disk was never a constraint: `/var/lib/docker` is 4.0K and `df -h /` stayed at 34G used; the real
  image store is the node's NVMe volume (resident images already total ~364GB).
- User authorization update: bench source is under git, so EP1+DPA may be debugged and fixed in
  source until it runs. Constraints recorded in CLAUDE.md and relayed to sweep-runner.

## 2026-09-11T08:30Z — C=4 PASSED on the first attempt (leader)
EP1 + DP-attention runs with no source change. `result_yihou.json` fields checked verbatim:
`"complete": true, "useful_output_tokens": 40000, "tp_size": 4, "ep_size": 1, "dp_size": 4,
"enable_dp_attention": true, "input_len": 70000, "output_len": 10000, "global_batch_size": 4,
"local_batch_size": 1, "moe_a2a_backend": "none", "realized_accept_length": 3.6134393063583814,
"verify_iterations": 2768, "effective_token_latency_ms_per_user": 6.245638818759471,
"output_tokens_per_second": 640.4468967986998`. `launch_status.json` = `{"exit_code":0,"launch_wall_seconds":958}`.
Phase split: load_pool_capture 823.9s dominates; decode itself 62.46s. Expect ~16 min per point.
Against the EP4 baseline (`sweeps/tp4_ep4_dpa_yihou_20260910-0452/summary.csv`, dpa=on, C=4:
tpot 7.199073620900163 ms, 555.6270446224225 tok/s) EP1 is 13.2% faster in TPOT at this point.
C=8 started. Teammate note: four teammate instances died back-to-back on server-side API 502
(08:09, 08:12, 08:17, 08:24); the leader is monitoring directly until the API stabilises.

## 2026-09-11T08:32:50Z — poll: 2/5 points complete (leader)
- C=8 PASSED: `"complete": true, "useful_output_tokens": 80000, "tp_size": 4, "ep_size": 1,
  "dp_size": 4, "enable_dp_attention": true, "global_batch_size": 8, "local_batch_size": 2,
  "input_len": 70000, "output_len": 10000, "realized_accept_length": 3.6134393063583814,
  "verify_iterations": 2768`, tpot 7.8147 ms, 1023.71 tok/s, `launch_status exit_code 0`,
  launch_wall 171s.
- The 958s -> 171s drop in launch_wall is explained by phase_seconds: `load_pool_capture` fell from
  823.9s to 62.2s because the AITER JIT cache (/tmp/yihou-aiter-b9a83742f631) and the page cache were
  cold only on the first point. 171s matches the EP4 baseline's 153-217s launch_wall range, so this
  is expected warm-cache behaviour, not a skipped stage: verify_iterations is still 2768 and decode
  time rose 62.5s -> 78.1s with concurrency as it should.
- Container `yihou-noep-dpa-0911` Up 18 minutes; all four GPUs at VRAM 85% (mem-fraction-static 0.85).
- Teammate still absent: four consecutive spawns died on server-side API 502. Given the remaining
  three points take ~3 min each, the leader continues to monitor directly rather than spawn a fifth
  instance that would likely die mid-point. Recorded here per the poll protocol.

## 2026-09-11T08:38Z — incident: local driver shell killed by host memory pressure
The background shell running the sweep driver was killed by the harness ("system is running low on
memory") while C=24 was starting. Only the LOCAL driver died: `pgrep -af profile_decode` inside
`yihou-noep-dpa-0911` still showed the C=24 python process, which ran to completion and wrote a full
`result_yihou.json` (complete true, 240000 tokens, TPOT 11.3441, 2115.63 tok/s). Because the driver
was gone, no `launch_status.json` exit-code attestation was written for it.
Action: MOVED (not deleted) that directory to `noep_dpa_on_c24_orphan_driver_yihou/` and reran C=24
with the same parameters through `run_decode.sh`. Rerun: `{"exit_code":0,"launch_wall_seconds":203}`,
complete true, 240000 tokens, TPOT 11.361972574423998, 2112.309270489213 tok/s — 0.16% from the
orphan run. The clean rerun is the reported point.
Likely cause of the host memory pressure: piping runtime output through `tail` when `runtime.log`
contains multi-megabyte single-line tqdm progress bars. Subsequent runs redirect to a file instead.

## 2026-09-11T08:5xZ — all five points complete
`collect_noep_sweep_yihou.py` exits 0; `results/summary_yihou.csv` has five `pass` rows. Report
written to `results/report.md`, including the comparison against
`sweeps/tp4_ep4_dpa_yihou_20260910-0452/summary.csv`. No source change was needed anywhere: the
sweep ran on the unmodified bench with parameters only (`--ep-size 1 --enable-dp-attention`).

## 2026-09-11T09:40Z — same-node EP4 control complete (leader)
To remove the node-to-node confound in the EP1-vs-EP4 comparison, EP4 + DPA-on was rerun on the SAME
node/container/allocation as this sweep (`control_iterations/`, driver `run_ep4_control_yihou.sh`).
All three points: complete true, exact C*10000 tokens, tp=4 ep=4 dp=4 dpa=true, accept
3.6134393063583814, verify_iterations 2768, launch exit_code 0.

| C | control EP4 TPOT ms | control tok/s | cross-node baseline TPOT | delta |
|---:|---:|---:|---:|---:|
| 4 | 7.205778 | 555.1101 | 7.199073620900163 | +0.09% |
| 16 | 10.968559 | 1458.7149 | 10.980900 | -0.11% |
| 24 | 12.321298 | 1947.8467 | 12.276800 | +0.36% |

Node-to-node variation is under 0.4%, so the previously published cross-sweep EP1-vs-EP4 comparison
stands as written; the control simply removes the caveat.

## 2026-09-11T09:41Z — KV headroom checked before the C=32/40/48 points (leader)
From `noep_dpa_on_c24_yihou/runtime.log` the per-rank KV pool is `#tokens: 2436864` and
`memory.target_initialized_bytes` is 108.26 GB at EVERY concurrency (the pool is sized by
mem-fraction-static, not by C). Reserved tokens scale with the LOCAL batch:
`reserved_tokens = (C/4) * 80064`. C=48 needs 12 * 80064 = 960768 tokens, 39% of the pool, and
`Memory pool end. avail mem` was still 41.6 GB at C=24. So no mem-fraction reduction is expected to be
needed; if a point nevertheless fails, the failure will be captured before any parameter is changed.

## 2026-09-11T09:32Z — C=32/40/48 driver chained
`run_extra_points_yihou.sh <control_pid> 32 40 48` (pid 2632413) waits on the control driver, then
runs the three remaining points through the unchanged `run_tp4_noep_dpa_sweep_yihou.sh`, same
parameters as the first five points. Crons re-armed: mission injection 4ac4456a (10 min), team poll
06e8f709 (20 min). Monitoring teammate `sweep-monitor3` spawned (read-only).

## 2026-09-11T10:40Z — C=32/40/48 complete, all eight points pass (leader)
`collect_noep_sweep_yihou.py` now covers C=(4,8,16,20,24,32,40,48) and exits 0. New points:

| C | local batch | TPOT ms | tok/s | decode s | launch wall s |
|---:|---:|---:|---:|---:|---:|
| 32 | 8 | 12.757144562341272 | 2508.3983209270114 | 127.571 | 219 |
| 40 | 10 | 14.3613898829557 | 2785.24574055834 | 143.614 | 237 |
| 48 | 12 | 15.429530137684196 | 3110.9178031784368 | 154.295 | 259 |

All three: complete true, exact C*10000 tokens, ep=1 tp=4 dp=4 dpa=true, accept 3.6134393063583814,
verify_iterations 2768, launch exit_code 0. **No mem-fraction change was needed** — the KV-headroom
calculation above held: C=48 reserves 960768 of 2436864 per-rank pool tokens.

## 2026-09-11T10:41Z — EP4 control extended to C=32/40/48 (leader)
The EP4 baseline only reached C=24, so the new no-EP points had no comparison partner. Launched
`run_ep4_control_yihou.sh 32 40 48` (pid 3456576, log control_driver2_yihou.log) on the same
node/container to complete the EP1-vs-EP4 table that CLAUDE.md names as the deliverable. This is a
small extension of the already-agreed comparison, not new scope.

## Teammate status (poll record)
`sweep-monitor3` (spawned 09:41) was unreachable by 10:41 — the same server-side API failure mode that
killed four teammate instances earlier today. Per the poll protocol this is recorded, not escalated;
`sweep-monitor4` was spawned in its place and the leader keeps monitoring directly in parallel so a
teammate death cannot stall the work.

## 2026-09-11T10:56Z — EP4 control complete at C=32/40/48; final report written
All six control points pass (complete true, exact C*10000 tokens, ep=4 tp=4 dp=4 dpa=true, accept
3.6134393063583814, verify_iterations 2768, launch exit_code 0): C=32 tpot 13.397674 / 2388.4743,
C=40 tpot 15.248736 / 2623.1683, C=48 tpot 16.430828 / 2921.3378.
`results/ep1_vs_ep4_same_node_yihou.csv` generated by a script that re-asserts completeness, token
counts, topology and exit codes for BOTH sides before computing any delta. `results/report.md`
rewritten with the eight-point table, the KV-headroom justification, the same-node EP1-vs-EP4
comparison, and the open item (the delta stops shrinking and widens slightly at C>=40, unexplained by
this sweep, single runs only). Work complete; the allocation (ends 15:54Z) and container are left
running and untouched.

## Teammate status (final)
`sweep-monitor4` also died on a server-side 502 (AnthropicVertex) at 2026-09-11T10:44:04Z, 3 minutes
after being spawned — the seventh teammate instance lost to the same API failure today. It died
before the EP4 control points finished, so it contributed no observations. All monitoring and
verification for this sweep was done by the leader directly. The "use an agent team" rule was
therefore only partially met; the cause was an external API outage, not a choice, and it is recorded
here rather than papered over.
