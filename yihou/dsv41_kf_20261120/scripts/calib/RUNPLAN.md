# RUNPLAN: profile Muse TP4 C40 with the team-calibrated profiler on crusoe v3-180

Date 2026-11-13. Author: subagent (plan only; nothing in this file was run on a node). Evidence marks:
[F] first-hand (file read / command run on this host), [A] arithmetic on [F], [I] inference, not established.
Background and tool choice: `STUDY.md` (same dir). Node: crsuse2-m2m-v3-180, agentx job 285544, grant until
stop_by 13:46Z [F coordination/grant.v3180.json]. All node commands go through the leader:

```
timeout <T> ~/.claude/skills/machine-pool/scripts/spur_remote.sh 'spur exec 285544 -- bash -c "<cmd>"' </dev/null
```

## 0. What this produces and the one-line design

- A per-rank, per-phase (VERIFY / DRAFT / EXTEND / OTHER) per-kernel table from the fork's rocprofiler-sdk
  json-tool for **20 VERIFY steps per rank** of the base stack (1106 TP4/EP4 C40 + reclaim0 + FOOWW), in the same
  buckets as the torch-profiler table (kf2/tasks/phase3/ANALYSIS.md 7.1: MoE MXFP4 12.66, MoE BF16 2.53, full attn
  9.72, SWA attn 1.36, dense 8.51, AR 6.08, norms ~1.8, total 42.62 ms per VERIFY step at bs 26; DRAFT 2.41) [F].
- Design: the json-tool is injected into every server process by env (`ROCP_TOOL_LIBRARIES`, team recipe [F]);
  our `hook/sitecustomize.py` keeps kernel/HIP-API tracing **paused through startup and CUDA-graph capture**
  (mode `window`), opens it when the host touches `start_window`, closes it after 20 VERIFY forwards (with one
  `torch.cuda.synchronize()`), and writes a step timeline (`ModelRunner.forward` t0/t1 in CLOCK_BOOTTIME, stage
  label = sglang `build_step_span_name` rule [F profile_utils.py:462-505]). The analysis assigns each kernel to a
  step through *dispatch -> correlation id -> launching HIP call CPU time -> enclosing forward* - the same rule
  research/a18-trace/analyze.py used on torch traces [F].
- Why mode `window` first: graphs are never captured under tracing, so the capture cannot be perturbed by the
  profiler; the cost is no capture-site attribution (graph_exec_id / capture_correlation_id may be 0). The
  per-phase table does not need it [I; verified in Step 1 check G2]. Mode `capture` (team behaviour) is the
  optional second run (Step 2b).
- **No server_launch.py change is needed** [F]: `server_launch.py` already applies `config["env"]` as `-e` flags
  and already mounts the NFS root at `/yihou/runtime` and the node disk at `/yihou/local`; the SDK build lives on
  the node disk and the hook on NFS. The only new inputs are a config JSON and the files in this directory.
  `LD_LIBRARY_PATH` is set explicitly to `<sdk>/lib:/opt/rocm/lib` because `-e` replaces the image value, which is
  `/opt/rocm/lib` [F research/infx_4881/image_1106.json Env]. The image sets no PYTHONPATH [F same file].

## Files (this directory)

| file | role |
|---|---|
| `prep_source.sh` | this host: sparse checkout of the fork + 13 submodules, pack `stage/rocprof-sdk-src.tar.gz`, cmake/ninja wheels |
| `node/env.sh` | paths: NFS `/shared_nfs/yihou/muse_agentx_opt_20261027/calib_profiler` (CP), node `/mnt/m2m_nobackup/yihou/muse_agentx_opt_20261027/calib_profiler` (LCP), 1106 image digest |
| `node/step0_build.sh`, `node/in_container_build.sh` | detached GPU-less build container `yihou-muse-calib-build`; records image facts |
| `node/step1_smoke.sh`, `node/in_container_smoke.sh` | 1-GPU smoke (`yihou-muse-calib-smoke-<UTC>`) |
| `smoke/probe_torch_graph.py`, `smoke/validate_probe.py` | torch CUDA-graph probe (5 kernel nodes, 20 replays + 10 eager matmuls) and its PASS/FAIL checks |
| `hook/sitecustomize.py` | in-server window control + step labels (no effect unless `CALIB_ENABLE=1`; chains any shadowed sitecustomize) |
| `config/calibprof-tp4-c40-v180p1.json` | base config + profiler env (copy of iterations/infx1106-tp4-nb-c40-v117a1/config.json with `env` additions) |
| `node/step2_profile.sh` | server + C40 load + trigger + drain + finalize + stop (modeled on scripts/prof_cru.sh) |
| `analysis/calib_trace.py`, `analysis/calib_table.py` | trace loader; per-rank VERIFY/DRAFT/EXTEND table (.md + .json) with the torch reference column |
| `analysis/selftest.py` | CPU-only self-test on synthetic json-tool documents; **ran on this host: `selftest OK`** [F] |

Node-side layout: build `LCP/build` (container `/yihou/local/calib_profiler/build`); smoke outputs `LCP/out/`;
server traces `NFS/profiles/calibprof-tp4-c40-v180p1-calib/` (container `/yihou/runtime/profiles/...-calib`).

## Step 0 - toolchain (about 30-60 min, mostly build) [plan; nothing executed]

0a. On this host (network): 
```
timeout 7200 bash /home/yihou/dev/git.16-19/infera.muse/yihou/muse_agentx_opt_20261027/research/calib_profiler/prep_source.sh
```
Branch default `diag/rocm72-capture-node-map-v35` (newest, all rocm72 fixes) [F STUDY 2.1]; `BRANCH=fix/rocprofiler-rocm72-scoped-strict`
is the minimal 7.2 set. The cmake helper skips `git submodule update` when a submodule's CMakeLists.txt exists
[F cmake/rocprofiler_utilities.cmake:158-200], so the packed tree builds offline [I].

0b. Stage to NFS (leader's usual push path; spur005 mounting /shared_nfs is [I]):
```
cd /home/yihou/dev/git.16-19/infera.muse/yihou/muse_agentx_opt_20261027/research
tar --exclude=calib_profiler/repo -czf - calib_profiler | timeout 900 ssh spur005 \
  'mkdir -p /shared_nfs/yihou/muse_agentx_opt_20261027 && tar -xzf - -C /shared_nfs/yihou/muse_agentx_opt_20261027'
# server config where step2/server_launch.py expects it (cwd workspace/scripts -> ../iterations/<tag>/config.json):
timeout 120 ssh spur005 'mkdir -p /shared_nfs/yihou/muse_agentx_opt_20261027/workspace/iterations/calibprof-tp4-c40-v180p1 && cp /shared_nfs/yihou/muse_agentx_opt_20261027/calib_profiler/config/calibprof-tp4-c40-v180p1.json /shared_nfs/yihou/muse_agentx_opt_20261027/workspace/iterations/calibprof-tp4-c40-v180p1/config.json'
```

0c. Build on the node (returns at once; ~20-40 min build [I]; do not run while a scored run is on the node -
the build uses `JOBS` CPU cores):
```
timeout 300 ~/.claude/skills/machine-pool/scripts/spur_remote.sh 'spur exec 285544 -- bash -c "bash /shared_nfs/yihou/muse_agentx_opt_20261027/calib_profiler/node/step0_build.sh 32"' </dev/null
# poll (every ~5 min):
timeout 120 ~/.claude/skills/machine-pool/scripts/spur_remote.sh 'spur exec 285544 -- bash -c "tail -25 /mnt/m2m_nobackup/yihou/muse_agentx_opt_20261027/calib_profiler/logs/build.log; cat /mnt/m2m_nobackup/yihou/muse_agentx_opt_20261027/calib_profiler/build/BUILD_OK 2>/dev/null"' </dev/null
```
Where binaries go: `LCP/build/lib/librocprofiler-sdk.so` and `LCP/build/lib/rocprofiler-sdk/librocprofiler-sdk-json-tool.so`
[F json-tool LIBRARY_OUTPUT_DIRECTORY, tests/tools/CMakeLists.txt]. Build target `rocprofiler-sdk-json-tool` [F];
the rocprofv3 tool target name `rocprofiler-sdk-tool` is [I] and optional.
`build.log` also answers STUDY open question 1 (image ROCm version, `/opt/rocm/lib/librocprofiler-sdk.so*`,
whether the image `rocprofv3` has `--attach`), cmake version, libdw/libelf headers, and whether the image already
ships a `sitecustomize.py` (ours chains it).
Pass: `BUILD_OK` with two sha256 lines; `nm -D` shows `json_tool_{start,stop,pause_until_graph,finalize_idle_worker}`;
`ldd` resolves `librocprofiler-sdk.so` to `LCP/build/lib` (rpath `$ORIGIN:$ORIGIN/..` [F]).

## Step 1 - smoke on one GPU (<= 15 min)

Run on an idle GPU outside the server's 0-3 (default 7); it can run while no server is up, or next to one.
```
timeout 1200 ~/.claude/skills/machine-pool/scripts/spur_remote.sh 'spur exec 285544 -- bash -c "bash /shared_nfs/yihou/muse_agentx_opt_20261027/calib_profiler/node/step1_smoke.sh 7"' </dev/null
```
What runs (inside the 1106 image, server env): (1a) probe mode `window` -> `validate_probe.py` (gating);
(1b) probe mode `capture` (informative: does capture attribution survive `json_tool_stop/start`? = STUDY open
question 3); (1c) team native hipcc probes `run_native.sh` (module 18 capture ids, memcpy 30 dispatches; team
validated on ROCm 10 only [F STUDY 5a]); (1d) image stock `rocprofv3 --kernel-trace` on a tiny torch loop
(baseline that the stock tool works on this image).
Gating checks for mode `window` (all must PASS): G1 every replay shows the same K >= 3 kernels launched by
`hipGraphLaunch`; G2 every window dispatch maps to a HIP API record inside a host bracket (the phase rule);
G3 nothing traced after stop; G4 no missing kernel names; G5 per replay, kernel span and summed time
<= hipEvent time + 0.05 ms. Informative: I1 causality violations (GPU start before its launch call; expect 0),
I2 graph_exec / capture-id coverage, `sdk_maps` must list only `LCP/build/lib/librocprofiler-sdk.so` (README
warning: wheels may load their own SDK [F]).

## Step 2 - Muse TP4 server profile at C40 (~1-1.5 h node time)

Precondition: Step 1a PASS; no `yihou-muse-opt-*` container; start no later than ~12:00Z (lease stop_by 13:46Z;
startup <= 30 min + load ramp + 20 steps + drain + finalize) [A].
```
timeout 120 ~/.claude/skills/machine-pool/scripts/spur_remote.sh 'spur exec 285544 -- bash -c "cd /shared_nfs/yihou/muse_agentx_opt_20261027/calib_profiler/node && nohup bash step2_profile.sh grant.v3180.json calibprof-tp4-c40-v180p1 40 20 > /shared_nfs/yihou/muse_agentx_opt_20261027/profiles/calibprof-tp4-c40-v180p1-calib.driver.log 2>&1 &"' </dev/null
# poll:
timeout 120 ~/.claude/skills/machine-pool/scripts/spur_remote.sh 'spur exec 285544 -- bash -c "tail -5 /shared_nfs/yihou/muse_agentx_opt_20261027/profiles/calibprof-tp4-c40-v180p1-calib.driver.log; ls /shared_nfs/yihou/muse_agentx_opt_20261027/profiles/calibprof-tp4-c40-v180p1-calib | head; cat /shared_nfs/yihou/muse_agentx_opt_20261027/profiles/calibprof-tp4-c40-v180p1-calib/calib-*.log | grep -c patched"' </dev/null
```
(`grant.v3180.json` must be the name the node scripts resolve via `YIHOU_MUSE_GRANT`, as `prof_cru.sh` does [F].)
Sequence in `step2_profile.sh`: `server_launch.py` (unchanged) -> `wait_then_client.py` (gate pdi 24, mem 0.86,
`--tp 4`, 900 s screen load, unscored) -> trigger at running >= 20 for 3 x 10 s (reference capture was bs 26
[F ANALYSIS 7]) -> `touch start_window` -> each TP rank opens at its next forward and closes after 20 VERIFY
forwards (`steps-<pid>.json`) -> stop client, wait running == 0 -> `touch finalize` -> each process writes
`trace-<pid>.json` + `finalized-<pid>` -> docker log to NFS -> `stop_owned.py`.
Graph-hang avoidance: (i) mode `window` traces nothing during capture; (ii) 20 steps (the torch profiler hung at
50-60 steps on 1106, 20-30 worked [F phase3_plan.md machine notes]); (iii) finalize only after the load is
drained; (iv) strict window and all `ROCPROFILER_EXPERIMENTAL_EVENT_*` diagnostics stay OFF (a strict violation is
`ROCP_FATAL` -> rank abort [F STUDY 6]).
Early checks while it runs: `calib-<pid>.log` has `patched ModelRunner.forward` in the 4 scheduler processes and
`sdk_maps` points at `LCP/build`; server reaches `/health` 200 within the 1800 s startup budget.

Step 2b (optional, only if 2 succeeded and capture-site attribution is wanted, and Step 1b showed I2 > 0):
same config with tag `calibprof-tp4-c40-v180p2` and `CALIB_MODE=capture` (traces capture from the first
`CUDAGraph.capture_begin`, stops on the first forward >= 60 s after the last `capture_end`; bigger trace,
capture under HIP-API tracing - higher hang risk [I]). Then also run the team's `inspect_graph.py` per trace.

## Step 3 - outputs and the comparison table

Outputs per scheduler rank in `NFS/profiles/calibprof-tp4-c40-v180p1-calib/`: `trace-<pid>.json` (json-tool),
`steps-<pid>.json` (window bounds, per-forward stage/bs/extend tokens), `calib-<pid>.log`, `finalized-<pid>`;
non-scheduler processes also leave small trace files. Server log: `NFS/iterations/calibprof-tp4-c40-v180p1/`.

Table (off-node, CPU only; copy the trace dir to `scratch/` or run on the node host with python3):
```
python3 -I research/calib_profiler/analysis/calib_table.py <trace_dir> --out <yihou dir>/calibprof-v180p1-table
python3 -I research/calib_profiler/analysis/calib_table.py <trace_dir> --out <yihou dir>/calibprof-v180p1-bs24-28 --bs-min 24 --bs-max 28
```
It prints per rank: window ms, dispatches, `unattributed_no_api`, `other_outside_forward` (ms), causality
violations, the `muse_verify_fly` grid groups and their full/SWA split (threshold `--full-attn-min-us 150`; expect
~16 full and ~46 SWA calls per VERIFY step [F ANALYSIS 7.1]), and per stage: steps, bs (median/min/max), median
GPU span per step, kernel ms per step, **bucket ms/step next to the torch reference**, top-25 kernels
(calls/step, ms/step, median us). Buckets = ANALYSIS 7.1 families (regexes in `calib_table.py`).
Comparison rules [I, to be applied when reading]: compare only VERIFY steps with bs near 26 (filter), rank 0
first then TP1-3; a bucket difference > 5 % with the same calls/step is a timing difference between the tools
(the subject of this study); a calls/step difference is a classification/regex problem, fix before reading
times. Also compare VERIFY GPU span per step with 42.62 ms and DRAFT total with 2.41 ms. Different node
(v3-006 vs v3-180) and different request mix: node-to-node spread up to ~9-10 % [F CLAUDE.md, KB] - treat
< 10 % totals as not different.
Profiler overhead outside the window [I: the intercept queue stays installed]: compare `Decode batch` gen
throughput at running 20-30 in this run's server.log with the base run logs via
`kf2/tasks/phase3/stepbins.py`; report the ratio, do not use the run for any score.

## Stop conditions

| id | condition | action |
|---|---|---|
| S0-deps | cmake < 3.24 and no staged wheel, or libdw/libelf headers missing and no apt | stop; fallback = STUDY option C (image rocprofv3 `--attach`, kernel names + durations only) if `build.log` shows `--attach` |
| S0-build | configure/build fails twice after fixing obvious deps | stop; record the first error; try `BRANCH=fix/rocprofiler-rocm72-scoped-strict` once |
| S1 | any G check FAILs in mode `window` | do not start Step 2; G2 FAIL alone means the phase rule is broken -> fix the analysis/hook first |
| S2-start | server not healthy within 1800 s, or a rank dies (`ROCP_FATAL`, HSA error, core) | stop_owned, keep logs, no retry with the profiler on this lease |
| S2-window | no `steps-*.json` 10 min after `start_window`, or `/health` fails during the window | drain + finalize anyway, stop server, report as hang |
| S2-final | `finalized` < number of steps files 20 min after `finalize` | stop server (atexit may still flush), report |
| S3 | `unattributed_no_api` + `other` > 5 % of VERIFY kernel time, or causality violations > 0 | report the table as provisional; do not compare buckets |
| lease | after ~12:00Z | do not start Step 2 |

## Rollback

Nothing in the image, `server_launch.py`, the overlays or the base config changes. Rollback = launch the base
config (`iterations/infx1106-tp4-nb-c40-v117a1/config.json` pattern) without the profiler env. Owned artifacts:
containers `yihou-muse-calib-build`, `yihou-muse-calib-smoke-*` (both `--rm`), `yihou-muse-opt-calibprof-*`;
files only under `LCP` (node disk) and `NFS/calib_profiler`, `NFS/profiles/calibprof-*`, `NFS/iterations/calibprof-*`,
`NFS/workspace/iterations/calibprof-*`. Cleanup, when wanted, deletes those literal paths only.

## Unknowns that this plan resolves on the node (do not assume)

1. Image ROCm/rocprofiler versions, `rocprofv3 --attach`, cmake version, libdw/libelf headers (Step 0 log).
2. Whether the fork SDK (1.5.0, develop base) builds against ROCm 7.2.0 headers (Step 0).
3. Whether the json-tool initialises at `import torch` so the first pause takes effect; the hook re-pauses every
   2 s until the window as a safety net, and the analysis keeps only dispatches launched inside the window.
4. Whether graph-replay dispatches carry the `hipGraphLaunch` correlation id (G1/G2), and whether capture
   attribution survives stop/start (1b I2).
5. Whether the DFlash draft model runs in the same scheduler process (expected 4 steps files; more is handled).
6. Profiler overhead with the intercept queue installed but contexts stopped.
