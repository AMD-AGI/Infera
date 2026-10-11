
## 2026-09-14 05:00-05:13 (bench-runner)
- Attempt 1 (`tp8_ep8_dpa_on_c128_yihou`) FAILED at CUDA-graph capture: `launch_status.json = {"exit_code":1,"launch_wall_seconds":779}`.
  Error: `Exception: Capture cuda graph failed: PermissionError: [Errno 13] Permission denied: 'nvcc'`
  raised from torch inductor's `subprocess.check_output(["nvcc","--version"])`.
- Reported to team-lead. Team-lead root-caused: image PATH began with `/root/.cargo/bin`
  (mode drwx------ root root); non-root PATH search into that dir raised EACCES -> Python
  surfaced PermissionError instead of FileNotFoundError. Not an OOM; no `nvcc` exists anywhere
  in the image regardless.
- Fix (team-lead): `create_container_yihou.sh` now passes explicit
  `-e PATH=/opt/venv/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/go/bin`
  (drops /root/.cargo/bin). Still non-root; no switch to root-container fallback.
- Actions taken:
  1. `mv iterations/tp8_ep8_dpa_on_c128_yihou iterations/aborted_nvcc_path_c128_yihou` (kept as evidence).
  2. `docker rm -f yihou-glm52-tp8ep8-0914` (our own container) + recreated via `create_container_yihou.sh`.
  3. Sanity check inside new container: `echo $PATH` shows no `/root/.cargo/bin`;
     `shutil.which("nvcc")` returns `None` cleanly (no exception).
  4. Relaunched: `setsid nohup bash scripts/run_tp8_ep8_dpa_yihou.sh 128 > logs/driver_c128_yihou.log 2>&1 &`
     under original iteration name `tp8_ep8_dpa_on_c128_yihou`. No benchmark parameters changed.
  5. Resuming monitoring.

## 2026-09-14 05:18-05:33 (bench-runner)
- Attempt 2 (`tp8_ep8_dpa_on_c128_yihou`, container recreated with PATH fix) got PAST CUDA-graph
  capture successfully (target verify ~130s x3 stages, draft decode 56.5s, draft extend 135.4s;
  KV cache allocated fine: #tokens=3448128/rank, KV size 153.18 GB, no OOM).
  Then FAILED at `allocate_batch -> batch.prepare_for_extend() -> alloc_req_slots`:
  `RuntimeError: alloc_req_slots runs out of memory ... req_to_token_pool.available_size()=6, num_reqs=16`
  `launch_status.json = {"exit_code":1,"launch_wall_seconds":491}`.
- Reported to team-lead. Team-lead root-caused (first-hand, via prepare_server_args + reading
  kv_cache_configurator.py:1880-1883): `max_running_requests` auto-derives to 48 for speculative
  decoding (never explicitly passed by us); with attn_dp_size=8, ReqToTokenPool size =
  `max_running_requests // attn_dp_size` = `48 // 8` = 6 slots/worker, but local batch = 128/8 = 16
  slots needed. NOT a KV-memory problem (KV check already passed; do not touch --mem-fraction-static).
  Note: spur packup at TP4/C=48 never hit this because there attn_dp_size=4 -> 48//4=12, exactly
  matching its local batch of 12 — it was sitting exactly at the limit.
- Fix (team-lead, applied to run_tp8_ep8_dpa_yihou.sh): added `--max-running-requests 128`
  (-> 128//8=16 slots/worker, exactly the requested local batch, no slack). This is a documented
  deviation from the packup's server_cli (adds a request-slot-table sizing flag), not a performance
  tuning knob — record it in REPORT.md deviations section alongside image digest and non-root container.
- Moved failed dir: `iterations/tp8_ep8_dpa_on_c128_yihou` -> `iterations/aborted_reqpool_c128_yihou`
  (confirmed already done).
- Relaunched (container unchanged, no recreate needed): `setsid nohup bash
  scripts/run_tp8_ep8_dpa_yihou.sh 128 > logs/driver_c128_yihou.log 2>&1 &` under original name.
- Verified `config_yihou.json` server_cli now contains `--max-running-requests 128`; all other
  flags unchanged (--tp-size 8 --ep-size 8 --dp-size 8 --cuda-graph-bs-decode 16
  --mem-fraction-static 0.85, etc.). Resuming monitoring.

## 2026-09-14 05:38-05:45 (bench-runner)
- Attempt 3 SUCCEEDED. Full lifecycle observed live: shard loading -> AITER JIT (all modules) ->
  CUDA-graph capture (target verify, draft decode, draft extend all completed) -> measured decode
  loop reached iteration 2768, useful=1280000/rank-sum.
- `launch_status.json = {"exit_code":0,"launch_wall_seconds":296}`.
- `result_yihou.json`: complete=true, useful_output_tokens=1280000, realized_accept_length=
  3.6134393063583814, verify_iterations=2768, tp_size=8/ep_size=8/dp_size=8/enable_dp_attention=true,
  local_batch_size=16, memory.reserved_tokens=1281024 (matches mission's expected value exactly).
- Ran `scripts/verify_point_yihou.py` -> `results/verify_c128_yihou.json`: verdict="pass", exit=0,
  problems=[].
- Reported success to team-lead.
- Post-processing: gzip -9 on runtime.log and console.log inside
  iterations/tp8_ep8_dpa_on_c128_yihou/ (in place; both were identical, 169119 bytes -> 11664 bytes
  each). Wrote results/REPORT.md (English) with command, measured TPOT/throughput/per-GPU
  throughput, phase_seconds, memory figures, environment, and a full "deviations from the spur
  packup" section (image digest, non-root container + 3 bind-mounts/PATH fix, added
  --max-running-requests 128, each with first-hand root-cause evidence). Listed all 5 aborted
  iteration dirs as kept evidence.
- Task complete pending team-lead review.

## 2026-09-14 05:46 (bench-runner) — Phase 2: concurrency sweep launched
- Re-read updated mission.md / CLAUDE.md: task extended to sweep C=48,64,96,128,160,192,224,256,
  288,320 at same TP8/EP8/DPA-on config. C=128 already done, reused.
- Container yihou-glm52-tp8ep8-0914 reused as-is (AITER JIT warm), not recreated.
- Launched: `setsid nohup bash scripts/run_tp8_ep8_dpa_yihou.sh 48 64 96 160 192 224 256 288 320 >
  logs/driver_sweep_yihou.log 2>&1 &` (driver pid 1191081), sequential single detached process.
  --max-running-requests=<C> passed per point automatically by the script.
- Starting monitoring loop, same discipline (wc -c / tail -c N only).

## 2026-09-14 05:48-05:52 (bench-runner)
- C=48 completed: exit_code=0, launch_wall_seconds=215, useful_output_tokens=480000,
  verify_iterations=2768. Gzipped runtime.log/console.log.
- C=64 started (model reload in progress; each sweep point is a fresh process re-loading the model).

## 2026-09-14 05:52-05:56 (bench-runner)
- C=64 completed: exit_code=0, launch_wall_seconds=225, useful_output_tokens=640000,
  verify_iterations=2768. Gzipped runtime.log/console.log.
- C=96 started (model loading, fast now - page cache warm, ~2-3s for shards).

## 2026-09-14 05:56-06:00 (bench-runner)
- C=96 completed: exit_code=0, launch_wall_seconds=256, useful_output_tokens=960000,
  verify_iterations=2768. Gzipped runtime.log/console.log.
- C=160 started (model loading).

## 2026-09-14 06:00-06:04 (bench-runner)
- C=160 completed: exit_code=0, launch_wall_seconds=303, useful_output_tokens=1600000,
  verify_iterations=2768. Gzipped runtime.log/console.log.
- C=192 started.

## 2026-09-14 06:04-06:08 (bench-runner)
- C=192 completed: exit_code=0, launch_wall_seconds=332, useful_output_tokens=1920000,
  verify_iterations=2768. Gzipped runtime.log/console.log.
- C=224 started.

## 2026-09-14 06:08-06:12 (bench-runner)
- C=224 completed: exit_code=0, launch_wall_seconds=(see launch_status.json), useful_output_tokens=2240000,
  verify_iterations=2768. Gzipped runtime.log/console.log.
- C=256 started.

## 2026-09-14 06:12-06:16 (bench-runner)
- C=256 completed: exit_code=0, launch_wall_seconds=(see launch_status.json), useful_output_tokens=2560000,
  verify_iterations=2768. Gzipped runtime.log/console.log.
- C=288 started (capacity risk zone flagged by team-lead; watching closely).

## 2026-09-14 06:16-06:22 (bench-runner)
- C=288 completed successfully: exit_code=0, useful_output_tokens=2880000, verify_iterations=2768.
  KV pool unchanged at #tokens=3448128/rank. No capacity issue observed. Gzipped logs.
- C=320 started (last point, highest capacity risk per team-lead's arithmetic: local_batch=40,
  40*80064=3202560 = 93% of the 3448128-token pool). Watching closely for OOM / capacity errors.

## 2026-09-14 06:22-06:30 (bench-runner) — sweep complete
- C=320 completed successfully: exit_code=0, launch_wall_seconds=401, useful_output_tokens=3200000,
  verify_iterations=2768. KV pool unchanged (#tokens=3448128/rank); local_batch=40 *
  reserved_tokens_per_request=80064 = 3,202,560 = 92.9% of pool, fit with no OOM/capacity error.
  Gzipped runtime.log/console.log.
- All 9 sweep points (48,64,96,160,192,224,256,288,320) completed with exit_code=0; combined with
  the pre-existing C=128 point, all 10 concurrency points are measured.
- Driver process (pid 1191081) exited; entire sweep finished, no failures at any C, including the
  highest-capacity-risk C=320.
- Ran `scripts/collect_sweep_yihou.py iterations --output results/sweep_summary_yihou.csv`:
  exit=0, all 10 rows verdict=pass, no problems.
- Verified realized_accept_length=3.6134393063583814 and verify_iterations=2768 identical across
  ALL 10 rows (no configuration drift across the sweep).
- Sweep summary (C, local_batch, TPOT ms, tok/s, tok/s/GPU):
    C=48   local=6   TPOT=12.083  tok/s=3972.46   tok/s/gpu=496.56
    C=64   local=8   TPOT=13.143  tok/s=4869.57   tok/s/gpu=608.70
    C=96   local=12  TPOT=16.238  tok/s=5912.10   tok/s/gpu=739.01
    C=128  local=16  TPOT=17.848  tok/s=7171.77   tok/s/gpu=896.47
    C=160  local=20  TPOT=20.843  tok/s=7676.61   tok/s/gpu=959.58
    C=192  local=24  TPOT=22.725  tok/s=8448.81   tok/s/gpu=1056.10
    C=224  local=28  TPOT=24.216  tok/s=9250.17   tok/s/gpu=1156.27
    C=256  local=32  TPOT=26.252  tok/s=9751.66   tok/s/gpu=1218.96
    C=288  local=36  TPOT=28.918  tok/s=9959.15   tok/s/gpu=1244.89
    C=320  local=40  TPOT=30.772  tok/s=10399.09  tok/s/gpu=1299.89

## 2026-09-14 06:35 (bench-runner) — Phase 3: EP1 sweep launched
- Launched after EP8 sweep + collector fully finished: `setsid nohup bash
  scripts/run_tp8_noep_dpa_yihou.sh 48 64 96 128 160 192 224 256 288 320 >
  logs/driver_noep_sweep_yihou.log 2>&1 &` (pid 1688461).
- Same container yihou-glm52-tp8ep8-0914 reused (not recreated) - single-variable EP8-vs-EP1
  comparison on same image/AITER cache/host.
- All 10 C values including C=128 (no EP1 point exists yet).
- Resuming monitoring loop.

## 2026-09-14 06:37-06:41 (bench-runner) — Phase 3 EP1 sweep progress
- EP1 C=48 completed: useful_output_tokens=480000, verify_iterations=2768. Gzipped logs.
- EP1 C=64 started.

## 2026-09-14 06:41-06:45 (bench-runner) — Phase 3 EP1 sweep progress
- EP1 C=64 completed: useful_output_tokens=640000, verify_iterations=2768. Gzipped logs.
- EP1 C=96 started.

## 2026-09-14 06:45-06:49 (bench-runner) — Phase 3 EP1 sweep progress
- EP1 C=96 completed: useful_output_tokens=960000, verify_iterations=2768. Gzipped logs.
- EP1 C=128 started.

## 2026-09-14 06:49-06:53 (bench-runner) — Phase 3 EP1 sweep progress
- EP1 C=128 completed: useful_output_tokens=1280000, verify_iterations=2768. Gzipped logs.
- EP1 C=160 started.

## 2026-09-14 06:53-06:58 (bench-runner) — Phase 3 EP1 sweep progress
- EP1 C=160 completed: useful_output_tokens=1600000, verify_iterations=2768. Gzipped logs.
- EP1 C=192 started.

## 2026-09-14 06:58-07:03 (bench-runner) — Phase 3 EP1 sweep progress
- EP1 C=192 completed: useful_output_tokens=1920000, verify_iterations=2768. Gzipped logs.
- EP1 C=224 started.

## 2026-09-14 07:03-07:08 (bench-runner) — Phase 3 EP1 sweep progress
- EP1 C=224 completed: useful_output_tokens=2240000, verify_iterations=2768. Gzipped logs.
- EP1 C=256 started.

## 2026-09-14 07:08-07:14 (bench-runner) — Phase 3 EP1 sweep progress
- EP1 C=256 completed: useful_output_tokens=2560000, verify_iterations=2768. Gzipped logs.
  Note: EP1 KV pool #tokens=3471936 (per rank), slightly different from EP8's 3448128 --
  observation only, not explained.
- EP1 C=288 started.

## 2026-09-14 07:14-07:20 (bench-runner) — Phase 3 EP1 sweep progress
- EP1 C=288 completed: exit_code=0, launch_wall_seconds=383, useful_output_tokens=2880000,
  verify_iterations=2768. No capacity issue. Gzipped logs.
- EP1 C=320 starting (last point).

## 2026-09-14 07:20-07:28 (bench-runner) — Phase 3 EP1 sweep complete
- EP1 C=320 completed: exit_code=0, launch_wall_seconds=404, useful_output_tokens=3200000,
  verify_iterations=2768. No capacity issue (KV pool #tokens=3471936/rank throughout EP1 sweep,
  consistently different from EP8's 3448128 -- observation only). Gzipped logs.
- All 10 EP1 points (48,64,96,128,160,192,224,256,288,320) completed exit_code=0, no failures.
- Ran `scripts/collect_sweep_yihou.py iterations --prefix tp8_noep_dpa_on_c --expect-ep 1
  --output results/sweep_summary_noep_yihou.csv`: exit=0, all 10 rows verdict=pass.
- EP8-vs-EP1 delta table (TPOT ms, tok/s; delta% = (EP1-EP8)/EP8*100):
    C=48   EP8=12.0832  EP1=10.5585  TPOT delta=-12.62%  | EP8=3972.46  EP1=4546.11  tok/s delta=+14.44%
    C=64   EP8=13.1428  EP1=11.9746  TPOT delta=-8.89%   | EP8=4869.57  EP1=5344.65  tok/s delta=+9.76%
    C=96   EP8=16.2379  EP1=14.9122  TPOT delta=-8.16%   | EP8=5912.10  EP1=6437.67  tok/s delta=+8.89%
    C=128  EP8=17.8478  EP1=17.3231  TPOT delta=-2.94%   | EP8=7171.77  EP1=7388.97  tok/s delta=+3.03%
    C=160  EP8=20.8425  EP1=19.7278  TPOT delta=-5.35%   | EP8=7676.61  EP1=8110.39  tok/s delta=+5.65%
    C=192  EP8=22.7251  EP1=21.9950  TPOT delta=-3.21%   | EP8=8448.81  EP1=8729.24  tok/s delta=+3.32%
    C=224  EP8=24.2158  EP1=24.2623  TPOT delta=+0.19%   | EP8=9250.17  EP1=9232.45  tok/s delta=-0.19%
    C=256  EP8=26.2519  EP1=25.6740  TPOT delta=-2.20%   | EP8=9751.66  EP1=9971.17  tok/s delta=+2.25%
    C=288  EP8=28.9181  EP1=28.3732  TPOT delta=-1.88%   | EP8=9959.15  EP1=10150.44 tok/s delta=+1.92%
    C=320  EP8=30.7719  EP1=30.4108  TPOT delta=-1.17%   | EP8=10399.09 EP1=10522.58 tok/s delta=+1.19%
  Note: sign flips at C=224 (EP1 slightly slower there, all other points EP1 faster). Flagging as
  open question per instructions, not explaining it -- each point is a single run, no repeats, so a
  few percent (and a sign flip near zero) is within possible run-to-run variation, unattributed.
- Next: update results/REPORT.md with this table and gzip already done during monitoring.

## 2026-09-14 07:28 (bench-runner) — Phase 3 complete, report updated
- Updated results/REPORT.md: added "Phase 3: EP8 vs EP1" section with full delta table, flagged
  the C=224 sign-flip as an open question (not explained), noted the KV pool size difference as
  an unexplained secondary observation, and updated the failed-attempts section to note both
  sweeps had zero new failures.
- Task complete pending team-lead review.

## 2026-09-14 (bench-runner) — accuracy fix per team-lead review
- Fixed the C=352 capacity-arithmetic paragraph in results/REPORT.md: now states both pool sizes
  (EP8=3,448,128 and EP1=3,471,936 tokens/rank) instead of only EP8's, and notes 3,522,816 (C=352's
  reservation need) exceeds both. Also clarified the +2.9GB max_memory_allocated_bytes growth was
  measured only on the EP8 sweep, not claimed for EP1.
- Both phases (EP8 sweep, EP1 sweep) confirmed accepted by team-lead; no further phases assigned.
  Going idle.
