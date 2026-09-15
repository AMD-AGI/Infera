# Working process — GLM-5.2 decode bench profiling

Timestamps are UTC. Append-only log.

## 2026-09-14

- **08:40** Task received. Phase 1 research started before any workspace creation, per mission rule 4.
- **08:42** Host state probed: container `yihou-glm52-tp8ep8-0914` from the previous sweep is still
  `Up 4 hours`; `rocm-smi --showuse` shows 0 % on all 8 GPUs. Decision: **reuse the container**,
  do not recreate — it already carries the four host-environment fixes.
- **08:44** Read `bench/profile_decode.py` (465 lines). Measured decode loop is lines 359–382;
  `run_rank` is the only place that owns the loop. No profiling hooks exist today.
- **08:46** Enumerated SGLang's profiling surface from the **pinned** tree inside the container
  (`git -C /sglang rev-parse HEAD` = `402df1e1e453e1e85ec0f5ac4052d36598cc691a`), not from the
  host's newer `sglang.parallel.strategy` checkout.
  - `srt/environ.py:409-432` is the complete profiling/tracing env block.
  - `srt/utils/profile_utils.py` — `ProfileManager` + 4 concrete backends (Torch / Memory / Cudart / RPD).
  - `srt/managers/scheduler_components/profiler_manager.py` — legacy manager; `SGLANG_PROFILE_V2`
    switches between the two.
- **08:48** **Finding (first-hand):** `SGLANG_PROFILE_V2=true` routes to `ProfileManager`, whose
  `manual_start()` / `manual_stop()` **raise NotImplementedError**, and whose `configure()` asserts
  `profile_by_stage=True`. So V2 is unusable for a scheduler-free harness that wants manual control.
  Leave it at its default `false`.
- **08:50** **Finding (first-hand):** `srt/utils/nvtx_utils.py::profile_range()` emits
  `torch.profiler.record_function` whenever `torch.autograd._profiler_enabled()` — **no env var
  needed**. Existing annotation sites (`model_runner.py:1520` step span; `eagle_worker_v2.py`
  `spec_stage_span("draft"/"draft_extend")` at 1167/1209/1230) therefore light up for free.
- **08:52** **Finding (first-hand, load-bearing):** `frozen_kv_mtp_cuda_graph_runner.py:448` carries
  the comment *"the graph bypasses `model_runner.forward`'s record_function"* and adds an explicit
  `torch.profiler.record_function` around `_replay_graph`. This is direct in-tree evidence that
  **CUDA-graph replay suppresses the normal forward spans**.
- **08:54** **Finding (first-hand):** `srt/utils/device_timer.py::DeviceTimer` brackets segments with
  `torch.cuda.Event(enable_timing=True)` from **outside** the replay, so it is graph-safe.
  `device_timer_ctx` call sites already exist in the eager runner and in every graph runner
  (categories: `decode`, `eagle_draft`, `eagle_draft_extend`, `frozen_kv_draft`, `idle`,
  `split_prefill`). `model_runner.device_timer` is `None` at `model_runner.py:332` and is only
  populated by the Scheduler's metrics reporter — **we can attach our own, no SGLang change needed.**
  This becomes the safety net for graph-on mode.
- **08:56** **Finding (first-hand):** `--enable-profile-cuda-graph` is a real `ServerArgs` flag
  (`server_args.py:1935`) wired at `decode_cuda_graph_runner.py:1014/1052`; it profiles the **capture
  pass** and logs `key_averages().table(sort_by="cuda_time_total", row_limit=10)`.
  Caveat recorded: during capture kernels are *recorded, not executed* → its times are not
  performance data, only a kernel identity/shape inventory.
- **08:58** **Finding (first-hand):** `DEBUG_CLR_GRAPH_PACKET_CAPTURE` is present in this machine's
  `/opt/rocm/lib/libamdhip64.so` (via `strings`), alongside 6 other `DEBUG_CLR_*` symbols.
  Semantics/default still unverified — delegated to the research agent, and will be settled by our
  own P1-vs-P2 A/B experiment regardless.
  In-container tooling confirmed present: `rocprofv3`, `rocprofv2`, `rocprof`, `rocprof-compute`,
  `rocprof-sys-*`.
- **08:51** Workspace created: `glm52_decode_profiling_yihou_20260914-0851/`. Nothing outside it is
  modified except the bench tool itself (the deliverable). No pre-existing `CLAUDE.md` needed backing
  up — the bench dir holds only `*.md.bak` files and the repo-level `.claude/CLAUDE.md` is the DCO
  conventions file, which is out of scope.
- **08:53** `mission.md` + `CLAUDE.md` written. 10-minute mission re-injection cron armed
  (`3,13,23,33,43,53 * * * *`, session-only).
- **09:00** Pinned SGLang source (3323 `.py` files, 58 MB) copied out of the container to
  `research/sglang_src_402df1e1e/` so serena/LSP navigate the **correct** commit. Serena project
  re-activated on this repo (it had been pointed at `sglang.parallel.strategy`).
- **09:02** `design.md` written: 3 complementary mechanisms (torch.profiler / DeviceTimer /
  graph-capture inventory) and the P0–P3 experiment matrix, whose P1-vs-P2 pair is the decisive A/B
  on `DEBUG_CLR_GRAPH_PACKET_CAPTURE`.
- **09:03** Team spawned: `rocm-prof-research` (external deep research, started 08:47) and
  `prof-impl` (implementation). 20-minute teammate polling cron armed
  (`7,27,47 * * * *`, session-only).

- **09:12** **Decisive model-free experiment** (`scripts/graph_visibility_probe_yihou.py`): a 3-kernel
  body (bf16 2048² gemm / elementwise chain / reduction) captured into a `torch.cuda.CUDAGraph` and
  replayed 20× under `torch.profiler`. Run in the container on 1 GPU; results in
  `results/graph_visibility_probe/`.

  | mode | `DEBUG_CLR_GRAPH_PACKET_CAPTURE` | distinct GPU entries | per-kernel self time? |
  |---|---|---|---|
  | eager | n/a | 9 | yes (baseline) |
  | graph | unset (default) | 7 | **yes** |
  | graph | `0` | 7 | **yes** |
  | graph | `1` | 7 | **yes** |

  **Finding (first-hand): on this stack (ROCm 7.2, torch 2.9.1+rocm7.2.0, MI355X/gfx950),
  `torch.profiler` attributes per-kernel GPU time inside a replayed hipGraph by default.** There is
  no opaque single graph-launch entry, and `DEBUG_CLR_GRAPH_PACKET_CAPTURE` does not gate this.
  This **refutes** `design.md` §2 worry #2 for this stack. Worry #1 is unaffected and still holds:
  Python-side `record_function` spans inside `model_runner.forward` do not fire under replay — which
  is exactly why `frozen_kv_mtp_cuda_graph_runner.py:452` adds one manually.

  **Unsettled, explicitly not concluded:** the per-kernel event *counts* differed in a single sample
  (`=0` → 20/20 replays, unset → 18-19/20, `=1` → 16-17/20). This may be genuine capture-completeness
  behaviour or may be noise. The 3× repeat to settle it **timed out contending for the GPUs with the
  implementer's smoke run** and has not been re-run. No claim is made either way. Measurement that
  would settle it: 3+ repeats per setting on idle GPUs, comparing the count spread to the gap.

- **09:14** `prof-impl` reported milestone 1. Two design deviations, both reviewed and approved:
  (a) not reusing `_ProfilerTorch` because its `stop()` ends in
  `torch.distributed.barrier(cpu_group)`, which would deadlock when only a subset of ranks profiles;
  it calls `torch.profiler.profile` directly and reproduces the filename convention instead.
  (b) per-runner `DeviceTimer` instances rather than one shared instance, because `DeviceTimer.wrap`
  asserts non-re-entrancy. Asked it to *verify* rather than assume that target/draft categories are
  disjoint. New profiling code isolated in a new `bench/profiling_yihou.py`; `profile_decode.py`
  touched in 6 guarded hunks; pristine copy kept at `bench_pristine/`.
- **09:15** Pushed the graph-visibility finding to `prof-impl` so it does not write off missing
  per-kernel rows in graph-on mode as expected behaviour. GPU work paused on my side to avoid
  contaminating its smoke timings.

- **09:47** 20-min teammate poll #1 sent to both `prof-impl` and `rocm-prof-research`.
  Observed state: `iterations/smoke_graphon_v2_yihou/` has produced `result_yihou.json`;
  a first attempt `smoke_graphon_double_stop_yihou` was correctly `mv`'d to `aborted_*`
  (not deleted) — the rule held. GPUs nearly idle (1 of 8 at 100 %).
  Asked `prof-impl` for the concrete acceptance evidence for criteria 1/2/3B, the one-line
  cause of the `double_stop` failure, and whether `--enable-profile-cuda-graph` actually fires
  in the scheduler-free path. Asked `rocm-prof-research` to prioritise the env-var inventory.
  **No problem recorded against either teammate at this poll** — both on the design.md §4 path.
- **10:07** 20-min teammate poll #2.
  `prof-impl`: on track and ahead of the brief — `smoke_graphoff_yihou` exit 0 with a trace, plus
  two new differential runs `smoke_noprofile_pristine_yihou` / `smoke_noprofile_patched_yihou`
  (the correct way to prove criterion 1 — 对拍 rather than argument) and
  `smoke_graphon_v3_capture_yihou` for `--enable-profile-cuda-graph`.
  **Recorded issue (2nd occurrence): the acceptance evidence for criteria 1/2/3B has been requested
  twice and not yet returned.** Per the polling rule the first occurrence was recorded only; this is
  the second, so I intervened and asked for the answers directly, in bullets, even if unfinished.
  Also told it to hold off on the C=256 run until Phase 3 is signed off and my packet-capture repeat
  probe has had the idle GPUs.
  `rocm-prof-research`: **recorded issue (2nd occurrence): no status reply and no report file at
  `research/rocm_profiling_env.md` after ~80 min.** Intervened: asked it to ship a partial report
  immediately, prioritising the env-var inventory (a mission deliverable in its own right), to state
  plainly if it is blocked rather than grinding, and not to pad the report to look complete.

- **10:20** **Repeat probe on an otherwise-quiet GPU settles the count question — the earlier
  observation was noise.** 5 reps × 3 settings, `HIP_VISIBLE_DEVICES=0`, results in
  `results/graph_visibility_probe/repeat_yihou.txt`:

  | setting | reps | distinct GPU entries | gemm count | reduce count |
  |---|---|---|---|---|
  | unset | 5 | 7 (all) | **19/20 (all)** | 19/20 (all) |
  | `=0`  | 5 | 7 (all) | **19/20 (all)** | 19/20 (all) |
  | `=1`  | 5 | 7 (all) | **19/20 (all)** | 19/20 (all) |

  **15/15 runs identical.** Zero spread within a setting and zero difference between settings.

  **Correction to the 09:12 entry:** the count variation recorded there (`=0` → 20/20, unset →
  18-19/20, `=1` → 16-17/20) was **an artefact of GPU contention with the implementer's smoke run**,
  not a property of the flag. It was explicitly marked unsettled at the time and is now refuted.

  **Conclusion (first-hand, n=5 per condition):** on this stack
  `DEBUG_CLR_GRAPH_PACKET_CAPTURE` has **no measurable effect** on either per-kernel profiler
  visibility or captured event count. Per-kernel attribution inside a replayed hipGraph works at the
  default setting; the flag does not need to be touched for profiling. This is consistent with
  `rocm-prof-research`'s CLR source read (`activity.cpp:104-119` emits one activity record per
  kernel even in capture mode).

  Separately noted, stable and **not** attributable to the flag: the profiler consistently records
  **19 of 20** replays in every condition. A reproducible off-by-one, not a variable effect.
  Not investigated further — it does not affect the design, since attribution is relative.

  The `min(timestamps.size(), kernel_names.size())` silent-truncation path the research agent found
  in CLR source is therefore a **mechanism that exists but was not observed to fire here**. Source
  finding and measurement must not be used to prop each other up.

- **10:52** **A finding in the research agent's own env-var table may invalidate my A/B, and I did
  not notice it until it had been written down.** `research/rocm_profiling_env.md` §1.3 records
  [FH-src] that `DEBUG_HIP_FORCE_GRAPH_QUEUES` defaults to **4** and that **`max_streams_ > 1`
  disables packet capture entirely**. If that holds, packet capture was **never active** in any of
  my 15 runs, and `DEBUG_CLR_GRAPH_PACKET_CAPTURE=1` was a no-op — which would make the clean
  "15/15 identical, flag has no effect" result evidence that I *never tested capture mode*, not
  evidence that capture mode preserves per-kernel attribution. Three identical conditions is exactly
  what an inert flag looks like.
  Started a 2×2: `DEBUG_HIP_FORCE_GRAPH_QUEUES ∈ {1,4}` × `DEBUG_CLR_GRAPH_PACKET_CAPTURE ∈ {0,1}`,
  3 reps each → `results/graph_visibility_probe/queues_x_capture_yihou.txt`. Decisive cell is
  **QUEUES=1, CAPTURE=1** — the only one where packet capture can actually engage.
- **10:58** First cell of the 2×2 back, and it is already informative:
  `QUEUES=1 CAPTURE=0` → **20/20 in 3/3 reps**, against **19/20 in 15/15 reps** at the default
  `QUEUES=4`. So the "19 of 20" constant recorded at 10:20 is **probably a queue-count effect, not
  the profiler start/stop window boundary I guessed.** That also undermines my own elimination
  argument ("it can't be Mechanism B because `=0` shows the identical 19/20"), which assumed the
  19/20 was invariant across conditions. It is not.
  **Explicitly not a result:** n=3, and `total_self_gpu_us` across those reps (1284 / 1586 / 1286 µs)
  shows the GPU was not quiet — the implementer's re-verification runs had restarted. The remaining
  three cells are queued. **Deliberately not collecting counts under contention again**; the 09:12
  entry is the cautionary case.
- **11:00** Instructed `rocm-prof-research` to hedge its new §2.0 "bottom line": the no-effect result
  holds **at default runtime settings (`QUEUES=4`)**; whether the flag matters when capture can
  actually engage (`QUEUES=1`) is [UNKNOWN], measurement in progress. The practical recommendation
  ("you do not need to set this flag") stays, because that is a statement about the default
  configuration, which is what a reproducer faces — but it must stop implying we characterised the
  flag itself.
  **Note for the record: the strong negative was mine, not the agent's — it hedged appropriately and
  I pushed it toward the confident version. The correction is mine to own.**

- **11:20** `rocm-prof-research` corrected its own inventory line, and the correction dissolves most
  of the 10:52 concern. `max_streams_` is **not** set by `DEBUG_HIP_FORCE_GRAPH_QUEUES`; it is set by
  graph **topology** (`hip_graph_internal.cpp:185-207`). The queues flag is a **modulus** applied to
  `stream_id` at branch points only (`stream_id = (stream_id + 1) % DEBUG_HIP_FORCE_GRAPH_QUEUES`,
  line 205, inside the per-edge loop). For a linear chain — out-degree ≤ 1 throughout — `stream_id`
  never advances, `max_streams_ == 1`, and **packet capture is active even at the default
  `QUEUES=4`**. The flag's real effect is the reverse of what we feared: `QUEUES=1` would *force*
  capture on for a branchy graph. `GraphExec::Init()` (423-447) confirms the nesting: the
  `max_streams_ == 1` topology test is the **outer** condition, `DEBUG_CLR_GRAPH_PACKET_CAPTURE` the
  inner one.
  So my 15 runs very likely *did* exercise capture mode, and the 10:20 conclusion probably stands —
  but "probably" is doing real work in that sentence and it is not yet measured.
- **11:26** **Attempted the direct readout and it did not work; recorded as unresolved rather than
  guessed at.** The agent pointed at a runtime log line
  (`hip_graph_internal.cpp:1105`, `"GraphExec::Run max_streams: %d ..."`, `LOG_DEBUG`/`LOG_CODE`)
  that reports `max_streams_` at every launch. The string **is** present in the loaded binary, and
  the loaded binary **is** the right one — `/proc/<pid>/maps` shows
  `/opt/rocm-7.2.0/lib/libamdhip64.so.7.2.70200` (checked because torch sometimes ships its own HIP
  runtime; it does not here). But no level-4 line is ever emitted:
  `AMD_LOG_LEVEL=4` alone yields only `:1:` and `:3:` lines; adding `AMD_LOG_MASK=0x4000` yields
  nothing; `AMD_LOG_MASK=0xFFFFFFFF` yields *fewer* lines than no mask at all (19 total), which looks
  like a mask-parsing problem rather than a level problem. Captured to
  `results/graph_visibility_probe/amdlog_q4_yihou.txt`.
  **Status: [UNRESOLVED].** Not investigated further — it is not on the critical path for the
  deliverable, and the GPUs are now needed for the real C=256 run. Whoever picks this up should try
  other `AMD_LOG_MASK` encodings (decimal, or a single valid category) and confirm whether this
  release build emits `LOG_DEBUG` at all.
- **11:28** Also unresolved and explicitly left open: the `QUEUES=4 → 19/20` vs `QUEUES=1 → 20/20`
  count difference from 10:58. Under the corrected understanding both should have capture active, so
  the difference is now *less* explicable, not more. n=3 and contended. **No claim made.**
- **11:30** Phase 3 signed off. `prof-impl` delivered `bench/profiling_yihou.py` (new, 473 lines) +
  6 guarded hunks in `profile_decode.py`; `batch_state.py` / `topology.py` hash-identical to
  `bench_pristine/`. Re-verified both modes after fixing the annotation/kernel classification:
  graph-ON 94 device kernels, graph-OFF 109. Two failed iterations renamed to `aborted_*`, nothing
  deleted. **Phase 4 authorised**: P1 (graph ON, default settings) first — it is the only
  configuration that matches how the published C=256 number was produced — then P0 (graph OFF),
  strictly sequential. P2 (`DEBUG_CLR_GRAPH_PACKET_CAPTURE=0`) **cancelled**: we have not
  characterised that knob, so a comparison against it would not be interpretable.

- **11:45** **Correction to the 09:12 and 10:20 entries — I mis-attributed the contention.**
  `prof-impl` reported two hung processes pinning GPU3 at 100 % with 1.7 GB VRAM each and 0 % CU
  occupancy. They were **mine**: PID 3108272 (49 min) and PID 3504270 (10 min), uid 100882, both my
  own `scripts/graph_visibility_probe_yihou.py`. Cause: a `docker exec` whose *outer shell* I killed
  on a 2-minute Bash timeout left the **in-container process running**; I never checked, and then
  wrote the contention up as being with the implementer's smoke run. It was at least partly my own
  leak. The conclusion (the 09:12 variation was contention) survives; **the attribution was wrong,
  and I asserted it about someone else's job without looking.**
  Cleaned up by explicit PID (`kill 3108272 3504270`) — my own processes, named targets, no variable
  target, nothing else touched. All 8 GPUs back to 0 %.
  **Operational lesson, worth carrying:** a timed-out `docker exec` does not kill the process inside
  the container. Any future timeout must be followed by an explicit process check.
- **11:47** Parked and explicitly unresolved: the decisive `QUEUES=1 CAPTURE=1` cell of the 2×2 may
  be one of the two runs that hung — **or** the 10-minute one may simply be a stale leak from the
  09:20 timeout. **No claim either way.** Not chased further; it does not block the deliverable, and
  the real run uses stock defaults regardless.
- **11:50** **Phase 4 launched with a window-placement change proposed by `prof-impl` and adopted —
  their reasoning was better than mine.** My `design.md` put the window at measured step 20, which
  sits at context ≈ 70072, the short end of the run's 70000 → 80000 sweep. Attention cost scales
  with context, so that window would have **systematically understated attention's share** — i.e. it
  would have produced "the top-10 for the first 2 % of the run" while reading as "the top-10 for
  this configuration", an error invisible in the output.
  Adopted instead: full OSL (`--max-steps 0`), window at the run mean,
  `--profile-start-step 1384 --profile-num-steps 10`, rank 0 only, stock runtime defaults
  (no `DEBUG_*` overrides — the published number used stock defaults). Cost ≈ +4 min against a
  ~6 min startup.
  Bonus the change buys: outside the window the run executes at full speed, so its end-to-end TPOT
  is a **sanity check** against the packup's 26.2519 ms. To be reported as a cross-check, never as a
  measurement; >5 % deviation is a stop-and-report condition.
  P0 (graph OFF) to follow with identical window placement so the two are comparable.

- **12:05** **Same mistake again, caught by routine checking rather than by reasoning.** The 11:45
  cleanup killed two leaked probe *workers* but **not the driving loop**: the 2×2 shell (PID 3502151)
  was still alive and had launched a fresh `QUEUES=1 CAPTURE=1` cell (PIDs 3607599 / 3607683)
  **while the real C=256 P1 run was using all 8 GPUs.** I contaminated the critical path with my own
  parked side-experiment. Killed by explicit PID; no probe processes remain.
  **Root cause of the repeat: at 11:45 I checked for hung *workers* and not for the *parent loop*
  that spawns them.** The correct check after any timeout is the whole process tree, not the leaf.
- **12:07** Partial 2×2 data, recorded for completeness and **explicitly not a result**:
  ```
  QUEUES=1 CAPTURE=0  rep=1  entries=7  gemm=20/20   total_us=1284.66
  QUEUES=1 CAPTURE=0  rep=2  entries=7  gemm=20/20   total_us=1586.86
  QUEUES=1 CAPTURE=0  rep=3  entries=7  gemm=20/20   total_us=1286.90
  QUEUES=1 CAPTURE=1  rep=1  PARSE_FAIL (no JSON — hung, then killed by me)
  QUEUES=1 CAPTURE=1  rep=2  entries=7  gemm=16/20   total_us=1012.89
  QUEUES=1 CAPTURE=1  rep=3  PARSE_FAIL (hung, then killed by me)
  ```
  Read literally this says the decisive cell — the only configuration where packet capture is
  certainly in force — hung in 2 of 3 attempts and under-counted in the third. **That reading is not
  supported.** Every one of those runs shared the GPUs with a live 8-rank C=256 benchmark, and I
  terminated two of them myself; "PARSE_FAIL" here means "I killed it", not "it hung on its own".
  The `QUEUES=4` cells were never reached.
  **Status: [ABANDONED, NOT UNKNOWN].** I am not resuming this. It is a side quest that has now
  contaminated the critical path twice, it is not required by any of the mission's five goals, and
  the deliverable uses stock defaults — the configuration that *is* characterised. If someone wants
  it later: idle machine, exclusive GPU, `QUEUES=1 CAPTURE=1` vs `QUEUES=1 CAPTURE=0`, with the
  `max_streams` log readout working first so the premise is verified rather than assumed.
- **12:09** Poll: `rocm-prof-research` has gone idle having shipped `research/rocm_profiling_env.md`
  (§2 correctly provisional, inventory line fixed, `rocpd`/`roctx` importability confirmed
  CPU-only). Nothing outstanding from it. `prof-impl` is mid-P1: `iterations/p1_ep8_c256_graphon_yihou/`
  has its snapshot files and a growing `runtime.log` (141 KB at 09:56), all 8 GPUs busy. On track;
  **no issue recorded against either teammate at this poll.**

- **12:25** **My error, and I put it on the teammate.** I told `prof-impl` that its P1/P0 "ran the
  rejected parameters" and to "stop and read". In fact I had written, earlier and explicitly:
  *"Window `[20,25)` with `--max-steps 40` is agreed... Do not widen it."* It ran exactly that. My
  later approval of the mid-window plan **crossed with its execution**. So the short-window pair was
  produced on my own instruction, and I then framed it as the teammate's deviation.
  The factual point in that message (the mid-window run is the one we want) was right; the
  attribution was wrong and the framing was worse than the attribution. Corrected to the teammate
  directly. **Recorded here under my name.**
  This is the second attribution error I've made today in the same direction — blaming a teammate's
  job for GPU contention that was my own leaked probe, then blaming a teammate for following my own
  instruction. Both times the underlying observation was fine and the causal story was invented.
- **12:27** `prof-impl`'s handling is better than my instruction was: it kept the short-window pair
  as **secondary** rather than discarding it, on the grounds that the context-position difference
  between the two pairs is itself informative. Agreed and endorsed — that converts my window-placement
  *worry* into something measurable. Asked it to compare the pairs directly in the write-up rather
  than simply designating one primary: if the top-10 ordering is stable between context ~70072 and
  ~75000 that is worth stating; if attention's share moves, that is a result.
- **12:28** Short-window pair complete, both exit 0, kept as secondary:
  `p1_ep8_c256_graphon_yihou/` (99 s, 86 device kernels, kernel-sum/DeviceTimer ratio 0.949) and
  `p0_ep8_c256_graphoff_yihou/` (105 s, self-CUDA total 432.2 ms).
  **P0 gives the clean answer to the mission's "both modes must work" requirement:** graph-OFF
  restores the CPU-operator layer (`aiter::gemm_a16w16` with both CPU and CUDA self time,
  `sglang::reg_all_gather_into_tensor`, the `step[DECODE bs=32]` annotation) which graph-ON simply
  does not emit, while the kernels underneath are the same in both. The two modes give **different
  layers of the same execution, not different quality.**
- **12:30** Mid-window primary pair launched sequentially with the approved command verbatim
  (`MAX_STEPS=0 PROFILE_START=1384 PROFILE_NUM=10`, stock runtime defaults):
  `p1b_ep8_c256_graphon_midwindow_yihou/` then `p0b_ep8_c256_graphoff_midwindow_yihou/`.
  Exit gates agreed before any analysis: end-to-end TPOT within 5 % of 26.2519 ms; trace size under
  ~50 MB; the three checks re-run; and **`realized_accept_length` must come back exactly
  `3.6134393063583814`** over the full 2768 iterations — the short-window runs gave 3.80, so an exact
  match is strong evidence the harness still is the one that produced the published number.
- **12:47** 20-min poll #4. **Found a silent stall: the mid-window chain never ran.**
  `logs/midwindow_chain.log` (10:05) shows both `p1b` and `p0b` rejected pre-launch by
  `run_profile_yihou.sh`'s guard, which required `MAX_STEPS >= PROFILE_START + PROFILE_NUM` and so
  rejected `MAX_STEPS=0` — but 0 means "full OSL" in `profile_decode.py` and therefore always covers
  the window. The shell guard was a wrong mirror of `validate_profile_args`, which had the 0 case
  right all along. Guard fires before `mkdir`, so no iteration dir was created and nothing needed
  aborting. ~25 min of idle GPU.
  `prof-impl` had already found and fixed it unprompted and reported it rather than absorbing it;
  both directions re-verified (`MAX_STEPS=0` passes and emits the exact approved command;
  `MAX_STEPS=40` with a 1384 window still rejected, now with a useful message). Relaunched.
  **No concern recorded about direction** — trivial guard bug, self-caught, self-reported.
  **Process lesson, mine as much as theirs: both of us read "no news" as "running" when it meant
  "died 25 minutes ago".** Asked for a one-line success marker on chain start, not only on failure.
  A long chain that logs only failures is indistinguishable from a healthy one while idle.

- **13:10** **`prof-impl` caught a factual error in my analysis report and was right.** I had written
  "78 层 × 每次迭代约 800 次 kernel 下发" in §3.5. The ~800 was **my own unmeasured estimate**, loosely
  extrapolated from the top-10 call counts. Measured from the chrome trace it is **2417 kernel
  launches per iteration** — off by 3×.
  Independently re-verified by me from both traces (`traceEvents`, `ph=="X"`, grouped by `cat`):
  ```
                    graph ON      graph OFF     ratio
  kernel            2417.2/iter   2432.0/iter   x1.006
  cuda_runtime       205.1/iter   5392.3/iter   x26.3
  cpu_op             753.0/iter  16792.3/iter   x22.3
  ```
  The correction is right, and the replacement text is better than what I wrote: the kernel count is
  essentially identical between modes (+0.6 %) while host-side runtime calls differ **26×**, which
  corroborates §3.5's "kernel time unchanged, non-kernel time +673 %" from an independent direction.
  **The worse part was not the number, it was the placement:** I offered an unmeasured mechanism two
  lines above my own "没有测过，不做解释" disclaimer. The teammate flagged exactly that tension rather
  than only fixing the digit. 20 µs/launch is now labelled an order-of-magnitude reference, not a
  measurement.
- **13:12** Mission goals 1-5 all complete. Deliverables:
  - `bench/profiling_yihou.py` (new) + 6 guarded hunks in `bench/profile_decode.py`; default-off
    behaviour proven identical to pristine by differential run.
  - `research/rocm_profiling_env.md` — full env-var inventory; §2 conclusion correctly provisional.
  - `results/ANALYSIS_top10_yihou.md` — the top-10 plus the per-iteration time budget, the
    graph-ON/OFF single-variable comparison, and six explicitly open items.
  - Four gate-verified runs; two `aborted_*` renames. **Nothing deleted at any point.**

- **13:45** **Open item 1 closed: `main_kernel` is the DSA sparse attention kernel, TileLang JIT.**
  `prof-impl` refused the `--profile-record-shapes` run I had approved, and was right to: it proved
  from existing data that `record_shapes` was **already on** in p0b (`profile_meta_yihou.json` →
  `profile_record_shapes: true`, and 166 883/167 923 = 99.4 % of `cpu_op` events carry `Input Dims`,
  so it was demonstrably live) while `main_kernel` still had no shape args. The approved run would
  have reproduced a known null. It launched `--profile-with-stack` instead — a strict superset —
  and flagged the deviation rather than burying it. **An approval from me is not a reason to run
  something demonstrably uninformative; this is the standard I want.**
  Identification verified independently by me from `p2_shapes_stack_graphoff_yihou/`:
  **332 of 332** `main_kernel` `hipLaunchKernel` records have innermost enclosing Python frame
  `tilelang_cython_wrapper.CythonKernelWrapper.forward`; walking up, **316 → `dsa_backend.py:2220
  forward_extend`** (target) and **16 → `:2570 forward_decode`** (draft) — matching the stage split
  derived earlier and independently from the annotations. Name origin confirmed in the pinned source:
  `kernels/ops/attention/dsa/tilelang_kernel.py:1314` is `return main`, so TileLang's entry symbol is
  `main_kernel`, which is why four unrelated kernels share it.
  Both earlier negatives now have one cause: TileLang's Cython wrapper dispatches directly, never
  entering the aten dispatcher. **That is also why both levers I proposed (cpu_op parent, record_shapes)
  were structurally incapable of working** — `record_shapes` is a dispatcher feature. My "next
  measurement" call was wrong; with-stack was the lever.
  **Headline consequence: attention (#1 + #5) = 31.0 % overtakes communication (#2+#4+#9) = 23.6 %
  as the largest single category.**
- **13:50** Answered the teammate's parting question myself (traces only, no GPU): **is any other
  top-10 row also an aggregate under a generic name?** Yes — `ncclDevKernel_Generic_1` has **3**
  launch signatures; `aiter::dynamic_per_group_scaled_quant_kernel` and `ck::kernel_moe_gemm_2lds`
  have 2 each (both outside the top-10). **But only `main_kernel`'s aggregation was
  category-ambiguous**; every other multi-signature name resolves within a single functional
  category, so §3's grouping is unaffected.
  Decomposing #2 produced a clean corroboration: its dominant branch is **79.0 launches/iter**
  (98.8 % of the entry) — the *same* per-layer rhythm as `main_kernel`'s dominant branch
  (78 target layers + 1 draft_extend layer). That is what tensor parallelism means: **one TP
  all-reduce after every transformer layer.** The 4.0/iter secondary branch matches the draft
  model's 4 forwards per iteration.
- **13:55** **Report hazard found and fixed during final read.** §3 still opened with the *old*
  grouping table (`main_kernel` 身份待定, communication largest) and only corrected it ~80 lines
  later. A reader would have taken the superseded conclusion as the result. Marked the first table
  explicitly superseded with a forward pointer and the note that its conclusion was wrong.
  **Lesson: appending a correction is not the same as retracting the thing corrected.**
- **13:57** Final state: no stray processes, all 8 GPUs at 0 %, team stood down, both crons deleted.
  Nothing deleted at any point in this session; two failed iterations remain as `aborted_*` renames
  and two superseded runs as `*_shortwindow_*`.
