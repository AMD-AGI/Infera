# Implementation notes — profiling for the scheduler-free GLM-5.2 decode bench

All statements below are first-hand: either a source read of the pinned tree
(`research/sglang_src_402df1e1e/`) or a smoke run whose output path is named.

## 1. What was added, and where

| File | Change | sha256 |
|---|---|---|
| `bench/profiling_yihou.py` | **NEW**, 473 lines. All profiling machinery. | `a84a1c1f4acecdb00bdb5b13adfbbe5f60f0bc32ce88043c71e296f5669c6b51` |
| `bench/profile_decode.py` | 6 hunks, 45 added lines, every one guarded. | `3e9971c780d32c971fdb9ce8725913367a9bfc3913a44b07ea56170d862ea65a` |
| `bench/batch_state.py`, `bench/topology.py` | **unchanged** (hashes match `bench_pristine/`) | — |

Pristine copies: `../bench_pristine/`.

The machinery lives in a separate module so the diff against the published bench stays
small enough to audit line by line. `profiling_yihou.py` imports only `json`, `os`,
`pathlib`, `time` at module level; torch and SGLang are imported inside methods, so
`--help` remains stdlib-only as `profile_decode.py`'s docstring promises.

The six hunks in `profile_decode.py`:
1. import the new module;
2. `make_parser()` → `add_profile_cli_args(parser)`;
3. `validate_args()` → `validate_profile_args(args)`;
4. `run_rank()` → `session = ProfileSession.create(args, rank, log)` (returns `None` unless `--profile`);
5. `run_rank()` → attach DeviceTimer after warmup; one `if session is not None: session.step_boundary(...)`
   in the loop, placed **before** `tick = time.perf_counter()`; `session.finish(...)` after the loop;
6. `main()` → `configure_profile_env(args)` before `mp.spawn`, and `config_yihou.json` records the
   `--profile*` keys only on a profiled run.

## 2. Two deliberate deviations from `design.md` §4 (both reviewed and approved)

**a. `_ProfilerTorch` is not reused.** Its `stop()` ends with
`torch.distributed.barrier(self.cpu_group)` (`srt/utils/profile_utils.py`). With the required
`--profile-ranks 0` default, only rank 0 would reach that barrier and the run would hang. We call
`torch.profiler.profile` directly and reproduce its filename convention verbatim:
`decode-<profile_id>-TP-r[-DP-r][-PP-r][-EP-r].trace.json.gz`
(observed: `decode-1789378711.934904-TP-0-DP-0-EP-0.trace.json.gz`).

**b. One `DeviceTimer` per runner, not the Scheduler's single shared instance.**
`DeviceTimer.wrap` asserts `not self._in_wrap` — it is not re-entrant. The Scheduler installs one
timer on target + draft (`metrics_reporter.py:192-203`) and gets away with it; rather than depend on
that ordering holding in a Scheduler-free driver, separate instances make a nested wrap impossible
by construction. Categories are disjoint across runners, so the accumulated totals are identical.
Attachment happens **after warmup**, so no `torch.cuda.Event` is ever recorded during graph capture.

Disjointness was **verified, not assumed** (`../iterations/smoke_graphon_v5_yihou`,
`../iterations/smoke_graphoff_v2_yihou`, `profile/device_timer_rank_0_yihou.json` → `by_runner`):

| mode | target categories | draft categories | intersection |
|---|---|---|---|
| graph ON | `target_verify` | `eagle_draft`, `eagle_draft_extend` | empty |
| graph OFF | `target_verify` | `decode`, `extend` | empty |

and `sum(by_category) == sum(by_runner)` exactly in both modes (2.830840 s and 2.630361 s), so the
split adds up with no double counting.

## 3. Gotchas found by the smoke runs (all fixed)

### Gotcha 1 — double stop of the torch profiler
`_stop()` did not latch, so the iteration after the window re-entered it and torch
  raised `RuntimeError: Can't disable Kineto profiler when it's not running`.
  Evidence: `../iterations/aborted_smoke_graphon_double_stop_yihou/runtime.log:952`.
  Fix: `step_boundary()` returns immediately once `stopped_at_step is not None`; `finish()` also
  closes a still-open window if the loop ended early (`accounting.complete`).
### Gotcha 2 — `record_function` spans silently corrupt the kernel ranking
**This is the one that would have produced a plausible, confidently-wrong top-10.** torch reports
  `step[TARGET_VERIFY bs=1]` as a `DeviceType.CUDA` row with 618 ms of self device time — time that
  **overlaps** the kernels beneath it. `device_type` alone cannot separate a span from a kernel.
  Fix: names are classified with `FunctionEvent.is_user_annotation` into
  `user_annotation` / `device_kernel` / `cpu_side`, and the kernel ranking
  (`top_by_self_device_time_device_kernels_only`) excludes annotations. **Rank GPU cost with that
  list; the `top_by_self_device_time` list is the unfiltered view and is not additive.**
  Reporting rule for the final top-10: **device_kernel rows only, with user_annotation rows shown
  alongside as context and never interleaved**, and the sum semantics stated. A reader who adds a
  percentage column and gets >100% will rightly distrust the whole report.

### Gotcha 3 — `--enable-profile-cuda-graph` needs a writable CWD (host-environment-specific)
This one belongs with the four host fixes already in `create_container_yihou.sh`: it is caused by
running as a non-root uid with `docker exec -w /`, and it will bite the next person on any
NFS/non-root setup.
  `decode_cuda_graph_runner._post_process_after_profile()` (line 814) writes
  `cuda_graph_runner_memory_usage.pickle` into the process CWD unconditionally. `run_decode.sh` uses
  `docker exec -w /`, which is EPERM for the non-root uid, and capture dies.
  Evidence: `../iterations/aborted_smoke_graphon_v3_capture_cwd_eperm_yihou/runtime.log`.
  Fix is in the runner, not the bench: `run_profile_yihou.sh` cd's into
  `<iteration>/graph_capture_cwd/` when `CAPTURE_PROFILE=1`. All eight ranks write that one filename,
  so the surviving pickle belongs to whichever rank wrote last — the per-rank `key_averages` tables
  go to `runtime.log` and the capture traces are rank-namespaced, so nothing else is lost.

## 4. Evidence per acceptance criterion

### Criterion 1 — no behavioural change with `--profile` absent
Patched code vs the pristine copy, identical args, same container, same session:
`../iterations/smoke_noprofile_patched_yihou/` vs `../iterations/smoke_noprofile_pristine_yihou/`.

- identical file set (20 files each);
- `config_yihou.json` differs only in the `result_dir` path;
- `result_yihou.json` key sets identical — nothing added, nothing removed;
- every value identical except wall-clock timings. `elapsed_seconds` 2.948152 (patched) vs
  2.949847 (pristine): patched 0.06 % **faster**, i.e. run-to-run noise;
- `steps_yihou.jsonl` identical in every non-timing field across all 20 steps.

Not claimed: zero cost in the absolute. The loop body gains one `if session is not None` pointer
test per iteration — no allocation, no syscall — and it sits before `tick = time.perf_counter()`, so
it is outside the per-step `seconds` that feeds `DecodeAccounting`.

### Criterion 2 — both CUDA-graph modes produce usable output
`../iterations/smoke_graphon_v5_yihou/` (graphs on, `target_graph_iterations=20`) and
`../iterations/smoke_graphoff_v2_yihou/` (`--disable-cuda-graph`, `target_graph_iterations=0`).
Both exit 0 and emit trace + top-N + DeviceTimer report.

### Criterion 3A — torch.profiler
**At default runtime settings, per-kernel GPU attribution survives hipGraph replay on this stack.**
Read that claim exactly as worded — see the scope note below it. Top device kernels, 5 profiled
iterations, rank 0, local batch 1:

| graph ON (94 kernels, 655.2 ms) | graph OFF (109 kernels, 679.4 ms) |
|---|---|
| `aiter add_rmsnorm_quant_kernel` 52.17 ms n=790 | `main_kernel` 50.49 ms n=830 |
| `main_kernel` 52.10 ms n=790 | `aiter add_rmsnorm_quant_kernel` 47.48 ms n=830 |
| `__amd_rocclr_copyBuffer` 35.37 ms n=815 | `Memcpy DtoD` 34.07 ms n=660 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16` 31.31 ms n=375 | `mfma_moe1_silu_mul_afp4_wfp4_bf16` 32.12 ms n=375 |
| `_gemm_a16_w16_kernel` 28.79 ms n=395 | `aiter allgather_vec` 29.36 ms n=445 |

This answers `design.md`'s P1 affirmatively **for the configuration we actually need to attribute**.

**Scope of the claim — do not strengthen it.** What is established is: *at default runtime settings*,
per-kernel rows are present. That is the same configuration in which the published C=256 number was
produced, so the attribution is valid for that number regardless of the underlying mechanism. What
is **not** established, and must not be written, is the stronger claim that "packet-capture mode
preserves per-kernel visibility" — the team lead's research found in CLR source that
`DEBUG_HIP_FORCE_GRAPH_QUEUES` defaults to 4 and that `max_streams_ > 1` disables packet capture
outright. If that holds, packet capture was never active in any run so far and
`DEBUG_CLR_GRAPH_PACKET_CAPTURE=1` was a no-op. A 2×2 (`QUEUES ∈ {1,4}` × `CAPTURE ∈ {0,1}`) is
pending on the lead's side. The knob stays wired in `run_profile_yihou.sh` via
`GRAPH_PACKET_CAPTURE=...`.

Corollary for the real run: per-kernel rows in graph-on mode are now the **expectation**, not a
hoped-for bonus. If the C=256 GLM-5.2 run does not produce them, that is a finding to investigate,
not a limitation to write off.

What graph-ON **does** lose is the CPU-side operator layer. Graph-OFF additionally names
`aiter::fused_moe_`, `aiter::gemm_a16w16`, `aiter::add_rmsnorm`,
`sglang::reg_all_gather_into_tensor` as ops with both CPU and CUDA time; graph-ON shows only raw
kernel symbols, because no Python executes inside the replay. Only 15 `hipGraphLaunch` calls appear
(3 graphs × 5 iterations).

### Criterion 3B — DeviceTimer fires in both modes, with **different category labels**

| Category | graph ON | graph OFF |
|---|---|---|
| `target_verify` | 2.465 s (87.1 %), n=20 | 2.327 s (89.8 %), n=20 |
| `eagle_draft` | 0.279 s (9.9 %), n=20 | — |
| `eagle_draft_extend` | 0.085 s (3.0 %), n=20 | — |
| `decode` | — | 0.191 s (7.4 %), n=80 |
| `extend` | — | 0.074 s (2.8 %), n=20 |

The target is `target_verify` in **both** modes — `design.md` expected `decode` for it, and that is
wrong: `decode` appears only with graphs off, and it is the **draft** runner's four eager forwards
per iteration (n=80 = 4 × 20). The graph runners label the same draft work `eagle_draft` /
`eagle_draft_extend`; the eager runner labels it `decode` / `extend`. The stage split is valid in
both modes, but the label mapping is mode-dependent and must be stated when comparing.

`idle` was **not** observed in either mode. Not investigated — DP-attention padding may simply never
have produced an idle forward at local batch 1. The measurement that would settle it is a run where
the DP shards carry unequal batch sizes.

### Criterion 3C — capture-time inventory

**Does `--enable-profile-cuda-graph` fire in the scheduler-free path? Yes — confirmed.** It is a
plain `ServerArgs` field read by the graph runner's `__init__`/`capture()`
(`decode_cuda_graph_runner.py:1014`), not something the Scheduler switches on, so calling
`target.init_cuda_graphs()` / `worker.init_cuda_graphs()` directly reaches it unchanged. The bench
passes the flag through `extra` into `ServerArgs` untouched. Evidence it actually ran: 24
`"Sorted by CUDA Time"` `key_averages` tables in
`../iterations/smoke_graphon_v4_capture_yihou/runtime.log` (3 runners × 8 ranks), and the pickle
side effect that broke the first attempt — SGLang code only reached on this path.

`CAPTURE_PROFILE=1` → `../iterations/smoke_graphon_v4_capture_yihou/`: 24 rank-namespaced capture
traces under `profile/graph_capture_profile/` (`DecodeCudaGraphRunner`,
`EAGLEDraftCudaGraphRunner`, `EAGLEDraftExtendCudaGraphRunner` × TP 0-7) plus 24 `key_averages`
tables in `runtime.log`. **These are kernel identity and shapes, not performance**: during capture
kernels are recorded, not executed. Every emitted metadata file carries
`cuda_graph_capture_profile_caveat` saying so.

### Criteria 4-6
- Only rank 0 emits a trace by default (`--profile-ranks 0`); `--profile-with-stack` defaults off.
  All ranks still emit their own `device_timer_rank_N_yihou.json` and `profile_meta_rank_N_yihou.json`
  — those are a few kB each, not traces.
- `profile_meta_yihou.json` (rank 0; per-rank copies alongside) records graph on/off, requested and
  actual window, all `--profile*` values, the full `SGLANG_*` / `DEBUG_CLR_*` / `HIP_*` / `AITER_*` /
  `TORCH_*` environment in effect, the pinned SGLang commit, module source paths, and
  `"is_performance_measurement": false`. Profiled runs additionally set `profiled: true` and
  `is_performance_measurement: false` in `result_yihou.json`, and `launch_status.json` carries the
  same marker.
- No measurement knob was touched. ISL/OSL/accept-length/mem-fraction/max-running-requests semantics
  are byte-identical; the only additions are the `--profile*` flags and use of the pre-existing
  `--max-steps`.

## 5. What I could not make work / did not verify

1. **`idle` category never observed** (see 3B). Unverified, not explained.
2. **`DEBUG_CLR_GRAPH_PACKET_CAPTURE` was never exercised.** The knob is plumbed through
   `run_profile_yihou.sh` (`GRAPH_PACKET_CAPTURE=...`), but since per-kernel attribution already
   works at the default, no A/B was run. Its semantics remain unverified beyond the `strings` hit.
3. **`MEM`, `CUDA_PROFILER` and `--profile-ranks all` code paths are unexercised.** They are
   implemented, and `RPD` is deliberately rejected by `validate_profile_args`, but no run covered
   them. **`--profile-with-stack` is no longer on this list** — exercised in
   `../iterations/p2_shapes_stack_graphoff_yihou/`, where it produced 715 218 `python_function`
   events and a usable Python call tree (trace 16.7 MB for 2 iterations, vs 7.96 MB for 10 without
   it, so budget roughly 40× the per-iteration trace size when enabling it).
4. **Everything in §4 above is at C=8 (local batch 1), ISL 2048, OSL 128.** Superseded for the
   headline results by the C=256 runs (`p1b_*`, `p0b_*`, `p2_*`); the smoke kernel numbers must not
   be carried forward.
5. **Profiled TPOT is not performance data.** Performance numbers remain the non-profiled packup
   runs. The full-OSL profiled run reproduces 26.3154 ms against the published 26.2519 ms (+0.24 %),
   which is a sanity check on the harness, not a measurement.
6. **`--profile-record-shapes` cannot see a kernel that bypasses the aten dispatcher.** Measured,
   not assumed: `p0b` ran with `record_shapes` on and 99.4 % of its `cpu_op` events carry
   `Input Dims`, yet `main_kernel` and its `hipLaunchKernel` carry no shape field at all. Shapes are
   attached by the dispatcher; TileLang's Cython wrapper never enters it. **The lever that works for
   such kernels is `--profile-with-stack`**, whose `python_function` events form a timestamped
   Python call tree independent of dispatch. That is what identified `main_kernel`
   (`results/ANALYSIS_top10_yihou.md` §3).
7. **Not checked: whether any other top-10 entry is also an aggregate under a generic name.**
   `main_kernel` turned out to be four distinct kernels sharing one TileLang-generated symbol, and
   nothing guarantees it is the only such row. The same launch-signature decomposition would settle
   it from the traces already on disk — no GPU time needed.
