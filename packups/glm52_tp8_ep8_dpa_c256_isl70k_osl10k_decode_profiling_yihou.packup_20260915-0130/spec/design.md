# Profiling design for the scheduler-free GLM-5.2 decode bench

All source facts below are **first-hand** (read from the pinned SGLang tree copied out of the
running container: `research/sglang_src_402df1e1e/`, commit `402df1e1e453e1e85ec0f5ac4052d36598cc691a`).

---

## 1. How SGLang profiles a *running server*, and why none of it is directly usable here

The server path is: HTTP `/start_profile` → `ProfileReq` → `Scheduler` →
`SchedulerProfilerManager` (`srt/managers/scheduler_components/profiler_manager.py`).
That manager has two implementations behind one env switch:

| `SGLANG_PROFILE_V2` | implementation | start/stop modes supported |
|---|---|---|
| `false` (**default**) | legacy inline code in `profiler_manager.py` | manual start/stop, `start_step`+`num_steps`, and by-stage |
| `true` | `srt/utils/profile_utils.py::ProfileManager` | **by-stage only** — `manual_start()`/`manual_stop()` `raise NotImplementedError`, and `configure()` asserts `profile_by_stage=True` |

**Consequence for us:** our harness has no Scheduler and no HTTP server, so neither manager is
driven. We also cannot simply set `SGLANG_PROFILE_V2=1` and call `manual_start()` — that path is
explicitly unimplemented. We drive the profiler ourselves from the bench loop.

What we *do* reuse, unchanged:

- `srt/utils/profile_utils.py::_ProfilerTorch / _ProfilerMemory / _ProfilerCudart / _ProfilerRPD`
  — the concrete profiler backends and their file-naming convention
  (`<prefix>-<profile_id>-TP-<r>[-DP-<r>][-EP-<r>]<suffix>.trace.json.gz`).
- `srt/utils/nvtx_utils.py::profile_range()` — emits `torch.profiler.record_function`
  **whenever a torch profiler is active; no env var required**. This is why SGLang's existing
  annotation sites light up for free as soon as we start a profiler.

### Annotation sites that will appear in our traces for free
| Span | Site | Note |
|---|---|---|
| `step[DECODE bs=N]` | `model_runner.py:1520` (`build_step_span_name`) | **eager path only** |
| `draft`, `draft_extend` | `eagle_worker_v2.py:1167,1209,1230` via `spec_utils.spec_stage_span` | CPU-side wrapper, present in both modes |
| `step[DRAFT_LOOP raw_bs=… bs=… topk=…]` | `frozen_kv_mtp_cuda_graph_runner.py:452` | explicitly added **because** "the graph bypasses `model_runner.forward`'s record_function" |

That last comment is the load-bearing fact for this whole design: **under CUDA-graph replay the
normal forward spans do not fire**, and SGLang worked around it by annotating the graph runner.

---

## 2. The CUDA-graph problem, stated precisely

A replayed hipGraph is submitted as a pre-built packet batch. Two separate things break:

1. **Python/CPU-side spans disappear** — nothing inside the graph executes Python, so
   `record_function` in `model_runner.forward` never runs. (First-hand: the
   `frozen_kv_mtp_cuda_graph_runner.py:448` comment + workaround.)
2. **Per-kernel GPU attribution may collapse** into a single opaque graph-launch entry, because the
   runtime does not dispatch kernels individually. This is what
   `DEBUG_CLR_GRAPH_PACKET_CAPTURE` is believed to control. **Verified so far: the symbol exists in
   this machine's `/opt/rocm/lib/libamdhip64.so` (`strings`).** Its exact semantics and default are
   pending the research agent's report and, decisively, our own A/B experiment.

### Three complementary mechanisms, chosen so that at least one works in each mode

| # | Mechanism | graph OFF | graph ON | what it yields |
|---|---|---|---|---|
| A | `torch.profiler` (CPU+CUDA) around a small window of decode iterations | ✅ full per-kernel + spans | ❓ per-kernel only if graph packet capture is disabled — **to be measured** | top-N kernel table, chrome trace |
| B | **`DeviceTimer`** (`srt/utils/device_timer.py`) attached by us to the model runners | ✅ | ✅ **graph-safe by construction** — CUDA events bracket the replay from outside | per-stage GPU seconds; see the corrected category table below |
| C | `--enable-profile-cuda-graph` (real `ServerArgs` flag) | n/a | ✅ | kernel **identity + shapes** captured during graph capture, plus SGLang's own `key_averages().table(row_limit=10)` |

**Mechanism B is the safety net.** `model_runner.device_timer` is initialised to `None`
(`model_runner.py:332`) and is only populated by the Scheduler's metrics reporter. Since we have no
Scheduler, we construct a `DeviceTimer` in the bench and assign it ourselves — no SGLang change.
The `device_timer_ctx(...)` call sites already exist in both the eager runner and every graph
runner, so the categories light up in both modes.

#### CORRECTION (2026-09-14, measured) — the categories this document predicted are wrong

An earlier draft of this section listed `decode` as an expected **graph-ON** category and implied it
was the target's. That came from reading the `device_timer_ctx(...)` call sites without checking
which runner owns each one. Measured (`iterations/smoke_graphon_v5_yihou`,
`iterations/smoke_graphoff_v2_yihou`, `profile/device_timer_rank_0_yihou.json`):

| Owner | graph ON | graph OFF |
|---|---|---|
| **target** runner | `target_verify` | `target_verify` |
| **draft** runner | `eagle_draft`, `eagle_draft_extend` | `decode`, `extend` |

- The target is `target_verify` in **both** modes — never `decode`.
- `decode` appears **only** with graphs off, and it is the **draft** runner's four eager forwards
  per iteration (n=80 over 20 iterations).
- The two owners' category sets are disjoint in both modes, and `sum(by_category)` equals
  `sum(by_runner)` exactly, so the stage split adds up with no double counting.
- `idle` was **not** observed in either mode. Unexplained; would need a run with unequal
  DP-shard batch sizes to test.

The stage split is valid in both modes, but **the label mapping is mode-dependent** and must be
stated whenever the two modes are compared, or it reads as "a stage vanished when graphs turned on".

#### CORRECTION (2026-09-14, measured) — worry #2 above does not reproduce

Per-kernel GPU attribution does **not** collapse under replay here. Two independent measurements:
the lead's model-free 3-kernel hipGraph probe (7 distinct GPU entries under
`DEBUG_CLR_GRAPH_PACKET_CAPTURE` unset / `=0` / `=1`) and this bench's graph-ON smoke (94 distinct
device kernels with per-kernel self time). Worry #1 — Python-side `record_function` spans not firing
inside the replay — **still stands** and is unchanged.

Claim scope: this is established **at default runtime settings**, which is the configuration the
published C=256 number was produced in. It is *not* established that packet-capture mode preserves
per-kernel visibility: `DEBUG_HIP_FORCE_GRAPH_QUEUES` defaults to 4 and `max_streams_ > 1` reportedly
disables packet capture entirely, so packet capture may never have been active in any run so far.
The `QUEUES ∈ {1,4}` × `CAPTURE ∈ {0,1}` 2×2 is pending.

**Mechanism C caveat, stated up front:** during graph *capture* kernels are recorded, not executed.
Its timings are therefore **not** performance data. It is a kernel-identity/shape inventory only,
and must be reported as such.

---

## 3. Environment variables in scope

SGLang (`srt/environ.py`, lines 409–432) — complete list of the profiling/tracing block:

| Var | Default | Effect here |
|---|---|---|
| `SGLANG_PROFILE_V2` | false | picks `ProfileManager`; **manual start/stop unimplemented** → leave false |
| `SGLANG_PROFILE_WITH_STACK` | true | python stack in trace; large + slow |
| `SGLANG_PROFILE_RECORD_SHAPES` | true | input shapes per op |
| `SGLANG_TORCH_PROFILER_DIR` | `/tmp` | output dir for traces and for graph-capture traces |
| `SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE` | false | export one combined capture trace |
| `SGLANG_GRAPH_BATCH_CAPTURE` | false | export one capture trace **per captured bs** (overridden by the previous one) |
| `SGLANG_ENABLE_NVTX_SCHEDULER` / `SGLANG_ENABLE_NVTX_OPERATIONS` | false | additionally emit nvtx ranges (needs `nvtx` pkg); `record_function` is independent of these |
| `SGLANG_ENABLE_METRICS_DEVICE_TIMER` | false | gates the Scheduler-side reporter only; **we attach `DeviceTimer` directly, so this is not required** |
| `SGLANG_RECORD_STEP_TIME` | false | scheduler step timing |
| `SGLANG_MEM_PROFILE_MAX_ENTRIES` | 100000 | allocator-history depth for `activities=["MEM"]` |

ROCm/HIP side (present in this machine's HIP runtime, confirmed by `strings`):
`DEBUG_CLR_GRAPH_PACKET_CAPTURE`, `DEBUG_CLR_MAX_BATCH_SIZE`, `DEBUG_CLR_BATCH_CPU_SYNC_SIZE`,
`DEBUG_CLR_BLIT_KERNARG_OPT`, `DEBUG_CLR_KERNARG_HDP_FLUSH_WA`, `DEBUG_CLR_LIMIT_BLIT_WG`,
`DEBUG_CLR_SYSMEM_POOL`. Tooling available in-container: `rocprofv3`, `rocprofv2`, `rocprof`,
`rocprof-compute`, `rocprof-sys-*`. The full annotated list is the research agent's deliverable and
lands in `research/rocm_profiling_env.md`.

---

## 4. Implementation plan for `bench/profile_decode.py`

**Opt-in, default off, zero behavioural change when off.** New flags:

```
--profile                       enable profiling (default: off)
--profile-start-step N          first measured decode iteration to profile (default 20)
--profile-num-steps K           how many iterations to profile (default 5)
--profile-activities CPU,GPU    comma list; also MEM / RPD / CUDA_PROFILER
--profile-ranks 0               comma list of TP ranks that emit traces (default "0")
--profile-dir PATH              default <result-dir>/profile
--profile-with-stack / --no-... default off (overhead + trace size)
--profile-record-shapes         default on
--device-timer                  attach DeviceTimer (default ON whenever --profile)
--profile-top-n 10              rows in the emitted top-N table
```

Placement inside `run_rank`: the profiler window sits **inside the measured decode loop** but the
run is a **dedicated profiling run** (small `--max-steps`), so no published TPOT is affected. The
harness must mark the result as profiled so nobody mistakes it for a performance point.

Outputs per profiled run, under `<result-dir>/profile/`:
- `*.trace.json.gz` (per emitting rank, SGLang naming convention)
- `top_ops_rank{R}_yihou.json` + `.txt` — `key_averages()` sorted by self CUDA time and by CPU time
- `device_timer_rank{R}_yihou.json` — per-category GPU seconds and share
- `profile_meta_yihou.json` — mode (graph on/off), env snapshot, window, versions

**Hard invariants to preserve:** when `--profile` is absent, every existing assertion, the
accounting, and the emitted `result_yihou.json` schema stay byte-compatible with the packup.

---

## 5. Experiment plan

Config fixed to the packup's C=256 EP8 point:
`--tp-size 8 --ep-size 8 --enable-dp-attention --batch-size 256 --max-running-requests 256
--input-len 70000 --output-len 10000 --accept-length 3.61 --mem-fraction-static 0.85` (+ the same
extra flags). Only `--max-steps` is reduced, for the profiling runs only.

| Run | Graph | `DEBUG_CLR_GRAPH_PACKET_CAPTURE` | Purpose |
|---|---|---|---|
| P0 | off (`--disable-cuda-graph`) | unset | ground truth per-kernel top-10 |
| P1 | on | unset (default) | ANSWERED YES at smoke scale; the real run re-confirms it at C=256 |
| P2 | on | `0` | no longer decisive (attribution already works at default); superseded by the QUEUES x CAPTURE 2x2 |
| P3 | on | unset | `--enable-profile-cuda-graph`: capture-time kernel identity |

P1 vs P2 is the decisive A/B for the graph question. **If P2 does not restore per-kernel
attribution, we report that as the finding** — we do not substitute P0's numbers for the graph-on
case. DeviceTimer runs in every one of P0–P3 and gives the stage split that is valid in all modes.

Smoke first, at a cheap config, before spending a C=256 startup.
