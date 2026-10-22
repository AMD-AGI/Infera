# ROCm 7.2 / PyTorch 2.9.1 profiling: environment variables, hipGraph behaviour, tooling

Target stack: 8× AMD MI355X (gfx950), ROCm 7.2.0, torch 2.9.1+rocm7.2.0, triton 3.7.0, inside
container `yihou-glm52-tp8ep8-0914`. Harness drives `TpModelWorker` + `EAGLEWorkerV2` directly —
no HTTP server, no Scheduler, so SGLang's `/start_profile` endpoint is unavailable.

**Evidence classes used throughout:**
- **[FH-src]** first-hand: I read the source, pinned to git tag `rocm-7.2.0` of `github.com/ROCm/clr`
  (the exact version in the container).
- **[FH-bin]** first-hand: output of a read-only command I ran in the container.
- **[FH-lead]** first-hand, but measured by the team lead, not by me.
- **[2H]** second-hand: docs, GitHub issues, blogs.
- **[UNKNOWN]** stated as unknown, with the experiment that would settle it.

No GPU work was run by me. Scratch files in `/var/tmp/yihou-rocm-prof-research/`. Nothing deleted.

---

## 0. What is actually installed (all [FH-bin])

```
ROCm                     7.2.0
torch                    2.9.1+rocm7.2.0.git7e1940d4
torch.profiler.supported_activities()  -> {ProfilerActivity.CPU, ProfilerActivity.CUDA}
rocminfo System Timestamp Freq.        -> 1000.000000 MHz   (relevant, see §2.4)
```

`libtorch_hip.so` links `libroctracer64.so.4`, `libroctx64.so.4`, `librocprofiler-register.so.0`
→ **Kineto on this build uses roctracer**, not rocprofiler-sdk.

Profiler binaries in `/opt/rocm/bin`: `rocprof`, `rocprofv2`, `rocprofv3` (**1.1.0**),
`rocprofv3-attach`, `rocprofv3-avail`, `rocprof-compute`, `rocprof-sys-{avail,causal,instrument,python,run,sample}`.

Python packages:
- `rpdTracerControl` / `rocmProfileData`: **NOT installed**.
- `rocpd`: **installed** at `/opt/rocm/lib/python3.10/site-packages/rocpd` — *not on the venv
  `sys.path`*, reachable with `PYTHONPATH=/opt/rocm/lib/python3.10/site-packages`.
  Submodules: `schema, importer, query, summary, time_window, pftrace, otf2, csv, __main__`.
  CLI: `python3 -m rocpd {convert,query,summary}`.
- `roctx`: **installed**, same path. Exposes `rangePush, rangePop, rangeStart, rangeStop, mark,
  profilerPause, profilerResume, nameOsThread, nameHipDevice, getThreadId`, plus
  `roctx.context_decorators.{RoctxRange, RoctxProfiler}`.
- `rocprofsys`: installed, same path.

**Consequence:** we can do roctx-annotated, window-limited rocprofv3 capture and SQLite analysis
**with nothing to build and nothing to install.**

---

## 1. Q3 — Complete profiling-related environment variable inventory

This is the primary deliverable. Names marked **[FH-bin]** were extracted from the shipped
binaries' string tables, so they are what *this* runtime actually looks at — not what some doc
claims. Semantics marked **[2H]** come from ROCm documentation.

### 1.1 PyTorch / Kineto

Extracted from `strings .../torch/lib/libtorch_cpu.so` **[FH-bin]**:

| Variable | Meaning | Default | AMD-specific |
|---|---|---|---|
| `KINETO_LOG_LEVEL` | libkineto verbosity, 0=verbose … 5=error | 2 [2H] | no |
| `KINETO_CONFIG` | path to libkineto config file. Fallbacks `/etc/libkineto.conf`, `/tmp/libkineto.conf` **[FH-bin]** | unset | no |
| `KINETO_USE_DAEMON` | enable dynolog daemon / IPC control path | 0 | no |
| `KINETO_IPC_SOCKET_DIR` | dynolog socket directory | — | no |
| `KINETO_GPU_FALLBACK` | internal fallback-profiler selection | — | no |
| `KINETO_PRIVATEUSE1_FALLBACK` | ditto, PrivateUse1 backend | — | no |
| `TORCH_PROFILER_ENABLE_COLLECTIVE_PROFILING` | extra NCCL/RCCL collective metadata | off | no |

**Negative result, worth recording:** I grepped the whole `TORCH_PROFILER_*`, `TORCH_CUDA_PROFILER_*`
and `LIBKINETO_*` families across `libtorch_cpu.so` and `libtorch_hip.so`. The *only* hit is
`TORCH_PROFILER_ENABLE_COLLECTIVE_PROFILING`. **There is no PyTorch env var in this build that
controls roctracer/ROCm profiling behaviour.** The `CUPTI_PROFILER_ENABLE_PER_KERNEL`,
`CUPTI_PROFILER_METRICS`, `CUPTI_PER_THREAD_BUFFER_ENABLED` strings that are present are dead code
on ROCm. **[FH-bin]**

**libkineto config-file keys** (settable only via a `KINETO_CONFIG` file, not as env vars) **[FH-bin]**:

```
ACTIVITIES_ENABLED                    ACTIVITIES_LOG_FILE
ACTIVITIES_MAX_GPU_BUFFER_SIZE_MB     ACTIVITIES_DURATION_MSECS / _SECS
ACTIVITIES_ITERATIONS                 ACTIVITIES_WARMUP_ITERATIONS / _PERIOD_SECS
ACTIVITIES_DISPLAY_CUDA_SYNC_WAIT_EVENTS
PROFILE_START_ITERATION               PROFILE_START_ITERATION_ROUNDUP
PROFILE_START_TIME                    PROFILE_WITH_STACK
PROFILE_REPORT_INPUT_SHAPES           PROFILE_WITH_FLOPS / _WITH_MODULES
PROFILE_MEMORY                        PROFILE_MEMORY_DURATION_MSECS  PROFILE_OP
EVENTS_ENABLED_DEVICES  EVENTS_DURATION_SECS  EVENTS_LOG_FILE
EVENTS_HEARTBEAT_MONITOR_PERIOD_SECS
```

`ACTIVITIES_MAX_GPU_BUFFER_SIZE_MB` is the one that bites on a long decode loop: if the device
activity buffer overflows, records are dropped silently.

### 1.2 roctracer

`strings /opt/rocm/lib/libroctracer64.so.4` yields exactly two env-var-shaped strings:
`ROCTRACER_LOG`, `ROCTRACER_4` **[FH-bin]**.

**There is no `ROCP_*` / `ROCPROF_*` env-var surface in roctracer in this build.** `ROCP_OUTPUT_DIR`,
`ROCP_METRICS` etc. belong to the legacy `rocprof` / `rocprofv2` wrapper scripts, not to the
Kineto path. If you set them expecting torch.profiler to change behaviour, nothing happens.

### 1.3 HIP / CLR runtime (`libamdhip64.so`)

Full list of env vars the runtime reads: 398 strings; the profiling/debug/graph-relevant subset,
all **[FH-bin]** for existence, with semantics from `flags.hpp` **[FH-src]** or docs **[2H]**:

**Serialization / attribution (distort timing — debug only, never for ranking):**

| Variable | Default | Effect |
|---|---|---|
| `AMD_SERIALIZE_KERNEL` | 0 | 1 = wait for completion *before* enqueue, 2 = *after*, 3 = both. AMD-specific. [2H] |
| `AMD_SERIALIZE_COPY` | 0 | same, for copies. AMD-specific. [2H] |
| `HIP_LAUNCH_BLOCKING` | 0 | 1 = serialize kernel enqueue; docs say "behaves the same as `AMD_SERIALIZE_KERNEL`". [2H] |

**Profiling enablement:**

| Variable | Default | Effect |
|---|---|---|
| `GPU_FORCE_QUEUE_PROFILING` | **false** [FH-src `flags.hpp:199`] | "Force command queue profiling by default" — run as if under rocprof. |
| `HIP_FORCE_QUEUE_PROFILING` | — | The **ROCm docs use this spelling** [2H], but **only `GPU_FORCE_QUEUE_PROFILING` exists in our `libamdhip64.so`** [FH-bin]. Prefer the `GPU_` form. Whether the `HIP_` form is a live alias is **[UNKNOWN]**. |
| `GPU_DEBUG_ENABLE` | false [FH-src] | "Enables collection of extra info for debugger at some perf cost". |
| `AMD_THREAD_TRACE_ENABLE` | — [FH-bin] | ATT / thread trace. |

**HIP graph control (see §2 for why these matter):**

| Variable | Default | Effect |
|---|---|---|
| `DEBUG_CLR_GRAPH_PACKET_CAPTURE` | **true** [FH-src `flags.hpp:240`] | Enable/disable AQL packet pre-capture for graph replay. See §2. |
| `DEBUG_HIP_GRAPH_BATCH_SIZE` | **256** [FH-src `flags.hpp:256`] | Packets per doorbell ring in the captured batch. |
| `DEBUG_HIP_FORCE_GRAPH_QUEUES` | **4** [FH-src] | **A modulus, not a setter — read §2.6 before drawing any conclusion from this row.** It caps how many streams a *branchy* graph spreads across; it does **not** set `max_streams_`, which comes from graph topology. The default 4 does **not** disable packet capture. Setting it to **1** *forces* `max_streams_ == 1`, i.e. forces capture **on**. |
| `DEBUG_HIP_FORCE_ASYNC_QUEUE` | false [FH-src] | Forces graphs into async queue mode; per `flags.hpp:253` requires `DEBUG_HIP_FORCE_GRAPH_QUEUES=1`. Used **only on the non-capture per-node path** (`hip_graph_internal.hpp:1173`) — no interaction with the capture decision (§2.6). |
| `DEBUG_HIP_GRAPH_DOT_PRINT` | false [FH-src] | Dump graph topology as a dot file. Useful to see what is in the captured graph. |
| `DEBUG_HIP_BLOCK_SYNC` | 50 [FH-src] | Blocks CPU synchronization until callback processing is done. |
| `DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING` | — | **DOES NOT EXIST in our runtime** [FH-bin: absent from the full env-var string table]. It is a ROCm 7.14 / TheRock-era flag. The widely-circulated "set it to 0 to recover 99% of missing kernels" advice from [ROCm/rocm-systems#9697](https://github.com/ROCm/rocm-systems/pull/9697) **does not apply to us.** |

**Logging (a profiler-free way to inspect dispatch timestamps):**

| Variable | Default | Effect |
|---|---|---|
| `AMD_LOG_LEVEL` | 0 [2H] | 0 off … 4 debug, 5 debug-extra. |
| `AMD_LOG_MASK` | 0x7FFFFFFF [2H] | Filter. `0x2` kernel/copy commands; `0x8` decode AQL packets; `0x10` queue contents; `0x20` signals; **`0x80000` timestamp details**; `0x40000` memory pools incl. graphs; `0x2000` raw AQL packet bytes. |
| `AMD_LOG_LEVEL_FILE` | stderr [2H] | Redirect the log. |

`AMD_LOG_LEVEL=3 AMD_LOG_MASK=0x80000` prints the per-signal
`Signal = (...), Translated start/end = ... Elapsed = ... ns` lines emitted from
`Timestamp::checkGpuTime` **[FH-src]** — i.e. you can see whether graph packets are getting
per-dispatch timestamps **without loading any profiler at all**. Extremely verbose; use on a
2-iteration run.

**Queue / dispatch topology (changes which trace lanes you see):**

| Variable | Default | Effect |
|---|---|---|
| `GPU_MAX_HW_QUEUES` | **4** [2H] | HW queues per device per process; extra HIP streams round-robin onto these. |
| `AMD_DIRECT_DISPATCH` | false [FH-src `flags.hpp:184`] | Direct kernel dispatch; changes the command/batch structure the profiler observes. |
| `GPU_NUM_COMPUTE_RINGS`, `DEBUG_HIP_DYNAMIC_QUEUES` | [FH-bin] | queue allocation. |
| `ROC_SIGNAL_POOL_SIZE` | **64** [FH-src `flags.hpp`] | "Initial size of HSA signal pool". **Directly relevant to §2.3.** |
| `ROC_AQL_QUEUE_SIZE` | 16384 [FH-src] | AQL queue size in packets. |
| `ROC_ACTIVE_WAIT_TIMEOUT` | 0 [FH-src] | Active-wait on GPU interrupt, µs. |

**HSA level** (`libhsa-runtime64.so.1` **[FH-bin]**, ~200 vars; profiling-relevant subset):
`HSA_AMD_TOOL_PRIORITY` (orders tool interception when several profilers load),
`HSA_DISABLE_PC_SAMPLING`, `HSA_ENABLE_SDMA`, `HSA_ENABLE_INTERRUPT`, `HSA_CU_MASK`,
`HSA_ENABLE_DEBUG`, `HSA_COREDUMP_*`. Plus the HSA profiling API entry points
`hsa_amd_profiling_{set_profiler_enabled,get_dispatch_time,get_async_copy_time,async_copy_enable}`.

**GPU isolation** (affects which ranks appear in a trace): `ROCR_VISIBLE_DEVICES` (preferred on
Linux [2H]), `HIP_VISIBLE_DEVICES`, `GPU_DEVICE_ORDINAL`, `ROC_GLOBAL_CU_MASK`.

### 1.4 RPD / rocpd

| Variable | Meaning | Evidence |
|---|---|---|
| `RPDT_AUTOFLUSH` | rpd_tracer: flush records continuously instead of at exit | [2H] rocmProfileData README |
| `RPDT_FILENAME` | output `.rpd` path | [2H] |

RPD is **not installed** here [FH-bin]. Installing it means building inside the container
(`apt install sqlite3 libsqlite3-dev libfmt-dev`, clone
[ROCm/rocmProfileData](https://github.com/ROCm/rocmProfileData), `make && make install`, then
`rocpd_python` and `rpd_tracer` setup.py) — exactly the recipe in
[SGLang `3rdparty/amd/profiling/PROFILING.md`](https://github.com/sgl-project/sglang/blob/main/3rdparty/amd/profiling/PROFILING.md),
which pins commit `976899e9` and patches out `RocmSmiDataSource.cpp`. **Given `rocpd` is already
installed, I do not recommend building RPD.**

---

## 2. Q1 + Q2 — `DEBUG_CLR_GRAPH_PACKET_CAPTURE` and graph profiling

The lead settled by experiment **[FH-lead, n=5 per condition]** that torch.profiler attributes
per-kernel GPU time inside a replayed hipGraph on this stack, and that this variable does not gate
that (§2.3.1). This section documents *what the variable does* in source, and records one source
mechanism that could in principle drop events but demonstrably does not here.

### 2.0 Bottom line up front

**Practical recommendation (solid, and all a reproducer needs):**
**you do not need to set `DEBUG_CLR_GRAPH_PACKET_CAPTURE` in order to profile on ROCm 7.2.**
At stock runtime settings, torch.profiler attributes per-kernel GPU time inside a replayed hipGraph
and this flag changes nothing about that (§2.3.1, n=5, 15/15 identical).

**Scope of that claim — read this before generalising it.** It was measured **only at default
runtime settings, `DEBUG_HIP_FORCE_GRAPH_QUEUES=4`**. We have **not** characterised the flag
itself. Two things are open:

1. **Was packet capture ever in force?** The `=1` condition only exercises capture if the probe
   graph had `max_streams_ == 1` (§2.6). Source analysis says a linear single-stream capture does,
   even at `QUEUES=4` — but this has **not been read off the running system**. One line settles it:
   `AMD_LOG_LEVEL=4 AMD_LOG_MASK=0x4000 <run> 2>&1 | grep "GraphExec::Run max_streams"`.
2. **Does the flag matter at `QUEUES=1`** (the setting that forces `max_streams_ == 1`, hence forces
   capture on)? **[UNKNOWN] — measurement in progress.** The decisive cell, `QUEUES=1 CAPTURE=1`,
   has not been run.

**Partial data that forbids the confident negative** [FH-lead, **n=3, GPU not fully quiet —
`total_self_gpu_us` spread 1284 / 1586 / 1286 µs — NOT a result**]:

| setting | gemm count |
|---|---|
| `QUEUES=4` (default), CAPTURE ∈ {unset, 0, 1} | 19/20 in 15/15 runs |
| `QUEUES=1`, `CAPTURE=0` | 20/20 in 3/3 runs |

If that holds, `DEBUG_HIP_FORCE_GRAPH_QUEUES` moves the count and the "19 of 20" constant is a
queue-count effect, not the start/stop window artefact we first assumed (§2.3.1, §5).

**Worth noting, because it cuts against the source reading:** at `CAPTURE=0` the runtime takes the
per-node path *regardless* of `max_streams_`, so if the probe graph were linear, `QUEUES` should
make **no** difference in that cell — yet 20/20 vs 19/20 is exactly that cell. If the difference
survives repetition on a quiet GPU, the most economical explanation is that **the probe graph is
not linear and `max_streams_ > 1` at `QUEUES=4`** — i.e. the original concern in §2.6 was right and
capture was never engaged. The `max_streams` log line in check (1) distinguishes these directly.
**n=3 and contaminated; this is a hypothesis, not a finding.**

§2.1-§2.2 document what the flag does in source; §2.3 records a truncation mechanism that exists in
source but was not observed to fire; §2.6 pins the preconditions.

**Parsing footgun, read this before reproducing anything:** the flag is parsed with
`strcmp(value, "true") == 0 || atoi(value) != 0` **[FH-src `flags.cpp:174`]**. The comparison is
case-sensitive and `atoi("TRUE") == 0`, so **`DEBUG_CLR_GRAPH_PACKET_CAPTURE=TRUE` silently
*disables* packet capture** — the exact opposite of what it reads like. Use `true` or `1`.
`0`, `false`, `off`, `no`, and empty all disable.

### 2.1 Definition and default — [FH-src]

`github.com/ROCm/clr`, tag `rocm-7.2.0`, `rocclr/utils/flags.hpp:240`:

```c
release(bool, DEBUG_CLR_GRAPH_PACKET_CAPTURE, true,                           \
         "Enable/Disable graph packet capturing")                             \
```

- **Type** bool, **default `true`**, **`release` flag** → active in shipping builds, not debug-only.
- Present in the shipped `/opt/rocm/lib/libamdhip64.so` **[FH-bin]**.
- Also confirmed on the `amd-staging` branch with the same default, so this is stable across 7.x.

**Value parsing**, `rocclr/utils/flags.cpp:174` **[FH-src]**:

```c
case Tbool:
  *(bool*)value_ = (strcmp(value, "true") == 0 || atoi(value) != 0) ? true : false;
```

So: `true`, `1`, any non-zero number → enabled. `0`, `false`, `off`, `no`, empty → disabled.
**Gotcha: `TRUE` in uppercase parses as *disabled*** (case-sensitive `strcmp`, and `atoi("TRUE")==0`).

An AMD test documents the variable by name:
`hip-tests/catch/unit/graph/hipGraphPerf.cc:492` does
`setenv("DEBUG_CLR_GRAPH_PACKET_CAPTURE", "true", 1)` and skips the perf test if it fails **[FH-src]**.

### 2.2 What it actually changes — [FH-src]

**Instantiate time**, `hipamd/src/hip_graph_internal.cpp`, `GraphExec::Init()`:

`hip_graph_internal.cpp:423-447` — note the **nesting order**: the topology check is the *outer*
condition, the flag is *inner*, so when `max_streams_ != 1` the flag is simply never consulted.

```c
if (max_streams_ == 1) {
  FindStreamsReqPerDev();
  if (max_streams_dev_.size() > 1) { /* ...CreateStreams per device... */ }
  if (DEBUG_CLR_GRAPH_PACKET_CAPTURE) {
    // For graph nodes capture AQL packets to dispatch them directly during graph launch.
    status = CaptureAQLPackets();
  }
} else {
  status = CreateStreams(max_streams_, hip::getCurrentDevice()->deviceId());
}
```

So **a multi-stream graph never uses packet capture, regardless of the flag.** But "multi-stream"
means *the graph branches*, not *`DEBUG_HIP_FORCE_GRAPH_QUEUES > 1`* — those are different things,
and confusing them is an easy and costly mistake. **See §2.6.**

**Launch time**, same file, `EnqueueGraphWithSingleList()`:

```c
amd::AccumulateCommand* accumulate = nullptr;
if (DEBUG_CLR_GRAPH_PACKET_CAPTURE) {
  accumulate = new amd::AccumulateCommand(*hip_stream, {}, nullptr);
}
for (int i = 0; i < topoOrder_.size(); i++) {
  if (topoOrder_[i]->GraphCaptureEnabled()) {
    for (auto& packet : topoOrder_[i]->GetAqlPackets())
      hip_stream->vdev()->dispatchAqlPacket(packet, topoOrder_[i]->GetKernelName(), accumulate);
  } else {
    topoOrder_[i]->SetStream(hip_stream);
    status = topoOrder_[i]->CreateCommand(topoOrder_[i]->GetQueue());
    topoOrder_[i]->EnqueueCommands(hip_stream);
  }
}
if (DEBUG_CLR_GRAPH_PACKET_CAPTURE) { accumulate->enqueue(); accumulate->release(); }
```

In 7.2.0 the batched variant is used: `vdev()->dispatchAqlPacketBatch(packets, kernelNames, accumulate)`,
which submits up to `DEBUG_HIP_GRAPH_BATCH_SIZE` (256) packets per doorbell ring.

**Summary of the two modes:**

| | `=1` (default) | `=0` |
|---|---|---|
| Instantiate | pre-builds AQL packets per node | nothing extra |
| Replay | writes pre-built packets straight into the HSA ring, batched doorbell | `CreateCommand` + `EnqueueCommands` per node — **ordinary eager-style dispatch** |
| Runtime object | **one** `amd::AccumulateCommand` (`CL_COMMAND_TASK`) for the whole graph | one `amd::Command` (`CL_COMMAND_NDRANGE_KERNEL`) per node |
| CPU launch cost | low (the point of graphs) | high |

Per-node opt-outs exist regardless of the flag (`GraphCaptureEnabled()` overrides **[FH-src]**):
cooperative kernels are never captured; memcpy nodes only for `hipMemcpyDeviceToDevice`;
memset nodes always.

### 2.3 Measurement: the flag has no effect. Plus a source mechanism that was *not* observed to fire.

**Two separate things live in this section, and they must not be used to support each other.**

#### 2.3.1 The measurement — [FH-lead], n=5 per condition

Model-free probe on this exact machine/container: bf16 gemm + elementwise + reduction captured into
a `torch.cuda.CUDAGraph`, replayed 20× under `torch.profiler`. 5 repetitions per setting on a quiet
GPU (`results/graph_visibility_probe/repeat_yihou.txt`):

| setting | reps | distinct GPU entries | gemm count | reduce count |
|---|---|---|---|---|
| unset | 5 | 7 (all) | 19/20 (all) | 19/20 (all) |
| `=0` | 5 | 7 (all) | 19/20 (all) | 19/20 (all) |
| `=1` | 5 | 7 (all) | 19/20 (all) | 19/20 (all) |

**15/15 runs identical. Zero spread within a setting, zero difference between settings.**

Conclusion: **`DEBUG_CLR_GRAPH_PACKET_CAPTURE` has no measurable effect on per-kernel profiler
visibility or on event count on this stack.** This independently confirms the source read in §2.3.2
(`activity.cpp:104-119` emits one activity record per kernel even in capture mode).

Note also that **`unset` ≡ `=1`**, since the default is `true` (§2.1) — those were never two
conditions. An earlier single-sample probe that appeared to show a count gradient
(20 / 18-19 / 16-17 across `=0` / unset / `=1`) is **refuted**: the spread was GPU contention with a
concurrent smoke benchmark, and the unset-vs-`=1` half of it was a same-condition comparison, i.e.
a pure noise estimate. It is recorded here only so the number is not resurrected later.

**One observed constant, NOT a `DEBUG_CLR_GRAPH_PACKET_CAPTURE` effect:** the profiler records
**19 of 20** replays in every one of these 15 runs — a reproducible off-by-one, identical across all
three capture settings.

**It is, however, probably not constant in general.** At `DEBUG_HIP_FORCE_GRAPH_QUEUES=1` the count
was **20/20** in 3/3 runs [FH-lead, n=3, GPU not fully quiet — not a result]. So the off-by-one may
be a **queue-count** effect rather than the profiler start/stop window artefact first assumed.
All 15 runs above were at the default `QUEUES=4`, so they cannot distinguish the two.
Not investigated; it does not affect relative attribution. Flagged so nobody mistakes it for a
capture-flag effect or for evidence of §2.3.2.

#### 2.3.2 A truncation mechanism that exists in source but was not observed to fire — [FH-src]

Stated for completeness, because it is a real code path and someone will find it. **There is
currently nothing to explain — see §2.3.1 — so this is not offered as an explanation of anything.**

**Mechanism A — silent truncation to `min(timestamps, kernel_names)`.**
`rocclr/platform/activity.cpp:104-114`:

```c
if (command.type() == CL_COMMAND_TASK) {
  auto timestamps   = static_cast<const amd::AccumulateCommand&>(command).getTimestamps();
  std::vector<std::string> kernel_names =
      static_cast<const amd::AccumulateCommand&>(command).getKernelNames();
  for (uint32_t i = 0; i < timestamps.size() && i < kernel_names.size(); i++) {
    record.begin_ns = timestamps[i].first;
    record.end_ns   = timestamps[i].second;
    record.kernel_name = kernel_names[i].c_str();
    function(ACTIVITY_DOMAIN_HIP_OPS, operation_id, &record);
  }
}
```

If fewer timestamps were harvested than kernels were dispatched, **the surplus kernels are
silently dropped, with no warning**. `CL_COMMAND_TASK` is *only* produced by the packet-capture
graph path, so this loop cannot affect `=0` or eager.

**Mechanism B — signal-ring reuse coalescing two dispatches into one timestamp.**
`rocclr/device/rocm/rocvirtual.cpp`, `HwQueueTracker::ActiveSignal()` (line 501 ff.) hands out
completion signals from a **ring buffer** `signal_list_` indexed by `current_id_`:

```c
auto temp_id = (current_id_ + 2) % signal_list_.size();
if (Hsa::signal_load_relaxed(signal_list_[temp_id]->signal_) > 0) {
  ... grow the list with a brand-new signal ...   // only if the GPU is still busy
}
if (!new_signal) { ++current_id_ %= signal_list_.size(); WaitCurrent(); WaitNext(); }
...
ProfilingSignal* prof_signal = signal_list_[current_id_];
prof_signal->flags_.done_ = false;
prof_signal->flags_.isPacketDispatch_ = false;
prof_signal->ResetCachedTiming();          // <-- previous dispatch's timing is destroyed
...
if (ts != nullptr) ts->AddProfilingSignal(prof_signal);   // line 610, same pointer pushed again
```

and `AddProfilingSignal` is a plain `signals_.push_back(signal)` with no dedup (`rocvirtual.hpp:147`).

At harvest, `Timestamp::checkGpuTime` skips anything already processed:

```c
auto process_signal = [&](ProfilingSignal* sig) {
  if (sig->flags_.done_) { return; }        // rocvirtual.cpp:178
  ...
};
```

and `ExtractSignalTiming` sets `signal->flags_.done_ = true` after pushing one
`addTimestamps(sig_start, sig_end)`.

**So:** in packet-capture mode all N packets of a replay share **one** `Timestamp` and **one**
signal ring. If the ring wraps within a single replay (N packets, ring size M < N), the same
`ProfilingSignal*` lands in `signals_` more than once; the second occurrence is skipped by the
`done_` guard, so **two dispatches yield one timestamp** — and Mechanism A then truncates the
kernel-name list to match. With `=0`, each node is its own `amd::Command` with its own `Timestamp`,
so this coalescing cannot occur.

Crucially, whether the ring wraps is **timing-dependent**: `ActiveSignal` only grows the ring when
the peeked signal 2-ahead is still busy. That would make the count run-to-run variable under GPU
contention.

> **[FH-lead] NEITHER MECHANISM WAS OBSERVED TO FIRE.** See §2.3.1 for the measurement
> (5 reps × 3 settings, 15/15 runs identical). **Mechanisms A and B are real code paths in CLR
> source that produced no observable effect here.** The source finding and the earlier timing
> observation must not be used to prop each other up — there is no observation left to explain.
>
> The 19/20 count is **not** Mechanism B **at `QUEUES=4`**: `=0` shows the identical 19/20 there,
> and coalescing is structurally impossible in that mode (each node gets its own `amd::Command` and
> `Timestamp`). **Caveat added after the fact:** that elimination assumed the 19/20 was invariant.
> It is not — `QUEUES=1 CAPTURE=0` gave 20/20 [FH-lead, n=3, contaminated]. The elimination still
> holds *within* the `QUEUES=4` runs, since `=0` and `=1` agree there, but it no longer rules
> Mechanism B out across queue settings. The `QUEUES=1 CAPTURE=1` cell is unrun. Not investigated —
> it does not affect relative attribution.

**If someone later wants to hunt Mechanism B deliberately** (not needed for our task): capture a
graph with a *known* kernel count K, replay R times under torch.profiler, and check
`sum(count) == K*R` per kernel name, for `=0` vs `=1`, while varying `ROC_SIGNAL_POOL_SIZE`
(default 64) across e.g. 8 / 64 / 1024, on an **idle** GPU. A ring smaller than the per-replay
packet count is the condition that would force a wrap. Status of Mechanism B: **[UNKNOWN] —
present in source, not demonstrated by any measurement we ran.**

**Third, weaker mechanism, worth knowing:** 7.2.0 passes kernel names by pointer —
`vcmd->setKernelNamesRef(&kernelNames)` in `dispatchAqlPacketBatch`, where `kernelNames` is in
some call sites a **caller-local vector** (`hip_graph_internal.cpp:855`). This is precisely the bug
class AMD fixed later in [#8735 / #9697](https://github.com/ROCm/rocm-systems/pull/9697)
("`setKernelNamesRef()` was called once per segment dispatch, **overwriting** the previous
segment's kernel names"). That fix targets the 7.14 segment scheduler which we do not have, so I
will **not** claim our path is affected — but it is the same pointer-to-local pattern.

### 2.4 A unit inconsistency I found and then ruled out — do not chase it

`rocvirtual.cpp:217` converts ticks→ns on the normal path (`final_start = start * ticksToTime_`),
but `addTimestamps(sig_start, sig_end)` at line 258 stores the **raw** cached HSA value.
`ticksToTime_ = 1e9 / frequency` (`rocdevice.cpp:1839`). On this host `rocminfo` reports
**System Timestamp Freq. = 1000 MHz** **[FH-bin]**, so `ticksToTime_ == 1.0` and the missing
multiply is a **no-op here**. Recorded only so nobody re-derives it and mistakes it for a bug.
It *would* matter on a part with a 100 MHz timestamp clock.

### 2.5 Where the real difficulty lies: attribution, not visibility

The per-kernel *durations* are trustworthy. The per-*step* attribution is not.

**[2H], but on our exact stack:** [sgl-project/sglang#31545](https://github.com/sgl-project/sglang/issues/31545)
— MI355X gfx950, **ROCm 7.2.0, torch 2.9.1+rocm7.2.0**. The reporter initially claimed total
invisibility, then corrected it: kernels *are* captured (~2873), but **92% of kernels and 88% of
GPU-busy time land after the last CPU step marker**. Mid-sequence step markers carry ~17 kernels /
~0.1 ms each. Per-decode-step wall time is unrecoverable in graph mode; `--disable-cuda-graph` is
their workaround. Per-kernel µs matched graph-on within ~2.6%.

**[2H]** [AMD-AGI/TraceLens#827](https://github.com/AMD-AGI/TraceLens/issues/827) — vLLM decode,
HIP graphs on: 40 `hipGraphLaunch` events vs 41,960 kernel events, **every graph kernel shares its
correlation id with a `hipGraphLaunch`**, so ~97% of GPU time collapses into one `hipGraphLaunch`
row categorised as "other" in any CPU-op-keyed summary. This is inherent to graphs — there is no
per-kernel `hipLaunchKernel` at replay time — not a ROCm defect.

**Practical reading for us:** `key_averages()` sums/means per kernel name are usable.
Anything keyed on the CPU op tree, input shapes, or per-step markers is not.

---

### 2.6 Exactly when is packet capture in force? — [FH-src]

This section exists because an earlier draft of §1.3 said "`max_streams_ > 1` disables packet
capture entirely" next to a row stating `DEBUG_HIP_FORCE_GRAPH_QUEUES` defaults to 4. That
juxtaposition invites the inference "the default queue count of 4 disables capture, so capture is
never active" — **which is wrong**, and it is worth spelling out why.

**`DEBUG_HIP_FORCE_GRAPH_QUEUES` does not set `max_streams_`. Graph topology does.**
`hip_graph_internal.cpp:185-207`:

```c
void Graph::ScheduleOneNode(Node node, int stream_id) {
  if (node->stream_id_ == -1) {
    node->stream_id_ = stream_id;
    max_streams_ = std::max(max_streams_, (stream_id + 1));         // line 191
    // (child graphs: max_streams_ = std::max(max_streams_, child->max_streams_))
    for (auto edge : node->GetEdges()) {
      ScheduleOneNode(edge, stream_id);
      // 1. Each extra edge will get a new stream from the pool
      // 2. Streams will be reused if the number of edges > streams
      stream_id = (stream_id + 1) % DEBUG_HIP_FORCE_GRAPH_QUEUES;   // line 205
    }
  }
}
```

The increment sits **inside the per-edge loop**, so `stream_id` only advances at a node with ≥2
outgoing edges. Consequences:

- **Linear chain** (every node out-degree ≤ 1, single root): `stream_id` stays 0 for the entire
  walk → `max_streams_ == 1` → **packet capture is ACTIVE, at the default `QUEUES=4`.**
- **Branchy graph**: `stream_id` advances at each branch, modulo `QUEUES` → `max_streams_ > 1` →
  capture disabled.
- `DEBUG_HIP_FORCE_GRAPH_QUEUES=1` makes the modulus 1, so `stream_id ≡ 0` always → `max_streams_
  == 1` → **capture forced ON even for a branchy graph.** The flag's effect is the *opposite* of
  "more queues ⇒ no capture ⇒ fewer profiler events".

**Direct readout — do not infer this, measure it.** `hip_graph_internal.cpp:1105-1108` logs
`max_streams_` on every launch:

```c
ClPrint(amd::LOG_DEBUG, amd::LOG_CODE,
        "GraphExec::Run max_streams: %d, on device: %d, total number of nodes: %d", ...);
```

`LOG_DEBUG` = level 4, `LOG_CODE` = mask `0x4000`. So:

```bash
AMD_LOG_LEVEL=4 AMD_LOG_MASK=0x4000 <your run> 2>&1 | grep "GraphExec::Run max_streams"
```

`max_streams: 1` ⇒ the capture path was taken. This is a CPU-side log line — no profiler, no extra
perturbation — and it settles the question in one run.

**Full list of conditions that silently prevent packet capture** (all [FH-src]):

| Condition | Where | Effect |
|---|---|---|
| `max_streams_ != 1` (graph branches) | `Init():423`, `Run():1110` | no capture; `RunNodes()` parallel path |
| `max_streams_dev_.size() > 1` (multi-**device** graph) | `Run():1123`, `UpdateAQLPacket():649` | `EnqueueMultiDeviceLinearGraph` instead |
| `instantiateDeviceId_ != launch_stream->DeviceId()` | `Run():1125` | per-node `CreateCommand`/`EnqueueCommands` |
| node is a cooperative kernel | `GraphCaptureEnabled()` override | that node only, falls to per-node branch |
| memcpy node that is not D2D | `GraphCaptureEnabled()` override | that node only |
| unrecognised node type (host, event, …) | base `GraphCaptureEnabled()` returns false | that node only — **mixed mode**, rest of graph still captured |

**There is no minimum graph size.** `DEBUG_HIP_GRAPH_BATCH_SIZE=256` is a chunking parameter
*inside* `dispatchGenericAqlPacketBatch` (staggered powers-of-2 growth up to that cap), **not a
threshold below which capture is skipped**. A 5-node graph is captured normally.

**`DEBUG_HIP_FORCE_ASYNC_QUEUE` does not interact with the capture decision.** Its only use is in
`GraphKernelNode::CreateCommand` (`hip_graph_internal.hpp:1173`) — i.e. the **non-capture**
per-node path — where it sets `hipExtAnyOrderLaunch` on siblings of a multi-edge dependency when
`DEBUG_HIP_FORCE_GRAPH_QUEUES == 1`.

---

## 3. Q4 — Tooling comparison for "rank the top-10 time-consuming items in a decode loop"

| Tool | Per-kernel GPU time under hipGraph replay | Overhead | Trace size, 1000s of iters | Verdict |
|---|---|---|---|---|
| `torch.profiler` + `key_averages()` | **Works** — confirmed by the lead's probe [FH-lead] and by the CLR reporting path [FH-src]. Per-*step* attribution does not (§2.5). | Enabling the profiler forces a completion signal per graph packet [FH-src] — profiled replay ≠ benchmarked replay. Magnitude [UNKNOWN]. | JSON per rank; GB-scale, Perfetto-hostile | **Use for the ranked kernel table over a 5-20 iteration window.** Do not read per-step wall time from it. |
| `rocprofv3 --kernel-trace` | Dispatch-level via HSA queue interception; *should* see every AQL packet regardless of graph. **[UNKNOWN] for 7.2** — no first-hand confirmation. Note our build is **1.1.0 and has NO `--hip-graph-trace`** [FH-bin `--help`]; per-graph-node attribution is a ROCm 7.14/10.x feature [2H]. | [UNKNOWN] | `-f rocpd` → SQLite | **Best candidate for a trustworthy ranking, zero code changes.** `rocprofv3 --attach <PID> --kernel-trace --stats -f rocpd` works on an already-running process — a good fit for a harness with no control plane. |
| RPD (`rpdTracerControl`) | Same roctracer backend as Kineto → expect the same behaviour. Unverified by anyone I found; the SGLang reporter says the RPD path "may exhibit similar deferral (not separately re-verified)" [2H]. | "low" [2H]; the related `rocm-trace-lite` quotes ~2-4% standard / near-zero in lite mode [2H] | SQLite — best for long runs | **Skip.** Not installed, needs an in-container build, and `rocpd` gives the same SQLite ergonomics for free. |
| `rocprof-sys` / omnitrace | [UNKNOWN] under graph replay | Sampling + instrumentation; heavier | OTF2 / Perfetto | Installed, but aimed at whole-application call-path analysis. **Overkill for a top-10 kernel ranking.** |

**Recommendation.** Run both and cross-check (对拍):
1. `torch.profiler` with `schedule(wait=N, warmup=1, active=5, repeat=1)`, `with_stack=False`,
   `record_shapes=False` → `prof.key_averages().table(sort_by="self_device_time_total", row_limit=10)`.
2. `rocprofv3 --attach <PID> --kernel-trace --stats -f rocpd --attach-duration-msec 2000`, then
   `PYTHONPATH=/opt/rocm/lib/python3.10/site-packages python3 -m rocpd summary`.

If the two top-10 lists agree, the ranking is trustworthy regardless of which backend has
attribution quirks. This directly de-risks the graph question at low cost.

**rocprofv3 options present in our 1.1.0 build [FH-bin `rocprofv3 --help`]:**
`--kernel-trace`, `--runtime-trace`, `--sys-trace`, `--hip-runtime-trace`, `--marker-trace`,
`--stats`, `--kernel-iteration-range`, `--kernel-include-regex`, `--kernel-rename`,
`--group-by-queue`, `--selected-regions`, `-P START:DURATION:REPEAT` with
`--collection-period-unit {hour,min,sec,msec,usec,nsec}`, `--pid/--attach PID`,
`--attach-duration-msec`, `--perfetto-buffer-size`, `--perfetto-buffer-fill-policy {discard,ring_buffer}`,
`--minimum-output-data`, `-f {csv,json,pftrace,otf2,rocpd}`, plus PC sampling (beta).
**Absent: `--hip-graph-trace`.**

---

## 4. Q5 — Practical gotchas

1. **8 ranks → 8 traces.** Kineto writes one file per process. Gate on rank unless you are
   specifically chasing collective imbalance. Write to a `yihou`-named scratch dir on **local**
   disk, not NFS.
2. **Window the capture.** Options, all available first-hand:
   - `torch.profiler.profile(schedule=torch.profiler.schedule(wait=N, warmup=1, active=5, repeat=1))`
     + `prof.step()` per decode iteration. 5 active iterations suffice for a kernel ranking.
   - Manual `prof.start()` / `prof.stop()` around iterations `[k, k+5)` — simplest in a
     scheduler-free harness where we own the loop.
   - rocprofv3: `--kernel-iteration-range`, `-P <delay>:<dur>:<repeat> --collection-period-unit msec`,
     or `--selected-regions` bracketed by `roctx.profilerResume(0)` / `roctx.profilerPause(0)` from
     the installed `roctx` Python package **[FH-bin]**.
3. **`with_stack=True` — leave it off.** It walks the Python stack per op; on a decode loop of
   thousands of small ops it is the dominant overhead and trace-size contributor, and it buys
   **nothing** for graph-replayed kernels — their Python stacks exist only at capture time [2H].
   Same reasoning for `record_shapes=True`.
4. **Warm up past graph capture.** The first iterations include graph instantiation,
   `CaptureAQLPackets()`, kernarg-pool growth and hidden-heap init [FH-src `GraphExec::Init`,
   `AllocKernelArgForGraphNode`]. Profiling those pollutes the ranking. Warm up at least as many
   iterations as there are captured graph shapes.
5. **`torch.cuda.synchronize()` before `prof.stop()`.** Given §2.5 (records flushed in a batch
   after the replay), this is not optional — sync, then stop, or in-flight graph work is missing
   or attributed past the boundary.
6. **Do not use `AMD_SERIALIZE_KERNEL=3` / `HIP_LAUNCH_BLOCKING=1` to "fix" attribution.** They
   serialize dispatch and change which kernels dominate; you would rank a different workload.
7. **Profiling perturbs graph replay itself** [FH-src, §2.3]: per-packet completion signals get
   attached and the signal-free fast path is abandoned. Absolute latency from a profiled run is an
   upper bound, not a benchmark number.
8. **Eager (`--disable-cuda-graph`) is a cross-check, not a substitute.** Per-kernel µs matched
   graph-on within ~2.6% [2H]; but eager loses dual-stream overlap and inflates step time.
9. **`ACTIVITIES_MAX_GPU_BUFFER_SIZE_MB`** — on a long capture, device-activity buffer overflow
   drops records silently. If kernel counts look short, suspect this before suspecting §2.3.
10. **Do not set `DEBUG_CLR_GRAPH_PACKET_CAPTURE`.** At stock runtime settings it made no difference
    to per-kernel visibility or event counts (§2.3.1, n=5 at `DEBUG_HIP_FORCE_GRAPH_QUEUES=4`), and
    if you do set it, `=TRUE` in uppercase silently *disables* packet capture (§2.0) — a way to slow
    down graph replay while believing you enabled something. Leave it alone. (Its behaviour at
    `QUEUES=1` is unmeasured — §2.0. That does not change the advice for stock configurations.)
11. **Benchmark on a quiet GPU.** The single-sample count variation that started this investigation
    was contention with a concurrent 8-GPU smoke run (§2.3.1). Any profiler count or duration taken
    while a neighbour job is running is unusable.

---

## 5. Explicitly unknown

- **Mechanisms A and B in §2.3.2 — settled negatively at `QUEUES=4` only.** They exist in CLR
  source; they were measured not to fire at default runtime settings (§2.3.1, n=5). **Not tested at
  `QUEUES=1`**, the setting that forces `max_streams_ == 1` and hence forces capture on. Whether
  they can be forced to fire (small `ROC_SIGNAL_POOL_SIZE`, or a graph whose packet count exceeds
  the signal ring) is **[UNKNOWN]** and not worth pursuing for this task.
- **Why the profiler records 19 of 20 replays.** First assumed a profiler start/stop window
  artefact; that is now doubtful, because `QUEUES=1 CAPTURE=0` gave **20/20** while every
  `QUEUES=4` run gave 19/20 [FH-lead, **n=3, GPU not fully quiet, not a result**]. Possibly a
  queue-count effect. Not investigated. Does not affect relative attribution.
- **Does `DEBUG_CLR_GRAPH_PACKET_CAPTURE` matter at `QUEUES=1`?** The decisive cell
  `QUEUES=1 CAPTURE=1` is **unrun**. **[UNKNOWN], measurement in progress.** Until it lands, the
  flag has been characterised **only at stock settings**, not in general.
- **Whether `max_streams_ == 1` held during the §2.3.1 probe**, i.e. whether packet capture was
  genuinely in force in the `=1` condition. Source says yes for a linear single-stream graph
  (§2.6); not yet read off the running system. One-line check in §2.0. **This is the only thing
  gating §2's conclusion from provisional to final.**
- **Whether `rocprofv3 --kernel-trace` sees graph-replayed dispatches on ROCm 7.2.0.**
  Settle with a 2 s `--attach` against the running harness: is the kernel count ~1 per graph node
  per replay, or ~1 per replay?
- **The exact flush timing of `AccumulateCommand` activity records** — the mechanism behind the
  "post-marker burst". I did not trace the completion-callback path to where `ReportActivity` fires.
- **Overhead magnitude** of torch.profiler vs rocprofv3 on this workload. No measurement.
- **Whether `HIP_FORCE_QUEUE_PROFILING` is a live alias** of `GPU_FORCE_QUEUE_PROFILING`.
- **`rocprof-sys` behaviour under hipGraph replay** — not investigated.

---

## Sources

First-hand source, `github.com/ROCm/clr` tag `rocm-7.2.0`:
`rocclr/utils/flags.hpp`, `rocclr/utils/flags.cpp`, `rocclr/platform/activity.cpp`,
`rocclr/platform/command.hpp`, `rocclr/device/rocm/rocvirtual.cpp`,
`rocclr/device/rocm/rocvirtual.hpp`, `rocclr/device/rocm/rocdevice.cpp`,
`hipamd/src/hip_graph_internal.cpp`. Plus `ROCm/hip-tests` `catch/unit/graph/hipGraphPerf.cc`.

Second-hand: [ROCm/rocm-systems#9697](https://github.com/ROCm/rocm-systems/pull/9697),
[sgl-project/sglang#31545](https://github.com/sgl-project/sglang/issues/31545),
[AMD-AGI/TraceLens#827](https://github.com/AMD-AGI/TraceLens/issues/827),
[HIP environment variables](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/env_variables.html),
[ROCm environment variables](https://rocm.docs.amd.com/en/latest/reference/environment-variables/index.html),
[Using rocprofv3](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html),
[ROCm/rocmProfileData](https://github.com/ROCm/rocmProfileData),
[SGLang AMD PROFILING.md](https://github.com/sgl-project/sglang/blob/main/3rdparty/amd/profiling/PROFILING.md),
[AMD blog: Optimizing DeepseekV3 Inference on SGLang](https://rocm.blogs.amd.com/software-tools-optimization/kernel-analysis-deep/README.html).
