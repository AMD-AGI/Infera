# Study: team-calibrated AMD profiler fork (jiejingzhangamd/rocm-systems)

Date: 2026-11-13. Evidence marks: [F] first-hand (file read / command run), [A] arithmetic on [F], [I] inference (not established).
Local blobless clone (branch `diag/rocm72-capture-node-map-v35`, 76 MB `.git`, no checkout): `./repo/`.
Nothing was built or run on a GPU in this study. No container was started.

## 1. Bottom line

- The fork is a **rocprofiler-sdk** fork only (no rocprofv3-CLI, rocprof-compute or rocprof-systems changes) [F: diffstat, 54 files, all under `projects/rocprofiler-sdk/`]. Base = upstream `develop` @ `a52dc9e73c` (2026-10-26) [F].
- "Calibrated" is **not** a numeric clock-correction factor. No such code is in the diff [F]. What the team changed is (a) **attribution**: every kernel replayed from a HIP/CUDA graph gets the graph-exec id, node ordinal, and the HIP call that captured it; (b) **timestamp auditing**: an opt-in mode that checks raw GPU times strictly instead of silently shifting them, and reports dropped records; (c) **ROCm 7.2.0 (ROCr) workarounds** for two runtime bugs that affect profiling [F: code and commit messages; the bug claims are the author's, not re-verified].
- Our 1106 serving image is ROCm 7.2.0 [F for the 1020 base image: `GPU_ARCH=gfx950-rocm720`, `TORCH_ROCM_VERSION=2.11.0+rocm7.2` in `research/infx_4881/image_1020.json`; the 1106 image is "1020 + 14 overlay layers" per `environment.md` [F], so 1106 = ROCm 7.2.0 is [I, high]]. The `rocm72` branches are therefore the ones that match our stack.
- Tool to use: build the fork's **rocprofiler-sdk** (with tests, to get `librocprofiler-sdk-json-tool.so`) or its **rocprofv3** for the same build, inside a container from the 1106 image, and inject it into the SGLang server through env vars. The team's own SGLang recipe uses the json-tool + a `sitecustomize.py`, not the rocprofv3 CLI [F].

## 2. What the fork contains (relative to upstream ROCm/rocm-systems)

### 2.1 Branches [F: `gh api .../compare/develop...<branch>`]

| Branch | Ahead of base | Content |
|---|---|---|
| `develop` | 0 (8461 behind upstream) | stale mirror, nothing of ours |
| `fix/rocprofiler-graph-memset-attribution` | 9 | graph attribution + memset |
| `feature/rocprofiler-graph-capture-attribution` | 10 | = above + merge PR #1; this is upstream PR **ROCm/rocm-systems#14887** (OPEN, 2026-11-03) |
| `fix/rocprofiler-rocm72-scoped-strict` | 13 | + ROCm 7.2 clock diagnostics, drop counters, wrap-safe intercept |
| `diag/rocm72-event-direct-plan-v31` | 14 | + HIP event packet / GPU-time diagnostics |
| `diag/rocm72-scoped-strict-event-signal-v34` | 16 | + strict-window catch-up bound 10 us -> 100 us, event-signal logging |
| `diag/rocm72-capture-node-map-v35` | 17 (newest, HEAD `ae7c95f4aa`, 2026-11-11) | + capture node/source identity logging |

All branches are linear supersets of each other [F: same commit prefix lists].

### 2.2 Authors and commits [F: `git log a52dc9e73c..HEAD`]

- 4 commits by **Jiawei Chen** (2026-10-24/25) = upstream PRs **#14204** and **#14211** (both OPEN), feature request issue **#14200** (OPEN): "attribute HIP graph replays to the context they were captured in". They add to `rocprofiler_kernel_dispatch_info_t` (taken from reserved padding, struct stays 128 B): `graph_exec_id`, `graph_node_id`, `capture_correlation_id`. Multi-stream captures are matched in the runtime's depth-first enqueue order. The commit message says MI350X, ROCm 7.2.3 was where multi-stream replays broke the naive capture-order matching.
- 13 commits by **Jiejing Zhang** (2026-11-03 .. 11-11). Commit email of three of them is `jiejizha@smc300x-ccs-aus-a16-19...`, the same host this study ran on [F: `hostname`]; `/home/jiejizha` is not readable by us [F].
  1. `62392bc462` / `8bc27cb00a` / `fc7f2bcdc5`: invalidate stale graph-exec plans after mutations, resolve **module-launched** kernel names, opt-in **copyBuffer** attribution (`ROCPROFILER_EXPERIMENTAL_GRAPH_COPYBUFFER=1`), refuse attribution on mixed graphs.
  2. `ffd992f6bc` / `f940c0ca26`: opt-in **fillBuffer (memset)** attribution (`ROCPROFILER_EXPERIMENTAL_GRAPH_FILLBUFFER=1`).
  3. `be5ac8dea6` "Add opt-in ROCm 7.2 profiler clock diagnostics": strict-window API, per-buffer drop counters in JSON metadata, `ROCPROFILER_EXPERIMENTAL_SINGLE_SIGNAL_WAIT=1` (code comment: "ROCr builds whose wait_any corrupts the satisfying-value array when a nonzero signal index completes").
  4. `6d413f8d78` "Reconstruct wrapped ROCr 7.2 legacy intercept batches": `ROCPROFILER_EXPERIMENTAL_WRAP_SAFE_INTERCEPT_BATCH=1` (code comment: "ROCr 7.2.0 can pass a pointer into the legacy intercept queue's ring even when this batch crosses its end"). Without it, the profiler would read packets past the ring end [I from the comment].
  5. `00a33399bc`, `bea913c7eb`, `ae7c95f4aa`: opt-in stderr diagnostics (`ROCPROFILER_EXPERIMENTAL_EVENT_{WAIT_DISPOSITION,WAIT_SIGNAL,BARRIER_PACKET,GPU_TIME}`, `..._CAPTURE_NODE_IDENTITY`). Code comment: "Keep this off for performance captures" [F].
  6. `a5094c9bcc`: strict-window host-clock catch-up limit 10 us -> **100 us** [F: `strict_bound_wait_limit_ns = 100000`]. The experiments README still says "10 microseconds" [F] - doc is stale.

### 2.3 Why (stated by the authors) [F: README, PR #14887 body, validation-summary.json]

- Purpose: make kernel traces of **CUDA-graph-replayed SGLang decode** attributable to the capture-time HIP call (and therefore to the Python/op context at capture). Stock SDK only had per-dispatch kernel names; replays carry no capture context.
- Measured on SGLang TP4 (DeepSeek-V4.1-Flash, DSPARK spec decoding, 524,288-token prompt, 128 output tokens, CUDA-graph decode, `ROCPROFILER_QUEUE_INTERPOSITION=0`, image `rocm/sgl-dev:v0.5.23-rocm10-mi35x-20261101`, i.e. **ROCm 10**, MI350-series): per-rank graph-dispatch resolution 2,682/7,085 (37.85%) -> 6,618/6,805 (97.25%) with module naming -> 6,270/6,270 (100%) with copyBuffer; rerun at `8bc27cb0`: 6,513/6,513 per rank, all 4 traces finalized.
- The authors explicitly disclaim: workload-specific coverage; no Python layer labels; no cross-stream critical path; no profiler-overhead estimate; queue interposition path not validated.
- The `rocm72` branches were added after that (11-10..11-11). Their validation on ROCm 7.2 is **not recorded** in the repo [F: no 7.2 entry in validation-summary.json].

## 3. How the timing differs from stock

Stock `adjust_profiling_time` (`tracing/profiling_time.hpp`) [F]:
- bounds = [CPU enqueue timestamp, CPU timestamp when the completion handler runs];
- if GPU end > bound end: the **whole interval is shifted earlier** (duration kept); if GPU start < enqueue: shifted later; if start > end: swapped with an ERROR log; only `ROCPROFILER_CI_STRICT_TIMESTAMPS=1` turns violations into `ROCP_FATAL`.
- Consequence [I]: stock traces can silently move kernels in time (by the CPU/GPU clock skew), which corrupts gaps/overlap analysis between kernels/streams but not per-kernel durations.

Fork [F: `kernel_dispatch/profiling_time.cpp`]:
- **Default behaviour is unchanged.** All new behaviour is opt-in.
- Opt-in strict window: a process calls `rocprofiler_experimental_set_dispatch_strict_window(start_ns, end_ns)` (exported C symbol in `librocprofiler-sdk.so`, SDK timestamp domain). Dispatches whose **CPU enqueue time** falls in the window: the host reads its clock *after* the HSA dispatch-time query, and if the raw GPU end is ahead of the host clock it spins up to 100 us for the host to catch up; then the strict checks apply and a violation is **`ROCP_FATAL` (process abort)**. The raw GPU interval is never shifted. Counts/max gap/total wait are written to JSON metadata (`strict_future_bound_*`).
- JSON metadata also gets `dropped_records` and `buffer_drop_observations` (stock tool ignored `drop_count`).
- So "calibrated timing" = raw GPU intervals with an audited guarantee of no silent shift inside the window, plus drop accounting. It does not change kernel durations [F code]. Spin-wait time is profiler overhead on the completion path [F comment].

## 4. Which tool for SGLang-in-docker on MI355X (gfx950)

### 4.1 Options

| Option | What you get | Graph-capture attribution | Notes |
|---|---|---|---|
| A. Fork SDK + **json-tool** + `sitecustomize.py` (team recipe) | per-PID JSON with HIP API + kernel dispatch + code objects | yes (capture traced from first `CUDAGraph.capture_begin`) | json-tool is a **test** tool: needs `-DROCPROFILER_BUILD_TESTS=ON` [F: `tests/tools/json-tool.cpp`]. Control via `json_tool_start/stop/pause_until_graph/finalize_idle_worker` C symbols. |
| B. Fork-built **rocprofv3** wrapping the server, `-P/--collection-period` | CSV/JSON/pftrace/rocpd, `--stats` | yes if `--hip-trace` covers capture time | options `--kernel-trace --hip-trace --memory-copy-trace --stats --output-format`, `-P start:dur:repeat`, `--collection-period-unit msec` exist [F: `source/bin/rocprofv3.py`]. Output is written at process exit [F: inference-framework doc] -> server must shut down gracefully. |
| C. **rocprofv3 --attach PID** (stock or fork) | as B, for a time window | **no** for graphs captured before attach [I] | target must be started with `ROCP_TOOL_ATTACH=1`, ptrace allowed [F: `using-rocprofv3-process-attachment.rst`]. Our `server_launch.py` already passes `--cap-add SYS_PTRACE --security-opt seccomp=unconfined` [F]. Whether the image's ROCm 7.2.0 rocprofv3 has `--attach` is unchecked. |

Recommendation [I]: **A** for graph-attributed decode traces (it is the only path the team validated on SGLang); **C** with the image's own stock rocprofv3 as the cheap first try if kernel names + durations are enough.

### 4.2 Build (option A/B) [F commands from upstream README + experiments README; not executed]

Build inside a throwaway container from the 1106 image so headers/libs match ROCm 7.2.0; write the build to a yihou dir on node-local disk (crusoe: `/mnt/m2m_nobackup/yihou/...`, not NFS - see memory "crusoe docker runtime dirs not on NFS").

```bash
# on this host (network clone; small):
cd /home/yihou/dev/git.16-19/infera.muse/yihou/muse_agentx_opt_20261027/research/calib_profiler/repo
git sparse-checkout init --cone
git sparse-checkout set projects/rocprofiler-sdk shared/sha256
git checkout diag/rocm72-capture-node-map-v35        # or fix/rocprofiler-rocm72-scoped-strict (minimal 7.2 set; 10 us catch-up)
# copy repo/ to the node's yihou dir, then inside a container of the 1106 image:
cmake -B /yihou/sdk-build -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_INSTALL_PREFIX=/yihou/sdk-install \
      -DROCPROFILER_BUILD_TESTS=ON -DGPU_TARGETS=gfx950 projects/rocprofiler-sdk
cmake --build /yihou/sdk-build --target all --parallel $(nproc)
python3 projects/rocprofiler-sdk/tests/graph-attribution-experiments/validate_scoped_strict.py /yihou/sdk-build/lib/librocprofiler-sdk.so   # no GPU needed
```
Unknowns: whether the 1106 image has cmake >= 3.24 [F: `cmake_minimum_required(VERSION 3.24)`] and all `-devel` headers; whether `-DGPU_TARGETS` is the right variable for the test apps. SDK version reported should be 1.5.0 [F: `VERSION`].

### 4.3 Runtime env (team recipe, `launch_sglang.sh` / `run_native.sh`) [F]

```bash
export LD_LIBRARY_PATH=$SDK_BUILD/lib:$LD_LIBRARY_PATH
export ROCPROFILER_REGISTER_LIBRARY=$SDK_BUILD/lib/librocprofiler-sdk.so
export ROCP_TOOL_LIBRARIES=$SDK_BUILD/lib/rocprofiler-sdk/librocprofiler-sdk-json-tool.so
export ROCPROFILER_TOOL_CONTEXTS=HIP_API_BUFFERED,KERNEL_DISPATCH_BUFFERED,CODE_OBJECT
export ROCPROFILER_QUEUE_INTERPOSITION=0
export ROCPROFILER_EXPERIMENTAL_GRAPH_COPYBUFFER=1         # private ROCclr ABI; revalidate per ROCm build
# ROCm 7.2.0 workarounds (not in the team's ROCm 10 SGLang recipe; [I] relevant for our 7.2.0 image):
export ROCPROFILER_EXPERIMENTAL_WRAP_SAFE_INTERCEPT_BATCH=1
# ROCPROFILER_EXPERIMENTAL_SINGLE_SIGNAL_WAIT=1             # only matters for the queue-interposition path [F: in queue_interposition.cpp]
```
The README warns that some ROCm Python wheels load their own SDK by absolute path; verify via the server log / `/proc/<pid>/maps` that the custom `librocprofiler-sdk.so` is the one loaded [F README].

Output: one `trace-<pid>.json` per process (`ROCPROFILER_TOOL_OUTPUT_FILE`, set per PID by `sitecustomize.py`), `buffer_records.kernel_dispatch` with `graph_exec_id/graph_node_id/capture_correlation_id`; `inspect_graph.py` / `summarize_run.py` turn it into per-rank CSV [F].

## 5. Recipe

### 5a. Validate the profiler on a tiny program (one GPU, minutes)

1. `validate_scoped_strict.py` on the built `.so` (no GPU) - must print `"status": "PASS"` [F script].
2. Team native probes on gfx950: `SDK_BUILD=... GRAPH_TEST_ARCH=gfx950 bash .../run_native.sh /yihou/out/native` (module probe: 18 exact capture ids; memcpy probe: 30 dispatches) [F README]. These were validated by the team on ROCm 10; on ROCm 7.2.0 a pass is new evidence.
3. Torch CUDA-graph probe (ours, to write in a yihou dir): capture a graph of ~5 ops (matmul, add, `copy_`) on one stream, replay 20x, `torch.cuda.synchronize()`. Run under
   `rocprofv3 --kernel-trace --hip-trace --memory-copy-trace --stats --output-format csv json -d /yihou/out/torchprobe -- python3 probe.py`
   (fork rocprofv3 from `$SDK_BUILD/bin`, with `ROCPROFILER_REGISTER_LIBRARY` set as in `run_memset_native.sh`). Pass criteria: kernel rows for the replays = 20 x kernel nodes; every replayed dispatch has non-zero `graph_exec_id` and `capture_correlation_id`; sum of kernel durations within a replay <= hipEvent-measured replay time; `dropped_records == 0`. Repeat with stock `/opt/rocm/bin/rocprofv3` to see the difference (stock: no capture ids).

### 5b. Profile the running Muse TP4 server for ~20 decode steps (not executed; design)

Constraints: 4 scheduler ranks, CUDA graphs, DFlash; torch-profiler windows of 50+ steps hung, 20-30 worked (cause not known).

1. Start the server via `server_launch.py` with the build dir bind-mounted read-only, the section-4.3 env, and `PYTHONPATH=<experiment dir>` so `sitecustomize.py` loads. Use a **modified** sitecustomize (ours, yihou dir) that:
   - calls `json_tool_pause_until_graph()` at import (no tracing during weight load) [F team behaviour];
   - calls `json_tool_start()` at the first `CUDAGraph.capture_begin` (so capture is traced and attribution plans are built) [F team behaviour];
   - **new** [I]: after capture finishes (e.g. a trigger file `capture_done`, or N seconds after the last `capture_end`), calls `json_tool_stop()`; then a watcher thread waits for `start_window` file, calls `json_tool_start()`, sleeps `WINDOW_MS`, calls `json_tool_stop()`, then `json_tool_finalize_idle_worker()` and writes `finalized-<pid>`.
   - Open risk: whether capture plans survive a stop/start of the contexts (the team never stopped between capture and requests) - validate in 5a first with the torch probe (capture, stop, start, replay).
2. `WINDOW_MS` = 20 x the decode step time measured from the server log at the same load [A once measured]. Drive load with our client, wait for steady running requests (as in `load_profile_trigger.sh`), then `touch start_window` on the trace dir.
3. Wait for 4 `finalized-<pid>` files, then summarize per rank. Stop the server with the normal drain path; do not reuse the trace dir (`finalize` marker) [F README].
4. Alternative cheap path (option C): start the server with `ROCP_TOOL_ATTACH=1`; then inside the container `rocprofv3 --attach <scheduler_pid> --attach-duration-msec <WINDOW_MS> --kernel-trace --output-format csv -d /yihou/...` per rank (or parent PID with `--attach-children`). Kernel names + durations only.

## 6. Risks

- Strict window = `ROCP_FATAL` on a violation -> aborts a scheduler rank -> TP group hangs/dies [F code; consequence I]. Do not enable the strict window on the first server run.
- copyBuffer/fillBuffer decoders use a **private ROCclr ABI**, validated on ROCm 10 only; ROCm 7.2.0 layout may differ -> attribution refused (0), by design not mislabeled [F README].
- WRAP_SAFE path is `ROCP_FATAL` on an invalid queue or packet count [F code].
- json-tool buffers all records in memory until finalize; trace size for the team's 512k-token run is not recorded. Long windows may grow memory [I].
- Diagnostic env vars write to stderr per event; README says keep them off for performance captures [F].
- Dispatch-time spin (up to 100 us) is per selected dispatch -> overhead inside the window [F].
- Only the queue-intercept path (`ROCPROFILER_QUEUE_INTERPOSITION=0`) was validated on SGLang [F].
- Team's SGLang runs used `--max-running-requests 1` (single request) [F launch_sglang.sh]; our C112-style load under profiling is unvalidated.

## 7. Open questions

1. Which rocprofiler-sdk / rocprofv3 version ships in the 1106 image, and does it support `--attach`? Check: `ls /opt/rocm/lib/librocprofiler-sdk.so*; /opt/rocm/bin/rocprofv3 --help | grep -i attach` inside the image.
2. Does the fork build against the image's ROCm 7.2.0 headers (cmake >= 3.24, devel packages)?
3. Do capture-attribution plans survive `json_tool_stop()`/`json_tool_start()` between capture and the window?
4. Is "calibrated" in the user's sense this fork's timestamp audit, or an internal calibration doc not in the repo? Ask Jiejing Zhang (author; works on this host).
5. Which branch does the team consider current? v35 (newest diag) vs `fix/rocprofiler-rocm72-scoped-strict` (minimal) vs PR #14887 branch.
