# Notes — gotchas, traps, and what is still open

Written as **what / why / how / context**. The first three would each have produced a
plausible-looking but wrong result.

---

## 1. `record_function` spans are reported as CUDA rows and double-count the kernels beneath them

**What.** `torch.profiler`'s `key_averages()` returns SGLang's `record_function` annotations
(`step[TARGET_VERIFY bs=32]`, `draft`, `draft_extend`) as rows with
`device_type == DeviceType.CUDA` and non-zero device time. `step[TARGET_VERIFY bs=32]` reads
**838.93 ms** over the 10-iteration window.

**Why it matters.** Sort by self device time without classifying and that annotation ranks **first**,
above every real kernel — while *containing* them. The whole top-10 becomes incoherent: the column
sums past 100 % and the headline is an annotation, not a kernel.

**How it is handled.** Rows are classified via `FunctionEvent.is_user_annotation` into
`user_annotation` / `device_kernel` / `cpu_side`. The ranking uses
`top_by_self_device_time_device_kernels_only`; annotations are reported **alongside, never
interleaved**. `top_ops_rank_0_yihou.json` carries a `how_to_read` field stating this.

**Context.** Found during smoke runs, not at C=256 — which is the only reason it did not reach the
report. There is nothing in the output of the naive version that looks wrong.

---

## 2. DeviceTimer category names change between CUDA-graph modes — a stage appears to vanish

**What.**

| mode | target categories | draft categories |
|---|---|---|
| graph ON | `target_verify` 90.1 % | `eagle_draft` 6.0 %, `eagle_draft_extend` 4.0 % |
| graph OFF | `target_verify` 90.3 % | `decode` 6.8 %, `extend` 2.9 % |

**Why it matters.** `decode` — the category you would naively expect to be the main one — appears
**only** in graph-OFF, and it is the **draft** runner's eager forwards, not the target's. Reading the
two modes side by side, it looks as though `eagle_draft` disappeared and a new `decode` stage
appeared. It is the same work under different labels, because the eager runner and the graph runners
annotate differently.

**How it is verified rather than assumed.** Per-runner category sets are disjoint in both modes, and
`sum(by_category) == sum(by_runner)` exactly, so there is no double counting. The `decode` count is
`n = 11072 = 4 × 2768` — exactly 4 draft forwards per iteration, which independently matches the
launch counts of the `draft`-stage `main_kernel` signatures (4.0/iter).

**Context.** The design doc originally listed `decode` as an expected graph-ON category. That was an
error from reading `device_timer_ctx` call sites without checking which runner owns them; corrected
in `spec/design.md`.

---

## 3. A generic kernel name can hide several unrelated kernels

**What.** `main_kernel` is **four** distinct kernels sharing one compiler-generated symbol. TileLang
compiles a function literally named `main` (`kernels/ops/attention/dsa/tilelang_kernel.py:1314`
ends `return main`), so every TileLang kernel in the build lands on the same name.

**Why it matters.** It was 24.4 % — rank 1 — and *unattributable*. Grouping it wrongly, or grouping
it at all before identification, would have put the headline on the wrong category. With it
identified as DSA sparse attention, **attention (31.0 %) overtakes communication (23.6 %)** as the
largest category; before identification the report said communication was largest.

**How it was resolved.** `--profile-with-stack` → `python_function` events form a timestamped Python
call tree → match each `hipLaunchKernel` by `ts` to its innermost enclosing frame. All 332 launches
resolve to `tilelang_cython_wrapper.CythonKernelWrapper.forward`; walking up, 316 →
`dsa_backend.py:2220 forward_extend` (target), 16 → `:2570 forward_decode` (draft).

**Two approaches that could not have worked, and why:**
- *cpu_op parent lookup* — these launches have **no aten ancestor**; the TileLang Cython wrapper
  dispatches directly.
- *`--profile-record-shapes`* — same reason. `record_shapes` is an **aten-dispatcher feature**; it
  cannot annotate a kernel that never enters the dispatcher. It was already enabled in the graph-OFF
  run (99.4 % of `cpu_op` events carry `Input Dims`, so it was demonstrably live) and `main_kernel`
  still had no shape args. Re-running with it would have reproduced a known null.

**Audit performed after the fact:** of the top 12 kernel names by total time, **four** are aggregates
— `main_kernel` (4 signatures), `ncclDevKernel_Generic_1` (3),
`aiter::dynamic_per_group_scaled_quant_kernel` (2), `ck::kernel_moe_gemm_2lds` (2). Only
`main_kernel`'s aggregation was **category-ambiguous**; the others resolve within one functional
category, so the grouping is unaffected. Decomposing #2 corroborated the structure: its dominant
branch is **79.0 launches/iter**, the same per-layer rhythm as `main_kernel`'s dominant branch
(78 target layers + 1 draft_extend layer) — one TP all-reduce after every transformer layer.

---

## 4. Contention corrupts profiler event counts — and it was our own leaked process

**What.** An early probe showed per-kernel event counts varying with
`DEBUG_CLR_GRAPH_PACKET_CAPTURE` (20/20 vs 18-19/20 vs 16-17/20), suggesting the flag dropped
events. On a quiet GPU the same probe gave **15/15 runs identical**. The variation was contention.

**Why it matters.** The apparent gradient was coherent enough to look like a finding, and CLR source
even contains a plausible mechanism (`activity.cpp:108` truncates to
`min(timestamps.size(), kernel_names.size())`, silently). Source finding + timing observation
propping each other up is exactly how a wrong conclusion gets shipped. **The mechanism exists in
source; it was not observed to fire.**

**How.** Never take profiler counts while anything else uses the GPUs. Also: **a `docker exec` whose
outer shell you kill on a timeout leaves the in-container process running.** Two such orphans pinned
a GPU for 49 minutes, and the contention was initially blamed on a colleague's job. After any
timeout, check the **whole process tree**, not just the leaf.

---

## 5. `--enable-profile-cuda-graph` crashes under `docker exec -w /`

**What.** `decode_cuda_graph_runner._post_process_after_profile()` unconditionally writes
`cuda_graph_runner_memory_usage.pickle` into the **CWD**. With `-w /` and a non-root user that is
EPERM, and capture dies outright (`evidence/aborted/aborted_smoke_graphon_v3_capture_cwd_eperm_yihou/`).

**How.** `run_profile_yihou.sh` `cd`s into a writable per-iteration directory when capture profiling
is enabled.

**Context.** Host-environment-specific, same family as the four container fixes. Also note: capture
profiling records kernels **during graph capture, where they are not executed** — its "timings" are
not performance data, only a kernel identity/shape inventory.

---

## 6. Other operational gotchas

- **`_ProfilerTorch` cannot be reused for a rank subset.** Its `stop()` ends with
  `torch.distributed.barrier(cpu_group)`; with `--profile-ranks 0` only rank 0 would reach it and the
  run deadlocks. The bench calls `torch.profiler.profile` directly and reproduces the filename
  convention instead.
- **`DeviceTimer.wrap` asserts non-re-entrancy**, so target and draft get separate instances.
- **`SGLANG_PROFILE_V2=true` is a dead end here**: `ProfileManager.manual_start/manual_stop` raise
  `NotImplementedError` and `configure()` asserts `profile_by_stage=True`. Leave it at its default.
- **The profiler stop must latch.** Without it the iteration after the window re-enters stop and
  torch raises "Can't disable Kineto profiler when it's not running"
  (`evidence/aborted/aborted_smoke_graphon_double_stop_yihou/`).
- **A guard that mirrors Python-side validation in shell will drift.** `run_profile_yihou.sh`
  rejected `MAX_STEPS=0` (which means "full OSL") and silently killed a chain for 25 minutes.
  Fixed, and the runner now writes `launch_started.json` before launching so "no news" is no longer
  ambiguous.
- **`with_stack` costs ~40× trace size per iteration.** 16.7 MB for 2 iterations vs 7.96 MB for 10
  without. Use it only to answer an identity question, with a 2-iteration window.

---

## 7. Open questions — measured limits, not speculation

1. **The 4.9 % gap** between summed kernel self time (887.8 ms) and the DeviceTimer window total
   (933.2 ms). Time inside the timer brackets occupied by no kernel. **Not measured.** A gap analysis
   on the trace timeline would settle it.
2. **The +42.3 % graph-OFF penalty** is entirely non-kernel time (48.62 ms/iter). Host runtime calls
   go up 26×, and 48.62 ms over 2432 launches is ~20 µs each — but that is an **order-of-magnitude
   reference, not a measurement of where the time goes.**
3. **Communication is 23.6 %, and EP1 beat EP8 at 9 of 10 concurrencies** in the throughput packup.
   These are suggestive together. **No EP1 profiling was run**, so the profile cannot be used to
   explain the throughput result. That is the obvious next experiment.
4. **Collective mix differs between graph modes** — `aiter::allgather_vec` is n=20/window graph-ON vs
   n=890 graph-OFF; `ncclDevKernel_Generic_1` n=791 vs n=840. **Why is unknown.** It does not affect
   the graph-ON ranking, which is the configuration of record.
5. **`idle` never fired** in any of the four runs. Needs unequal DP-shard batch sizes to trigger.
   Unexplained, not chased.
6. **Only rank 0 was traced.** Per-rank kernel distribution is unverified; with DP attention the
   ranks are symmetric by construction, but that was not measured.
7. **`DEBUG_CLR_GRAPH_PACKET_CAPTURE` is plumbed but was never exercised** — attribution already
   works at stock defaults. **The flag itself is uncharacterised.** `research/rocm_profiling_env.md`
   §2 is explicitly *provisional*: whether packet capture was ever in force during the probes depends
   on `max_streams_`, which is set by graph **topology** (`hip_graph_internal.cpp:185-207`), not by
   the queues flag, and was never read off the running system — the `LOG_DEBUG` line that reports it
   never emitted despite the string being present in the loaded binary. **Do not set this flag.**
