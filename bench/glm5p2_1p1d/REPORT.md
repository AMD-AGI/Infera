# GLM-5.2 Agentic 1P1D experiment report

Status: complete  
Workspace revision: `8f04a68cc5c33aa0f8dae69615aad09a8ee36fd6`

## Goal and acceptance criteria

Deploy `GLM-5.2-MXFP4` as one prefill node plus one decode node through
Infera/SGLang, then establish:

1. Correctness: one prefill and one decode worker remain registered; normal chat
   is coherent; the GLM tool parser emits a valid OpenAI function call.
2. Transport: the container sees an RDMA device, Mooncake does not report TCP
   fallback or a null GID, and a cross-node bandwidth probe succeeds.
3. Feature state: prefill and decode resolve to the intended DP-attention shape;
   EAGLE produces acceptance-length evidence without DP deadlock.
4. Concurrency: fixed-shape sweeps at 8, 16, 32, 64 and 128 complete with zero
   failed requests, worker loss, scheduler traceback or retraction.
5. Agentic stability: AIPerf `inferencex-agentx-mvp`, concurrency 14, 3600
   seconds, pinned `semianalysis_cc_traces_weka_062126`, with zero failed
   requests.
6. Optimization: each safe PR group is measured against the same baseline
   rather than enabling all unmerged changes at once.

## Execution plan

### Phase A — environment and image

- Select two idle nodes and record GPU/container usage.
- Probe peer-memory, ODP, GID and in-container provider capability.
- Build the checked-out `deploy/docker/Dockerfile.sglang`.
- Repeat registration-mode and cross-node fabric probes with that exact image.

### Phase B — functional baseline

- Start etcd.
- Start prefill and decode legs over the safe RDMA mode.
- Start the Infera router after both workers are healthy.
- Run chat, tool-call, worker, log and feature checks.

### Phase C — load baseline

- Run fixed 8K-input/1K-output concurrency sweeps.
- Run the one-hour AgentX workload.
- Preserve per-request artifacts and engine logs.

### Phase D — optimization A/B

- Classify every PR in `glm5.2_single_opt.md` as correctness, compatibility or
  performance work.
- Build the smallest dependency-complete image groups.
- Re-run smoke and representative synthetic arms after each group.
- Run the full AgentX test only on groups that pass the shorter gates.

## Environment findings

Node scan at 2026-09-02 17:13 UTC+8:

- `crsuse2-m2m-136` and `crsuse2-m2m-140` were initially idle and used for
  image builds and the first fabric probe.
- `135`, `138` and `139` had active SGLang/VLLM allocations and were rejected.
- During the first fabric probe, unrelated `glm52_pd_ab` workloads started on
  both 136 and 140 and occupied about 90% VRAM on all GPUs. The experiment
  stopped only its own hanging preflight container and did not touch those
  workloads.
- `crsuse2-m2m-137` and `crsuse2-m2m-138` had become fully idle by the rescan,
  with about 89 GB free on each root disk. They are the final prefill and decode
  nodes.
- `142` was also idle at the rescan, and remains the spare node.
- `141` was also GPU-idle, but only about 21 GB remained on its root disk; it
  was rejected to avoid deleting unrelated image caches before the baseline
  build.
- Both selected nodes expose eight active `ionic_0..7` ports at 400 Gb/s and
  one active `mlx5_0` port at 200 Gb/s.
- Both use `amdgpu-dkms 1:6.14.14.30100100-2212064.24.04`; host `rocm-core` is
  7.0.1.
- The model is present at
  `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`.

## Registration-mode findings

The initial hardware probe used the existing image only as a diagnostic vehicle;
the result will be repeated with the newly built experiment image.

The initially selected nodes reported the same capabilities; the final pair
will be probed again with the final image:

- 8 × gfx950 GPUs.
- `CONFIG_PCI_P2PDMA=y`.
- No peer-memory module.
- Mooncake was built with `ibv_reg_dmabuf_mr`.
- Ionic rails support dma-buf but do not expose ODP.
- `mlx5_0` supports both dma-buf and ODP.

Consequences:

- Bare `ibv_reg_mr` over all rails is blocked because peer-memory is absent.
- The safe no-pin mode is dma-buf on `mlx5_0`, with
  `MC_GID_INDEX=3`, `MC_MS_FILTERS=mlx5_0`,
  `MC_MS_AUTO_DISC=0`, and `MOONCAKE_DISABLE_HIP_DMABUF=0`.
- This safe mode caps the available KV-transfer link at 200 Gb/s.
- Multi-rail ionic is viable only as capped mode C. Registration pins and
  duplicates the KV pool, so it requires a GLM-5.2-specific token cap before it
  is safe to launch. It will be considered after the baseline.

## Image findings

The repository Dockerfile currently pins SGLang `v0.5.17-rocm720-mi35x` and
applies:

- a rebuilt Mooncake with cross-host routing and dma-buf support;
- four GLM-5.2 DSA patches needed by PD + DP-attention + EAGLE;
- the chunked-prefill Mooncake early-send correctness fix;
- ROCm HiCache host-allocation fixes;
- the current Infera source and Rust router.

Initial baseline image build:

- Tag: `infera/engine-sglang:glm52-1p1d-8f04a68`
- Build hosts: `crsuse2-m2m-136`, `crsuse2-m2m-140`
- State: succeeded on both.

The first full fabric probe completed its 9×9 `ib_write_bw` matrix but failed
when the Mooncake wrapper decoded a child process's non-UTF-8 logging with
strict UTF-8. Rank 0 raised `UnicodeDecodeError`; rank 1 then waited for a peer.
The run was stopped and no bandwidth result is claimed from it.

The next run exposed a second probe assumption: it read GID 1 from `ionic_0`
and reused it for every device, including `mlx5_0`, which requires GID 3.
The preflight now accepts an explicit device filter, derives the GID after
filtering, and passes the same filter into Mooncake's GPU tests. Seven focused
unit tests pass for output decoding, device filtering and degraded-link
detection.

Final experiment image:

- Tag: `infera/engine-sglang:glm52-1p1d-8f04a68-pf2`
- Build hosts: `crsuse2-m2m-137`, `crsuse2-m2m-138`
- State: succeeded on both.

Layered A/B images:

- `glm52-1p1d-8f04a68-corr-1` applies only
  [SGLang #37133](https://github.com/sgl-project/sglang/pull/37133), retaining
  GLM-5.2's MoE correction bias in fp32 at parameter load and the Aiter
  boundary. Upstream measured that bf16 collapses 238 bias values to 8 and
  changes the selected top-8 expert set for 98.5% of tokens. This is the only
  active semantic correction in the added layers.
- `glm52-1p1d-8f04a68-runtime-1` adds
  [SGLang #37118](https://github.com/sgl-project/sglang/pull/37118), exposing
  the plain-torch DSA head-gate graph helpers on HIP. Prefill graphs are
  auto-disabled in this PD configuration, but the decode leg captures full
  EAGLE target-verify and draft graphs, so this merged graph-compatibility
  change is retained in the accepted runtime layer.
- `glm52-1p1d-8f04a68-opt-1` additionally carries the dependency-free
  `fp8_mqa_logits` offset-gate hunk from
  [Aiter #5121](https://github.com/ROCm/aiter/pull/5121). Source verification
  showed that the pinned older Aiter does not contain the later `BLOCK_M=2`
  abort and already falls back to global load/store for large buffers.
  Near-limit runtime testing without this hunk also passed, so `opt-1` is kept
  only as the historical experimental arm and is no longer the default.
- All layers use the locally built `pf2` image as their parent. Build-time
  `git apply --check`, fail-closed source-anchor replacement and `compileall`
  passed on both nodes.
- [SGLang #37124](https://github.com/sgl-project/sglang/pull/37124),
  [#37134](https://github.com/sgl-project/sglang/pull/37134), and
  [Aiter #5122](https://github.com/ROCm/aiter/pull/5122) require newer,
  dependency-complete kernel and caller layouts and were not partially
  applied. The current Infera startup barrier covers only the co-scheduling
  part of [SGLang #37152](https://github.com/sgl-project/sglang/pull/37152);
  the rest was not backported. HiCache remains disabled.
  [SGLang #37130](https://github.com/sgl-project/sglang/pull/37130) has no
  standalone upstream diff, and
  [#37117](https://github.com/sgl-project/sglang/pull/37117) is closed without
  code.

Corrected targeted fabric run:

- Both directions: `ib_write_bw` 1/1 links succeeded over `mlx5_0`, GID 3.
- 138 → 137 bandwidth: 21.78 GB/s.
- 137 → 138 bandwidth: 6.56 GB/s.
- Mooncake host-memory RDMA: 5.12–5.52 GB/s; TCP control: 0.08 GB/s.
- Mooncake real VRAM transfer: all 8 GPUs passed in both directions with byte
  verification, 16.72–18.01 GB/s.
- Artifact:
  `results/preflight_fabric_targeted_20260902_102312/infera_preflight_report.html`.

## Results

### Baseline deployment

Launch completed in 8 minutes 44 seconds:

- Prefill: `10.245.153.247:30001`, TP8/EP8, resolved DP size 1 and
  DP-attention false.
- Decode: `10.245.157.237:30002`, TP8/EP8/DP8, resolved DP size 8 and
  DP-attention true.
- Router: `10.245.153.247:8000`, `kv-aware`, two active workers, KV block size
  64.
- Every Mooncake worker installed the RDMA transport on `mlx5_0` with GID 3.
  Decode logs contain RDMA-ready acknowledgements from real routed requests.

Smoke test:

- Normal chat returned the correct sentence `The capital of France is Paris.`
- Required tool choice returned a valid `get_weather({"city":"Paris"})` OpenAI
  function call.
- Zero `MC_FORCE_TCP`, `GID is NULL`, `KVTransferError`, DSA row mismatch,
  Python traceback or GPU memory-fault log signatures.
- EAGLE acceptance length: 3 samples, median 2.98, range 2.67–3.16.
- Artifact: `results/smoke_20260902_104135/`.

### Baseline synthetic load

8K input / 1K output, concurrency 8, 80 measured requests:

- 80/80 successful in 131.63 seconds.
- Output throughput: 622.35 tokens/s; total throughput: 5,601.19 tokens/s.
- Median TTFT: 498.61 ms; P99 TTFT: 9,369.76 ms.
- Median TPOT: 11.08 ms; P99 TPOT: 13.38 ms.
- Median ITL: 27.56 ms; P99 ITL: 32.74 ms.
- No worker loss after the arm.
- Artifact:
  `results/synthetic_baseline_c8_retry_20260902_104259/`.

The remaining baseline arms all completed:

- C16: 160/160 successful in 148.83 s; 1,100.82 output tok/s; median/P99
  TTFT 512.53/12,901.30 ms; median/P99 TPOT 12.32/15.07 ms; cache hit 49.7%.
- C32: 320/320 successful in 157.74 s; 2,077.34 output tok/s; median/P99
  TTFT 517.07/3,266.43 ms; median/P99 TPOT 14.12/17.43 ms; cache hit 49.7%.
- C64: 640/640 successful in 215.56 s; 3,040.29 output tok/s; median/P99
  TTFT 1,615.98/11,423.11 ms; median/P99 TPOT 16.41/20.49 ms; cache hit 50.1%.
- C128: 1,280/1,280 successful in 489.95 s; 2,675.19 output tok/s; median/P99
  TTFT 30,814.53/43,133.58 ms; median/P99 TPOT 16.17/19.84 ms; cache hit 0.5%.

Across the five arms, 2,480/2,480 measured requests succeeded. The post-sweep
smoke check still found two active workers, valid chat and tool-call outputs,
no Mooncake/DSA/GPU-fault signature, and no nonzero `#retracted-req`. Decode
logs contained 6,074 EAGLE acceptance samples (median 2.50, range 1.57–3.81).
C64 is the throughput knee: C128 adds severe prefill queueing, drops output
throughput by 12.0%, and raises median TTFT to 30.8 seconds.

- Sweep artifact:
  `results/synthetic_baseline_sweep_20260902_104653/`.
- Post-sweep artifact:
  `results/smoke_post_sweep_20260902_110656/`.

### Optimization gate

The optimized image launched successfully after a clean stop and passed the
same worker, chat, tool-call, Mooncake, DP-attention and MTP checks. The first
in-place restart hit a transient free-port race in Infera's automatically
allocated KV-event ZMQ endpoint (`Address already in use`). No listener
remained after the process exited; a clean stop/retry succeeded. `launch.sh`
now removes the old router and both legs together and waits five seconds before
starting the new deployment.

The current [SGLang #37134](https://github.com/sgl-project/sglang/pull/37134)
sampling fix is not safe to transplant alone into the v0.5.17 tree: it adds a
new Triton rejection sampler plus argument-resolution changes and remains
unmerged upstream. Without it, ROCm EAGLE verify silently uses argmax for
non-greedy requests. The benchmark scripts therefore pin temperature 0 so their
requested sampling contract matches the executed path.

- Smoke artifact:
  `results/smoke_optimized_smoke_20260902_112320/`.

Optimized synthetic results:

- C16 cold-cache gate: 160/160 successful in 146.10 s; 1,121.44 output tok/s.
  This is not a direct baseline comparison because the baseline C16 inherited
  half of its prompt prefixes from the prior C8 arm.
- C32 first run: 320/320 successful in 174.91 s; 1,873.43 output tok/s.
- C64: 640/640 successful in 217.02 s; 3,019.82 output tok/s; median/P99
  TTFT 1,613.36/10,143.54 ms; median/P99 TPOT 16.88/20.37 ms.
- C128: 1,280/1,280 successful in 489.07 s; 2,680.02 output tok/s; median/P99
  TTFT 30,328.32/43,156.66 ms; median/P99 TPOT 16.50/20.38 ms.
- A cache-controlled C16→C32 repeat produced 320/320 successful C32 requests
  in 163.04 s at 2,009.85 output tok/s, with a 49.7% cache hit rate.

All four first-sweep arms wrote complete metrics before the wrapper returned
status 2: `bench.sh` was edited live after the process had parsed part of the
case branch, and the old shell encountered the changed text only after C128.
The final file passes `bash -n`, and the subsequent C16→C32 invocation exited
zero; this was not an engine or request failure.

At the two saturation points with matching cache behavior, output throughput
changed by -0.7% at C64 and +0.2% at C128. The C32 repeat was -3.2%. Median
TPOT rose by roughly 1.6–2.9%, consistent in direction with the fp32 routing
cost. The combined image is therefore a compatibility/correctness validation,
not an 8K-context throughput win. The follow-up isolation below narrows the
production composition and removes the unnecessary Aiter hunk.

After 2,880 measured optimized requests, the post-sweep smoke still passed:
two workers remained active, all searched fatal signatures were zero, and
7,079 EAGLE samples had median acceptance length 2.49 (range 1.77–4.00).

- Sweep artifact:
  `results/synthetic_optimized_sweep_20260902_112416/`.
- Controlled-repeat artifact:
  `results/synthetic_optimized_repeat_20260902_114435/`.
- Post-sweep artifact:
  `results/smoke_optimized_post_sweep_20260902_115107/`.

### Follow-up patch isolation

The final PR/source mapping showed that the combined `opt-1` layer was broader
than required by the pinned runtime, so #37133-only and #37118+#37133 images
were built independently on both nodes.

The #37133-only `corr-1` image passed worker, chat, tool-call, Mooncake,
DP-attention and MTP smoke checks. Its cache-controlled C32→C64 run completed
960/960 requests:

- C32 cold cache: 1,946.70 output tok/s; median/P99 TPOT 13.87/17.25 ms.
- C64 with the same 50.1% inherited cache state as the original sweep:
  2,905.81 output tok/s; median/P99 TPOT 16.37/20.36 ms.

The `runtime-1` image, adding #37118 but not Aiter #5121, passed the same smoke
checks and completed another 960/960 requests:

- C32 cold cache: 1,831.03 output tok/s; median/P99 TPOT 14.45/18.31 ms.
- C64 at 50.1% cache hit: 2,820.84 output tok/s; median/P99 TPOT
  16.75/20.61 ms.

These follow-up arms ran hours after the original A/B and showed higher
prefill/TTFT variance; they do not demonstrate a #37118 throughput gain.
Decode TPOT remained in the same range, and all requests succeeded. #37118 is
therefore retained for the active decode graph surface, not advertised as a
performance optimization.

Without the Aiter #5121 hunk, `runtime-1` served a 70,017-token request in
4.78 seconds and a 250,016-token request in 24.16 seconds. The reproducible
post-AgentX run of `long_context.sh` repeated the 250,016-token case in
24.83 seconds. This validates the pinned Aiter fallback near the configured
262,144-token limit and removes #5121 from the production requirement.

- #37133-only smoke:
  `results/smoke_corr_only_smoke_20260902_2244/`.
- #37133-only C32→C64:
  `results/synthetic_corr_only_c32_c64_20260902_2251/`.
- Runtime smoke:
  `results/smoke_runtime_smoke_20260902_2314/`.
- Runtime C32→C64:
  `results/synthetic_runtime_c32_c64_20260902_2315/`.
- Runtime final smoke:
  `results/smoke_runtime_final_20260902_2333/`.
- Reproducible near-limit request:
  `results/long_context_runtime_post_agentx_20260903_0003/`.

### AgentX-MVP

AIPerf 0.12 accepted the `inferencex-agentx-mvp` scenario locks, forced
streaming, injected `ignore_eos=true`, selected first-turn-prefix cache busting
and set the scenario's 10-second global idle-gap cap. From the pinned 393-trace
dataset, 220 traces fit the 262,144-token context limit and reconstructed into
1,393 conversations.

The snapshot warmup completed 16/16 requests with zero errors or cancellation
in 250.86 seconds. Its global idle cap skipped 37,474.64 seconds of recorded
idle time without changing request order.

The C14, 3,600-second profiling window sent 785 requests. All 785 record
exports are present and report `was_cancelled=false`, no error,
no context-overflow skip and no OSL mismatch:

- 68,984,000 input and 633,562 output tokens.
- 0.2173 requests/s and 175.35 output tokens/s.
- Median/P99 TTFT: 764.36/4,868.80 ms.
- Median/P99 request latency: 3,674.62/64,389.31 ms; maximum 241.74 s.
- Median/P99 ITL: 11.79/17.79 ms.
- Reported prompt-cache read ratio: 95.25%.
- Average measured request concurrency was 1.94 (maximum 6), below the C14
  lane target because this scenario preserves each trajectory's recorded
  think time and join dependencies.

AIPerf 0.12 did not finalize the run automatically. The final response record
was exported 15 seconds after the sending cutoff and there were no remaining
benchmark TCP connections or engine requests, but one future DAG join remained
tracked with an outstanding delayed child. With `--benchmark-grace-period
inf`, that non-wire branch kept the completion event pending indefinitely;
even the 1,800-second HTTP timeout was irrelevant. After 59 minutes of
quiescence, the controller was stopped gracefully to force aggregate export.
It exited zero and summarized all 785 records, but correctly stamped
`submission_valid=false` with reason `run_cancelled`.

This is an AIPerf control-plane finalization issue, not a failed model request:
cleanup reported one future join and no branch errors, while both engines had
zero running or queued requests. `bench.sh` now defaults to a finite 600-second
grace period so delayed branch bookkeeping is bounded.

The 900-second minimum-duration control confirmed that behavior:

- 127/127 requests completed; zero cancelled requests, request errors,
  context-overflow skips or OSL mismatches.
- Both wire requests present at the sending cutoff completed within 21 seconds.
- The same single future join remained, so the runner used the full 600-second
  grace, found `in_flight=0`, and cleaned up without cancelling a request.
- AIPerf exited zero with `was_cancelled=false` and
  `submission_valid=true`.
- 12,885,686 input and 132,630 output tokens; 88.42 output tokens/s.
- Median/P99 TTFT: 853.27/5,221.58 ms.
- Median/P99 request latency: 5,942.53/85,609.32 ms.
- Reported prompt-cache read ratio: 95.64%.

The post-run smoke passed with both workers active, valid chat and tool-call
responses, zero searched Mooncake/DSA/GPU-fault signatures, and 12,442 EAGLE
samples with median acceptance length 2.55 (range 1.68–4.00).
The final smoke after the control also passed; the cumulative decode log then
contained 13,532 EAGLE samples with median acceptance length 2.56.

- Full-duration artifact:
  `results/agentic_optimized_agentx_20260902_115144/`.
- Scenario-valid control artifact:
  `results/agentic_optimized_agentx_control_20260902_140027/`.
- Post-run smoke artifact:
  `results/smoke_optimized_post_agentx_20260902_140007/`.
- Final smoke artifact:
  `results/smoke_optimized_final_20260902_143033/`.

The accepted `runtime-1` image was then given its own 900-second,
scenario-valid control:

- Warmup completed 16/16 requests with no error or cancellation.
- Profiling completed 131/131 requests; all 147 exported warmup plus profiling
  records report no cancellation, request error, context-overflow skip or OSL
  mismatch.
- 13,139,698 input and 134,780 output tokens; 89.85 output tok/s.
- Median/P99 TTFT: 677.01/5,216.65 ms.
- Median/P99 request latency: 6,389.90/75,964.04 ms; maximum 244.12 s.
- Median/P99 ITL: 11.13/16.56 ms.
- Reported prompt-cache read ratio: 95.21%.
- The same future join consumed the full 600-second grace, but all 131 wire
  requests had already completed. AIPerf exited zero with
  `was_cancelled=false`, zero errors/cancellations and
  `submission_valid=true`.

The subsequent near-limit request and final smoke both passed. Both workers
remained active, every searched Mooncake/DSA/GPU-fault signature was zero, and
the decode log contained 3,094 EAGLE samples with median acceptance length
2.54 (range 1.68–4.00).

- Runtime AgentX control:
  `results/agentic_runtime_agentx_control_20260902_2335/`.
- Runtime post-AgentX smoke:
  `results/smoke_runtime_post_agentx_20260903_0003/`.

## Final verdict

The 1P1D deployment is accepted for the tested greedy agentic workload. Use
`infera/engine-sglang:glm52-1p1d-8f04a68-runtime-1`, keep prefill DP-attention
off and decode DP-attention at DP8, and retain `mlx5_0`/GID 3 as the safe
no-pin Mooncake path on these nodes. C64 remains the synthetic throughput knee;
C128 is stable but adds severe TTFT without increasing throughput.

#37133 is the required routing-correctness backport. #37118 is retained as a
merged HIP graph-compatibility layer because the decode EAGLE graphs are
active, but it has no demonstrated throughput gain in these runs. Aiter #5121
is excluded from the default: the pinned source lacks the later abort and
passed a 250,016-token fallback test without that hunk. `opt-1` remains
available only for historical A/B reproduction.

The one-hour AgentX load ran on `opt-1`; the narrowed `runtime-1` image
separately passed the valid 900-second control, near-limit context test and
post-run smoke with no model or transport failures. Non-zero-temperature EAGLE
sampling remains outside the accepted envelope until the newer
rejection-sampling stack is integrated and revalidated. AIPerf 0.12's
delayed-join finalization issue remains visible, but the finite grace setting
makes runs bounded and yields valid submissions with zero failed or cancelled
wire requests.
