# GLM-5.2 internal decode benchmark

## Context
Implement the task in `/home/yihou/dev/git/sglang/mission.md`: a lightweight, configurable, scheduler-free SGLang internal-forward benchmark inspired by SIKL `profile_decode.py`, incorporating the optimized GLM-5.2 stack from `/shared_nfs/yihou/packups/glm52_mix_repro.packup_20260909-100729`. Finish with a successful concurrency-16, ISL=70,000, OSL=10,000 decode-only run and a reproducible report.

The packup defines 3.61 as **expected accepted tokens per speculative iteration, including the bonus token**, not a cache-hit probability. SIKL only measures a repeated fixed-context decode step and has no MTP implementation. We must execute real draft/verify/draft-extension costs rather than multiply ordinary decode throughput by 3.61.

## Recommended design
Use the packup's existing optimized image and its exact source APIs, not a speculative port of the optimization set onto current main:
- Image: `sha256:b9a83742f631d7321c6b75aceb031a8b347c70f97c62ef8c62370c686853de7d`.
- SGLang fork: `402df1e1e453e1e85ec0f5ac4052d36598cc691a`.
- AITER fork: `2c71811b32c8ce2e1266aedaec199df7d90f597d`.
- Read-only weights: `/shared_nfs/models/GLM-5.2-MXFP4`.

Build a thin executable wrapper around real `TpModelWorker` and `EAGLEWorkerV2`. A fixed `ScheduleBatch` is only a data container; never instantiate `Scheduler`, HTTP serving, admission queues, or PD workers. Reuse `forward_batch_generation()` for real draft -> target verification/sample -> draft extension. Keep real model weights and MoE routing; do not silently copy SIKL's synthetic-routing override.

Configure TP8 / EP1 / DP1, FP8 e4m3 KV, FlyDSL DSA decode and required bootstrap prefill support, AITER top-k, fused allreduce and fused QK norm/rope, EAGLE steps=5 / draft tokens=6 / top-k=1, and the packup's acceptance simulation method. No hierarchical prefix-cache/offload system is needed for fixed resident synthetic-prefix state; report that difference rather than claim a full serving reproduction.

### Input state and workload
- Physically initialize target and draft KV plus DSA index keys/scales for each synthetic prefix, using valid numeric ranges and real allocator/page mappings. Metadata-only fake cache is not acceptable.
- Use a short untimed model bootstrap against the populated prefix to obtain valid model-produced draft recurrent state. Do not substitute the worker's zero-request idle input for an active batch.
- Start the measured loop at exactly 70,000 committed context tokens per request. Allocate capacity for 16 * 80,000 tokens plus page/speculative reserve, derived from actual model and pool layout.
- Advance real batch state every iteration: draft input, GPU/CPU sequence lengths, request committed/allocated lengths, token IDs, output lifetime/stream completion, and decode mode.
- Measure until each request has 10,000 emitted tokens. Define and test terminal acceptance clipping and distinguish useful emitted tokens from extra terminal computation. Report realized acceptance separately from expected 3.61.
- Primary result: useful emitted output tokens / measured complete internal iteration loop time (draft, verify, extension and required wrapper bookkeeping). Report synchronized wall time and optionally phase GPU times; exclude model load, prefix initialization, graph capture and warmup. Optional fixed-context samples do not replace the full progression run.
- Label all results as synthetic-prefix, simulated-acceptance performance; no text-quality or production-throughput claim.

## Workspace and operational boundaries
After approval, create `/shared_nfs/yihou/playground/glm52_decode_internal_yihou_20260909_<timestamp>/` and keep implementation, environment, logs, temporary files and per-iteration artifacts there. Write its English `CLAUDE.md` in task/background/context/references/principles order. Back up any existing target CLAUDE.md before replacement; no root project CLAUDE.md currently exists. Leave original repository sources, packup and other experiments untouched; no commits or pushes.

Expected files:
- `CLAUDE.md`, `design.md`, `working_process.md`.
- `bench/profile_decode.py`: CLI, real worker initialization, fixed-batch decode loop, timing.
- `bench/batch_state.py` if needed: small isolated KV/bootstrap/state/accounting helpers.
- `tests/`: CPU accounting tests and documented GPU smoke checks.
- `scripts/run_decode.sh`: existing-allocation validation, unique Docker container and launch configuration.
- `iterations/000_research/`, then one directory per experiment; `results/report.md` and machine-readable results.

Use only an existing authorized allocation. Current candidate is job **126175 on crsuse2-m2m-055**; revalidate ownership and node immediately before execution. **Never use crsuse2-m2m-234 or crsuse2-m2m-036; never submit/allocate/cancel jobs.** No blanket process cleanup: identify ownership and active experiments before touching anything; leave other users' containers and supervisors intact. The existing container on 055 uses another user's mounts and currently has only idle shell processes; it was not touched.

The Spur namespace on 055 has empty `/dev/dri`, while sysfs shows eight AMD devices. Validate GPU access in our own Docker container before diagnosing a hardware failure. The pinned image is not currently listed on 055: inspect actual Docker backing storage, verify/load the external archive if feasible, and verify image/source identity. Do not infer host capacity from namespace `df`, prune images, or change the host daemon. Block and ask if authorized resources are insufficient.

Use Docker with a unique `yihou` container name, read-only model mount, workspace mount, and image-keyed JIT cache. Allow roughly 30 minutes for graph/JIT startup, observe logs/cache progress instead of restarting blindly. Do not delete any file whose name lacks `yihou`; prefer retaining artifacts. Clean up only this task's processes/container, without cancelling the pre-existing hold.

## Research anchors and reuse
- SIKL `profile_decode.py:24-47`: physical prefix initialization and direct forward timing; `common.py:258-315` KV/index population; `common.py:483-519` synchronized replay timing; `common.py:744-762` synthetic routing that we will not copy by default.
- Pinned fork `python/sglang/srt/managers/tp_worker.py:302-388`: scheduler-independent worker constructor.
- Pinned fork `python/sglang/srt/managers/scheduler.py:956-991`: target/draft initialization, shared pools, attention backends, graphs; `:3761-3772` result-state relay (read/reuse behavior only).
- Pinned fork `python/sglang/srt/speculative/eagle_worker_v2.py:1043-1089,1139-1234`: complete speculative worker lifecycle and forward.
- Pinned fork `eagle_worker_common.py:461-660`, `eagle_utils.py:1041-1114`, `spec_utils.py:358-393`: verify results, allocation bookkeeping and acceptance semantics.
- Packup `scripts/repo/Dockerfile` and `glm52/launch_sglang.sh`: source pins, optimization flags and backend-engagement markers.
- Local current `python/sglang/benchmark/one_batch.py:497-548`: ordinary-forward reference, not assumed identical to pinned fork.

Read required runtime/speculative/env-var skills before corresponding implementation. Prefer LSP/Serena for source navigation. Any sizeable core-library change requires a separate user check-in; default deliverable is a standalone wrapper.

## Execution and validation
1. Preserve a research snapshot and hashes. Recheck evolving packup after environment preparation and before the final run; distinguish changed documents from changed executable inputs. Initial second pass already verified eight key files against the current manifest; final audit was still in progress.
2. Use the existing research teammate for an independently owned bounded implementation component once interfaces are agreed. Leader owns integration and GPU execution; no agent verifies leader's work. Establish session-only 10-minute mission reminders and 20-minute team checkpoints after approval. Record a newly observed team issue first and intervene at the next checkpoint only if unresolved. Timers auto-expire after seven days and are removed when the task ends.
3. CPU tests: expected/realized acceptance math (bonus included), per-request output cap, total output accounting, sequence progression, reserve calculations, deterministic seed/config serialization.
4. Own-container environment probe: GPU count/architecture, actual driver/runtime, source import locations/commits, model config identity, image ID, capacity and competing workload check.
5. Small eager smoke: initialize real target/draft workers and nonempty speculative state, then execute a few iterations; assert finite outputs, valid indices, state progress and full draft/verify/extension execution.
6. Graph smoke at small lengths, then concurrency 16 / 70K resident prefix. Confirm target/draft/extension graph use or clearly report fallback. Confirm FlyDSL sparse MLA decode and fused gfx950 DSA indexer engagement, not merely successful startup.
7. Execute full 16 x (70K input context + 10K emitted output) benchmark. Log exact emitted count=160,000, iteration count, realized acceptance distribution, synchronized duration, output tokens/s total/per GPU, per-user effective token latency, memory usage and relevant graph/backend evidence.
8. Re-run a bounded representative segment for stability, not a statistical-equivalence claim; perform differential checks against the ordinary forward/reference for shared state invariants, while acknowledging no reference MTP performance target exists.
9. Publish English reproducibility report and concise Chinese user summary. Include commands, image/source/model provenance, each iteration's hypothesis/change/evidence, timing boundary, limitations and any blockers. Do not report completion before the requested full workload succeeds.
