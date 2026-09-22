# Aligned C80 follow-up

User authorized matching yihou C80 settings while retaining cross-rank routing:
3600-second profiling, 10 warmup requests/lane, P/D max-running and graph caps
256, SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=1. Prefill HiCache remains enabled
(ratio 1.5, write_through, kernel, page_first); decode MTP simulation stays 3.61.
No router affinity patch. Existing image/SGLang nightly unchanged (20260916),
so this aligns selected knobs, not the complete yihou 20260917 image environment.

Config: scripts/config.phase-c-aligned.sh.
Runtime root: /perf_apps/liyingli/bench_agentx/glm52-c80-20260922.
Planned launch artifacts: results/launch-aligned.
Planned benchmark artifacts: results/agentx-c80-aligned.

Current status: previous P/D/router/etcd containers stopped. HiCache GPU memory
release is delayed despite no KFD processes and released host memory. User
explicitly instructed waiting until BOTH nodes are idle before launch. Do not
reset GPUs or relaunch while residual VRAM remains. Decode is already idle;
prefill is draining slowly. scripts/wait_nodes_idle.py probes all 8 GPUs/node,
requiring <=2% VRAM and <=5% utilization on every GPU, failing closed on missing
data. Active log: runtime results/wait-idle-aligned.log. Timeout is 5400 seconds.
Current image source confirms the grouped-topk environment knob is recognized.
Next: stop prior deployment, check idle GPUs, launch aligned configuration,
validate runtime flags, run C80 to completion, audit records and compare with
yihou C80 plus both prior AUS points. Preserve raw artifacts separately.

Update: BOTH nodes passed idle gate at 2026-09-22 12:35:48 UTC, all 16 GPUs at
0.096% VRAM and 0% busy. The independent harness check also passed all GPUs.
Evidence: analysis/wait-idle-aligned.txt. Aligned P/D containers then launched;
model loading in progress. No GPU reset was needed.

Update: P/D/router all healthy. C80 client running at runtime
results/agentx-c80-aligned. Runtime snapshots confirm grouped_topk=1 and
max_running=256 on both workers, warmup/lane=10, duration=3600, concurrency=80.
Decode graph capture covers per-rank batch sizes through 32. Dataset preparation
is in progress; profiling has not started yet. Initial direct HiCache snapshot
is results/cache-aligned/before.prom (host used=0, dropped=0).

Update: warmup finished at 13:09:10 UTC, runner reports 884 completed, zero
errors/cancellations, 1279.16 seconds; raw-record error audit still required.
Profiling started 2026-09-22 13:09:10 UTC; 3600-second sending window ends around
14:09:10 UTC, then drain/export. Active execution session 72990 (SSH runner).

COMPLETED: runner exited 0. Profiling 9737 sent, 9724 valid, 0 errors, 13 cancelled.
Raw audit: warmup 877 valid + 7 empty-content errors, all excluded from profiling.
Report: aligned-comparison.md, numeric comparison aligned-comparison.json.
Aggregate, CSV, logs, service snapshots and HiCache snapshots copied to workspace;
full raw AIPerf exports remain at shared runtime path with a checksum manifest.
Services remain running; benchmark client exited. No further benchmark work pending.
