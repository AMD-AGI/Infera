# TP4/EP4 internal-decode DPA sweep

## Context
The user moved the proven benchmark into Git for traceable development and colleague sharing. Use only `/home/yihou/dev/git/infera.dev.yihou.sglang.bench.fast.script/glm52_decode_internal_yihou_20260909_1057` for subsequent code/debug work. Repository branch is `dev.yihou.sglang.bench.fast.script`, clean at `4d982fa`; bench/launcher hashes match the proven source. Preserve previous workspace and packups.

Run ten configurations: TP4/EP4, global concurrency4,8,16,20,24, DPA off/on. Keep ISL70000,OSL10000,expected bonus-inclusive accept length3.61 and the same optimized image/source pins, real weights/routing, synthetic physical prefix and scheduler-free actual draft/verify/draft-extension loop.

Historical timing, verified from original command/result file mtimes: complete prior TP8 run182.94s wall (78.175s measured decode); first cold eager smoke676.82s; hot graph/long-context smoke109–116s. These are observed approximate launch-to-result times, not predictions for new TP/EP kernels.

## Resource ownership
The user explicitly assigned **node056/job130737** to this sweep and **node249/job130740** to their other fake-input PD benchmark. Do not operate249. Never use234/036, request allocations or cancel existing holds. Coordinate any reassignment with the other session before use.

User explicitly authorized stopping non-our-party GPU workloads on the held056, while preserving own-party work, supervisors and system/cluster collectors. Inspect exact container/PID ownership and identity immediately before stopping. Observed056 GPU task: `zebra3d-kf10-sharded-cell-130825` (`bc2cc42f0a3e`, /home/usaxena/Zebra3D mount,8 Python processes). Preserve `crusoe-amd-log-collector` and `crusoe-amd-exporter`. Stop only verified authorized targets, not broad pkill or Docker pruning. Do not delete their files/containers. Peer fake-prefill-decode-benchmark confirmed no work on056.

056 lacks the pinned optimized image; after cleanup validate GPU drain/driver and actual containerd backing capacity, then load the existing read-only archive. Use a unique yihou container with new Git benchmark path mounted, read-only model mount, explicit four-GPU visibility and image-keyed JIT cache. Sequential configurations on the same four GPUs avoid concurrent-run interference. Do not touch host package/daemon configuration.

## Implementation approach
Extend the existing wrapper rather than create a separate benchmark or switch to online/PD serving. Preserve the old TP8/EP1/DP1 mode as a regression reference.

Use four GPU processes for both modes:
- DPA off: TP4/EP4/DP1, attentionTP4, local attention batch equals global concurrency.
- DPA on: TP4/EP4/DP4, attentionTP1, each attention replica owns concurrency/4 requests (1,2,4,5,6). User concurrency always means global, not per-replica.

Parameterize topology, local batch, graph batch sizes and result aggregation from one tested descriptor. Populate real pinned-fork ParallelState and required cross-DP forward metadata/collectives for target, draft, verify and bootstrap. Use supported pinned MoE communication/backend defaults explicitly and log resolved selections. Do not merely bypass EP/DP guards while retaining TP8-derived state.

Keep physical KV/index initialization and sequence-progress validation. Partition request identities/data by attention-DP replica, preserve TP-equivalent state within each replica, and ensure speculative acceptance/control-flow stays collective-compatible. Aggregate useful tokens once per attention-DP replica, not once per TP rank; global output must equal concurrency*10000. Preserve per-rank/per-DP evidence rather than assume all rank JSONs must be identical.

## Files and workspace
Before editing, back up the benchmark's existing CLAUDE.md as `CLAUDE.glm52-internal-decode.<YYYYMMDD-HHMM>.md.bak` and replace it with current task/resources/references. Leave repository `.claude/CLAUDE.md` DCO instructions unchanged. Create `sweeps/tp4_ep4_dpa_yihou_<timestamp>/` for plans, process log, source/patch snapshots and per-iteration artifacts.

Modify only benchmark files needed for this task:
- `bench/profile_decode.py`: EP/DPA CLI, worker/forward initialization, local/global batch semantics, aggregate reporting and timing phases.
- `bench/batch_state.py` or a small `bench/topology.py`: tested topology/aggregation helpers.
- `tests/test_batch_state_yihou.py` and focused topology tests.
- `scripts/create_container_yihou.sh`, `scripts/run_decode.sh`: remove stale hardcoded resource/path assumptions via explicit validated inputs, retain guards and dry-run.
- A small sweep launcher/collector plus benchmark README usage.

Record Git HEAD, dirty diff and code hashes per run. No automatic commit/push; user has not requested one. Any later requested commit requires repository DCO sign-off and assistant coauthor trailer.

## Validation and execution
1. CPU tests first: TP4EP4 topology for both modes, global/local concurrency mapping, invalid divisibility rejection, replica-aware counts and TP8 backward compatibility. Syntax/CLI/dry-run checks on launch scripts; never silently accept contradictory flags.
2. Probe assigned container environment and verify exact image/SGLang/AITER identities. Establish clean GPU baseline after authorized cleanup.
3. Small eager/graph smoke for DPAoff and DPAon before the sweep. Check real model/DSA/EP path, valid initialized prefix, finite bootstrap, correct local/global token lengths and actual graph executions. Use known-good wrapper/pinned production metadata preparation as differential references.
4. Execute all ten full-output points sequentially, increasing concurrency per mode. Keep warmup10, seed1234, same memory/optimization settings and timing boundary. Allow compilation time with progress checks. Failed/OOM points are diagnosed and rerun, not fabricated or silently skipped.
5. Each result records global and per-DP useful output count, max-rank synchronized decode elapsed, global outputtokens/s, /4GPU throughput, effective TPOT(ms/token/request), realized acceptance, graph counts and memory. Also record model-load/capture/warmup/total wall time so duration estimates are measured.
6. Produce DPAoff/on comparison tables for conc4,8,16,20,24, with code/image/model provenance and all raw evidence. Keep simulated-prefix/acceptance and non-client-ITL limitations explicit.

## Execution discipline
Use one independent research/implementation teammate for topology work; leader owns GPU operations, integration and conclusion. No subagent verification of leader work. Load required speculative/runtime/env skills before corresponding edits. Follow per-iteration evidence-driven debug loop. New task mission reminders/team checkpoints are session-only and cancelled at completion. Large core-library patches or a material topology/measurement change require user check-in.
