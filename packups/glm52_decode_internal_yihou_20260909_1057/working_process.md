# Working process

## Target
Complete scheduler-free TP8 GLM-5.2-MXFP4 decode: 16 requests, 70000 initial context, 10000 useful emitted tokens/request, expected accept length3.61. Real draft/verify/draft-extension, synthetic physical prefix state, real routing. No PD or Scheduler.

## Iterations
- [000_research](iterations/000_research/): established fixed-context reference limits, pinned worker boundary, optimization provenance. First and second packup read differed in later-image-load evidence; eight key files matched then-current manifest. Plan approved; no runtime success yet.
- [001_environment](iterations/001_environment/): in progress. Revalidated job126175/node055. Own Docker probe sees8 MI355X with0% utilization/VRAM; PyTorch reports8 devices. First read-only probe failed only because tempfile had no writable directory; repeat with workspace TMPDIR passed. Docker uses containerd snapshotter; read-only mount of actual /var/lib/containerd shows28TiB filesystem,24TiB available (not the65GiB /var/lib/docker root view). Archive hash/zstd/load now running. Other user's idle container left untouched.

- [002_eager_smoke_yihou](iterations/002_eager_smoke_yihou/): PASS exit0, real TP8/EAGLE,bs16/ISL1024/OSL16,maxsteps2,warmup0,eager. All8 ranks identical;128 useful tokens,context1024->1032,complete=false.8 FlyDSL decode and8 fused-indexer markers.99.52s includes first-use JIT and is not usable performance evidence. Source freeze lifted for bounded invariant guards; next graph smoke adds warmup.

- [003_graph_smoke_yihou](iterations/003_graph_smoke_yihou/): running bs16/ISL1024/OSL32,warmup3,maxsteps4,graphs enabled. Adds tested state/terminal/layout guards and actual target/draft/extension graph execution counters.13 CPU tests pass; source hashes verified/frozen. Goal: no state divergence and all three graph paths actually replay after warmup.

- Iteration003 completed PASS exit0: all8 ranks agree;target/draft/extension execute counts4/4/4;context1024->1040;256 useful tokens;0.094634s after3 warmups. No state-assertion failures.
- [004_long_context_smoke_yihou](iterations/004_long_context_smoke_yihou/): running unchanged code,bs16/ISL70000/OSL10000 capacity,warmup5,maxsteps16. Validates actual sparse long-context graph path and full capacity before complete progression.

- Iteration004 completed PASS exit0 at12:50Z: all8 ranks agree,16/16/16 graph executes,1,281,024 slots allocated,context70000->70063,1008 useful tokens. Long-context state invariants pass.
- [005_full_target_yihou](iterations/005_full_target_yihou/): running full16/70000/10000/3.61,unchanged frozen sources,warmup10,no maxsteps cap. Goal160000 useful tokens,all final contexts80000,all graph paths counted.

- Iteration005 completed PASS exit0:160000 useful tokens,all16 contexts80000,8-rank agreement,2768/2768/2768 graph executes.78.174991s,2046.690351 outputtokens/s,realized accept length3.6134393.32 terminal extra tokens excluded from useful output. No core-code patch.
- [006_repeat_segment_yihou](iterations/006_repeat_segment_yihou/): running identical target configuration with maxsteps256 to compare against the first256 steps of005,not against its whole-context average. Runtime source remains frozen.

## Team checkpoints
- 2026-09-09T12:06Z: decode-research research complete; assigning bounded wrapper implementation. Leader handles environment and GPU execution. No unresolved team-direction issue recorded.

- 2026-09-09T12:34Z: mission re-anchored to16/70000/10000/3.61; no scope drift. Teammate implementation handed off and intentionally paused during runtime freeze. First smoke has run9min; eight ranks remain alive and new clang/ninja jobs are actively compiling, so no hang inference or restart. Newly noted observability gap: Spur buffered console output; subsequent launcher already writes runtime.log inside the container. Queued post-completion guard/state assertions remain known pending work, not a teammate-direction issue. Packup full7993-file manifest check passed at12:23Z; no need to repeatedly rehash unchanged material at this checkpoint.

- 2026-09-09T12:46Z: mission unchanged, target16/70000/10000/3.61. Iteration003 completed target-verify capture (~9s), draft-decode capture (~2.9s), draft-extend capture (~1.64s) after cached startup; now warmup3. Capture success is not yet proof of measured replay; await result counters. No new teammate-direction issue or resource change.

## Completion
Full target005 and matched repeat006 passed. Report: results/report.md.006 first256-step sum differs from005 by-0.034756%,with identical acceptance/context trajectory. Final CPU13/13 and launcher4/4 tests pass. All benchmark processes exited and GPU memory reported0%; only this task's owned container was stopped,not deleted. Existing allocation and other user's container preserved. Original repository git status unchanged. No commits/pushes or forbidden-node access.

## Resource constraints
Never node234 or036; no allocation submission/cancellation. Existing holds are not ours to release. Preserve all other users' work. No broad process cleanup or file deletion.
