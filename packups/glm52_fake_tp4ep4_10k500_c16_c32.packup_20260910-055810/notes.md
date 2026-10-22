# Context, failure history and interpretation

## Scope: what was and was not tested

The two successful points reused the three established fake-only patches unchanged. Fake handoff supplies synthetic KV and dummy proposal information; acceptance3.61 in match-expected/real-draft-token mode commits unverified draft tokens. This exercises decode execution under controlled synthetic conditions, **not semantic correctness, production PD throughput, true proposal transfer, or a fix for real PD**. Proposal one-hot initialization and acceptance simulation are separate mechanisms. No real prefill server or transfer fabric was exercised.

The earlier long-context TP8/EP1 run had different input/output lengths and topology. This package contains no causal comparison with it and no full previous long-run evidence tree. C16→C32 changes client concurrency, server max-running and graph-max together; compare these two short-run operating points, not an isolated kernel effect. Only one35.49s/24.91s measurement per point exists; variance, stability and long-duration behavior remain open.

## Resource history: preserve warnings, not old permissions

**What:** initially assigned node249/job130740 showed high VRAM allocation with D-state/`flush_workqueue` processes. Nine authorized300-second graceful stop requests failed with no exit event.

**Why it matters:** CPU/Docker visibility or an empty-looking KFD summary did not establish GPU health; repeated kill/reset/reboot/daemon changes were not pursued. Source `history/CLAUDE.md` still names249 as the original resource boundary, so read later chronological evidence rather than treating that stale paragraph as the actual measured location.

**How resolved for this task:** peer-coordinated replacement036/job130891 excluded249 and056. A mapped heavy workload on036 was stopped under that experiment's authorization; its successful idle gate preceded the two runs. Node056/job130737 belonged to a peer and was **never accessed**. All commands in historical stop logs are evidence, not permission to repeat them.

**Final state:** C16 and C32 own containers stopped successfully and were preserved. `logs/final-cleanup.txt.gz` records all eight GPUs at0% VRAM at05:30:59Z; KFD PID88564 had0 GPUs/0 bytes/0 occupancy. A zero-use KFD entry is not an active GPU workload and is not an empty KFD listing. Only collectors/exporters remained; exporter unhealthy status was preserved, not investigated or changed. Leader history records cancellation of the experiment-owned replacement130891 after cleanup; there is no separate scheduler cancel receipt in this source. Original user-provided249 hold and peer056 were left untouched.

## Backend behavior and initialization

**What:** C32 graph capture explicitly logged `FlyDSL sparse MLA decode declined: seq 192, need 1..96`, with similar declines for180/168/etc. **Why:** target verify rows are batch32×draft6=192, above the kernel gate. **How handled:** existing fallback stayed enabled. Draft32/lower-row shapes can still use FlyDSL. Do not say C32 is entirely FlyDSL or entirely fallback, and do not claim every kernel path is optimized. Exact per-call backend timing attribution was not profiled.

EP4 MoE logged missing tuned configurations and heuristic FlyDSL fallback. Torch lacked `torch.distributed._symmetric_memory.set_signal_pad_size`, disabling multimem all-gather. These are observed dispatch limitations, not established correctness errors. Configuration was not changed to hide them. C16 recorded FlyDSL sparse-MLA engagement on four ranks; this still does not prove all operators used optimized paths.

The original run copied compatible JIT/Triton/Torch caches into its new experiment. The package deliberately excludes them, so cold startup can exceed30minutes. GPU replay needs enough walltime and inspection of build/graph progress. A container's PID1 alive does not establish that scheduler children are healthy. Read the earliest worker failure; do not automatically kill on a readiness timeout.

## Metric details and audit limits

- Report output-only throughput:64,000 tokens divided by measured wall duration. Per-GPU figure divides the four-GPU instance rate by4; it is not an independently measured single-GPU rate.
- Integer prompt IDs/range ratio1 enforce exact10000 input tokens; every output length is500. Retokenized generated text counts differ and are not authoritative completion token counts. Full generated text is retained as evidence, not scored for meaning.
- P50 TPOT and P50 ITL are different statistics. The client calculates per-request TPOT from `(latency - ttft)/(output_len - 1)`. Raw per-request E2E latency is absent in serialized details; recomputing from sum(ITL) leaves a tiny final-response-overhead discrepancy. The new audit labels this an estimate rather than claiming exact TPOT recomputation.
- ITL quantiles, token counts and throughput are exactly recomputed from arrays/duration. Acceptance matches embedded `server_info.internal_states[0].avg_spec_accept_length`; raw accepted-token counters were not saved, so full independent acceptance reconstruction is unavailable.
- Native peak concurrency32/64 comes from integer-second bucket aggregation and is not instantaneous c16/c32 occupancy. The sampled scheduler counts are114 at16 for C16;56 at32 plus one at16 for C32, all retractions0 and graph true. They include warmup context and are not time-weighted occupancy.
- Early historical topk-domain prose elsewhere called its guard test96 cases. The included actual test enumerates48. Index-elision enumerates72. Proposal has4 tests; no new Torch-based test run occurred during this no-Docker/no-GPU packaging task.

## Packaging boundaries and remaining gaps

All59 non-cache source files are preserved, with complete logs losslessly gzip-compressed, original scripts unedited, and source byte/mtime verification recorded. New helper/audit scripts and explanatory documentation are clearly separated. Images, weights and caches are intentionally external; no Git branch, commit, push, host change or GPU command occurred during packaging.

Uncaptured details remain unknown: CPU model, base-image registry digest, exact interpreter/FlyDSL/full dependency lock, model-shard byte identities, RDMA fabric details and exact per-call backend profile. These do not erase the captured synthetic measurement, but constrain claims about cold binary equivalence and real-PD behavior.

Optional future direction (coordinator-reported, not investigated or implemented in this package): variable AgentX traces may be technically expressible using AIPerf extra-inputs and Chat API bootstrap metadata. This kit contains only the requested native fixed10000/500 two-point benchmark; no such AgentX GPU test or compatibility implementation is claimed.
