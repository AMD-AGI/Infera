# TP4/EP4 DPA sweep working process

## Goal
Ten full-output points: global concurrency4,8,16,20,24; DPAoff/on; ISL70000,OSL10000,expected bonus-inclusive accept length3.61. Same4 GPUs and pinned optimized stack. Code authoritative in new Git repo baseline4d982fa. Preserve old experiments.

## Research
DPAoff TP4EP4DP1 attentionTP4 local batchC. DPAon TP4EP4DP4 attentionTP1 local batchC/4. Required global token metadata must be base local counts per replica; speculative scaling happens inside ForwardBatch. Real AITER EP path with a2a=none still communicates. Detailed research pending handoff.

## Iterations
- 000_environment: user explicitly assigned056/job130737 and authorized clearing non-our-party GPU workloads, retaining collectors. Peer confirmed no own work on056 and uses249. Inspected exact Zebra3D container bc2cc42f0a3e (usaxena mounts,8 Python workers); stopped with60s grace, no file/container removal. Collectors preserved. GPU drain still being checked. Storage probe and pinned image loading next.

## Implementation checkpoint
Research complete:4-process DP4 uses attentionTP1/localC/4; default AITER MoE a2a none provides EP gather/reduce. ForwardBatch scales base DP counts itself. dpa-implement owns bench/tests; leader owns scripts/environment. Image loaded with verified hash/ID; four-GPU container created. Launcher tests5/5 pass after parameterizing JOB_ID/NODE/CONTAINER and rejecting forbidden/peer nodes. No blanket GPU kill or host changes. Peer reports reassignment from249 to036 under its own authorization; both remain off-limits to this task.

- Environment follow-up: a new GPU workload h24-tsa-diag-130886 appeared while image loaded. Inspected exact container6a664906ba50 with ruijyang mounts and matching KFD PIDs; stopped under user's explicit056 cleanup authorization. All GPU memory now0–1%,only zero-VRAM KFD entries remain. Own four-GPU container verifies torch2.9.1/4MI355X and exact source pins. Collectors preserved; no files/containers deleted.

## Leader checkpoint 2026-09-10T05:16Z
Task target re-read unchanged.056/job130737 still RUNNING with about6h36m remaining. Teammate has written focused topology tests (05:15), implementation not yet handed off; no runtime benchmark started. Earlier missing-worktree command issue was resolved by permitted explicit authoritative cwd, without permission-setting changes. No new direction/information issue requiring intervention; retain single ownership of bench/tests. Peer has moved to036/job130891 under its own authorization;249 and036 remain unavailable to this task. Ten-point sweep pending, no fabricated metrics.

## GPU iterations
- 001_ep4_off_smoke_yihou: started after leader23CPU+5launcher tests pass. TP4EP4 DPAoff,global/localC4,ISL1024,OSL32,warmup3,maxsteps4,graphs enabled. Source snapshot+Git diff captured. Goal valid four-rank topology and actual full speculative graph execution. Newly observed vLLM container on056 had only sleep (no GPU process),left untouched; collectors retained.

## Checkpoint 2026-09-10T05:27Z
Current task re-read; target unchanged. Smoke001 still loading model shards with advancing progress (~56% in latest stream); no result yet and no hang evidence. Four-rank distributed init/AITER custom-allreduce JIT succeeded. Bench/tests frozen after23CPU tests and5launcher tests. No changes to forbidden/peer nodes or allocation ownership.

## Leader checkpoint 2026-09-10T05:32Z
Teammate implementation is complete and intentionally frozen; no missing-information or path issue. Smoke001 shard progress reached282/282 but no post-load/graph/result log yet. Four worker processes remain alive; no new JIT build directory beyond core/allreduce. First observation of a possible post-load stall is recorded, not yet a hang conclusion. Next checkpoint should compare log/process progress; do not restart simply for startup duration.

## Checkpoint 2026-09-10T05:37Z
Previously noted post-shard quiet period resolved without intervention: runtime advanced to CUDA graph capture atbs4, module_norm built in54.9s, now compiling rope module. This is verified compilation progress, not a stalled worker. Target and scope unchanged; keep sources frozen and await smoke result.

## Smoke results
001 DPAoff PASS exit0:TP4EP4DP1,global/localC4,64 useful tokens,context1024->1040;all4 ranks graph counters4/4/4. Cold launch891s;load/pool/capture802.27s,bootstrap34.10s,warmup7.77s,decode0.0842s. 002 DPAon started same workload/globalC4 withlocalC1,source unchanged,frozen;validating replica aggregation andDP collectives.

## Full sweep start
002 DPAon PASS exit0:TP4EP4DP4,globalC4/local1,perDPoutputs16each=64global,contexts1040,all4ranks3graphcounters4. Hot launch102s,load/pool/capture63.77s,decode0.10294s. Both modes now GPU-validated. Started ten full points sequentially via run_tp4_ep4_sweep_yihou.sh;code frozen,each run recordsHEAD/diff/hashes/snapshot and wall time. Failure stops sweep rather than skipping silently.

## Checkpoint 2026-09-10T05:50Z
Shell execution verified through actual sequential results: DPAoffC4 exit0,40000 useful outputs,TPOT6.15049ms,launch153s;DPAoffC8 exit0,80000 outputs,TPOT7.99908ms,launch163s. Each completed rank's target/draft/extension executes2768 times. C16 automatically started and measured decode log reachediteration600/context72175 with target_graph=True;four active worker processes. No shell quoting/early-exit issue observed. Ten-point sweep remains incomplete.

## Leader checkpoint 2026-09-10T05:52Z
DPAoffC16 completed exit0,complete=true,TPOT10.28611ms,1555.50tokens/s,total185s. Sweep automatically advanced toC20. Three of ten points complete. Teammate remains intentionally frozen; no new information/direction issue. No intervention required.

## Checkpoint 2026-09-10T05:57Z
Four points complete: DPAoffC20 exit0,200000outputs,TPOT11.66706ms,1714.23tokens/s,total208s. C24 running measured loop. Backend caveat verified in logs: DPAoffC20 target verify has120rows andC24 has144rows,exceeding FlyDSL sparse MLA96-row support; each rank logs declined and falls back. Decode engagement marker from another phase does not negate this. Preserve default fallback in this requested sweep and report explicitly; no kernel change mid-sweep. MoE also reports heuristic FlyDSL configs for some shapes. Task remains in progress.

## Checkpoint 2026-09-10T06:07Z
Eight of ten points complete,all exit0/full useful counts. DPAoffC24 TPOT12.63220ms,1899.91tokens/s,total217s. DPAonC4/8/16 TPOT7.19907/8.87046/10.98086ms,throughput555.63/901.87/1457.08,total153/168/191s. DPAonC20 started. Lower-concurrency DPAon is slower in this observed sample; wait for all points and backend audit before conclusion. Code/task/resource scope unchanged,no intervention needed.

## Constraints
Only056;no249/234/036 access. No allocation requests/cancellations. No commits/pushes. New task instructions backed up. Session timers c84db82d(10min),77bf5649(20min),expire7days;cancel on completion.
