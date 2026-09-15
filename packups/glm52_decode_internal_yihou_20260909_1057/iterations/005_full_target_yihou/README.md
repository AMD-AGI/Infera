# Iteration005: complete mission workload

## Goal
Run16 fixed requests with70000 synthetic physically populated prefix tokens each, emitting exactly10000 useful output tokens/request through actual draft/verify/draft-extension, expected accept length3.61 including bonus. No Scheduler or PD. Real weights and routing.

## Configuration
Same frozen code/image/backends as successful long-context smoke004. Increase warmup to10 complete speculative iterations and remove maxsteps cap. Restart/bootstrap to exact ISL after warmup. Full command in command.txt; snapshots and packup input hashes retained.

## Pass criteria
Exit0; complete=true;160000 useful output tokens; all16 final contexts80000; all8 rank outputs consistent; no state/layout/finite assertion failure; actual graph counts equal verify iterations for all three stages. Report realized acceptance and terminal over-compute without counting it as useful output. Timing excludes initialization/load/capture/bootstrap/warmup and includes full internal loop plus synchronization/bookkeeping.

## Result
PASS exit0. All8 rank results identical; complete=true;160000 useful emitted tokens; all16 accounting contexts80000.2768 iterations and2768 successful executes in each target/draft/draft-extension graph. Expected accept length3.61;realized3.6134393. Raw accept tokens160032 include32 terminal extra tokens excluded from useful throughput. Measured78.174991s;2046.690351 outputtokens/s total;255.836294/GPU;7.817499ms effective token latency/user.8 decode and8 fused-indexer markers. Full-loop median iteration28.209ms. Synthetic prefix/simulated acceptance only; no correctness/production-throughput claim.
