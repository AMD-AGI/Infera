# Iteration002: real eager speculative smoke

## Hypothesis and target
The pinned TpModelWorker and EAGLEWorkerV2 can run without Scheduler when given real pools and a fixed ScheduleBatch. Test two complete speculative iterations at TP8,bs16,ISL1024,OSL16,expected accept length3.61; no graphs, no warmup. Success means finite bootstrap tensors, valid state progression and incomplete-result accounting, not mission completion.

## Changes
First standalone driver; no SGLang/AITER core changes. Real weights/routing; SIKL-equivalent physical random prefix/index initialization. Source snapshot retained beside command.txt.

## Observation
At12:25Z eight rank processes active; module_custom_all_reduce JIT compiling. At12:27Z module_custom_all_reduce.so exists and GPUs show21% HBM, consistent with model load progress. Spur exec buffers stdout, so console.log initially empty despite live work. Subsequent launcher uses an inner container tee to runtime.log; current run is not restarted just for logs.

## Result
PASS for the smoke target: exit0, all8 rank JSON results identical,2 iterations,8 emitted tokens/request, context1024->1032,128 useful tokens total, complete=false as intended.8 FlyDSL decode and8 fused-indexer engagement markers. Bootstrap finite checks passed. Measured99.52s includes first-use JIT (no warmup), so it is NOT a performance measurement. All benchmark processes exited; only owned container sleep remains. Post-completion guard/state assertions are now unfrozen for bounded implementation before graph smoke.
