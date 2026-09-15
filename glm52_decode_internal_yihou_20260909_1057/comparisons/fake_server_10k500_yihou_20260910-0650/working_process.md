# Internal versus fake decode-server comparison

## Goal
Compare C16/C32 TP4EP4DP1 ISL10000 OSL500, primarily validating internal wrapper semantics. Reference server:128requests after16warmups,simaccept3.61. No trueP-to-D transfer or semantic correctness claim.

## Iterations
- 000_environment: user approved271/job131080 and cleanup. Peer confirmed no work on271. Inspected h24-tsa-diag-131095 ruijyang mounts and stopped exactcontainer;no file/container deletion. Pinned image verified/loaded;actual containerd filesystem17TiB free;ownedcontainer yihou-glm52-compare-20260910,fourGPUs,zeroVRAM baseline.
- baseline_c16_yihou: PASS exit0,8000usefuloutputs,136iterations,4.637333s,TPOT9.274666ms,1725.1295tokens/s,actualaccept3.683824. All4rank3graphcounts136. Coldlaunch911s.
- baseline_c32_yihou: PASS exit0,16000outputs,136iterations,5.970033s,TPOT11.940066ms,2680.0523tokens/s,actualaccept3.683824. All4rank3graphcounts136. Hotlaunch116s.

## Reference source evidence
- Packup server-info flattened:TP4EP4DP1,DPAoff,overlap enabled,mem0.85,rejectionsamplingtrue,graphmaxC,seeds197604707(C16)/1017015168(C32). Current baseline uses synchronous wrapper/seed1234/onewave;not equivalent yet.
- serving.py657-768: native client tracks chunk completion_tokens,divides interchunk gap by num_new_tokens;firsttext chunk becomesTTFT. calculate_metrics1134-1135 TPOT=(latency-TTFT)/(output_len-1).
- serving.py1424-1442:16warmup requests use firstprompt with outputmin(requested,32),not16decodeiterations. benchmark_duration1604 evaluated after server_info HTTP query,not just device execution.
- Pinned fake receiver conn.py124-134 only marks metadata sent;no KV write. memory_pool.py3994-4001 zero-initializes MLAKV. MetadataBuffers utils.py352/382-399 zero output/topk/hidden andDSAindices-1. decode.py1909-1910 appends handoffoutput0 before decoding. Patched build_eagle_disagg_draft_input fills topkp1 and full-vocabonehotat0,hidden0,bonus0.
- Need label leftover KV reuse after serverwarmup/refill;per-wave zero reset is controlled reference-style initialization,not byte-identical history.

## Aligned runs and findings
- fake_c16_yihou PASS execution:128requests,64000outputs,63872computeddecode,8waves,38.24686s,total-throughput1673.34. New summarize_waves TPOT formula erroneously multiplied totalrequests instead of wave concurrency; observed regression test, fixed formula,derived corrected_summary saved separately. Correct average decodeTPOT9.58088ms,wave median8.88165ms. Original raw file retains incorrect76.647ms field,DO NOT use it.
- fake_c32_yihou PASS execution with correctedformula:128requests,64000outputs,63872decode,25.96271s,2465.07outputs/s,average decodeTPOT13.00737ms,wave median11.73330ms.
- Outliers retained:C16wave3step29=2.9469s;C32wave2step70=2.7190s. No JIT line around these;cause not established.
- path_check_yihou:executionexit0,but numerical equivalence NOT PASS. Same intended fake reset,2steps:accept/seq lengths match,bonus/topk IDs differ,hiddenrelativeL2~0.807/1.049,allfinite. Could be reset/nondeterminism/kernel-path difference;no rootcauseclaim.
- path_repeat_yihou:running graph/graph_repeat/eager/eager_repeat to separate within-mode repeatability from cross-mode divergence. No backendpatch or artificial acceptance change.

## User decision: superseded on 2026-09-10
Stop numerical-path investigation. The repeated graph and repeated eager runs also differ; root cause remains unresolved. Execution success does not establish numerical equivalence. User requested reverse validation of the prior70K/10K ten-point sweep using actual fake-input server runs. New work is in ../reverse_fake_server_sweep_yihou_20260910-0837/. The following work-in-progress note is historical, not an active assignment.

## Historical work in progress
comparison-implement owns bench/tests repeatedwaves + opt-in fake-server state + warmup-request mode;leader owns reference metrics/environment/run+conclusion. DefaultSIKL preserved. No timing-fitting patches. GPU idle pending smoke-ready. No215/249/234/036operations;peer movedto215/job131219.
