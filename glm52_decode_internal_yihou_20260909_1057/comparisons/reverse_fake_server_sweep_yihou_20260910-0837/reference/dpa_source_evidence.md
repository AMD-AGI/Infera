# Fixed-source DPA admission and graphs

Independent read-only source research in fixed image confirmed SGLangHEAD402df1e1e453e1e85ec0f5ac4052d36598cc691a.

Paths below relative to /sglang/python/sglang/srt/:
- layers/dp_attention.py325-339:attnDP=dp_size,attnTP=tp_size/attnDP/CP. TP4DP4CP1 -> localattentionTP1.
- mem_cache/kv_cache_configurator.py1873-1886:max_running_requests global value is integer-divided by attnDP,then limited by token_capacity/2. C4/8/16/20/24 ->local1/2/4/5/6.
- managers/scheduler.py1029-1043,4299:worker effective localcap consumed and exposed as effective_max_running_requests_per_dp.
- disaggregation/decode.py2288-2302:admission=min(reqpoolsize,max_running)-runningbatch;not multiplied back by DP.
- server_args.py5159-5188:max graph bucket itself appended. Each separatepoint uses localmax1/2/4/5/6;max5 includes5,max6 includes6. Defaultmax6 buckets[1,2,4,6],so tail5 maypad6.
- server_args.py8823-8838:old --cuda-graph-max-bs and new --cuda-graph-max-bs-decode share argparse destination;retain originalalias without duplicate flags.
- model_executor/runner/base_cuda_graph_runner.py64-96:localreqpool limits graphlist;utils/common.py3804-3816 givesalignment1 without TBO. Do not enable TBO.
- model_executor/runner/decode_cuda_graph_runner.py639-646,674-688,1308-1315:MLPgather selects max localrankbatch,not global sum.
- server_args.py4041-4056:auto->round_robin in PDdecode;managers/data_parallel_controller.py738-767:RR acrossactiveworkers,not remaining-slots-aware. ExplicitRR chosen forDPAon;globalcapacity doesnotguarantee allranks equallyoccupied.
- arg_groups/pd_disaggregation_hook.py37-117:fake onlydecode,noDCP>1,no stagingbuffer,no decoderadixcache. Extra decodeallocslots2*(C/DP) do notincrease runningcap.
- disaggregation/decode.py549-552:fake receiver.init(0) skips prefillrouting;doesnotforce decodeDP0.

## Default difference retained, not hidden
arg_groups/overrides.py2407-2427 enables TP LM-head all-to-all bydefault for PDdecode+DPattention+TP=DP>1+CP1,unlike ordinary/internal mixmode. The requested primary run preserves fake-server defaults;do not silently disable or tune to match the internal benchmark. Save resolved enable_tp_lm_head_all_to_all and explicitly report this difference. If results warrant a separate ablation,it requires a separately labeled follow-up,not replacement of primary results.
