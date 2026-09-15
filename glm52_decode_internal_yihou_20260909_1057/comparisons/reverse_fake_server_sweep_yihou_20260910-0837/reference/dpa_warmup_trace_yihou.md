# Fixed-source warmup trace (2026-09-10)

Read-only source research, SGLang402df1e1. Paths relative to /sglang/python/sglang/srt/.

- entrypoints/http_server.py2156-2183: four concurrent warmup requests,input4/output8,explicit routed_dp_rank0..3. managers/data_parallel_controller.py738-755 respects explicitrank. Not a demonstrated round-robin routing failure.
- disaggregation/fake/conn.py107-131:fake transfer immediatelysuccessful aftermetadata;disaggregation/utils.py180-202 skipsmetadatareadinessgate. No realnetworkKVwait.
- disaggregation/decode.py2183-2187,2342-2367 receives/polls fake;2246-2268 seeds prebuiltEAGLE beforeDPMLPsync.
- speculative/eagle_disaggregation.py113-120:overlappublish+stash. speculative/overlap_utils.py464-472:onHIP resolve_seq_lens_cpu hostsynchronizes publish_readyevent. eagle_worker_v2.py1203-1232:draft,verify,on_publish,draft_extend. No rankstacks available,actualhangsiteunproven.
- arg_groups/speculative_hook.py85-89:--disable-overlap-schedule selects synchronousEAGLEV2. decode.py2139-2170 runs/processes immediately instead of2183-2227 resultqueue. dp_attn.py314-322 also changesDPsync fromCPUGloo todevicegroup. Diagnostic only,not equivalentperformancecurve.
- http_server.py806-809:server_info awaits scheduler internalstate,not pureHTTPhealth.2393-2397:skip-server-warmup suppressesrequests andmarksUp,but cannotprovefirstdecodeworks.

Observed: disablingTP LM-headall-to-all alone stillstallswarmup. Nextsinglepoint keeps thatflag and additionallydisablesoverlap;retain originaloverlapfailure,neverlabeldiagnostic asmatchingservermethod. No kernelpatch or newframework.
