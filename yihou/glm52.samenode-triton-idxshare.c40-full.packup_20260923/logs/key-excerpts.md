# Key log excerpts — the lines each conclusion rests on

Full logs stay in the experiment workspace at
`bench/glm5p2_pd/results/yihou-triton-idxshare/rounds/` (gitignored; the decode
log alone is 16 MB and the aiperf `server_metrics_export.json` is 1.3 GB). Only
the load-bearing lines are reproduced here, per the user's lean-packup decision.

### The same-node start gate fired — and was still insufficient
```
starting prefill-0 on crsuse2-m2m-137
 + docker + run + -d + --init + --name + glm52-pd-yihou-sn-p4d4-prefill-0 + --network + host + --ipc + host + --shm-size + 32g + --device + /dev/kfd + --device + /dev/dri + --device + /dev/infiniband + --group-add + video + --grou
same-node gate: waiting for the already-started leg(s) on crsuse2-m2m-137 to answer /health
prefill-0 on crsuse2-m2m-137: healthy
starting decode-0 on crsuse2-m2m-137
 + docker + run + -d + --init + --name + glm52-pd-yihou-sn-p4d4-decode-0 + --network + host + --ipc + host + --shm-size + 32g + --device + /dev/kfd + --device + /dev/dri + --device + /dev/infiniband + --group-add + video + --group
prefill-0 on crsuse2-m2m-137: healthy
decode-0 on crsuse2-m2m-137: FAILED: container stopped: status=exited exit=143 oom=False
```

### Round 1 — decode lost the RCCL start race despite the gate
```
2026-09-23T08:13:03.159073700Z     raise RuntimeError(f"NCCL error: {error_str}")
2026-09-23T08:13:03.159074540Z RuntimeError: NCCL error: unhandled cuda error (run with NCCL_DEBUG=INFO for details)
2026-09-23T08:13:08.295824642Z RuntimeError: Rank 0 scheduler died during initialization (exit code: -3). If exit code is -9 (SIGKILL), a common cause is the OS OOM killer. Run `dmesg -T | grep -i oom` to check.
2026-09-23T08:13:11.915356885Z RuntimeError: sglang subprocess exited with code 1 before reporting ready
```

### Round 2 — manual retry ~9 min later, clean
```
2026-09-23T08:22:05.226377359Z I0923 08:22:05.226337  1360 transfer_engine_impl.cpp:274] Topology discovery complete. Found 1 HCAs.
2026-09-23T08:22:05.256834553Z I0923 08:22:05.256793  1360 transfer_engine_impl.cpp:412] HIP transport installed for intra-node GPU P2P
2026-09-23T08:22:07.912913436Z I0923 08:22:07.912850  1448 transfer_engine_impl.cpp:274] Topology discovery complete. Found 1 HCAs.
2026-09-23T08:22:07.960027677Z I0923 08:22:07.959965  1448 transfer_engine_impl.cpp:412] HIP transport installed for intra-node GPU P2P
2026-09-23T08:22:10.381592685Z I0923 08:22:10.381561  1521 transfer_engine_impl.cpp:274] Topology discovery complete. Found 1 HCAs.
2026-09-23T08:22:10.404268389Z I0923 08:22:10.404238  1521 transfer_engine_impl.cpp:412] HIP transport installed for intra-node GPU P2P
2026-09-23T08:22:13.891393774Z I0923 08:22:13.891352  1589 transfer_engine_impl.cpp:274] Topology discovery complete. Found 1 HCAs.
2026-09-23T08:22:13.990455366Z I0923 08:22:13.990394  1589 transfer_engine_impl.cpp:412] HIP transport installed for intra-node GPU P2P
2026-09-23T08:24:42.655136271Z [2026-09-23 08:24:42 DP0 TP0] max_total_num_tokens=2226368, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=16, context_len=1048576, available_gpu_mem=39.67 GB
2026-09-23T08:24:51.873661100Z [2026-09-23 08:24:51] End of disaggregation warmup
2026-09-23T08:24:51.977090836Z [2026-09-23 08:24:51] The server is fired up and ready to roll!
```

### Prefill: HiCache host pool AND the separate DSA indexer allocation
```
2026-09-23T08:10:52.713348595Z [2026-09-23 08:10:52 DP2 TP2] Allocating kv hierarchical KV host pool: 3423488 tokens, 153.81 GB host memory.
2026-09-23T08:10:52.751329980Z [2026-09-23 08:10:52 DP1 TP1] Allocating kv hierarchical KV host pool: 3423488 tokens, 153.81 GB host memory.
2026-09-23T08:10:52.786871388Z [2026-09-23 08:10:52 DP3 TP3] Allocating kv hierarchical KV host pool: 3423488 tokens, 153.81 GB host memory.
2026-09-23T08:10:52.807285027Z [2026-09-23 08:10:52 DP0 TP0] Allocating kv hierarchical KV host pool: 3423488 tokens, 153.81 GB host memory.
2026-09-23T08:11:32.650178021Z [2026-09-23 08:11:32 DP3 TP3] Allocating 35.25 GB host memory for DSA indexer (layout=page_first).
2026-09-23T08:11:32.660056994Z [2026-09-23 08:11:32 DP0 TP0] Allocating 35.25 GB host memory for DSA indexer (layout=page_first).
2026-09-23T08:11:41.780893884Z [2026-09-23 08:11:41 DP1 TP1] Allocating 35.25 GB host memory for DSA indexer (layout=page_first).
2026-09-23T08:11:42.181885319Z [2026-09-23 08:11:42 DP2 TP2] Allocating 35.25 GB host memory for DSA indexer (layout=page_first).
```

### Round 3 — the deliverable passed, full window
```
08:48:31.863 NOTICE   Phase warmup (warmup) sending complete | sent=444, completed=442, in_flight=2 | sessions: sent=43, completed=13 (runner.py:1039)
08:48:33.867 NOTICE   Phase warmup (warmup) complete | completed=444, cancelled=0, errors=0 | sessions: completed=14, cancelled=0 | elapsed=890.76s (runner.py:1162)
08:48:34.006 NOTICE   Phase profiling (profiling) started | phase_index=0 | profiling_index=0 | target: 3600.0s duration (runner.py:593)
Validated aiperf request error rate: 0/4157 = 0.000% <= 10.000%
AgentX passed: /home/yihou/dev/git/infera.glm52.pd/bench/glm5p2_pd/results/yihou-triton-idxshare/rounds/003-agentx-c40/out/agentx_conc40.json
```

### Server-side counters as the HiCache tier warmed
```
08:48:49.946 INFO       trace  theoretical_prefix_cache_hit=96.7% (records_manager.py:1345)
08:49:19.972 INFO       trace  theoretical_prefix_cache_hit=96.3% (records_manager.py:1345)
08:49:50.033 INFO       trace  theoretical_prefix_cache_hit=96.3% (records_manager.py:1345)
08:50:20.133 INFO       trace  theoretical_prefix_cache_hit=96.3% (records_manager.py:1345)
08:50:50.267 INFO       trace  theoretical_prefix_cache_hit=96.3% (records_manager.py:1345)
08:50:50.267 INFO       srv    prefix_cache_hit=90.0% unique_in_srv=2,356,561 kv_usage=30.0% cpu_kv_usage=99.9% queue=5r/1w tput_in_srv=173,610/s tput_out_srv=569/s (records_manager.py:1345)
08:51:20.485 INFO       trace  theoretical_prefix_cache_hit=96.2% (records_manager.py:1345)
08:51:50.784 INFO       trace  theoretical_prefix_cache_hit=96.0% (records_manager.py:1345)
08:52:21.072 INFO       trace  theoretical_prefix_cache_hit=96.0% (records_manager.py:1345)
08:52:21.072 INFO       srv    prefix_cache_hit=90.2% unique_in_srv=4,047,912 kv_usage=33.0% cpu_kv_usage=99.8% queue=6r/0w tput_in_srv=183,022/s tput_out_srv=651/s (records_manager.py:1345)
08:52:51.392 INFO       trace  theoretical_prefix_cache_hit=96.1% (records_manager.py:1345)
08:52:51.392 INFO       srv    prefix_cache_hit=91.0% unique_in_srv=4,212,472 kv_usage=35.0% cpu_kv_usage=99.9% queue=7r/0w tput_in_srv=182,716/s tput_out_srv=634/s (records_manager.py:1345)
```

