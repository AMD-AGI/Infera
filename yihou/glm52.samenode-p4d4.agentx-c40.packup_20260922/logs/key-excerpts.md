# Key log excerpts — the lines each conclusion rests on

Full server logs stay in the experiment workspace at
`bench/glm5p2_pd/results/yihou-samenode-p4d4/rounds/<round>/` (gitignored, up to
17 MB each). Only the load-bearing lines are reproduced here.

### Defect 2 — SGLang internal port block overlap (round 6)
source: rounds/006-bringup-1/launch/server-logs/prefill-0.log (full log left in the workspace)
```
2026-09-22T06:49:56.336833841Z [2026-09-22 06:49:56] server_args={'model_path': '/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4', 'tokenizer_path': '/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4', 'tokenizer_mode': 'auto', 'tokenizer_b
2026-09-22T06:50:07.810057661Z     self.init_ipc_channels(port_args, server_args)
2026-09-22T06:50:07.810058421Z   File "/sgl-workspace/sglang/python/sglang/srt/managers/detokenizer_manager.py", line 124, in init_ipc_channels
2026-09-22T06:50:07.810074780Z zmq.error.ZMQError: Address already in use (addr='tcp://127.0.0.1:29236')
2026-09-22T06:50:07.810311167Z [2026-09-22 06:50:07] Received sigquit from a child process. It usually means the child failed.
2026-09-22T06:50:09.784774673Z RuntimeError: sglang subprocess exited with code -9 before reporting ready
```

### Defect 3 — RCCL init race, exact line (round 8)
source: rounds/008-nccl-debug/launch/server-logs/prefill-0.log (full log left in the workspace)
```
2026-09-22T06:57:47.356804930Z crsuse2-m2m-276:717:717 [0] NCCL INFO Check P2P Type isAllDirectP2p 1 directMode 0
2026-09-22T06:57:47.356807990Z [2026-09-22 06:57:47] crsuse2-m2m-276:717:1686 [0] /longer_pathname_so_that_rpms_can_support_packaging_the_debug_info_for_all_os_profiles/src/out/ubuntu-22.04/22.04/build/rccl/hipify/src/transport/p2p.cc:256 N
2026-09-22T06:57:47.356808900Z crsuse2-m2m-276:717:1686 [0] [FATAL ERROR]: HIP failure: 'invalid argument'
2026-09-22T06:57:47.362238205Z     raise RuntimeError(f"NCCL error: {error_str}")
2026-09-22T06:57:47.362239015Z RuntimeError: NCCL error: unhandled cuda error (run with NCCL_DEBUG=INFO for details)
```

### Node-276 segfault (round 9)
source: rounds/009-gpu-swap/launch/server-logs/decode-0.log (full log left in the workspace)
```
2026-09-22T07:13:25.070180166Z Fatal Python error: Segmentation fault
2026-09-22T07:13:25.070280414Z   File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 1894 in run_event_loop
2026-09-22T07:13:25.121356728Z Fatal Python error: Segmentation fault
2026-09-22T07:13:25.121455396Z   File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 1894 in run_event_loop
2026-09-22T07:13:25.244890836Z Fatal Python error: Segmentation fault
2026-09-22T07:13:25.244971605Z   File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 1894 in run_event_loop
2026-09-22T07:13:25.275976124Z Fatal Python error: Segmentation fault
2026-09-22T07:13:25.276104752Z   File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 1894 in run_event_loop
```

### Co-location exonerated — solo leg on 276 still segfaults (round 12)
source: rounds/012-solo-decode/decode-0.log (full log left in the workspace)
```
2026-09-22T07:54:41.168627263Z Fatal Python error: Segmentation fault
2026-09-22T07:54:41.228454386Z Fatal Python error: Segmentation fault
2026-09-22T07:54:41.390889863Z Fatal Python error: Segmentation fault
2026-09-22T07:54:41.477253923Z Fatal Python error: Segmentation fault
```

### JIT cache exonerated — wiped cache on 276 still segfaults (round 13)
source: rounds/013-jitwipe-276/decode-0.log (full log left in the workspace)
```
2026-09-22T08:10:11.620851296Z Fatal Python error: Segmentation fault
2026-09-22T08:10:11.624003867Z Fatal Python error: Segmentation fault
2026-09-22T08:10:11.699736168Z Fatal Python error: Segmentation fault
2026-09-22T08:10:11.755065447Z Fatal Python error: Segmentation fault
```

### Same run on 137 SERVES (round 14)
source: rounds/014-solo-137/decode-0.log (full log left in the workspace)
```
2026-09-22T08:10:24.183700518Z [2026-09-22 08:10:24] End of disaggregation warmup
2026-09-22T08:10:24.281524137Z [2026-09-22 08:10:24] The server is fired up and ready to roll!
```

### RCCL race reappears on 137 with concurrent starts (round 15)
source: rounds/015-samenode-137/launch/server-logs/decode-0.log (full log left in the workspace)
```
2026-09-22T09:05:38.609555296Z     raise RuntimeError(f"NCCL error: {error_str}")
2026-09-22T09:05:38.609556116Z RuntimeError: NCCL error: unhandled cuda error (run with NCCL_DEBUG=INFO for details)
2026-09-22T09:05:46.796588949Z RuntimeError: sglang subprocess exited with code 1 before reporting ready
```

### Staggered start: clean (round 16)
source: rounds/016-stagger-137/decode-0.log (full log left in the workspace)
```
2026-09-22T09:15:24.765279877Z I0922 09:15:24.765240  1361 transfer_engine_impl.cpp:274] Topology discovery complete. Found 1 HCAs.
2026-09-22T09:15:24.769605917Z I0922 09:15:24.769521  1361 transfer_engine_impl.cpp:412] HIP transport installed for intra-node GPU P2P
2026-09-22T09:15:27.343803674Z I0922 09:15:27.343767  1448 transfer_engine_impl.cpp:274] Topology discovery complete. Found 1 HCAs.
2026-09-22T09:15:27.348152024Z I0922 09:15:27.348112  1448 transfer_engine_impl.cpp:412] HIP transport installed for intra-node GPU P2P
2026-09-22T09:15:30.573892865Z I0922 09:15:30.573851  1521 transfer_engine_impl.cpp:274] Topology discovery complete. Found 1 HCAs.
2026-09-22T09:15:30.628454051Z I0922 09:15:30.628422  1521 transfer_engine_impl.cpp:412] HIP transport installed for intra-node GPU P2P
2026-09-22T09:15:33.450384838Z I0922 09:15:33.450330  1589 transfer_engine_impl.cpp:274] Topology discovery complete. Found 1 HCAs.
2026-09-22T09:15:33.514640461Z I0922 09:15:33.514605  1589 transfer_engine_impl.cpp:412] HIP transport installed for intra-node GPU P2P
2026-09-22T09:24:44.836440004Z [2026-09-22 09:24:44] End of disaggregation warmup
2026-09-22T09:24:44.934187539Z [2026-09-22 09:24:44] The server is fired up and ready to roll!
```

### Mooncake exonerated — last mooncake line vs crash time (round 11)
source: rounds/011-mclog-segv/launch/server-logs/decode-0.log (full log left in the workspace)
```
2026-09-22T07:40:42.706666744Z I0922 07:40:42.706610  1414 topology.cpp:127] Device ionic_0 port 1 is available
2026-09-22T07:40:42.706869610Z I0922 07:40:42.706823  1414 transfer_engine_impl.cpp:274] Topology discovery complete. Found 1 HCAs.
2026-09-22T07:40:42.734209889Z I0922 07:40:42.734180  1414 transfer_engine_impl.cpp:412] HIP transport installed for intra-node GPU P2P
2026-09-22T07:40:44.730544773Z I0922 07:40:44.730468  1478 topology.cpp:127] Device ionic_1 port 1 is available
2026-09-22T07:40:44.730696700Z I0922 07:40:44.730643  1478 transfer_engine_impl.cpp:274] Topology discovery complete. Found 1 HCAs.
2026-09-22T07:40:44.734290251Z I0922 07:40:44.734273  1478 transfer_engine_impl.cpp:412] HIP transport installed for intra-node GPU P2P
2026-09-22T07:40:47.087091328Z I0922 07:40:47.087004  1551 topology.cpp:127] Device ionic_2 port 1 is available
2026-09-22T07:40:47.087237856Z I0922 07:40:47.087167  1551 transfer_engine_impl.cpp:274] Topology discovery complete. Found 1 HCAs.
2026-09-22T07:40:47.127742258Z I0922 07:40:47.127709  1551 transfer_engine_impl.cpp:412] HIP transport installed for intra-node GPU P2P
2026-09-22T07:40:50.009824171Z I0922 07:40:50.009750  1625 topology.cpp:127] Device ionic_3 port 1 is available
2026-09-22T07:40:50.009967029Z I0922 07:40:50.009912  1625 transfer_engine_impl.cpp:274] Topology discovery complete. Found 1 HCAs.
2026-09-22T07:40:50.097636983Z I0922 07:40:50.097587  1625 transfer_engine_impl.cpp:412] HIP transport installed for intra-node GPU P2P
2026-09-22T07:44:28.708322040Z Fatal Python error: Segmentation fault
2026-09-22T07:44:28.722961371Z Fatal Python error: Segmentation fault
2026-09-22T07:44:28.881912259Z Fatal Python error: Segmentation fault
2026-09-22T07:44:28.885041828Z Fatal Python error: Segmentation fault
```

