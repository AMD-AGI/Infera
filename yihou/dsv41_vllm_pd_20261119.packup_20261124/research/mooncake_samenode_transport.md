# Mooncake same-node P->D transport on ROCm (2026-11-21) — facts + open questions

Image Mooncake: mooncake-transfer-engine-rocm 0.3.14 (tag v0.3.14 = e1ea5d31, 2026-11-03). Upstream main is 43 commits ahead
(e.g. a6a93b41 #4953 "HipTransport: submit copies asynchronously, one stream and event per device", 748af1a5 #3548 "Register HIP
IPC memory at allocation granularity"). Source snippets: source/mooncake_snip/{multi_transport.cpp (main), hip_transport.cpp (main),
hip_transport.v0314.cpp}.

Code facts (read):
- multi_transport.cpp (main) selectTransport: for a segment registered under several protocols, priority hip=4 unless
  MC_DISABLE_HIP is set (-> 0); hip is used only if the target is local-IPC-reachable (same host), else falls back to rdma/tcp.
- hip_transport (v0.3.14 ~l.460-474 and main ~l.570-588): sets the device context from the SOURCE pointer, then
  relocateSharedMemoryAddress() -> openIPCHandle() -> hipIpcOpenMemHandle(..., hipIpcMemLazyEnablePeerAccess); on failure the submit
  returns Status::Memory("device memory not registered") — exactly the logged chain.

Runs (first-hand):
| run | node | transport setup | result |
|---|---|---|---|
| e7-pd-c256-11210342 | openai n06-25 | RDMA container, HIP allowed | 254,318 tok/s/GPU; 0 IPC errors logged |
| e7-pd-c512-11210519 | openai n06-25 | RDMA container, HIP allowed | warmup: 99x hipIpcOpenMemHandle 201 -> stopped |
| e7h-pd-c512-11210548 | openai n06-25 | RDMA + MC_DISABLE_HIP=1 | 0 failures until preempted |
| e7th-c256-11211349 | crusoe v3-200 | TCP + MC_DISABLE_HIP=1 | no KV moved: tcp "no status frame within 30 s", "Local segment descriptor not found" |
| e7u-c256-11211500 | crusoe v3-200 | TCP, HIP allowed | 16.6M tok reached D, then 62x hipIpcOpenMemHandle 201 -> D engine exception |
| e7s* RDMA container | crusoe v3-186/v3-129 | RDMA, any | Triton Code 209 at kernel warmup (no ib_peer_mem on crusoe nodes) |

Open (not established): why hipIpcOpenMemHandle intermittently fails with 201 "invalid device context" (P containers only see their
own 2 GPUs; whether IPC to a GPU not visible in the importing process is the issue is untested); why Mooncake TCP + MC_DISABLE_HIP
does not move device memory. Candidate tests: (1) PD with all 4 GPUs visible to both roles (GPU_SELECT=device-ids within one
container set); (2) Mooncake main (newer HipTransport) wheel; (3) cross-node PD on two crusoe nodes (no IPC) once RDMA crash is fixed.

Update 15:5xZ (code read): our kit runs engines with GPU_SELECT=device-ids — serve.sh passes the whole /dev/dri + /dev/kfd and sets no
HIP_VISIBLE_DEVICES (serve.sh l.285/303); vLLM main source read finds --device-ids -> ParallelConfig.assigned_physical_gpu_ids with
no HIP/CUDA_VISIBLE_DEVICES export in the paths grepped. So each P/D process most likely sees all 8 GPUs, which makes candidate
test (1) "GPU visibility" unlikely to be the cause (not verified on the 60bd image; a /proc/<pid>/environ check on a live P worker
would settle it). Next candidates: (2) newer Mooncake HipTransport (main, 43 commits ahead incl. #4953/#3548); (3) cross-node PD.
