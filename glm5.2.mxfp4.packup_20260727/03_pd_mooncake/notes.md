# Notes — 03 PD mooncake (honest failure analysis)

## What failed and why

**conc=64 over mooncake TCP: only 50/256 requests succeeded.** Both engines stayed alive
(health=200) and decode generated fine (TPOT ~13 ms), but under concurrent load the prefill log
floods with:

    KVTransferError(bootstrap_room=...): Decode instance could be dead,
    remote mooncake session 10.2.122.10:<port> is not alive

i.e. the **mooncake TCP KV-transfer sessions drop under concurrency**. Median TTFT ballooned to
~51 s (the KV handoff, not the compute). This is a fixed-per-request TCP transfer/bootstrap cost
that collapses at conc=64. Single requests and the temp=0 probe (low concurrency) work correctly —
so the *model + PD wiring are correct*; only the *TCP transport throughput* fails.

## Why we were on TCP at all — the RDMA driver dead-end (from the reference program)

mooncake's intended cross-node path is RDMA. On this stack it does not work, for two stacked reasons
(documented by the reference program `/mnt/vast/jiejing/crusoe_glm_52/PD_disaggregation.md`, on its
nodes; we did NOT independently re-verify on ours — see "Honest boundary" below):

1. **Transport mis-selection.** mooncake's `multi_transport.cpp` hardcodes `protocol_priority`
   hip > cxl > rdma > tcp, so for a cross-node send it picks intra-node **HIP-IPC**, which fails
   cross-node with `hipIpcOpenMemHandle Error 17/201`. There is no env to disable hip (compiled
   `#ifdef USE_HIP`); the reference patched the .so to flip priority.
2. **GPU-direct DMA fault.** With RDMA forced, the GPU KV buffer registration path dead-ends:
   - dma_buf path (`ibv_reg_dmabuf_mr`) needs kernel `CONFIG_PCI_P2PDMA` + `CONFIG_DMABUF_MOVE_NOTIFY`;
   - peermem path (`ibv_reg_mr` via amdgpu `ib_peer_mem`) registers + warms up 200-OK but the first
     REAL transfer SIGKILLs the prefill (amdgpu-peermem GPU-direct instability over ionic).
   → the only correct-output path left is `MC_FORCE_TCP=1`, which is too slow for conc.

## Honest boundary — what WE tested vs. inherited

- We ran **only the TCP path** on chi2878/chi2879 (correct output, conc=64 fail — reproduced here).
- We did **NOT** independently run the mooncake RDMA path on our nodes. We took the RDMA dead-end
  as established by the reference program and went straight to TCP per the mission owner's decision
  ("先做 mori，mooncake 最后议").
- **Caveat worth flagging:** our nodes actually report `CONFIG_PCI_P2PDMA=y` AND
  `CONFIG_DMABUF_MOVE_NOTIFY=y` (see ../environment.md) — which is the *opposite* of the reference's
  chi2866 (kernel-136, lacked P2PDMA). So it is POSSIBLE mooncake RDMA behaves differently on
  chi2878/chi2879. We did not test it. If someone wants to push mooncake to a conc=64 pass, the
  RDMA path on these specific P2PDMA-enabled nodes is the untested lever — start there, not TCP.
  (`engine.sh MODE=rdma` carries the scaffolding: it mounts the reference's protocol-priority-patched
  `mooncake_engine_rdmaprio.so` — that 67 MB .so is NOT copied into this packup; it lives at
  `/mnt/vast/jiejing/crusoe_glm_52/patched/mooncake_engine_rdmaprio.so`.)

## Reproducible takeaway

For a working GLM-5.2 PD on this cluster **today**, use **MoRI RDMA (02_pd_mori)** — it passes
conc=64. mooncake is correct-but-throughput-blocked here.

## Logs

`logs/prefill.log` — grep `KVTransferError` for the session-death flood; `logs/decode.log` — grep
`TcpTransport: listen` to confirm TCP was actually selected, and `Decode batch` for the working
(but starved) decode.
