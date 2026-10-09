# Working process — synthetic expert8 PD

| Round | Goal | State |
|---|---|---|
| 000-research | Baseline, reference, RDMA GPU byte validation; diagnose NFS TMPDIR | PASS:8 directed transfers40.11–42.44GB/s |
| 001-component | Exact-image8-expert loader/router and client tests | PASS focused filter/route/graph/scaling; full EP4 forward pending |
| 002-bringup | Initial1P1D launch | FAIL: inherited JSON default added extra brace; fixed source order |
| 003-fixed-c32 | 4096/1024 C32,32 warmup +256 measured | PASS256/256 measured,1424.02output tok/s,184.09s |
| 004-json-override | Clean GPU precheck + valid JSON relaunch | PASS physical8/EP4, graphs, single+burst smoke |

## Initial evidence — 2026-09-23
- SSH works; both138/136 GPU0–7 idle approximately284MiB/card.136 has zero-VRAM gpuagent plus monitoring containers.138 no running containers.
- Both have exact image fd7220a57b7d3b58efd875c41f7a9ef46b93469581102d96cbeb6f5451e91d35.
- Selected ionic0–3 GID1 RoCEv2 nonzero; rail prefixes correspond across nodes. Transfer not yet tested.
- User's T2f reference: EP1, MTP+HiCache on; user explicitly approved MTP/HiCache off here. EP4 is a requested change.
- Parent worktree dirty from other sessions; captured initial diff stat. Root CLAUDE backed up, new child CLAUDE written, root untouched.
- LSP documentSymbol succeeded for topology and mooncakeperf. Host LSP reports torch/mooncake imports unavailable, expected host environment; no runtime conclusion follows.
- Serena CLI available; workspace-scoped runtime integration pending.
- All runtime and measurement actions remain pending; no results inferred.
- Original teammate reported local Docker socket failure; resolved route is node-local Docker over SSH136, not login-node Docker. Replacement teammate expert8-implementation is extracting exact image files.

## Round000 completed
- All8 directed GPU/RDMA checks passed byte verification,40.11–42.44GB/s, rc0. Evidence: rounds/000-research/rdma-tmpfs/summary.json.
- HIP first GPU kernel faults with TMPDIR on shared NFS. Native stack+HIP log show failed code-object unbundle. Same minimal torch fill passes on workspace-scoped container tmpfs; all launchers now use that workaround. Exact NFS/COMGR mechanism remains open. Failed initialization-order hypothesis preserved in round README.
- LSP and Serena symbol queries succeeded; Serena initially selected bash-only, child config corrected to include Python.
- Mock HTTP client tests passed measured-token enforcement, discovery shape and bounded concurrency. Not model smoke; model runtime pending.
- Harness shell syntax and effective TP4/DP4/EP4/DPA1, MTP0/HiCache0 validated. No root/shared files changed by this task.
