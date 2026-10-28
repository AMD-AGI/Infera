> Redacted historical snapshot. Measurements and correctness statements refer to the original run, not these published bytes. Revalidate the published source and adapters on the target stack before reuse.

# Environment (2026-10-23 23:20Z .. 2026-10-24 04:18Z)

| run | engines up (UTC) | done (UTC) |
|---|---|---|
| llr5 | 10-23 23:20 | 10-24 00:52 |
| llr6 | 10-24 00:52 | 02:26 |
| llr7 | 02:50 | 04:18 |

- Hardware: compute-node-03 (prefill PP4 x TP1, router, AgentX client) and compute-node-04 (decode TP4 / DP4 DPA, EAGLE
  5/1/6), 8 x MI355X each, GPU4-7 used; RDMA ionic_0..7 registered (`RDMA_DEVICES`), traffic seen on ionic_4..7 only;
  `MC_GID_INDEX=1`. Hardware / driver capture: previous pack-up `environment.md` (same nodes, same week).
- Image: `historical-sglang-image-b` (sglang 0.5.22.dev20261018+g8dee3f6bdb), ID and build recipe in the previous
  pack-up's `image/`.
- Kit: Infera `06796e13` examples/sglang_2p1d_glm5.2 + contributor `config/kit-v6-pp4.diff` + registry hooks (this `kit/`).
- KF overlay: `REDACTED_LOCAL_PATH` as of 10-23 23:20Z (md5 per file in
  `kf_overlay/MD5SUMS.source`), owned by the kernel-optimization session.
- Model: `REDACTED_LOCAL_PATH` (served as glm5.2-mxfp4).
- Bench: AgentX C80, 3600 s, 10 warmup per lane, `SIM_ACCEPT_LEN=3.61`.
[Private access details omitted.]
