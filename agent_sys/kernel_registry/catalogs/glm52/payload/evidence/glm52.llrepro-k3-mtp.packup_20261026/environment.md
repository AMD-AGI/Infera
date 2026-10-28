> Redacted historical snapshot. Measurements and correctness statements refer to the original run, not these published bytes. Revalidate the published source and adapters on the target stack before reuse.

# Environment (2026-10-26)

- Prefill: compute-node-03 GPU4-7 (MI355X, gfx950), decode: compute-node-04 GPU4-7; Mooncake RDMA over ionic_0..7, MC_GID_INDEX=1.
- Image: `historical-sglang-image-b`, id `sha256:f69f49e39cbf97e933c742f61d0d9a3c45914e51ab01c99ac4b6eb859858a967`
  (built 2026-10-23; build recipe in `registry/glm52.llrepro-pp4-dpa4.packup_20261024/image`).
- AgentX: InferenceX commit 918524ff94045b3f091115f1051c22a8588edf2b, dataset semianalysisai cc-traces-weka-062126, CONC=96, 3600 s.
- Model: `REDACTED_LOCAL_PATH`.
- Offline work: pool node compute-node-06 (cluster cluster, job REDACTED), same image (copied 10-26).
- Run times: llr25 05:4xZ, llr26 09:1xZ, llr27 10:5xZ, llr28 12:17Z (all 2026-10-26).
[Private access details omitted.]
