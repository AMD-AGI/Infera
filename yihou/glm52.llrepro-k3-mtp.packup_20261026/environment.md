# Environment (2026-10-26)

- Prefill: crsuse2-m2m-135 GPU4-7 (MI355X, gfx950), decode: crsuse2-m2m-136 GPU4-7; Mooncake RDMA over ionic_0..7, MC_GID_INDEX=1.
- Image: `infera-sglang:v0522-fdbafb60`, id `sha256:f69f49e39cbf97e933c742f61d0d9a3c45914e51ab01c99ac4b6eb859858a967`
  (built 2026-10-23; build recipe in `yihou/glm52.llrepro-pp4-dpa4.packup_20261024/image`).
- AgentX: InferenceX commit 918524ff94045b3f091115f1051c22a8588edf2b, dataset semianalysisai cc-traces-weka-062126, CONC=96, 3600 s.
- Model: `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`.
- Offline work: pool node crsuse2-m2m-190 (crusoe OAI, job 242135), same image (copied 10-26).
- Run times: llr25 05:4xZ, llr26 09:1xZ, llr27 10:5xZ, llr28 12:17Z (all 2026-10-26).
- Secrets: none packed; cluster access via spur login (SSH), registry not needed (image loaded locally).
