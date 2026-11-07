<!-- Draft only, not opened. Opening the PR is a public action: needs the user's go-ahead.
     Branch: AMD-AGI/Infera feat/sglang-pd-glm5.2-agentx (bf61a7c7), base main 44a294f8. -->

# feat(examples): GLM-5.2 SGLang PD kit with the tuned AgentX recipe

## What

`examples/sglang_pd_glm5.2_agentx/`: a self-contained kit that deploys GLM-5.2 (MXFP4) as SGLang prefill /
decode disaggregation behind the Infera router on MI355X and runs InferenceX AgentX against it.

- Shapes: **4+4** (default; one TP1 x PP4 prefill + one TP4 / DP4-attention decode, 8 GPUs, one node or two),
  **2P1D** (`PREFILL_WORKERS=2`, 8-GPU decode, 16 GPUs), **1P2D** (`DECODE_WORKERS=2`, 12 GPUs).
- Recipe knobs: `MTP_STEPS` (EAGLE K3 / K4 / K5; `sweep.sh` scores K at InferenceX's golden acceptance
  2.99 / 3.33 / 3.61), `DECODE_REQUEST_WEIGHT` (router DP balance), `KF_KERNELS`, `DRAFT_MOE_FP8`.
- `engine/up.sh` checks both images against `patches/BASE_MD5SUMS` before mounting the whole-file patches;
  `engine/smoke.sh`, `engine/eval.sh`, `engine/agentx/{agentx,sweep}.sh` and `summarize.py` cover
  correctness and scoring.
- `pyproject.toml`: ruff skips the example's copies of SGLang files (`patches/`) and the KF kernels
  (`kf_overlay/`), which are kept byte-for-byte as validated.

Commits: f7bb6860 (kit), e6a0d9cd (kit must be readable by dockerd), bdcfa953 (`sweep.sh` under `set -e`),
347b9b38 (`MC_DISABLE_HIP=1`), bf61a7c7 (`MTP_STEPS`, `DECODE_WORKERS=2`).

## Why `MC_DISABLE_HIP=1`

With prefill and decode on one node, Mooncake picks its HIP (XGMI peer copy) transport for the KV, about
1 GB/s per transfer under load: one-node 4+4 C104 scored 40,807 tok/s/GPU against 53,666 on two nodes.
`MC_DISABLE_HIP=1` keeps it on RDMA loopback on the destination's rail: 53,209 (two nodes 53,666, -0.9%).
`MC_DISABLE_HIP_TRANSPORT=1` only works in images built with Infera's `transfer_engine_impl.diff`.

## Results (AUS MI355X, AgentX 3600 s, tok/s/GPU; interactivity = per-user tok/s)

One-node 4+4 Pareto with this kit (v2 image = Infera main 44a294f8):

| point | tok/s/GPU | intvty p50 / p90 | TTFT p90 |
|---|---|---|---|
| DP balance 400k + K5 C32 | 23,530 | 121.6 / 106.8 | 2.39 s |
| DP balance 400k + K5 C48 | 32,819 | 108.0 / 97.8 | 2.76 s |
| DP balance 400k + K5 C64 | 43,245 | 91.3 / 81.6 | 3.92 s |
| DP balance 400k + K3 C104 | 53,460 | 73.0 / 65.4 | 11.27 s |

- DP balance (`DECODE_REQUEST_WEIGHT=400000`) moved every measured point up and right (C32 / C48 / C64 / C104).
- K5 pays up to about C64 (busiest decode rank under ~14-16 running), K3 from C104.
- Two-node 4+4 C104 with the same kit and image: 53,055 (one node 52,409, pkg two-node 53,666).
- 1P2D (12 GPUs): C96 38,094 @ p90 84.3, C128 37,999 @ 78.2; bound by the single PP4 prefill, below 4+4.
- Correctness on one node (`MC_DISABLE_HIP=1`): GSM8K 1,319 0.970, lm-eval 0.9682 / 0.9689, 16k needle 64/64,
  100k needle 16/16, real MTP acceptance 3.62. 1P2D gate: smoke 6/6, probe 193/200, 16k 32/32, acceptance 3.69.

## Known limits

- K5 at C88 aborts the decode (`HSA_STATUS_ERROR_OUT_OF_RESOURCES`, 3/3) with `--mem-fraction-static 0.95`:
  K5's graphs leave 2.1 GB free per decode GPU and the run grows 3.6-4.9 GB beyond the static pools; with
  0.93 it completes. The README says to use K3 there.
- `DRAFT_MOE_FP8=1` (draft MoE in FP8, sglang#45800 backport) is not InferenceX-submittable; the
  submission recipe keeps the draft in the precision it ships in.
- KF MoE kernels cover TP4 only; at TP8 (2P1D decode) the MoE falls back to aiter.

## Test plan

- [x] `up.sh` image check: the 11 replaced files match `BASE_MD5SUMS` in the main 44a294f8 image
- [x] one-node production correctness suite (above)
- [x] AgentX sweeps: one-node 4+4 K3 / K5, DP balance, two-node 4+4, 1P2D
- [x] `ruff check` on the kit's own Python (patches / kf_overlay excluded)
