# Working process

## R0 — research (2026-10-08)

- Reference packup decode leg: `server-info.29257.json` → TP4/DP4/EP1, EAGLE 5/6/1,
  dsa triton/triton, `speculative_attention_mode=prefill`, `--disable-custom-all-reduce`.
- sglang `a9fb1c3238` `deepseek_v2.py:dispatch_attn_forward_method`: TARGET_VERIFY and
  DRAFT_EXTEND_V2 use the prefill backend when the mode is `prefill` → this is the
  "decode-side prefill". PREBUILT runs no forward.
- Workload from `agentx_conc40.json` and decode-0.log (profiling window 08:50–09:48):
  ISL p50 110,498; OSL p50 390 / mean 987; running-req per DP rank p50 4 (mode 3).
- Tool: decode internal bench (only one with EAGLEWorkerV2). Ported to the image's
  sglang (no-arg initialisers; `get_model().is_startup_weight_load_overlap`), plus an
  opt-in module/attn-backend shape recorder (`bench/shape_capture_yihou.py`).
  Diff: `patches/profile_decode.port-and-shapes.yihou.diff`. Unit tests 60/60 pass.

## R1 — node

- Burst job 200390 landed on 207 with `ReqTRES=cpu=1` and `ROCR_VISIBLE_DEVICES=-1`
  (no GPUs; another tenant on GPU0). Cancelled.
- Resubmitted with `--gres=gpu:mi355x:8`: job **200392** on **crsuse2-m2m-251**,
  `ROCR_VISIBLE_DEVICES=0..7`, all GPUs 0 % at check time.
- `docker load` of the deployment image tar (66.7 GB) from `/shared_nfs`.
- Image loaded; ID reads `291f148a...` (manifest digest under this node's containerd store), but
  **all 72 RootFS DiffIDs match** the reference environment.md → same filesystem. sglang
  `0.5.19.dev20260917+ga9fb1c3238`, AITER `4ad99832`, triton 3.7.0 confirmed in-container.
- Container `glm52-dsp-yihou` (GPUs 4-7, the deployment's decode GPUs), root, env mirrored from
  bench-harness engine.sh/config.sh. Second container `glm52-dsp-g03-yihou` on GPUs 0-3 for
  parallel shape-only runs.

## R2 — smoke (bs16, ISL 110K, graph OFF, 4 steps, shape step 2)

- Passed first try with the port. Artifact present, no Traceback. Weight load 670 s cold (NFS).
- Captured one iteration: draft DECODE x4 (bs4/4 tok) → TARGET_VERIFY (bs4/24 tok) →
  DRAFT_EXTEND_V2 (bs4/24 tok, extend 6 each).
- Findings: MLP/MoE see 96 tokens (24 x 4 DP ranks gathered); topk 9 = 8 routed + 1 fused
  shared expert (257 experts); indexer runs on 21 layers (0-2, 6, 10, ..., 74); other layers take
  `prev_topk_indices`. Verify passes q as fp8 [T,64,576]; draft extend passes q_nope [T,64,512]
  + q_rope [T,64,64] separately.

## R3 — P1 (ISL 110K) and P2 (ISL 240K), graph OFF, profiler steps 8-9, shapes step 12

- Both passed (exit 0, artifacts present). Run concurrently on 4-7 / 0-3 — timings from these
  are NOT used; they are eager-mode shape runs.
- Weight-load straggler again: 3 ranks ~230 s, the 4th ~13 min, others wait in
  `dist_barrier_after_load` (py-spy). Not investigated.
- Kernel→op→shape attribution script `scripts/kernel_shapes.yihou.py`. Triton kernels have no
  host op in the trace → added a `JITFunction.run` recorder to the shape capture for a re-run.
- Note: sglang names the draft-extend forward's span `step[DRAFT ...]`; the enclosing
  `draft_extend` annotation is what separates it.

## R4 — P1g graph ON: graphs captured, never replayed (false "graph on")

- Exit 0, but `graph_execution_counts = {target 0, draft 0, draft_extend 0}` and every step
  `target_graph=False`, although capture logged verify/draft/draft-extend graphs for bs=[4].
- Cause (code read, sglang a9fb1c3238): `decode_cuda_graph_runner.can_run_graph` requires
  `forward_batch.can_run_decode_cuda_graph` under DP/MLP sync; the scheduler sets it in
  `dp_attn.py:_update_gather_batch`. The tool still set the old name `can_run_dp_cuda_graph`.
- Fix: `topology.prepare_dp_metadata` also sets `can_run_decode_cuda_graph`
  (`patches/topology.graph-gate.yihou.diff`). Kept the bad run as `results/p1g_..._NOGRAPH`.
- Relaunched as `p1g2_bs16_isl110k_on`.
- Lesson: check `graph_execution_counts`, not the flag passed.

## R5 — graph ON fixed, Triton/function shape capture, batch variant

- `p1g2`: `graph_execution_counts = 30/30/30`, `target_graph=True` every step. Ranking source.
  Graph-ON trace shows **no kernels for the draft-decode graph** (its hipGraphLaunch correlates
  to 0 kernels); verify and draft-extend graphs are fully visible.
- `p3` (B=4) / `p4` (B=6): Triton launch args + module params. Draft MTP MoE is **bf16**
  (`w13 bf16[257,1024,6144]`), target MoE MXFP4; all attention linears bf16.
- `p5`: wrappers on dsa_indexer helpers → `aiter_paged_mqa_logits` q fp8[24,32,128],
  weights f32[24,32], block table i32[24,1720], out f32[24,110080].
- Deliverable: `analysis/decode_side_prefill_shapes.yihou.md`. Full fork diff:
  `patches/decode-bench.all.yihou.diff`.

## Phase 2 (2026-10-15) — workset export; stopped for handoff

- flashinfer-bench schema read from source (HEAD 15618a67) + dataset (da915083).
- Batch model, workset generator (stdlib), 13 definitions / 80 workloads in `phase2-workset/dataset/`.
- Bench fork: DeviceTimer per-call samples, `--eager-window` (untested).
- Burst job 217099 queued ~25 min then cancelled at the user's request (login-node limits make
  the agent unable to continue). Handoff: `phase2-workset/HANDOFF.md` (TODO-1..5).
