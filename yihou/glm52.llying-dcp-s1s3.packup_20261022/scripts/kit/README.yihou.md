# llkit: llying's best config adapted for same-node P4D4 (stages S1/S2/S3), dcp_pd 2026-10-22

Plan of record: `../design/llying_dcp_plan.md`. One 30-min AgentX C80 run per stage (user directive).

- `orig/` = Infera `origin/feat/sglang-mix-glm5.2-tp4-v0522` @ `d395c3b3`, `examples/sglang_2p1d_glm5.2` (git archive).
- `kit-v6-pp4.nodeepseek.diff` = llying `config/kit-v6-pp4.diff` without its `patches/deepseek_v2.py` part
  (the PP fix is a commit on our branch instead). Applied with `patch -p2`.
- NOTE: kitedit originally set PREFILL_TP=1 in agentx.sh metadata; InferenceX divides by prefill TP + decode TP, so S1/S2 per_gpu values are over 5 GPUs -- use total/8. Reverted to PREFILL_TP=4 (llying convention) after S1/S2.
- `kitedit.yihou.py` = the plan section 1 edits (each anchored, asserted once):
  DECODE_SSH=local; MC_TE_FILTERS_OVERRIDE; DISAGG_IB_DEVICE per-GPU map; custom AR ON (drop
  --disable-custom-all-reduce); INDEX_SHARE (default true); DECODE_LAYOUT dpa4|dcp4 (dcp4 = TP4 EP4 DCP4, mem 0.85,
  graph/max-running 96, SGLANG_DP_USE_GATHERV=0); PREFILL_ENV/DECODE_ENV extra env; decode 30,000-pid burn;
  llying's deepseek_v2.py mount removed, SGLANG_OVERLAY mounted ro on both legs; --init on engine/router/client
  containers (job-bound guard); down.sh stops and saves logs, never removes.
- `kit/` = result (orig + diff + edits; env.sh section 5 adds the knob defaults).
- `stages/S{1,2,3}.env` = per-stage site + knobs (needs NK, NODE_IP). Containers `yihou-dcp-pd-<NK>-s<stage>-*`.
- `render/` = argv rendered with stubbed docker/ip (`render/render.yihou.sh <s>`); S1->S2 differs only in the decode
  layout/env, S2->S3 only in the prefill offloader flags.
- `run_stage.yihou.sh <s>` = node-side driver: preflight -> up.sh -> (smoke if sim off) -> agentx.sh -> down.sh if DOWN=1.
Overlay: /shared_nfs/yihou_glm52/dcp_pd_20261021/src/sglang-ab7a898b77/python/sglang (built by benchrun).
Image: infera-sglang:v0522-d395c3b3-1020-mc4340 (built by benchrun; IMAGE= overrides).
