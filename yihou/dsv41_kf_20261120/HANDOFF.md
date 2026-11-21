# Handoff from the PD leader session (2026-11-20) — read fully before planning

## Who is who
- You: kernel teammate, own tmux session `yihou-dsv41-kf`, own workspace (this dir). You are the leader of your own agent team.
- PD leader: Claude session working in yihou/dsv41_vllm_pd_20261119 (same worktree). It owns PD experiments and its own pool requests.
  Report milestones to the user (Chinese) and append a one-line status to ../dsv41_vllm_pd_20261119/coordination/kf_status.md
  (create it; append only — the only file you may write in the leader's workspace).

## Model + deployment to profile (first-hand facts from the leader's runs)
- Model RO: openai `/perf_apps/chaox/models/DeepSeek-V4.1-Flash` (476 G, NFS); crusoe `/shared_nfs/models/DeepSeek-V4.1-Flash`.
  DeepseekV41ForCausalLM, 40 layers, hidden_size 5120, hc_mult 4, hc_sinkhorn_iters 20 (config.json, verified), 384 experts top-6 FP4,
  compressed + SWA attention, DSA indexer, Engram host-offloaded, DSpark 5 drafts.
- Image: `vllm/vllm-openai-rocm:nightly-60bd6dd2edd175a1c55fd3488f871e4c3d2ef27d@sha256:67a3a0fb79893fc5a0fcff5f9b739ba8fec7e920875a12e0574c2e9f50cd814c`
  (vllm 0.34.0rc1.dev571, aiter 0.1.29) = InferenceX CI image.
- Recipe R2 (the "~400k" colocated point): config `../dsv41_vllm_pd_20261119/configs/r2-dp2-umbp-c384.env`, DP2 on 2 GPUs + UMBP host tier,
  vllm-router 0.1.14, AgentX c384, synthetic acceptance 3.51. Measured 407,049 tok/s/GPU (run e2-r2-c384-11191138). R1 (DP2, no UMBP, c128)
  is the simpler variant (256-267k) if UMBP gets in the way of profiling.
- Kit (launch/bench): `../dsv41_vllm_pd_20261119/scripts/kit/` — serve.sh, run_point.sh, stage.sh, launch_remote.sh, stop.sh; how-to in
  `../dsv41_vllm_pd_20261119.packup_20261120/REPRODUCE.md`, gotchas in its `notes.md` (each cost a run: HOME NFS not mountable by docker on openai,
  VLLM_ENGINE_READY_TIMEOUT_S, local TOKENIZER dir, --ulimit nofile, owner-only kit dir). Copy what you need into your workspace / your NFS stage
  dir; do not edit the leader's kit in place.
- AgentX client: venv `/perf_apps/yihou/dsv41_vllm_pd_20261119/venv-agentx`, HF cache `/perf_apps/yihou/dsv41_vllm_pd_20261119/hf_home`
  (dataset cc-traces-weka-062126 @23f152f6), InferenceX stage `/perf_apps/yihou/dsv41_vllm_pd_20261119/stage/inferencex-e2e` — read-only reuse OK.

## Method (what "Muse latest profiling + op capture + KernelForge" points to — verify, these are pointers)
- agent_sys: `/home/yihou/dev/git.16-19/infera/agent_sys/examples/02b_profiling_qwen36`, `03_kernelforge_one_kernel`,
  `llm_e2e_performance_optimization/e2e-flow` (assets/build_workset.task = operator workset capture).
- Skills: infera-agent-sys -> agent-sys-prior-work (knowledge repo handoffs for Muse profiling / worksets), agent-sys-kernel-forge
  (Hyperloom campaign; scripts/workset/*), pd-agentx-tuning references/kernel-profiling-and-optimization.md, kernel-artifacts.
- Muse project (latest, READ-ONLY): `/home/yihou/dev/git/infera.muse/yihou/muse_agentx_opt_20261027/` (kf2/, research/kf-prep/brief.md,
  PROGRESS_REPORT_20261118.md) and `/home/yihou/dev/git/infera.muse/yihou/muse_vllm_20261118/` (Muse on vLLM). Find the newest method
  there first-hand (grep "profil", "workset", "kf"), and check the knowledge repo for newer Muse handoffs.

## mHC prior work (research/aklr_mhc.md in the leader's workspace; first-hand GitHub reads by a teammate, key facts re-verified)
- AMD-AGI/AKLR main has NO mHC kernel for hidden 5120 (only 4096 and 7168). Open PR #88 (zhaoan12-prc): DSv4 mHC boundary + RMSNorm
  for hidden 5120, Triton (GEAK), MI355X, bf16/MXFP8 + DSpark draft boundary; validator 1.02-1.48x; sglang 0.5.23 routes only, no vLLM.
  Merged #271 lets AKLR serve aiter mhc_pre / mhc_fused_post_pre via aiter plugins. Unknown: whether our vLLM image calls aiter mHC entry
  points at all (needs a code read of vllm/model_executor/models/deepseek_v41* in the image).
- llying observed (second-hand) that vLLM's mHC in-graph multi-stream fork matters for the DP4 CUDA-graph stall
  (research/colleagues/llying_alignment_20261120.md) — be aware when changing mHC code paths under CUDA graphs.

## Machines
- Pool manager session `machines-and-clusters-b1`; skill machine-pool; `P=~/.claude/skills/machine-pool/scripts/pool.py`.
  File YOUR OWN openai request (kind batch, count 1, --hours as needed, --notify <your session name from ListAgents>).
  Start only on "granted". openai: ssh to the node only while holding the job; scratch /data/yihou/...; NFS /perf_apps/yihou/...
  The leader holds openai request r-1119160508-bd2b (x2, pending) and crusoe r-1120164431-f35f — do not touch them.
- openai preemption requeues to another node: keep state on NFS, make runs restartable.

## Secrets
- Hyperloom API key: file `.secrets/hyperloom_api_key` in this dir (mode 0600, git-excluded). Read it into the env var the Forge tooling
  expects at launch; never print it, never write it into logs, configs, commits or packups.

## Deliverables
- `results/hot_ops.md`: hot-op table (decode / prefill), method, run ids, evidence level; explicit answer "is mHC hot?".
- If hot: `workset/` (mHC shapes from the live run) + Forge campaign status/results; integration plan for vLLM.
- Commits: `git commit -s`, English, short; add only your workspace files (never .secrets/).

## REVISION 2026-11-20 ~09:45Z (user): profile a PD deployment, preferably Infera + vLLM; optimise mHC on prefill AND decode
- Reference Infera + vLLM PD stack (colleague llying/liyingli, second-hand, READ-ONLY): workdir
  `/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/` (scripts/, configs/, infera-overlay/, vllm-overlay/, holder/, logs/);
  a git copy of his record (README.zh-CN.md, B300-ANALYSIS.zh-CN.md, configs, scripts) is in
  `../dsv41_vllm_pd_20261119/research/colleagues/llying_20261120/llying/vllm-dsv41-pd-20261119/`; summary in
  `../dsv41_vllm_pd_20261119/research/colleagues/llying_alignment_20261120.md`. Copy what you need into your workspace.
- His stack: image vllm nightly-9c1d45c2 (dev897), Infera 0.3.3 (`pip install amd-infera==0.3.3 --no-deps`) + his overlay, Mooncake
  with per-GPU NIC pinning (MC_TOPO=pergpu), prefill DP2 (TP1/rank) + decode DP4 on one 8-GPU node (2P1D) or 1P1D.
  Required fixes he found: Mooncake STOPPED-prefill KV send (vllm-overlay / llying-001/vllm#3), ZMQ MAX_SOCKETS (mooncake_connector_v2.diff).
- Known blocker (his runs): decode DP4 + golden synthetic acceptance + CUDA graphs stalls in AgentX warmup; eager decode works (43x,
  291,400 tok/s/GPU) and decode TP4 with graphs works (43t, 141,900). Profiling eager decode != graph-mode decode timing: state which
  mode each profile used; prefer a graph-mode decode (e.g. TP4) for decode kernel shares, or wait for a stall fix from llying.
- Ownership: llying owns the PD bring-up and the DP4 stall; you only need a working PD stack to profile. Do not change his files.
- Our PD kit (vllm-router based) is the fallback: `../dsv41_vllm_pd_20261119/scripts/kit/serve_pd.sh` + configs/e7*.env (MC_STOPPED_FIX=1),
  not yet run successfully (crusoe crash "Triton Error [HIP] Code 209 no kernel image" at kernel warmup, cause unknown).
