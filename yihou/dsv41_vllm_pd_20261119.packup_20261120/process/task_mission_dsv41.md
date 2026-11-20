# Mission: DeepSeek-V4.1-Flash + MI355X + vLLM — PD disaggregation, max AgentX throughput (+50%)

(User task, 2026-11-19 ~08:00Z. Re-read this whole file every 10 minutes. Work in English; report to the user in Chinese.)

## Background
- We optimise mainstream LLMs on the InferenceX AgentX leaderboard; further optimisation should be based on the vLLM engine.
- Prior projects: GLM-5.2/5.3 on SGLang, big gains (/home/yihou/dev/git/infera.glm52.pd); Muse on SGLang, big gains, now
  moving to vLLM (/home/yihou/dev/git/infera.muse/yihou/muse_vllm_20261118).
- New: DeepSeek V4.1 Flash on MI355X, PD-disaggregated serving, objective (provisional) = max throughput.
- DSv4.1 Flash has released / in-PR mixed performance numbers on both SGLang and vLLM.

## Task phases
1. Collect official DSv4.1 model-structure material (incl. transformers modeling_xxx); learn the model structure.
   Collect ALL existing DS Flash V4/V4.1 performance numbers from InferenceX / sglang / vllm (release, main, PRs);
   analyse, learn, pick the best and reproduce.
2. With phase-1 material: reproduce vLLM performance, profile, analyse bottlenecks, state the possible advantages of PD
   deployment, judge feasibility overall.
3. If PD optimisation is chosen: for the given scenario and engine, using agent_sys skills, knowledge base, handbook
   (pd-agentx-tuning), iterate experiments/analysis. Target: +50% throughput.

## User decisions
- 2026-11-19 ~10:15Z: +50% is relative to the PUBLIC best MI355X number = vLLM colocated DP2+UMBP c384 427,451 tok/s/GPU
  (InferenceX CI 2026-11-18, PR #5036) -> target ~641k tok/s/GPU.
- 2026-11-19 ~10:15Z: Dynamo + vLLM (NVIDIA PD, far ahead of us) must be studied in depth; every transferable point is in scope
  of our experiments.

## Resources
- Machines: ask the pool-manager session `machines-and-clusters-b1`; prefer openai machines; start only on "granted".
- Before work, look for already-downloaded models: found /perf_apps/chaox/models/DeepSeek-V4.1-Flash (476G, NFS,
  readable read-only from openai nodes).

## Rules (MUST, at all times)
### General
1. Inject this mission.md every 10 minutes so it never falls out of context.
2. Work as an agent team.
   a. Leader runs a 20-minute poll to make sure teammates are on the right path and not missing information.
   b. First time a poll finds an issue: only record it; intervene/discuss only if still unresolved at the next poll.
   c. While serving/benching: one watchdog teammate checks service and bench progress every 10 minutes.
3. Strictly follow the user-level CLAUDE.md; any deviation must be asked interactively.
4. Research / gather / analyse first, then plan, then the sub-workspace (all temp experiment activity inside it), then
   CLAUDE.md (back up the old one), then work.
5. Run in docker containers; do not operate the host directly.
6. Use LSP and serena.
7. Never delete any file whose path does not contain the substring "yihou".
8. English for work; Chinese only for reports to the user.
### Task related
1. CUDA-graph capture can take ~30 min; wait, and watch the build/cache dirs to observe progress.
2. Use the existing debug skills (iterative-debug-loop, systematic-debugging, pd-agentx-tuning handbook, ...).
3. After each test round clean owned garbage on disk; create files only under yihou sub-dirs (e.g. /perf_apps/yihou,
   /data/yihou on openai nodes, /shared_nfs/yihou on crusoe); never HOME — so every deletion can be checked for "yihou".

## Our names
Workspace ./ (yihou/dsv41_vllm_pd_20261119 in infera.dsv41), containers yihou-dsv41-*, NFS /perf_apps/yihou/dsv41_vllm_pd_20261119,
node-local /data/yihou/dsv41_vllm_pd_20261119 (openai). Other people's dirs (/perf_apps/chaox, /perf_apps/yanfwang, ...) and
sibling projects (glm52, muse) are READ-ONLY for us.
