# TASK (2026-11-19) — DeepSeek-V4.1-Flash + MI355X + vLLM: PD disaggregation for max AgentX throughput (+50%)

Workspace `yihou/dsv41_vllm_pd_20261119/` (research/ notes/ scripts/ configs/ results/ coordination/ debug/ source/).
Mission (re-read every 10 min): `yihou/dsv41_vllm_pd_20261119/mission.md`. Phase-1 synthesis + Phase-2 plan:
`yihou/dsv41_vllm_pd_20261119/research/SYNTHESIS.md` (sections 0, 3, 5).

1. Phase 1 (DONE 11-19 08:40Z): model structure + all DSv4/4.1 Flash numbers (InferenceX/vLLM/SGLang/colleagues).
2. Phase 2 (NOW): reproduce vLLM colocated recipes (R4 TP2 c32 107,989 -> R1 DP2 c128 271,329 -> R2 DP2+UMBP c384 427,451,
   InferenceX CI), profile, measure connector-forced kernel-path cost (E4), KV facts (E5), decode/prefill curves (E6),
   1P1D bring-up (E7, Mooncake first), go/no-go PD vs colocated at equal GPUs (E8).
3. Phase 3 (if PD chosen): iterate with the pd-agentx-tuning handbook to +50% over the PUBLIC best 427,451 -> ~641k tok/s/GPU
   (user 11-19). Dynamo+vLLM NVIDIA PD recipes: absorb every transferable point (research/dynamo_vllm.md).

## Background / key facts (first-hand unless noted; details in research/*.md)
- Model RO: `/perf_apps/chaox/models/DeepSeek-V4.1-Flash` (NFS, 476G; DeepseekV41ForCausalLM, 40 layers CED, 384E top6 FP4,
  compressed + SWA attention, DSA indexer, Engram host-offloaded, DSpark 5 drafts). Weights 285.5 GiB excl. Engram -> >=2 GPUs/instance.
- Best MI355X: vLLM colocated DP2+UMBP c384 427,451 tok/s/GPU (p90i 15.8). No vLLM PD of DSv4.1 on any AMD GPU exists.
- PD blockers: MoRIIO rejects DSv4.1 on main (vllm#68513 draft); any non-UMBP connector disables AITER sparse-MLA fast path
  (rocm.py:80-109, `only_restores_canonical_kv`); Mooncake on ROCm needs a custom build (recipes#1262). KV per request is small
  (0.29-0.43 GB at ISL 173k) -> wire not expected to bind.
- AgentX contract: agentx-harness 1.0.14, cc-traces-weka-062126@23f152f6, 3600 s, 10 warmups/lane, seed 42, fresh server per
  point, synthetic acceptance 3.51 (DSpark 5), metric total tok/s / (all P+D GPUs).

## Machines
Pool manager session `machines-and-clusters-b1`; request r-1119160508-bd2b (openai batch x2, 12 h, notify infera-dsv41-0a).
Start only on "granted". openai: scratch `/data/yihou/dsv41_vllm_pd_20261119`, NFS `/perf_apps/yihou/dsv41_vllm_pd_20261119`;
preemption requeues to another node -> keep state on NFS, restartable runs. Release when done; never scancel pool jobs.

## Core principles
- Agent team: leader 20-min poll (coordination/team-polls.md; record first, intervene on second sighting); watchdog teammate every
  10 min while serving/benching.
- One variable per scored run; 3600 s official for decisions; prove every knob from the live process.
- Docker only, containers `yihou-dsv41-*`; one AITER JIT dir per server; graph capture can take ~30 min.
- Never delete paths without "yihou"; no recursive delete with a variable target; others' dirs (/perf_apps/chaox, yanfwang, ...) and
  sibling projects are READ-ONLY. Suspend, don't conclude. English in files, Chinese to the user.
- Commits: `git commit -s`.
