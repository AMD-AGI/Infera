# Owned PR index for the upstreaming target repositories

Checked: 2026-11-02T04:46:05.537397+00:00. Author filter: `dorado269`. Repositories: `AMD-AGI/Infera`, `sgl-project/sglang`, `kvcache-ai/Mooncake`, `ROCm/aiter`.

**53 authored PRs: 14 open (7 ready / 7 draft), 21 merged, 18 closed unmerged.** This is an author-wide inventory for these four repositories, not an assertion that every entry belongs to the GLM-5.2 patch series. It excludes PRs authored by other people even when we contribute evidence there.

The GLM-5.2 core subset contains **13 PRs: 11 open (6 ready / 5 draft), 1 merged, 1 closed**. Infera #221 is a separate kernel artifact catalog; #164 is comparison tooling; #208 is general agent_sys tooling. Ready is GitHub's non-draft state, not approval or a clean CI result.

The earlier report refreshed only the old tracker and missed the subsequently opened SGLang #47276/#47278 and Infera #221. This author-wide audit corrects that omission. Future maintenance must discover new authored PRs before refreshing the known list.

## Counts

| Repository | Total | Open | Merged | Closed unmerged |
|---|---:|---:|---:|---:|
| AMD-AGI/Infera | 36 | 3 | 21 | 12 |
| sgl-project/sglang | 16 | 10 | 0 | 6 |
| kvcache-ai/Mooncake | 1 | 1 | 0 | 0 |
| ROCm/aiter | 0 | 0 | 0 | 0 |

## Complete authored list

| PR | Purpose / original title | State | Scope |
|---|---|---|---|
| [AMD-AGI/Infera #4](https://github.com/AMD-AGI/Infera/pull/4) |  feat(image): support DeepSeek V4 on MI325X (gfx942) for atom + sglang | MERGED | Historical / other work; see PR scope |
| [AMD-AGI/Infera #7](https://github.com/AMD-AGI/Infera/pull/7) | Upgrade vLLM base image to stable v0.25.1 (digest-pinned) | MERGED | Historical / other work; see PR scope |
| [AMD-AGI/Infera #15](https://github.com/AMD-AGI/Infera/pull/15) | Upgrade SGLang base image to v0.5.15.post1 (official release repo) | MERGED | Historical / other work; see PR scope |
| [AMD-AGI/Infera #18](https://github.com/AMD-AGI/Infera/pull/18) | (WIP)Fix MoRIIO MLA page-length PD corruption on v0.25.1 + GLM-5.1 regression test | CLOSED | Historical / other work; see PR scope |
| [AMD-AGI/Infera #19](https://github.com/AMD-AGI/Infera/pull/19) | feat(sglang): unified Mooncake image — one engine.so, every PD path runtime-decided | MERGED | Historical / other work; see PR scope |
| [AMD-AGI/Infera #25](https://github.com/AMD-AGI/Infera/pull/25) | Feature/Fix: support GLM for all engines mix and vllm moriio pd & adding all to test | MERGED | Historical / other work; see PR scope |
| [AMD-AGI/Infera #29](https://github.com/AMD-AGI/Infera/pull/29) | Yihou.dev.sglang.mooncake.experiment | CLOSED | Historical / other work; see PR scope |
| [AMD-AGI/Infera #31](https://github.com/AMD-AGI/Infera/pull/31) | fix: persist e2e worker logs to shared NFS — and fix the first crash they exposed | MERGED | Historical / other work; see PR scope |
| [AMD-AGI/Infera #33](https://github.com/AMD-AGI/Infera/pull/33) | feat(preflight): mooncake KV-registration mode selector | MERGED | Historical / other work; see PR scope |
| [AMD-AGI/Infera #34](https://github.com/AMD-AGI/Infera/pull/34) | sglang DSA: enable PD + DP-attention + EAGLE MTP on GLM-5.2 / gfx950 (4 patches) | CLOSED | Historical / other work; see PR scope |
| [AMD-AGI/Infera #58](https://github.com/AMD-AGI/Infera/pull/58) | sglang DSA: enable PD + DP-attention + EAGLE MTP on GLM-5.2 / gfx950 (3 patches, rebased onto main) | MERGED | Historical / other work; see PR scope |
| [AMD-AGI/Infera #59](https://github.com/AMD-AGI/Infera/pull/59) | [wip]fix(kvaware+kvd): two bugs blocking KV-aware routing + kvd together on SGLang, with image, docs and repro | CLOSED | Historical / other work; see PR scope |
| [AMD-AGI/Infera #79](https://github.com/AMD-AGI/Infera/pull/79) | fix(glm5.2): the remaining fixes for agentic serving — DSA indexer, ROCm hicache, PD/kvd wiring | MERGED | Historical / other work; see PR scope |
| [AMD-AGI/Infera #84](https://github.com/AMD-AGI/Infera/pull/84) | docs(examples): add a GLM-5.2 1P1D SGLang deployment example | MERGED | Historical / other work; see PR scope |
| [AMD-AGI/Infera #87](https://github.com/AMD-AGI/Infera/pull/87) | docs(sglang): record the MTP penalty-accounting defect we filed but do not patch | CLOSED | Historical / other work; see PR scope |
| [AMD-AGI/Infera #94](https://github.com/AMD-AGI/Infera/pull/94) | chore(examples): drop the GLM-5.2 results/ directory | MERGED | Historical / other work; see PR scope |
| [AMD-AGI/Infera #95](https://github.com/AMD-AGI/Infera/pull/95) | docs(patches): give every patch a schema-checked upstream status record | CLOSED | Historical / other work; see PR scope |
| [AMD-AGI/Infera #123](https://github.com/AMD-AGI/Infera/pull/123) | feat(agent_sys): add env_mgr layered environment manager | MERGED | Historical / other work; see PR scope |
| [AMD-AGI/Infera #124](https://github.com/AMD-AGI/Infera/pull/124) | feat(agent_sys): task_graph — task-management substrate for the agent loop | MERGED | Historical / other work; see PR scope |
| [AMD-AGI/Infera #132](https://github.com/AMD-AGI/Infera/pull/132) | feat(agent_sys): the agent work system — specs, designs, and a running implementation | CLOSED | Historical / other work; see PR scope |
| [AMD-AGI/Infera #142](https://github.com/AMD-AGI/Infera/pull/142) | [draft][agent_sys] task package init | CLOSED | Historical / other work; see PR scope |
| [AMD-AGI/Infera #147](https://github.com/AMD-AGI/Infera/pull/147) | [draft]Dev.yihou.aiopt.kernel opt module | MERGED | Historical / other work; see PR scope |
| [AMD-AGI/Infera #150](https://github.com/AMD-AGI/Infera/pull/150) | Dev.yihou.aiopt.kernel opt module | CLOSED | Historical / other work; see PR scope |
| [AMD-AGI/Infera #151](https://github.com/AMD-AGI/Infera/pull/151) | GLM-5.3 series: engine image, example kits, and parked e2e cases | CLOSED | Historical / other work; see PR scope |
| [AMD-AGI/Infera #154](https://github.com/AMD-AGI/Infera/pull/154) | feat(agent_sys): AgentsView as the o11y panel, one project per run | MERGED | Historical / other work; see PR scope |
| [AMD-AGI/Infera #155](https://github.com/AMD-AGI/Infera/pull/155) | [Draft] agent_sys with agent config | CLOSED | Historical / other work; see PR scope |
| [AMD-AGI/Infera #156](https://github.com/AMD-AGI/Infera/pull/156) | feat(agent_sys): label every runtime directory with the name it alrea… | MERGED | Historical / other work; see PR scope |
| [AMD-AGI/Infera #157](https://github.com/AMD-AGI/Infera/pull/157) | [draft]Dev.yihou.aiopt.task package.concat | CLOSED | Historical / other work; see PR scope |
| [AMD-AGI/Infera #163](https://github.com/AMD-AGI/Infera/pull/163) | GLM-5.3 (glm_moe_dsa): mixed, 1P1D, and a Kubernetes recipe — plus overlay support for SGLang bases | MERGED | Historical / other work; see PR scope |
| [AMD-AGI/Infera #164](https://github.com/AMD-AGI/Infera/pull/164) | [Compare only][Draft]fast bench script | DRAFT | Comparison-only tooling |
| [AMD-AGI/Infera #167](https://github.com/AMD-AGI/Infera/pull/167) | feat(agent_sys): an agent work system, and the five-stage LLM e2e optimisation flow | MERGED | Historical / other work; see PR scope |
| [AMD-AGI/Infera #208](https://github.com/AMD-AGI/Infera/pull/208) | feat(agent_sys): Claude Code skills to create, pack, run and share agent_sys tasks | READY FOR REVIEW | General agent_sys tooling |
| [AMD-AGI/Infera #218](https://github.com/AMD-AGI/Infera/pull/218) | fix: avoid SIGPIPE in Mooncake artifact checks | MERGED | GLM-5.2 core patch |
| [AMD-AGI/Infera #220](https://github.com/AMD-AGI/Infera/pull/220) | agent_sys: add simple kernel artifact catalogs and skill | MERGED | General agent_sys tooling |
| [AMD-AGI/Infera #221](https://github.com/AMD-AGI/Infera/pull/221) | agent_sys: catalog GLM-5.2 KernelForge artifacts and test history | DRAFT | GLM-5.2 kernel artifact catalog (draft) |
| [AMD-AGI/Infera #223](https://github.com/AMD-AGI/Infera/pull/223) | ci(release): build the Rust router once and bundle it in a Linux x86_64 wheel | MERGED | Release packaging |
| [kvcache-ai/Mooncake #5005](https://github.com/kvcache-ai/Mooncake/pull/5005) | [TransferEngine] Add opt-in high-priority HIP copy streams | DRAFT | GLM-5.2 core patch |
| [sgl-project/sglang #33059](https://github.com/sgl-project/sglang/pull/33059) | [Fix] [AMD] dsa: Align Aiter paged rows with DP-attention metadata | DRAFT | GLM-5.2 core patch |
| [sgl-project/sglang #33643](https://github.com/sgl-project/sglang/pull/33643) | [Bug] Accumulate sampling penalties per accepted token under speculative decoding | CLOSED | Historical / other work; see PR scope |
| [sgl-project/sglang #33968](https://github.com/sgl-project/sglang/pull/33968) | [ROCm] Fix HiCache host-pool allocator: hipHostRegister's device pointer is not the host VA | CLOSED | Historical / other work; see PR scope |
| [sgl-project/sglang #33970](https://github.com/sgl-project/sglang/pull/33970) | [PD] Make the mooncake KV transfer wait on the prefill forward that wrote the pages | CLOSED | Historical / other work; see PR scope |
| [sgl-project/sglang #33973](https://github.com/sgl-project/sglang/pull/33973) | [Fix] dsa: Remove host syncs from rank-divergent eager paths | CLOSED | Historical / other work; see PR scope |
| [sgl-project/sglang #45483](https://github.com/sgl-project/sglang/pull/45483) | [Fix] spec: init the active attention backend for eager IDLE batches | CLOSED | Historical / other work; see PR scope |
| [sgl-project/sglang #45484](https://github.com/sgl-project/sglang/pull/45484) | [Fix] [AMD] offloader: keep is_shuffled on device copies of offloaded weights | READY FOR REVIEW | GLM-5.2 core patch |
| [sgl-project/sglang #46819](https://github.com/sgl-project/sglang/pull/46819) | [Perf] [AMD] disaggregation: Cache DCP pack metadata on the gather stream | DRAFT | GLM-5.2 core patch |
| [sgl-project/sglang #46820](https://github.com/sgl-project/sglang/pull/46820) | [Fix] metrics: Avoid blocking the scheduler on KV metrics backpressure | READY FOR REVIEW | GLM-5.2 core patch |
| [sgl-project/sglang #46822](https://github.com/sgl-project/sglang/pull/46822) | [Fix] [AMD] dsa: Skip the indexer on eager idle ranks | READY FOR REVIEW | GLM-5.2 core patch |
| [sgl-project/sglang #46824](https://github.com/sgl-project/sglang/pull/46824) | [Fix] disaggregation: Relayout DCP-sharded DSA draft KV with the target | DRAFT | GLM-5.2 core patch |
| [sgl-project/sglang #46826](https://github.com/sgl-project/sglang/pull/46826) | [Feat] [AMD] disaggregation: Allow DSA DCP decode behind DCP1 prefill | DRAFT | GLM-5.2 core patch |
| [sgl-project/sglang #46828](https://github.com/sgl-project/sglang/pull/46828) | [Perf] [AMD] dsa: Allow retaining GLM kv_b BF16 under DCP | CLOSED | GLM-5.2 core patch |
| [sgl-project/sglang #46830](https://github.com/sgl-project/sglang/pull/46830) | [Fix] dsa: Remove host syncs from rank-divergent eager paths | READY FOR REVIEW | GLM-5.2 core patch |
| [sgl-project/sglang #47276](https://github.com/sgl-project/sglang/pull/47276) | [Perf] [AMD] dsa: Faster gfx950 fused indexer logits and top-k for MTP verify rows | READY FOR REVIEW | GLM-5.2 core patch |
| [sgl-project/sglang #47278](https://github.com/sgl-project/sglang/pull/47278) | [Perf] [AMD] dsa: gfx950 split-K decode plan for 64-head fp8 sparse MLA | READY FOR REVIEW | GLM-5.2 core patch |

See [PR_STATUS.md](PR_STATUS.md) for GLM-5.2 maintenance and blockers, [PR_PACKUP_MAP.md](PR_PACKUP_MAP.md) for experiment mapping, and [PR_SUMMARY_zh-CN.md](PR_SUMMARY_zh-CN.md) for the Chinese handoff. The two new kernel PRs were inspected, not newly tested or opened in this correction. Their performance/correctness results remain attributed to their PR reports.
