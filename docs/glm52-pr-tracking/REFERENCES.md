# References and maintenance

## Performance and patch provenance

All archive links below pin commit `b563462fa4a20b4a6f563d279de14ddee455b343` in the public Infera repository. They remain valid when this documentation branch is rebased onto main, which does not contain the experiment pack-ups. PR URLs intentionally point to live upstream discussions.

| Reference | Source | What it establishes |
|---|---|---|
| R1 | [C104 archive](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.llrepro-c104-conc.packup_20261027/README.md) | llr37 48,713; concurrency-only change and correctness caveat |
| R2 | [Final C104 archive](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.llrepro-c104-tail.packup_20261028/README.md) | Three-run mean 48,204; llr51 48,806 unreplicated; llr50/52 not carried |
| R3 | [Raw llr37 result](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.llrepro-c104-conc.packup_20261027/results/llr/llr37-c104/agentx_conc104.json) | Input/output accounting and reported performance metrics |
| R4 | [llr51 site](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.llrepro-c104-tail.packup_20261028/sites/llr51.site.env) | Only decode memory fraction changes from the llr37 site |
| R5 | [K3, FP8 draft and GEMM pack-up](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.llrepro-k3-fp8-gemm-mem.packup_20261027/README.md) | 47.9k baseline, both Aiter CSVs and per-trial outcomes |
| R6 | [Base PP4/DPA4 kit](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.llrepro-pp4-dpa4.packup_20261024/README.md) | Frozen image and S13 model source |
| R7 | [KF kernel investigation](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.llrepro-kf-ar-topk.packup_20261024/README.md) | Historical overlay family; not proof of final #47276/#47278 identity |
| R8 | [Overlap/custom-AR sweep](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.llrepro-garblefree-sweep.packup_20261028/README.md) | Real-acceptance failure context, llr42 alias and rejected trials |
| R9 | [DCP PD experiments](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.dcp4-pd.packup_20261022/README.md) | pdBe-r1, R16–R25 and separate DCP patch evidence |
| R10 | [DCP gather experiments](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.dcpx-pack-fix-best.packup_20261023/README.md) | dcpx8c/9/10 and the metadata/producer-wait evidence boundary |
| R11 | [OffloaderV2 experiments](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.v0521-offloaderv2.packup_20261015/README.md) | ofs0c/ofg3c/ofg2c; distinct from HiCache and the 48k stack |
| R12 | [c44 transfer experiments](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012/README.md) | Metrics, HIP IPC/priority and rejected transfer trials |

## Reproduction inputs and unresolved identities

- [Public reproduction procedure](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.patch-inventory_20261027/PUBLIC_REPRODUCE.md): image build, artifact reconstruction, topology and benchmark commands.
- [Source and image pins](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.patch-inventory_20261027/repro/SOURCE_PINS.json); [artifact checksums](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.patch-inventory_20261027/repro/SHA256SUMS); [source-only kit reconstruction](https://github.com/AMD-AGI/Infera/blob/b563462fa4a20b4a6f563d279de14ddee455b343/yihou/glm52.patch-inventory_20261027/repro/prepare_llr37.py).
- Frozen Infera source: [fdbafb6065](https://github.com/AMD-AGI/Infera/tree/fdbafb6065dae6d1c53936c7d2cf1f5022a514d0); Mooncake [faae8dd4a6](https://github.com/kvcache-ai/Mooncake/tree/faae8dd4a6309c3ecd47e0721a83b0250d686fa2) plus inherited RDMA patches.
- Official SGLang base: `lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261018`; exact registry digest is in SOURCE_PINS.json. The historical integrated image name is a local tag, not a public pull URL. Rebuild from the pinned public sources.
- Public model/trace revisions and retrieval references are in the reproduction guide. Their byte identity to historical local weights/trace files is unverified; record exact hashes in a new run. Current-main kernel ports have separate evidence and do not certify this frozen overlay.

## Daily refresh and team handoff

Owner: yihou until a named delegate accepts a row. The configured maintenance host runs daily at **09:00 UTC / 17:00 Asia/Shanghai**. It discovers new authored PRs, checks repository visibility, refreshes CI/review/heads and republishes these Markdown files to the open tracking PR branch. A timestamp older than 36 hours needs operator investigation. The host and authenticated GitHub/SSH access must remain available.

The public account ledger includes historical work outside the GLM-5.2 core set. Internal repository records are kept in the restricted handoff. Requested reviewers are not assumed assignees. Ready means available for review, not approved; skipped/draft/permission gates do not establish functional-test coverage.

For each substantive change, record the named owner, last action, next actor, exact tested head, commands, run IDs, unresolved review and dependency order. Update the curated notes before refreshing; otherwise generated text would overwrite an unrecorded manual table edit. New PRs and changed heads/activity are flagged for triage. Keep evidence qualification separate from GitHub state.

The maintained source is [handoff_notes.json](https://github.com/AMD-AGI/Infera/blob/dev/pd_opt/glm_5.2_agentx/yihou/glm52.patch-inventory_20261027/maintenance/handoff_notes.json); [maintenance tools](https://github.com/AMD-AGI/Infera/tree/dev/pd_opt/glm_5.2_agentx/yihou/glm52.patch-inventory_20261027/maintenance) regenerate the public export. Machine-specific cron/config/logs stay outside this PR. The experiment workspace and uncommitted pack-ups are not staged by the publisher.

This documentation PR (#228) is permanently Draft by owner instruction: **for view and track only; never mark Ready or merge**. Daily publication updates its branch and verifies the Draft flag before collection and publication, restoring Draft if necessary. It does not promote, merge or edit runtime PRs. If the tracker is closed, preserve artifacts and ask its owner how to resume; the account collector continues daily. Never push generated changes directly to main.

Validation work uses isolated checkouts. Keep `git commit -s` under the configured identity; no assistant/co-author trailers. Allocate GPU checks through the machine pool; do not use the live experiment nodes. Matching archived hardware evidence is reusable, while changed implementations and missing model matrices need specific additional coverage.
