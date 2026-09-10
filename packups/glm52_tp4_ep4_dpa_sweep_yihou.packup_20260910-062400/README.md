# TP4/EP4 DPA decode sweep reproduction kit

**Ran:** 2026-09-10. **Packaged:** 2026-09-10. **Status:** all ten full points and two smokes passed their execution/count checks. Source unchanged; complete logs gzip-compressed with user approval. No GPU rerun during packaging.

## Goal
Sweep TP4/EP4, global concurrency4/8/16/20/24, DPAoff/on, ISL70000, OSL10000, expected bonus-inclusive acceptance3.61. Real target/draft compute, real weights/routing, physical synthetic KV, no Scheduler/PD. Original task is preserved in [history/CLAUDE.md](history/CLAUDE.md), with [plan](evidence/plan.md) and full [process log](evidence/working_process.md).

| Concurrency | DPA off TPOT ms | DPA on TPOT ms | Off output tok/s | On output tok/s |
|---:|---:|---:|---:|---:|
|4|6.150486|7.199074|650.355|555.627|
|8|7.999084|8.870461|1000.115|901.870|
|16|10.286111|10.980857|1555.496|1457.081|
|20|11.667059|11.807833|1714.228|1693.791|
|24|12.632195|12.276834|1899.907|1954.901|

All points emitted exactlyC*10000 useful tokens; all four ranks executed2768 target/draft/draft-extension graphs. Realized acceptance3.613439306. Full single-point wall time153–217s; ten-point sum1850s (~30min50s), excluding setup/smokes. First cold smoke891s.

**Important:** DPAon uses64Qheads, unsupported by this pinned FlyDSL sparse MLA path, so **all DPAon cases fall back**. DPAoffC20/C24 verify also falls back because120/144rows exceed96. Thus these compare the fixed stack's dispatch/fallback behavior, not pure DPA under an identical attention kernel. TPOT is average complete internal-loop cost, not client ITL/P50/P90. Synthetic acceptance is not correctness-valid output generation.

## Contents
- [REPRODUCE.md](REPRODUCE.md): fresh output directory, explicit existing allocation/container, exact image/model validation, smoke+sweep commands.
- [environment.md](environment.md): pins, hardware evidence and external inputs/gaps.
- [evidence/report.md](evidence/report.md): complete original result report; [summary.csv](evidence/summary.csv): machine-readable table.
- [evidence/iterations](evidence/iterations): ten full points, two smokes and setup logs; full rank/step outputs, config, source snapshots, code diffs and timing.
- [scripts/original](scripts/original): exact runnable bench, scripts and tests; no dependency on old scratch source.
- [notes.md](notes.md), [patches/README.md](patches/README.md): semantics, failures and changes.
- [provenance/source-files.json](provenance/source-files.json): original paths, hashes, byte counts, mtimes; Git HEAD/branch/diff also retained.
- [audit.md](audit.md): offline verification; MANIFEST.sha256: complete delivered-byte checksums.

Image archive and model weights are deliberately external and not copied. No copied source file exceeds4MB. Generated Python caches are excluded; all source, result JSON/JSONL and original log streams retained. No commit/push was requested or performed.
