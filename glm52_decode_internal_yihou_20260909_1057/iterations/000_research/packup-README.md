# GLM-5.2-MXFP4 mix: completed experiment reproduction kit

**Experiment:** 2026-09-08; historical close-out/correction: 2026-09-09 UTC.  
**Packaged:** 2026-09-09, from immutable-on-read source artifacts.  
**Scope:** real-verification correctness plus ONE TP8 / EP1 / DP1, concurrency-1 AgentX simulated-acceptance performance point. No PD/decode-only experiment is included.

## Result and interpretation

The aggregated server answered the curl probe with Jupiter, produced coherent three-turn chat output, and completed all 1,319 GSM8K questions, 5-shot, with `ACC_LEN=` (real verification). GSM8K exact match is **0.9681576952**, reported standard error **0.0048363486**, identical for strict-match and flexible-extract.

The separate performance run used **simulated acceptance length 3.61**. These measurements are **not correctness-valid output throughput or production throughput**. This is one run versus one baseline run; numerical closeness does **not** establish statistical equivalence or significance.

| Metric | 2026-09-07 baseline | 2026-09-08 run | Observed delta |
|---|---:|---:|---:|
| Total tokens/s/chip, input + output | 2172.7358 | 2184.4203 | +0.538% |
| P90 interactivity, tokens/s/user | 261.0966 | 261.7801 | +0.262% |
| Output tokens/s/chip | 16.3256 | 16.4135 | +0.538% |
| Median ITL, seconds | 0.00368 | 0.00363 | -1.359% |
| Median TTFT, seconds | 0.85274 | 0.85016 | -0.303% |
| Profiled / total / errored records | 249 / 260 / 0 | 249 / 260 / 0 | Same counts |

**Original success criteria:** preserved verbatim in [the original task](CLAUDE.glm52-mix.20260909-1007.md.bak), first section. Deployment and requested checks completed; the instruction “sane accuracy” supplied no numerical pass threshold, so no retrospective threshold is invented. The requested comparison is available above. No formal tolerance or statistical-equivalence criterion was specified.

## Start here

- [REPRODUCE.md](REPRODUCE.md): ordered commands, original and explicitly labeled bootstrap adaptations.
- [environment.md](environment.md): actual pins, historical environment evidence and missing snapshots.
- [notes.md](notes.md): corrections, operational caveats, evidence limitations.
- [audit.md](audit.md): offline self-audit and coverage, **not a cold rerun**.
- [patches/README.md](patches/README.md): existing fixes and optional packaging-only pinning adaptation.
- [results/perf/results.csv](results/perf/results.csv): canonical one-point comparison data.
- [results/correctness](results/correctness): complete evaluation outputs and lossless compressed per-sample evidence.
- [scripts/repo](scripts/repo): original tracked launcher/build/bench/eval/collection source, copied verbatim. Its older result directories are upstream reference material, not additional experiments run here.
- [dependencies](dependencies): complete tracked InferenceX/aiperf snapshots, pinned bootstrap and dataset revision evidence.
- [history/work.checkpoint.summary.md](history/work.checkpoint.summary.md): complete append-only historical log including final correction. Historical prose contains unsupported claims; **read notes.md before relying on it**.
- [logs/original](logs/original): full original workspace logs, lossless gzip; intentional original-log duplicates are preserved, but the superseded partial result tree is not.
- [provenance/source-files.json](provenance/source-files.json): source-to-package mapping, original hashes, sizes and timestamps.
- [MANIFEST.sha256](MANIFEST.sha256): delivered file checksums.

## Excluded dependencies and gaps

The 23,912,216,852-byte Docker archive and model weights are **not copied**. The archive's absolute path, actual rechecked checksum and pinned build alternative are in environment.md/REPRODUCE.md. Dataset contents are not copied; exact revisions and retrieval commands are included. No credentials are included.

Historical GPU kernel-driver/fabric details, host Docker backing-store proof, full model-weight identity hashes, exact chat input text and a complete wheel-hash-locked runtime remain unavailable. Archive load was validated later on2026-09-09 with matching identity ([separate addendum](provenance/later-image-validation.md)); a cold rebuild/rerun and statistical equivalence have not been verified. The old source directory was not edited, moved or deleted.
