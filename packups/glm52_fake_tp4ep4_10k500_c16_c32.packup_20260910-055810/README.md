# TP4/EP4 fake decode: 10000 input / 500 output, c16 and c32

**Ran:** 2026-09-10, node `crsuse2-m2m-036`, job130891, GPUs0–3 (four MI355X). **Status:** both requested points completed; synthetic execution only, not real-PD correctness or production performance.

The task was to repeat the established fake-decode/simulated-acceptance setup at TP4/EP4 and exact ISL10000/OSL500, at concurrency16 and32. Preserve EAGLE5 steps / 6 draft tokens / topk1, simulated acceptance3.61 (`match-expected`, `real-draft-token`), 16 warmups then128 measured requests per point, and report actual lengths, successes, occupancy and backend fallback. Memory fraction was0.85, FP8 KV, HiCache off, TP4/EP4/DP1. Original task and work history are verbatim in `history/CLAUDE.md` and `history/working_process.md`; no separate originating spec file existed.

| Observation | c16 | c32 |
|---|---:|---:|
| Success / measured requests |128/128|128/128|
| Errors |0|0|
| Every request input/output tokens |10000/500|10000/500|
| Measured duration, seconds |35.493531|24.911468|
| Output tok/s, four-GPU instance |1803.145482|2569.097920|
| Output tok/s/GPU (instance rate / 4) |450.786370|642.274480|
| Reported P50 TPOT, ms |8.775409|11.632978|
| Reported P90 TPOT, ms |8.911237|13.871815|
| Observed acceptance length |3.597335|3.618010|

C32 versus C16: output throughput **+42.48%**, P50 TPOT **+32.56%**, P90 TPOT **+55.67%**. These are single short-run observations, not repeat-run significance or topology-only speedup. No minimum throughput or TPOT threshold was specified. Request count/length criteria passed; batch and backend evidence were captured. Sampled batch counts: C16 114 entries at16; C32 56 at32 and one at16; all sampled retracted counts zero.

**Important qualification:** C32 target verification has192 rows (32×6), above the FlyDSL sparse-MLA96-row gate. The log explicitly records decline and fallback; lower-row/draft shapes can still use FlyDSL. Neither point proves that every kernel path is optimized. Synthetic KV does not encode prompt semantics, and simulated acceptance commits unverified draft tokens. The three fixes do **not** establish that true PD is fixed.

## Reproduce and audit

Read [REPRODUCE.md](REPRODUCE.md) for offline checks and exact ordered replay steps on a **new** authorized workspace. Do not run historical scripts directly: they hard-code the old experiment path. `scripts/prepare_workspace.py` is a **new, separately labeled relocation helper**, not part of the measured run. It preserves original script bytes and only substitutes runtime workspace, container name and optional port in generated copies.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/audit_package.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/recompute_metrics.py
```

## Contents

- `rounds/c16/`, `rounds/c32/`: full untrimmed JSONL (including all outputs/timing arrays), exact commands, container metadata, readiness and losslessly gzipped logs.
- `results/REPORT.md`, `results/summary.json`: verbatim original report; `results/recomputed.json`: new offline calculation.
- `scripts/original/`: the three verbatim launch/benchmark/readiness scripts. `reference/` also preserves previous-launcher references and adaptation diff.
- `source/`: three measured modified files; `source/baseline/`: exact three original files plus the benchmark metric implementation, not a reconstructed repository.
- `patches/`: three exact patches and [what/why/how](patches/README.md); `scripts/tests/`: three verbatim regression tests.
- [environment.md](environment.md): actual captured environment, pinned image, external dependencies and missing captures.
- [notes.md](notes.md): failure history, interpretation and safe replay boundaries.
- `history/`, `logs/`: all non-cache source history/state/operational evidence. Historical language/content is verbatim.
- `provenance/source_manifest.json`, `MANIFEST.sha256`, [audit.md](audit.md): source mapping, SHA256s, audit scope and gaps.

The source was copied, not moved or modified. All59 non-cache source files are retained, including full logs; compression is lossless. Image layers/archive, model weights and all JIT/Triton/Torch/HF caches are excluded as requested. This is a bounded reproduction kit, not an entire Git repository or the earlier long-run archive.
