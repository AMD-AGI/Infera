# Offline audit scope

The packaged artifact audit and CPU tests pass:283 source records verify byte-for-byte after optional gzip decompression;ten full points have correct global count/topology/TPOT arithmetic,replica aggregation and actual per-rank graph counts. All12 run bench snapshots match delivered bench sources.23 CPU tests and5 launcher tests pass. No GPU rerun was performed.

Final sealing also checks current original source bytes, REPRODUCE.md Bash syntax, entry-document links, selected known-format secret patterns and manifest hashes. Machine-readable final results are in provenance/offline-audit.json; source mapping in provenance/source-files.json; timing in provenance/run-times.json. Logger timestamps and result mtimes are artifact evidence, not exact measurement boundaries; launch_status.json contains measured launcher wall seconds.

## Limits
This audit does not establish natural-input correctness,statistical significance,continuous node exclusivity,cold image rebuild equivalence,current allocation availability or missing056CPU/RAM/RDMA details. Secret pattern scanning is heuristic and excludes credential stores;zero known-pattern hits is not universal proof of no secrets. Original archived documents keep historical paths; new REPRODUCE.md is the normative relocated recipe.

Image/weights are external,not copied. No selected file exceeds4MB. User approved complete gzip logs. Originals untouched;Python caches omitted from package,not deleted. No commit/push or container/node action during packaging.

## Recheck
```bash
KIT=/home/yihou/dev/git/infera.dev.yihou.sglang.bench.fast.script/packups/glm52_tp4_ep4_dpa_sweep_yihou.packup_20260910-062400
(cd "$KIT" && sha256sum -c MANIFEST.sha256)
PYTHONDONTWRITEBYTECODE=1 python3 "$KIT/scripts/audit_packup_yihou.py"
```
