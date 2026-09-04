#!/usr/bin/env python3
"""What `seed_table` runs: republish a Magpie CSV as a `kernel_table` handoff.

This file imports nothing from `agent_sys`. It is package data, run as a
subprocess by `agent.backends.program.ProgramExecutor`, and everything it needs
arrives as an environment variable.

| | |
|---|---|
| `AD_SEED_CSV` | the gap-analysis CSV to republish |
| `AGENT_SYS_OUTPUT_KERNEL_TABLE` | the pre-allocated, granted `<store>/<hid>/v<N>/content/` |
| `AGENT_SYS_TASK_PACKAGE` | the staged copy of this package, for `assets/lib` |

**This is a mock and says so in its own README.** The CSV it copies is a
GLM-5.2 1P1D decode profile, not GLM-5.3-Flash: the shapes are usable for
exercising the pipeline, and the operator mix is not the target model's. A
consumer that treats it as the target model's profile is reading it wrong, so
the README it writes states the provenance rather than leaving the reader to
find out from it.

**No host path reaches the handoff.** `handoff/locality.py` scans every file
under `content/` and refuses a seal when it finds an absolute path outside an
anchored allow-list — `/apps/...` and `/data/...` are both outside it. The rule
is spec §7: a handoff names its dependencies and nothing about the machine that
produced it. So provenance here is `(filename, sha256, size, rows)`, which
identifies the artefact exactly and names no machine.
"""

import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(
    0,
    str(
        Path(
            os.environ.get("AGENT_SYS_TASK_PACKAGE")
            or os.environ["AGENT_SYS_DEMO_PACKAGE"]
        )
        / "assets"
        / "lib"
    ),
)

import csv_io  # noqa: E402 — the path insert above is what makes it importable

README = """# kernel_table

## Purpose

Kernel-level GPU time from a Magpie gap analysis: one row per device kernel,
carrying its call count, total and average self time, share of profiled GPU
time, and the distinct input shapes the profiler recorded for it.

This copy was produced by `seed_table`, which is a **mock** standing in for
`profiling-demo`'s `kernel_scan` until the two graphs are joined. It republishes
an existing CSV rather than running a profile.

Provenance of this copy:

- file: `{filename}`
- sha256: `{sha256}`
- size: {size} bytes
- header shape: {shape} ({columns} columns)
- rows: {rows}
- copied at: {stamp}

The source directory is deliberately not recorded: `handoff/locality.py` refuses
to seal content carrying an absolute host path, and the sha256 above identifies
the artefact without naming a machine.

{provenance_note}

## Schema

`items/text.json`:

```json
{{"source": {{"filename": "gap_analysis.csv", "sha256": "...", "size": 0}},
  "enriched": false,
  "columns": ["Name", "Calls", "..."],
  "repo_paths": {{}},
  "totals": {{"rows": 0, "pct_total_sum": 0.0}},
  "kernels": [
    {{"name": "...", "calls": 0, "self_us": 0.0, "avg_us": 0.0,
      "pct_total": 0.0, "input_shapes": "[a,b]x[c,d]; ...",
      "launcher": {{}}}}
  ]}}
```

`kernels[].launcher` is **optional and absent here, and no longer absent
upstream.** It is the slot for the Python call-site a `with_stack: true` profile
carries — `source_file`, `line`, `function`, `launch_api`, `sample_count`, named
after Hyperloom's `LauncherFrame`, plus `container_root`, `owner` and
`path_form`. `profiling-demo`'s `kernel_scan` now fills it from a short
`with_stack` window, and `identify` reads it as resolution level 1 instead of
falling back to name-based search. See DESIGN.md section 4.4, and
`../README.md`'s "Joining up with profiling-demo" for what the two layouts are.

This mock still has none, because it republishes a CSV and a CSV carries no call
stacks. That is why `assets/lib/kernel_table.launcher_coverage` distinguishes
"resolution found nothing" from "nobody looked": read as the former, a run
against this mock would look like a failed capture.

`items/gap_analysis/` holds the CSV exactly as Magpie wrote it.
"""

GLM52_NOTE = """**This is a GLM-5.2 1P1D decode profile, not GLM-5.3-Flash.** The
shapes are real and usable for exercising the pipeline, but the operator mix
differs from the target model, which serves DSA through TileLang, KDA through
Triton, and MoE through the Triton runner. Do not read the ranking below as a
statement about GLM-5.3-Flash."""


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            f"{name} is not set. A task body writes its output into the "
            f"pre-allocated <store>/<hid>/v<N>/content/, exported under "
            f"AGENT_SYS_OUTPUT_<KIND>; without it this body has nowhere to write."
        )
    return value


def main() -> int:
    source = Path(os.environ.get("AD_SEED_CSV") or "").expanduser()
    if not source.is_file():
        raise SystemExit(
            f"AD_SEED_CSV does not name a readable file: {source!s}\n"
            f"Supply it with --var seed_csv=/path/to/gap_analysis.csv"
        )

    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()

    rows, meta = csv_io.read_gap_analysis(source)
    records = [csv_io.to_record(r) for r in rows]
    pct_sum = sum(r["pct_total"] or 0.0 for r in records)

    dst = Path(_required("AGENT_SYS_OUTPUT_KERNEL_TABLE"))
    items = dst / "items"
    items.mkdir(parents=True, exist_ok=True)

    (items / "gap_analysis").mkdir(exist_ok=True)
    shutil.copy2(source, items / "gap_analysis" / source.name)

    document = {
        # No directory, by rule. See the module docstring.
        "source": {"filename": source.name, "sha256": digest, "size": len(payload)},
        "enriched": meta["enriched"],
        "columns": meta["columns"],
        # `repo_paths` are host paths from an enriched CSV's comment header, and
        # the seal refuses them. Kept as names only, so a reader knows which
        # repositories were indexed without learning where they sat.
        "repo_variables": sorted(meta["repo_paths"]),
        "totals": {"rows": len(records), "pct_total_sum": round(pct_sum, 3)},
        "kernels": records,
    }
    (items / "text.json").write_text(json.dumps(document, indent=2), encoding="utf-8")

    note = GLM52_NOTE if "glm5.2" in str(source).lower() else ""
    (dst / "README.md").write_text(
        README.format(
            filename=source.name,
            sha256=digest,
            size=len(payload),
            shape="enriched (Magpie --find-kernel-sources)" if meta["enriched"] else "base",
            columns=len(meta["columns"]),
            rows=len(records),
            stamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            provenance_note=note,
        ),
        encoding="utf-8",
    )

    print(
        f"seed_table: {len(records)} kernels from {source.name} "
        f"({'enriched' if meta['enriched'] else 'base'}, "
        f"pct sum {pct_sum:.2f}, sha256 {digest[:12]}) -> {dst}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
