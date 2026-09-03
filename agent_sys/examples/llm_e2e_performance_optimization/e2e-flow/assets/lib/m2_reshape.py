#!/usr/bin/env python3
"""Reshape one sealed stage-2 handoff into the shape this package's kind declares.

`assets/lib/mock.sh` copies bytes, which is right for the four kinds whose
sealed sample already has the layout the kind declares. **`kernel_table` is not
one of them**, and `MOCK-MAP.md` records only adaptation (A) for it, so this is
the correction:

    sealed `stage2-profiling/kernel_table`   content_type: reproducible
      items/result/text.json
      items/result/gap_analysis/gap_analysis.csv
      items/result/{top_kernels,launchers}.json
      items/{command,watchout}, items/env/, items/logs/

    this package's `profiling_mode_on.kernel_table`   content_type: structured_text
      items/text.json          the record
      items/table.csv          Magpie's export
      items/schema             this package's kernel_table.schema.json, byte-identical
      items/env/               the environment record and the capture's provenance

`structured_text` declares `text.json` / `text.yaml` / `text.xml` / `schema` and
nothing else, so `check_items` refuses any other top-level item that the kind's
`items_schema` does not declare — which here is `env` and `table.csv` and no
more. `top_kernels.json` and `launchers.json` are dropped rather than smuggled
in: both are derivable from `text.json`, which carries every row and every
kernel's `launcher` block, and `check_kernel_table` recomputes the head's share
from the CSV rather than reading anybody's summary of it.

The README is rewritten rather than copied, because a content type's required
sections are part of the type: `structured_text` requires `Purpose` and
`Schema`, and the sealed README has neither.

    m2_reshape.py kernel_table <sealed content dir> <output dir>
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import schema as schema_lib  # noqa: E402 — the path insert above is what makes it importable

README = """# kernel_table

## Purpose

Every GPU kernel in the captured window, ranked by the CUDA time it owns. This is
stage 3's input: the list of operators worth optimising is chosen from here, and
the choice is made over the whole table rather than its head, because a kernel is
bucketed before it is ranked.

## Schema

`items/text.json` is the record and validates against `items/schema`, which is a
byte-identical copy of this package's `assets/schemas/kernel_table.schema.json`.
`check_kernel_table` checks both the validation and the identity, so a reader who
has this handoff and not the package can still tell what it is holding.

`items/table.csv` is Magpie's own export, with the columns
`Name, Calls, Self CUDA total (us), Avg time (us), % Total, Input Shapes`. It is
the artefact and `text.json` is the record: `source.sha256` in the record is over
this CSV, so the two can be shown to describe one ranking rather than two.

`items/env/` carries the environment record every handoff in this flow carries,
plus the capture's own manifest and the engine command line that produced it, so
this ranking and the traces behind it can be matched by digest.
"""


def kernel_table(sealed: Path, out: Path) -> int:
    result = sealed / "items" / "result"
    record = result / "text.json"
    export = result / "gap_analysis" / "gap_analysis.csv"
    for path, what in ((record, "the record"), (export, "Magpie's export")):
        if not path.is_file():
            print(f"m2_reshape: {what} is missing at {path}", file=sys.stderr)
            return 1

    items = out / "items"
    (items / "env").mkdir(parents=True, exist_ok=True)

    shutil.copy2(record, items / "text.json")
    shutil.copy2(export, items / "table.csv")
    # Copied from `assets/schemas/`, not from the handoff, and byte for byte:
    # `check_kernel_table` compares them and a re-serialisation would fail it.
    shutil.copy2(schema_lib.schema_path("kernel_table"), items / "schema")

    sealed_env = sealed / "items" / "env"
    if sealed_env.is_dir():
        for entry in sealed_env.iterdir():
            if entry.is_file():
                shutil.copy2(entry, items / "env" / entry.name)

    (out / "README.md").write_text(README, encoding="utf-8")
    print(
        f"m2_reshape: kernel_table -> {out} "
        f"(text.json, table.csv, schema, env/{len(list((items / 'env').iterdir()))})",
        file=sys.stderr,
    )
    return 0


SHAPES = {"kernel_table": kernel_table}


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[0] not in SHAPES:
        print(f"usage: m2_reshape.py {{{'|'.join(SHAPES)}}} <sealed content> <output>", file=sys.stderr)
        return 2
    return SHAPES[argv[0]](Path(argv[1]), Path(argv[2]))


__all__ = ["kernel_table", "SHAPES", "main"]


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
