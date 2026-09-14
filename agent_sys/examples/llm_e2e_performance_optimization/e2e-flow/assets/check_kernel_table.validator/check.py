#!/usr/bin/env python3
"""`check_kernel_table` — usability, strong. Eight rules over a ranking.

**One definition, used by two modules.** Mission M3.5: m2 produces the table and
m3 reads it, and the two demo packages each carried a copy with different `args`
and different `min_rows` — over two artefacts that turned out to be different
things, a real 124-kernel capture and a 34-row synthetic seed. The spec lives
once, in `steps/common.yaml`; this is its one body.

`usability` and not `completeness`, because a table that arrived with every
column and a hundred rows can still be unusable: if the top twenty-five kernels
account for a tenth of the time, there is no operator worth picking and m3 would
be ranking noise.

`strong` is still honest. Every rule is a count, a sum or a file comparison, and
none of them is approximately right.

The rules:

1. **The record validates against the package's schema** (`assets/schemas/
   kernel_table.schema.json`), resolved by name from the same loader the
   producer used. Mission G2.
2. **The schema copy the handoff carries is byte-identical to the package's.**
   CONTRACT §3.4: a `structured_text` handoff is self-describing, and this is
   what stops the copy being a private fork that quietly says something else.
3. **Magpie's columns, exactly**, in the CSV that travels beside the record. A
   rename upstream stops here rather than at whatever parses the CSV next.
4. **Enough rows to rank.**
5. **The shares add up to a whole run** — too low and the trace covered a
   fraction of the work; above 100 and something is double-counted across ranks.
6. **The head accounts for enough of the run to be worth acting on.**
7. **`Input Shapes` is populated.** That column exists only when the capture
   asked the profiler for `record_shapes`, and without it every roofline
   downstream is impossible — but the table looks entirely normal, so nothing
   else would notice.
8. **The record carries every row of the CSV**, under the field names m3 reads.
   Rules 3-7 judge the CSV; this is the one that checks the thing m3 is actually
   handed, and the failure it catches is a consumer receiving a top-N where it
   expected a whole table. **The schema cannot express this** — a 25-row
   document is a perfectly valid `kernel_table` — which is exactly why the rule
   is here and not there.

Optionally, rule 9: enough of the head carries a launcher frame, when the round
was asked for a stack window. Off unless `min_launchers_in_top_n` is set,
because a round taken with `--var stack_window_s=0` resolves none by design.

**The layout is `structured_text`'s and not the demo's.** The record is
`items/text.json`, the export is `items/table.csv`, the schema copy is
`items/schema`. The sealed sample was produced as a `reproducible` handoff with
everything under `items/result/`; the mock reshapes it, and this body reads the
kind's layout rather than the sample's.
"""

import csv
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import schema as schema_lib  # noqa: E402 — the path insert above is what makes these importable
import workset_io as W  # noqa: E402 — `write_report` and `CRASH_MARKER`
import zone  # noqa: E402

#: The four columns every rule below reads. Reported once when they are absent,
#: rather than as four more failures that all say the same thing.
LOAD_BEARING = ("Name", "Self CUDA total (us)", "% Total", "Input Shapes")


def _fail(reasons: list, message: str) -> bool:
    reasons.append(message)
    return False


def _number(row: dict, key: str) -> float:
    try:
        return float((row.get(key) or "0").strip().replace(",", ""))
    except ValueError:
        return 0.0


def check(content: Path, args: dict, reasons: list) -> bool:
    document, ok_doc = _read_record(content, args, reasons)
    ok_csv, rows = _check_csv(content, args, reasons)
    ok = ok_doc and ok_csv
    if document is not None:
        ok = _check_document(document, rows, args, reasons) and ok
    return ok


def _observed_shape(content: Path) -> str:
    """Which layout this artefact actually has — m3's ask, as the consumer.

    `kernel_table` is one of the three cross-stage seams in
    `handoff.analysis.md`: **one kind name over two `content_type`s.** This body
    reads the `structured_text` layout (`items/text.json`, `items/table.csv`,
    `items/schema`); the sealed sample was produced as `reproducible` with
    everything under `items/result/`, and the mock reshapes it.

    So "items/text.json is missing" has two very different causes — a producer
    that wrote a malformed `structured_text` artefact, and a producer that
    handed over the `reproducible` shape unreshaped. **Naming the shape turns a
    refusal m3 has to reproduce into one they can act on**, which is the whole
    of what they asked for.
    """
    has_text = (content / "items" / "text.json").is_file()
    has_result = (content / "items" / "result").is_dir()
    if has_text and not has_result:
        return "structured_text (items/text.json) — the shape this validator grades"
    if has_result and not has_text:
        return ("reproducible (items/result/) — NOT the shape this validator grades; "
                "this looks like the sealed layout before m2_reshape.py")
    if has_text and has_result:
        return "both items/text.json and items/result/ present — ambiguous layout"
    return "neither items/text.json nor items/result/ — not a kernel_table layout at all"


def _read_record(content: Path, args: dict, reasons: list):
    """Rules 1 and 2."""
    path = content / "items" / "text.json"
    if not path.is_file():
        _fail(
            reasons,
            "items/text.json is missing — the CSV is the export but this is the "
            "document the operator-selection stage reads",
        )
        return None, False
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(reasons, f"text.json is not readable JSON: {exc}")
        return None, False

    ok = True
    name = args.get("schema") or "kernel_table"
    try:
        schema_lib.validate(name, document)
    except schema_lib.SchemaError as exc:
        for line in str(exc).splitlines()[1:]:
            ok = _fail(reasons, line.strip())
        ok = _fail(reasons, f"text.json does not validate against the {name!r} schema")

    # Rule 2. `structured_text` carries its own schema so the artefact is
    # self-describing; identity with the package's is what keeps it honest.
    carried = content / "items" / "schema"
    if not carried.is_file():
        ok = _fail(
            reasons,
            "items/schema is missing — a structured_text handoff carries the schema it "
            "was written against, so a reader who has the artefact and not this "
            "package can still tell what it is",
        )
    else:
        try:
            ours = schema_lib.schema_path(name).read_bytes()
        except schema_lib.SchemaError as exc:
            return document, _fail(reasons, str(exc))
        if carried.read_bytes() != ours:
            ok = _fail(
                reasons,
                f"items/schema is not byte-identical to the package's "
                f"{name}.schema.json — the handoff describes itself with a private "
                f"fork of the schema, and a producer and its validator are then "
                f"grading against two different documents",
            )
    return document, ok


def _check_csv(content: Path, args: dict, reasons: list):
    """Rules 3-7, over Magpie's own export."""
    csv_path = content / "items" / "table.csv"
    if not csv_path.is_file():
        return _fail(reasons, "items/table.csv is missing"), []

    try:
        with csv_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            header = list(reader.fieldnames or [])
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        return _fail(reasons, f"table.csv is not readable: {exc}"), []

    ok = True
    for column in args.get("require_columns") or []:
        if column not in header:
            ok = _fail(reasons, f"table.csv has no column {column!r} (header: {header})")
    if any(c not in header for c in LOAD_BEARING):
        _fail(reasons, "table.csv is missing the columns the remaining rules read")
        return False, rows

    min_rows = int(args.get("min_rows", 20))
    if len(rows) < min_rows:
        ok = _fail(reasons, f"table.csv has {len(rows)} data row(s), floor is {min_rows}")

    pct_sum = sum(_number(r, "% Total") for r in rows)
    low = float(args.get("min_pct_total_sum", 80.0))
    high = float(args.get("max_pct_total_sum", 100.5))
    if not low <= pct_sum <= high:
        ok = _fail(reasons, f"the '% Total' column sums to {pct_sum:.2f}, want {low}..{high}")

    if args.get("require_input_shapes", True):
        with_shapes = sum(1 for r in rows if (r.get("Input Shapes") or "").strip())
        if with_shapes == 0:
            ok = _fail(
                reasons,
                "no row carries Input Shapes — the capture did not ask the profiler "
                "for record_shapes, and no roofline can be built from this table",
            )

    # Rule 6. Recomputed from the CSV rather than read from the producer's own
    # summary, for the reason every recomputation here exists: a number a
    # producer wrote about itself is not evidence.
    # `.get(key, default)`, not `x or default`: an explicit `top_n: 0` must reach
    # the arithmetic and fail honestly on a 0% head share, rather than being
    # silently rewritten to 25 and passing.
    top_n = int(args.get("top_n", 25))
    times = sorted((_number(r, "Self CUDA total (us)") for r in rows), reverse=True)
    total = sum(times)
    share = 0.0
    if total <= 0:
        ok = _fail(reasons, "the 'Self CUDA total (us)' column sums to zero")
    else:
        share = 100.0 * sum(times[:top_n]) / total
        floor = float(args.get("min_top_n_share_pct", 50.0))
        if share < floor:
            ok = _fail(
                reasons,
                f"the top {top_n} kernels account for {share:.1f}% of CUDA time, "
                f"floor is {floor}% — this table has no head worth acting on",
            )

    reasons.append(
        f"(note) {len(rows)} kernels, top {top_n} cover {share:.1f}%, shares sum to {pct_sum:.2f}"
    )
    return ok, rows


def _check_document(document: dict, rows: list, args: dict, reasons: list) -> bool:
    """Rules 8 and 9: the document m3 reads, and its launcher frames."""
    ok = True
    kernels = document.get("kernels")
    if not isinstance(kernels, list):
        # The schema already said so; saying it twice helps nobody.
        return False

    # Rule 8. Every row, not the head. m3 buckets every kernel before it sorts,
    # so a truncated table costs it candidates rather than noise.
    if rows and len(kernels) != len(rows):
        ok = _fail(
            reasons,
            f"text.json carries {len(kernels)} kernel(s) against {len(rows)} row(s) in "
            f"table.csv — m3 classifies every row before it ranks, so a partial table "
            f"silently drops candidates",
        )
    claimed = (document.get("totals") or {}).get("rows")
    if claimed is not None and int(claimed) != len(kernels):
        ok = _fail(
            reasons,
            f"totals.rows says {claimed} and the kernels array holds {len(kernels)}",
        )

    # Rule 9, off by default.
    want = int(args.get("min_launchers_in_top_n", 0) or 0)
    summary = document.get("launchers") or {}
    if want > 0:
        if not summary.get("available"):
            return _fail(
                reasons,
                f"no launcher frames were resolved ({summary.get('reason') or 'no reason recorded'}) "
                f"and this round wanted at least {want} in the head. Set "
                f"--var stack_window_s=0 and min_launchers_in_top_n to 0 to say that is intended",
            ) and ok
        head = kernels[: int(args.get("launcher_head_n", 25))]
        with_frames = [k for k in head if (k.get("launcher") or {}).get("source_file")]
        if len(with_frames) < want:
            ok = _fail(
                reasons,
                f"{len(with_frames)} of the top {len(head)} kernel(s) carry a launcher "
                f"frame, floor is {want}. Resolved {summary.get('resolved')} of "
                f"{summary.get('wanted')} overall, {summary.get('unmapped')} outside a "
                f"known container root",
            )
        reasons.append(
            f"(note) launchers: {len(with_frames)}/{len(head)} of the head, "
            f"by owner {summary.get('by_owner')}, by form {summary.get('by_path_form')}"
        )
    return ok


def main() -> int:
    args = zone.args()
    results = {}
    # **The reasons have to outlive stdout.** This one is read by two stages
    # (M3.5), so a refusal nobody can read costs whichever of us did not run it
    # a reproduction. m3 asked, as the consumer, that a refusal name *which*
    # shape was graded — see `_observed_shape`.
    findings: dict[str, tuple[list[str], list[str]]] = {}
    for hid in zone.inputs():
        content = zone.content_of(hid)
        reasons: list = []
        notes_extra: list = []
        if content is None:
            results[hid] = False
            reasons.append("no staged content for this handoff")
        else:
            notes_extra.append(f"layout: {_observed_shape(content)}")
            try:
                results[hid] = check(content, args, reasons)
            except Exception as error:  # noqa: BLE001
                # `W.CRASH_MARKER` rather than the literal, so m3's `DID NOT RUN`
                # heading follows a reworded message instead of silently
                # reverting to `REFUSED` — their point, and it is the reason the
                # constant exists.
                reasons.append(
                    f"{W.CRASH_MARKER} — {type(error).__name__}: {error}. "
                    f"An instrument failure, not a finding: nothing here was graded."
                )
                reasons.append(traceback.format_exc())
                results[hid] = False
        # **`(note)` lines are notes, not problems**, and the split matters:
        # `write_report`'s heading is `REFUSED if problems else passed`, so
        # filing an informational line as a problem prints `REFUSED` above a
        # verdict of `true`. Measured on a real `kernel_table` — verdict
        # `{"h-kt": true}`, heading `## h-kt: REFUSED` — which is the same
        # heading-contradicts-its-own-text defect m3 had just removed for
        # crashes, reintroduced by me one field over. These bodies keep both
        # kinds in one `reasons` list; the prefix is what tells them apart.
        problems = [str(r) for r in reasons if not str(r).lstrip().startswith("(note)")]
        notes = [str(r) for r in reasons if str(r).lstrip().startswith("(note)")]
        findings[hid] = (problems, notes + notes_extra)
        print(f"check_kernel_table: {hid} {'PASS' if results[hid] else 'FAIL'}")
        for reason in reasons:
            print(f"  - {reason}")
    # Before the verdict, so a crash in the writer cannot take the reasons with it.
    W.write_report("check_kernel_table", findings, results)
    zone.write_verdict(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
