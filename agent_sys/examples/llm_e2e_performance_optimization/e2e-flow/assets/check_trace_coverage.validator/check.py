#!/usr/bin/env python3
"""`check_trace_coverage` — completeness, strong. Six rules over a capture.

Carried across from `../../profiling-demo/assets/check_trace_coverage.validator/`,
which had five of them and had been driven to a real cluster run. Rule 0 is new
and is mission M2.2.2: *"profile result 应至少被对应解析工具正确 load"*.

**The question is not "did eight files arrive".** A profiler window that opened
on an idle scheduler produces eight perfectly well-formed trace files holding
nothing. On disk that is indistinguishable from a good capture: same count, same
names, plausible sizes. The difference is inside, and the only way to see it is
to decompress and parse.

The rules:

0. **This validator opens a trace itself** and counts what is in it, with a
   reader of the Chrome Trace Format that torch and perfetto both write and
   read. That is what makes rules 1-3 evidence rather than a producer's account
   of itself: everything below is read from a manifest the *producer* built, and
   a manifest is a claim until something re-derives one of its rows.
1. One readable trace per tensor-parallel rank.
2. Every rank holds GPU kernels, above a floor a warm-up artefact cannot reach.
3. Every rank's time span is plausible for the window that was asked for.
4. The files on disk match the manifest, by name and by size, so the manifest
   describes *these* traces.
5. The measurement window carries **no** Python stacks, and the stack window
   carries them.

Rule 5 checks both halves of one decision, and it is a rule because both halves
fail silently. A measurement window taken with `with_stack` on is 13x the bytes
it should be, measured, and is otherwise a perfectly good trace — the only
symptom is a handoff nobody wants to move. A stack window taken with `with_stack`
off holds no `python_function` events at all, and the launcher resolution
downstream then reports "resolved 0" as if the frames were simply not findable.

**Extending rule 2 to check completeness against the sglang source and the model
structure is deferred** — `../todo.md` T1, and mission M2.8.2 says 先不做. Today
this cannot tell a trace that captured every layer from one that captured the
first two and stopped, because it has no model of what "every layer" means.
"""

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import trace_stream  # noqa: E402 — the path insert above is what makes these importable
import workset_io as W  # noqa: E402 — `write_report`; nothing else from it
import zone  # noqa: E402


def _fail(reasons: list, message: str) -> bool:
    reasons.append(message)
    return False


def check(content: Path, args: dict, reasons: list) -> bool:
    manifest_path = content / "items" / "result" / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _fail(reasons, f"manifest.json is not readable JSON: {exc}")

    ranks = manifest.get("ranks")
    if not isinstance(ranks, list) or not ranks:
        return _fail(reasons, "manifest.json lists no ranks")

    ok = True

    want = int(args.get("expect_ranks", 8))
    if len(ranks) != want:
        ok = _fail(reasons, f"expected {want} rank(s), the manifest lists {len(ranks)}")

    # Rule 4 first: everything below is a claim about these files.
    traces = content / "items" / "result" / "traces"
    on_disk = {p.name: p.stat().st_size for p in traces.glob("*.trace.json.gz")} if traces.is_dir() else {}
    if len(on_disk) != len(ranks):
        ok = _fail(reasons, f"{len(on_disk)} trace file(s) on disk against {len(ranks)} in the manifest")
    for row in ranks:
        name = row.get("file")
        if name not in on_disk:
            ok = _fail(reasons, f"{name!r} is in the manifest and not on disk")
        elif on_disk[name] != row.get("bytes"):
            ok = _fail(
                reasons,
                f"{name}: {on_disk[name]} bytes on disk, manifest says {row.get('bytes')}",
            )

    seen_ranks = set()
    floor = int(args.get("min_gpu_kernels_per_rank", 1000))
    min_span = float(args.get("min_span_s", 1.0))
    # The window was asked for in seconds. A trace much longer than that means the
    # stop did not take effect; much shorter means it never really started.
    window_s = float(args.get("window_s", 0) or 0)
    max_span = window_s * float(args.get("max_span_ratio", 4.0)) if window_s else None

    for row in ranks:
        name = row.get("file", "?")
        rank = row.get("rank")
        if rank in seen_ranks:
            ok = _fail(reasons, f"rank {rank} appears twice; {name} duplicates it")
        seen_ranks.add(rank)

        if not row.get("readable"):
            ok = _fail(reasons, f"{name} could not be parsed: {row.get('error', 'unknown')}")
            continue
        kernels = int(row.get("gpu_kernels") or 0)
        if kernels < floor:
            ok = _fail(reasons, f"{name} holds {kernels} GPU kernel event(s), floor is {floor}")
        span = float(row.get("span_s") or 0.0)
        if span < min_span:
            ok = _fail(reasons, f"{name} spans {span}s, floor is {min_span}s")
        elif max_span is not None and span > max_span:
            ok = _fail(reasons, f"{name} spans {span}s against a {window_s}s window")

    if None in seen_ranks:
        ok = _fail(reasons, "at least one trace file has no readable rank in its name")

    # Rule 5a. The measurement window must not carry stacks. Counted rather than
    # trusted: the capture sends `with_stack: false` explicitly, and this is what
    # catches an engine that ignored it.
    stacks_in_measurement = sum(int(r.get("python_functions") or 0) for r in ranks)
    if stacks_in_measurement:
        ok = _fail(
            reasons,
            f"the measurement window carries {stacks_in_measurement} python_function "
            f"event(s); it was taken with with_stack on, which costs ~13x the bytes "
            f"for a trace nobody would keep at that size",
        )

    ok = _reparse(traces, ranks, args, reasons) and ok
    ok = _check_stack_window(content, args, reasons) and ok

    reasons.append(
        f"(note) {len(ranks)} rank(s), "
        f"{sum(int(r.get('gpu_kernels') or 0) for r in ranks)} GPU kernel events"
    )
    return ok


def _reparse(traces: Path, ranks: list, args: dict, reasons: list) -> bool:
    """Rule 0. Open a trace here and check the manifest told the truth about it.

    Mission M2.2.2 asks that the profile result be *loaded by the corresponding
    analysis tool*. It cannot be loaded by torch's here — this body runs on the
    login node, where there is no torch and no GPU, and a validator that needed
    one would have to reach a compute node to grade a handoff. So it is loaded
    by a reader of the format torch writes and perfetto reads, streaming rather
    than `json.load` because a rank of this capture is 3.4 million events.

    **One rank by default, and the default is a cost decision rather than a
    principle.** A full pass over one 65 MB rank is ~14 s measured on this
    cluster; eight ranks plus a stack window is over two minutes, which would
    make the cheapest-first ordering of a validation phase meaningless. One rank
    is enough for what this rule is for: a manifest that was fabricated, or
    built from a different capture, does not survive one row being re-derived.
    `verify_ranks: -1` checks every rank, `0` trusts the manifest entirely and
    says so in the args rather than silently.
    """
    how_many = int(args.get("verify_ranks", 1))
    if how_many == 0:
        reasons.append("(note) verify_ranks=0 — the manifest was not re-derived from any trace")
        return True

    readable = [r for r in ranks if r.get("readable") and (traces / str(r.get("file"))).is_file()]
    if not readable:
        return _fail(reasons, "no readable trace file to re-parse; the manifest is ungrounded")
    # Largest first: the biggest rank is the one where a fabricated count is
    # least likely to have been guessed close.
    readable.sort(key=lambda r: int(r.get("bytes") or 0), reverse=True)
    chosen = readable if how_many < 0 else readable[:how_many]

    ok = True
    for row in chosen:
        name = str(row.get("file"))
        errors: list[str] = []
        try:
            counts = trace_stream.count_categories(traces / name, errors=errors)
        except (OSError, EOFError, ValueError) as exc:
            ok = _fail(reasons, f"{name} did not load: {type(exc).__name__}: {exc}")
            continue
        if errors:
            ok = _fail(reasons, f"{name} is not a well-formed chrome trace: {'; '.join(errors)}")
            continue

        for field in ("gpu_kernels", "python_functions"):
            claimed, actual = int(row.get(field) or 0), int(counts[field])
            if claimed != actual:
                ok = _fail(
                    reasons,
                    f"{name}: the manifest claims {claimed} {field} and the file holds "
                    f"{actual} — the manifest does not describe this capture, so every "
                    f"count in it is a number about something else",
                )
        # Spans are floats derived from microsecond timestamps; a re-derivation
        # agreeing to the millisecond is agreement, and demanding equality would
        # fail on rounding.
        claimed_span, actual_span = float(row.get("span_s") or 0.0), float(counts["span_s"])
        if abs(claimed_span - actual_span) > 0.01:
            ok = _fail(
                reasons,
                f"{name}: the manifest claims a {claimed_span}s span and the file spans "
                f"{actual_span}s",
            )
        reasons.append(
            f"(note) re-parsed {name}: {counts['events']} events, "
            f"{counts['gpu_kernels']} GPU kernels, {counts['span_s']}s — manifest agrees"
        )
    return ok


def _check_stack_window(content: Path, args: dict, reasons: list) -> bool:
    """Rule 5b: the short `with_stack` window, when the round was asked for one.

    Separate from the loop above because it is a different capture with different
    expectations: it is seconds long, it deliberately holds only some of the
    ranks, and its whole purpose is the `python_function` events the measurement
    window must not have. Judging it by the measurement window's rules would fail
    a correct stack window on every one of them.
    """
    want_ranks = int(args.get("expect_stack_ranks", 0) or 0)
    if want_ranks <= 0:
        return True

    manifest_path = content / "items" / "result" / "stacks_manifest.json"
    if not manifest_path.is_file():
        return _fail(
            reasons,
            "items/result/stacks_manifest.json is missing — the round was asked for a "
            "stack window and this handoff carries none, so no launcher frame can be "
            "resolved from it. Set --var stack_window_s=0 to say that is intended",
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _fail(reasons, f"stacks_manifest.json is not readable JSON: {exc}")

    ranks = manifest.get("ranks")
    if not isinstance(ranks, list) or not ranks:
        return _fail(reasons, "stacks_manifest.json lists no ranks")

    ok = True
    if len(ranks) != want_ranks:
        ok = _fail(
            reasons,
            f"the stack window carries {len(ranks)} rank file(s), expected {want_ranks}",
        )

    stacks = content / "items" / "result" / "stacks"
    on_disk = {path.name for path in stacks.glob("*.trace.json.gz")} if stacks.is_dir() else set()
    for row in ranks:
        if row.get("file") not in on_disk:
            ok = _fail(reasons, f"stack trace {row.get('file')!r} is in the manifest and not on disk")

    floor = int(args.get("min_python_functions_per_stack_rank", 10000))
    for row in ranks:
        name = row.get("file", "?")
        if not row.get("readable"):
            ok = _fail(reasons, f"{name} could not be parsed: {row.get('error', 'unknown')}")
            continue
        frames = int(row.get("python_functions") or 0)
        if frames < floor:
            ok = _fail(
                reasons,
                f"{name} holds {frames} python_function event(s), floor is {floor} — "
                f"the window was taken with with_stack off, or it opened on an idle "
                f"engine. Either way no launcher frame can be resolved from it",
            )
        # A stack window with frames and no kernels resolves nothing: the
        # correlation that binds a frame to a kernel needs both sides.
        if int(row.get("gpu_kernels") or 0) <= 0:
            ok = _fail(reasons, f"{name} holds no GPU kernel event to attribute a frame to")

    reasons.append(
        f"(note) stack window: {len(ranks)} rank(s), "
        f"{sum(int(r.get('python_functions') or 0) for r in ranks)} python_function events"
    )
    return ok


def main() -> int:
    args = zone.args()
    results = {}
    # **The reasons have to outlive stdout.** A validator's stdout is kept
    # nowhere, so a zone holds `args.json`, `inputs.json`, `materials.json` and
    # `verdict.json` and not one word about why. This validator is the one that
    # judges a capture nobody can re-take cheaply — a refusal here costs a
    # bring-up and a three-minute load to reproduce, so losing its reason is
    # the most expensive instance of that bug in this stage.
    findings: dict[str, tuple[list[str], list[str]]] = {}
    for hid in zone.inputs():
        content = zone.content_of(hid)
        reasons: list = []
        if content is None:
            results[hid] = False
            reasons.append("no staged content for this handoff")
        else:
            try:
                results[hid] = check(content, args, reasons)
            except Exception as error:  # noqa: BLE001
                # A crash is not a refusal. `verdict.json` is `dict[str, bool]`
                # and cannot say so (todo.md T29), so `False` is written because
                # a check that did not execute has established nothing — and the
                # text below is the only place the difference exists.
                reasons.append(
                    f"THIS VALIDATOR DID NOT RUN — {type(error).__name__}: {error}. "
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
        findings[hid] = (problems, notes)
        print(f"check_trace_coverage: {hid} {'PASS' if results[hid] else 'FAIL'}")
        for reason in reasons:
            print(f"  - {reason}")
    # Before the verdict, so a crash in the writer cannot take the reasons with
    # it, and always rather than only on failure.
    W.write_report("check_trace_coverage", findings, results)
    zone.write_verdict(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
