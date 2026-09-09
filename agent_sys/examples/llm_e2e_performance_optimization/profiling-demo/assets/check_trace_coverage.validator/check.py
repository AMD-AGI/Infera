#!/usr/bin/env python3
"""`check_trace_coverage` — completeness, strong.

The question is not "did eight files arrive" but "did the window catch the load".
Those look identical on disk: a profiler window that opened on an idle scheduler
produces eight perfectly well-formed traces holding nothing.

So the rules are counts, read from the manifest the capture built by
decompressing and parsing every rank:

1. one readable trace per tensor-parallel rank
2. every rank holds GPU kernels, above a floor a warm-up artefact cannot reach
3. every rank's time span is plausible for the window that was asked for
4. the files on disk match the manifest, so the manifest describes these traces
5. the measurement window carries **no** Python stacks, and the stack window
   carries them

Rule 4 is what stops the other three from being a description of something else.

Rule 5 checks both halves of one decision, and it is a rule because both halves
fail silently. A measurement window that arrived with `with_stack` on is 13x the
bytes it should be, measured, and is otherwise a perfectly good trace -- the only
symptom is a handoff nobody wants to move. A stack window that arrived with
`with_stack` off holds no `python_function` events at all, and the launcher
resolution downstream then reports "resolved 0" as if the frames were simply not
findable. `python_functions` in the manifest is what separates those.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import store  # noqa: E402 — the path insert above is what makes it importable


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
    # trusted: `capture.sh` sends `with_stack: false` explicitly, and this is what
    # catches an engine that ignored it.
    stacks_in_measurement = sum(int(r.get("python_functions") or 0) for r in ranks)
    if stacks_in_measurement:
        ok = _fail(
            reasons,
            f"the measurement window carries {stacks_in_measurement} python_function "
            f"event(s); it was taken with with_stack on, which costs ~13x the bytes "
            f"for a trace nobody would keep at that size",
        )

    ok = _check_stack_window(content, args, reasons) and ok

    reasons.append(
        f"(note) {len(ranks)} rank(s), "
        f"{sum(int(r.get('gpu_kernels') or 0) for r in ranks)} GPU kernel events"
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

    on_disk = {
        path.name
        for path in (content / "items" / "result" / "stacks").glob("*.trace.json.gz")
    } if (content / "items" / "result" / "stacks").is_dir() else set()
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
    args = store.args()
    results = {}
    for hid in store.inputs():
        content = store.staged_content(hid) or store.content_dir(hid)
        reasons: list = []
        if content is None:
            results[hid] = False
            reasons.append("no published content for this handoff")
        else:
            results[hid] = check(content, args, reasons)
        print(f"check_trace_coverage: {hid} {'PASS' if results[hid] else 'FAIL'}")
        for reason in reasons:
            print(f"  - {reason}")
    store.write_verdict(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
