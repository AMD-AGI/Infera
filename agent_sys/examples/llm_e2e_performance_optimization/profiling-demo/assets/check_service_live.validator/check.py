#!/usr/bin/env python3
"""`check_service_live` — completeness, strong.

Five rules over a deployment record, every one of them decided by looking at a
file that either says the thing or does not:

1. the shape `reproducible` promises is there and non-empty
2. exactly `expect_workers` worker registered, in `expect_disagg_mode`
3. the arithmetic probe answered `expect_arithmetic`
4. the engine log tail carries none of `fault_patterns`
5. the record's own `cuda_graph` / `profiling_enabled` match what its round means

Rule 5 is the one that is easy to leave out and the one that catches the
expensive mistake. The two rounds this package runs differ only in those two
flags, and a `serve_profiled` that quietly came up with graphs on would still
serve, still smoke-test clean, and still produce a trace -- a trace in which
every decode step is one graph launch and no kernel is attributable. Checking the
record against its own declared round is what makes that loud.

Reads only files. Nothing here calls the endpoint; see entry.sh.
"""

import gzip
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import store  # noqa: E402 — the path insert above is what makes it importable

#: Non-empty means something a reader would accept as evidence, not merely a file
#: that exists. A zero-byte `workers.json` is the shape of a curl that timed out.
MIN_BYTES = 2

#: The logs are gzipped in the handoff. Not a size decision: `handoff.locality`
#: scans every UTF-8 file for absolute paths, and an engine log is almost
#: entirely false positives of the kind that module's own docstring predicts —
#: container-internal paths, HTTP routes, an etcd key prefix. Compressing keeps
#: the bytes exactly where substituting them would corrupt the one artefact whose
#: value is being faithful. See temp/bugs/002.
REQUIRED = {
    "result": ["smoke.txt", "workers.json", "models.json", "health.txt"],
    "env": ["deployment.json", "gpu.txt", "image.txt", "engine_argv.txt", "router_cmd.txt"],
    "logs": ["mix_up.log.gz", "worker.tail.log.gz"],
}


def _fail(reasons: list, message: str) -> bool:
    reasons.append(message)
    return False


def shape_ok(content: Path, reasons: list) -> bool:
    """Rule 1. The items `reproducible` declares, with substance in them."""
    ok = True
    for item, files in REQUIRED.items():
        base = content / "items" / item
        if not base.is_dir():
            ok = _fail(reasons, f"items/{item}/ is missing")
            continue
        for name in files:
            path = base / name
            if not path.is_file():
                ok = _fail(reasons, f"items/{item}/{name} is missing")
            elif path.stat().st_size < MIN_BYTES:
                ok = _fail(reasons, f"items/{item}/{name} is empty")
    if not (content / "items" / "watchout").is_file():
        ok = _fail(reasons, "items/watchout is missing")
    # `script` or `command`; the type requires one and this kind writes
    # `command`, because a copied bring-up script cannot pass the locality seal.
    command = content / "items" / "command"
    if not command.is_file():
        ok = _fail(reasons, "items/command is missing")
    elif command.stat().st_size < MIN_BYTES:
        ok = _fail(reasons, "items/command is empty")
    return ok


def workers_ok(content: Path, args: dict, reasons: list) -> bool:
    """Rule 2. One worker, and it is the aggregated one.

    Two workers is not a stricter version of one: it means a registration from a
    previous round is still in etcd, and the router would then split the load
    across a worker that exists and one that does not.
    """
    path = content / "items" / "result" / "workers.json"
    try:
        listed = json.loads(path.read_text(encoding="utf-8")).get("workers")
    except (OSError, json.JSONDecodeError, AttributeError) as exc:
        return _fail(reasons, f"workers.json is not readable JSON: {exc}")
    if not isinstance(listed, list):
        return _fail(reasons, "workers.json has no 'workers' list")

    want_n = int(args.get("expect_workers", 1))
    if len(listed) != want_n:
        return _fail(reasons, f"expected {want_n} worker(s), found {len(listed)}")

    want_mode = str(args.get("expect_disagg_mode", "mixed"))
    ok = True
    for worker in listed:
        mode = worker.get("disagg_mode")
        if mode != want_mode:
            ok = _fail(reasons, f"worker {worker.get('worker_id')} is {mode!r}, want {want_mode!r}")
        if worker.get("status") != "active":
            ok = _fail(reasons, f"worker {worker.get('worker_id')} is {worker.get('status')!r}")
    return ok


def smoke_ok(content: Path, args: dict, reasons: list) -> bool:
    """Rule 3. The arithmetic probe.

    `mix_smoke.sh` asks "Compute 17 * 23. Reply with only the number." and prints
    SMOKE_ARITHMETIC_OK when the answer contains the expected value. Both the
    marker and the value are checked: the marker alone would pass a smoke script
    that failed to run and left a stale file, and the value alone would pass a
    model that emitted it inside an apology.
    """
    path = content / "items" / "result" / "smoke.txt"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return _fail(reasons, f"smoke.txt is not readable: {exc}")
    if "SMOKE_ARITHMETIC_OK" not in text:
        return _fail(reasons, "smoke.txt does not carry SMOKE_ARITHMETIC_OK")
    want = str(args.get("expect_arithmetic", "391"))
    if want not in text:
        return _fail(reasons, f"smoke.txt does not contain the expected answer {want!r}")
    return True


def log_ok(content: Path, args: dict, reasons: list) -> bool:
    """Rule 4. No fault lines in the engine log tail."""
    path = content / "items" / "logs" / "worker.tail.log.gz"
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except (OSError, EOFError, gzip.BadGzipFile) as exc:
        return _fail(reasons, f"worker.tail.log.gz is not readable: {exc}")
    ok = True
    for pattern in args.get("fault_patterns") or []:
        found = re.search(re.escape(str(pattern)), text, re.IGNORECASE)
        if found:
            line = text[: found.start()].count("\n") + 1
            ok = _fail(reasons, f"engine log line {line} matches fault pattern {pattern!r}")
    return ok


def round_ok(content: Path, reasons: list) -> bool:
    """Rule 5. The processes that ran agree with what the round name means.

    **Read from the observed command lines, not from a field the producer wrote
    about itself.** `deployment.json` records what the run asked for; a
    self-declared `cuda_graph: 1` is exactly as true as the producer says it is,
    which is no use for catching the failure this rule exists to catch. The
    engine's argv and the router's invocation are what the processes have.
    """
    env = content / "items" / "env"
    try:
        record = json.loads((env / "deployment.json").read_text(encoding="utf-8"))
        argv = (env / "engine_argv.txt").read_text(encoding="utf-8", errors="replace")
        router = (env / "router_cmd.txt").read_text(encoding="utf-8", errors="replace")
    except (OSError, json.JSONDecodeError) as exc:
        return _fail(reasons, f"the environment capture is not readable: {exc}")

    name = record.get("round")
    # baseline: graphs on so the number is quotable, profiling off so the
    # deployment is not perturbed. profiled: the exact opposite, on both axes.
    expected = {"baseline": (1, 0), "profiled": (0, 1)}.get(name)
    if expected is None:
        return _fail(reasons, f"deployment.json has an unknown round {name!r}")
    want_graph, want_profile = expected

    ok = True

    # mix_worker.sh passes `--cuda-graph-backend-decode full` or `disabled`, one
    # argument per line in this capture. Anything else means the recipe moved and
    # this rule should be re-read rather than guessed at.
    lines = [line.strip() for line in argv.splitlines()]
    if "--cuda-graph-backend-decode" in lines:
        backend = lines[lines.index("--cuda-graph-backend-decode") + 1]
    else:
        backend = None
    observed_graph = {"full": 1, "disabled": 0}.get(backend)
    if observed_graph is None:
        ok = _fail(reasons, f"engine_argv.txt has no readable decode graph backend (found {backend!r})")
    elif observed_graph != want_graph:
        ok = _fail(
            reasons,
            f"round {name!r} wants cuda_graph={want_graph}, the engine is running "
            f"--cuda-graph-backend-decode {backend}",
        )

    observed_profile = 1 if "--enable-profiling" in router else 0
    if observed_profile != want_profile:
        ok = _fail(
            reasons,
            f"round {name!r} wants profiling_enabled={want_profile}, the router was "
            f"started {'with' if observed_profile else 'without'} --enable-profiling",
        )

    # A drift between what was asked for and what is running is worth naming even
    # when both happen to satisfy the round: it means a bring-up fell back.
    requested = record.get("requested") or {}
    if observed_graph is not None and int(requested.get("cuda_graph", -1)) != observed_graph:
        ok = _fail(
            reasons,
            f"deployment.json asked for cuda_graph={requested.get('cuda_graph')} "
            f"but the engine is running with {observed_graph}",
        )

    if record.get("disagg_mode") != "mixed":
        ok = _fail(reasons, f"deployment.json says disagg_mode={record.get('disagg_mode')!r}, want 'mixed'")
    return ok


def check(content: Path, args: dict, reasons: list) -> bool:
    """All five rules, all of them run.

    Deliberately not short-circuited. A validator that stops at the first failure
    makes the operator rerun a 13-minute deployment once per problem; running
    every rule costs nothing here and reports the whole set.
    """
    results = [
        shape_ok(content, reasons),
        workers_ok(content, args, reasons),
        smoke_ok(content, args, reasons),
        log_ok(content, args, reasons),
        round_ok(content, reasons),
    ]
    return all(results)


def main() -> int:
    args = store.args()
    results = {}
    for hid in store.inputs():
        # `materials.json` first: it is the declared route. `content_dir` is the
        # fallback for a run with no env_mgr wired, where nothing was staged.
        content = store.staged_content(hid) or store.content_dir(hid)
        reasons: list = []
        if content is None:
            # No published content is not a pass. A missing artefact and a bad
            # one have to report differently, but neither is a verdict of True.
            results[hid] = False
            reasons.append("no published content for this handoff")
        else:
            results[hid] = check(content, args, reasons)
        verdict = "PASS" if results[hid] else "FAIL"
        print(f"check_service_live: {hid} {verdict}")
        for reason in reasons:
            print(f"  - {reason}")
    store.write_verdict(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
