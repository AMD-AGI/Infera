#!/usr/bin/env python3
"""`check_profiling_evidence` — completeness, strong. Five rules over the merge.

Mission M2.9: *"bench_result、profiling result、magpie standardized output
result，三者整合进一个大的统一 handoff"*. `profiling_evidence` is stage 2's only
export; m3 and m5 consume it rather than the four pieces.

**The merge is the one place the two lines meet, so it is the one place that can
notice they were not run against the same deployment.** Each line's own
validators grade that line's artefacts and cannot see the other's. Everything
below is a cross-part rule, and none of it can be checked anywhere else in the
flow.

The rules:

1. **Every part arrived**, as a non-empty directory under `items/result/`, and
   `items/env/parts.json` accounts for exactly those parts and no others.
2. **The parts describe one deployment.** Same container, same image digest,
   same node — read from each part's own environment record as it was carried
   into the merge, not from the merged one, because the merged one is what a
   merge that got this wrong would write.
3. **The two benches replayed the same load.** Same trace, same window, same
   concurrency ceiling. Without this the pair is two measurements of two
   different things, `profiling_mode_off` is not a control for
   `profiling_mode_on`, and the number m5's stock arm must reproduce
   (M5.1.3.1) describes a load nobody will run again.
4. **The ranking was derived from this trace.** The kernel table records the
   capture's own kernel total; the trace manifest records it independently.
   When they disagree, the table ranks a different capture — which is exactly
   what happened once already in this flow's history, where a stage-3 run was
   fed a synthetic seed table and every validator downstream passed.
5. **The merged environment record agrees with its parts**, so a consumer that
   reads only `items/env/environment.yaml` — which is every consumer, since
   that is the point of the merge — is reading something the parts support.

`check_environment` grades the merged record's *shape* against the schema. This
grades its *agreement* with what went into it. Neither substitutes for the other.
"""

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import workset_io as W  # noqa: E402 — `write_report`; nothing else from it
import zone  # noqa: E402 — the path insert above is what makes it importable

#: What every part must agree on: the machine and the software it ran.
#:
#: **`container` is deliberately not here, and that is not a weakening.** The
#: two lines are two bring-ups by design — they differ in the engine's CUDA
#: graph setting, which cannot be changed under a running engine — and CONTRACT
#: §5.2 forbids either of them reusing a container name it did not create. So
#: the two lines legitimately name different containers, and requiring one would
#: fail every correct run. This was written the other way first and the
#: validator refused a correct merge, which is the right way round to find out.
ACROSS_LINES = ("node", "image_id")

#: What parts **from the same line** must agree on. `trace` and `kernel_table`
#: come from the same bring-up as `bench_profiling_mode_on`, and this is the rule
#: with teeth: it catches a trace or a ranking folded in from a different
#: profiled run, which is the one substitution that would leave every other
#: number in this handoff looking right.
WITHIN_LINE = ("node", "image_id", "container", "endpoint")


def _fail(reasons: list, message: str) -> bool:
    reasons.append(message)
    return False


def _read_json(path: Path, reasons: list, what: str):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(reasons, f"{what} is not readable JSON: {exc}")
        return None


def _read_yaml(path: Path, reasons: list, what: str):
    try:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, Exception) as exc:  # noqa: BLE001 — yaml raises its own tree
        _fail(reasons, f"{what} is not readable YAML: {exc}")
        return None


def parts_ok(content: Path, args: dict, reasons: list):
    """Rule 1. Returns `(ok, parts)`; `parts` is the manifest's rows by name."""
    want = list(args.get("require_parts") or [])
    result = content / "items" / "result"
    ok = True

    for name in want:
        base = result / name
        if not base.is_dir():
            ok = _fail(reasons, f"items/result/{name}/ is missing")
        elif not any(p.is_file() for p in base.rglob("*")):
            ok = _fail(reasons, f"items/result/{name}/ holds no files")

    manifest_path = content / "items" / "env" / "parts.json"
    if not manifest_path.is_file():
        _fail(
            reasons,
            "items/env/parts.json is missing — it is what says which handoff each part "
            "came from, and without it the merge is four directories with no provenance",
        )
        return False, {}
    manifest = _read_json(manifest_path, reasons, "parts.json")
    if manifest is None:
        return False, {}

    rows = manifest.get("parts")
    if not isinstance(rows, list):
        return _fail(reasons, "parts.json has no 'parts' array"), {}

    by_name = {}
    for row in rows:
        name = (row or {}).get("name")
        if not name:
            ok = _fail(reasons, f"parts.json holds a row with no name: {row!r}")
            continue
        if name in by_name:
            ok = _fail(reasons, f"parts.json lists {name!r} twice")
        by_name[name] = row
        for field in ("source_kind", "environment"):
            if field not in row:
                ok = _fail(reasons, f"parts.json row {name!r} has no {field!r}")

    missing = [n for n in want if n not in by_name]
    extra = [n for n in by_name if n not in want]
    if missing:
        ok = _fail(reasons, f"parts.json does not account for {missing}")
    if extra:
        ok = _fail(
            reasons,
            f"parts.json claims part(s) {extra} that this kind does not declare — a "
            f"consumer reading the manifest would look for a directory nobody agreed to",
        )
    return ok, by_name


def one_deployment(parts: dict, args: dict, reasons: list) -> bool:
    """Rule 2, in two halves — across the lines, and within each line."""
    if not args.get("require_same_environment", True):
        reasons.append("(note) require_same_environment is off — the parts were not compared")
        return True

    ok = True
    seen: dict[str, dict] = {}
    for name, row in sorted(parts.items()):
        env = (row or {}).get("environment") or {}
        missing = [f for f in ACROSS_LINES if not env.get(f)]
        if missing:
            ok = _fail(
                reasons,
                f"parts.json row {name!r} does not record {missing} — a part that will "
                f"not say which deployment it came from cannot be compared with one "
                f"that will",
            )
            continue
        seen[name] = env

    for field in ACROSS_LINES:
        values = {name: env.get(field) for name, env in seen.items()}
        if len(set(values.values())) > 1:
            ok = _fail(
                reasons,
                f"the parts disagree about {field}: {values} — stage 2's two lines ran "
                f"on different machines or different software, so neither is a control "
                f"for the other and the trace does not explain the clean bench",
            )

    # The within-line half. Grouped by the `line` the merge recorded, because
    # which bring-up a part came from is not derivable from its name: `trace` and
    # `kernel_table` both belong to the profiler-attached line.
    by_line: dict[str, dict] = {}
    for name, env in seen.items():
        line = (parts[name] or {}).get("line")
        if not line:
            ok = _fail(
                reasons,
                f"parts.json row {name!r} does not say which line produced it, so it "
                f"cannot be checked against the other parts of that line",
            )
            continue
        by_line.setdefault(line, {})[name] = env

    for line, members in sorted(by_line.items()):
        if len(members) < 2:
            continue
        for field in WITHIN_LINE:
            values = {name: env.get(field) for name, env in members.items()}
            if len(set(values.values())) > 1:
                ok = _fail(
                    reasons,
                    f"the {line} parts disagree about {field}: {values} — these came "
                    f"from one bring-up and cannot have. One of them was folded in "
                    f"from a different run, and every other number here would still "
                    f"look right",
                )

    if seen:
        reasons.append(
            "(note) "
            + ", ".join(f"{line}: {sorted(m)}" for line, m in sorted(by_line.items()))
            + f" — all on {next(iter(seen.values())).get('node')}"
        )
    return ok


def same_load(content: Path, args: dict, reasons: list) -> bool:
    """Rule 3."""
    # **`.get(key, default)` and not `x or default`.** An explicitly empty list is
    # an operator saying "do not compare loads", and `or` cannot tell that from
    # an absent key — it silently reinstates the pair. Same class of fault as
    # `reverify_shapes: 0` being unreachable in m3's guard, raised 2026-09-03.
    names = list(args.get("compare_load_of", ["bench_profiling_mode_off", "bench_profiling_mode_on"]))
    if not names:
        reasons.append("(note) compare_load_of is empty — the two benches' loads were not compared")
        return True
    configs = {}
    ok = True
    for name in names:
        path = content / "items" / "result" / name / "env" / "load.json"
        if not path.is_file():
            ok = _fail(reasons, f"items/result/{name}/env/load.json is missing")
            continue
        cfg = _read_json(path, reasons, f"{name}/env/load.json")
        if cfg is not None:
            configs[name] = cfg
    if len(configs) < 2:
        return False

    # `profiler_window` legitimately differs — that is the axis — and so does
    # `round`. Everything that describes the *load* must not.
    for field in ("trace", "trace_window_ms", "concurrency_ceiling", "served_model_name"):
        values = {name: cfg.get(field) for name, cfg in configs.items()}
        if len(set(json.dumps(v, sort_keys=True) for v in values.values())) > 1:
            ok = _fail(
                reasons,
                f"the two benches disagree about {field}: {values} — they replayed "
                f"different loads, so one is not a control for the other",
            )
    if ok:
        any_cfg = next(iter(configs.values()))
        reasons.append(
            f"(note) both benches replayed {any_cfg.get('trace')!r} over "
            f"{any_cfg.get('trace_window_ms')} at concurrency {any_cfg.get('concurrency_ceiling')}"
        )
    return ok


def table_matches_trace(content: Path, args: dict, reasons: list) -> bool:
    """Rule 4. The ranking's own account of its input, against the trace's."""
    table_path = content / "items" / "result" / "kernel_table" / "text.json"
    trace_path = content / "items" / "result" / "trace" / "manifest.json"
    if not table_path.is_file() or not trace_path.is_file():
        return _fail(
            reasons,
            "cannot tie the ranking to the trace: "
            f"{'kernel_table/text.json ' if not table_path.is_file() else ''}"
            f"{'trace/manifest.json ' if not trace_path.is_file() else ''}missing",
        )
    table = _read_json(table_path, reasons, "kernel_table/text.json")
    trace = _read_json(trace_path, reasons, "trace/manifest.json")
    if table is None or trace is None:
        return False

    claimed = (table.get("source") or {}).get("trace_gpu_kernels")
    actual = (trace.get("totals") or {}).get("gpu_kernels")
    if claimed is None:
        return _fail(reasons, "the kernel table does not record how many kernels its trace held")
    if actual is None:
        return _fail(reasons, "the trace manifest carries no totals.gpu_kernels")
    if int(claimed) != int(actual):
        return _fail(
            reasons,
            f"the ranking was built over {claimed} GPU kernel event(s) and this "
            f"handoff's trace holds {actual} — the table ranks a different capture, "
            f"and every share in it is a share of something else",
        )
    reasons.append(f"(note) the ranking and the trace agree on {actual} GPU kernel events")
    return True


def merged_env_agrees(content: Path, parts: dict, reasons: list) -> bool:
    """Rule 5."""
    path = content / "items" / "env" / "environment.yaml"
    if not path.is_file():
        # `check_environment` reports the absence; this rule has nothing to add.
        return True
    record = _read_yaml(path, reasons, "environment.yaml")
    if not isinstance(record, dict):
        return _fail(reasons, "environment.yaml is not a mapping")

    fixed, runtime = record.get("fixed") or {}, record.get("runtime") or {}
    ok = True
    for name, row in sorted(parts.items()):
        env = (row or {}).get("environment") or {}
        for field in ACROSS_LINES:
            if env.get(field) and fixed.get(field) and env[field] != fixed[field]:
                ok = _fail(
                    reasons,
                    f"the merged environment says {field}={fixed[field]!r} and part "
                    f"{name!r} says {env[field]!r} — a consumer reading only the merged "
                    f"record would be told something its own evidence contradicts",
                )

    # The merged record's `runtime` describes **one** of the two bring-ups, and
    # it cannot describe both. What it may not do is describe a third: a
    # container name that belongs to no part is a record of a deployment this
    # handoff carries no evidence of.
    merged_container = runtime.get("container")
    containers = {
        (row or {}).get("environment", {}).get("container")
        for row in parts.values()
    } - {None}
    if merged_container and containers and merged_container not in containers:
        ok = _fail(
            reasons,
            f"the merged environment names container {merged_container!r}, which is "
            f"none of the parts' ({sorted(containers)}) — the record describes a "
            f"deployment this handoff holds no evidence of",
        )
    elif merged_container:
        reasons.append(
            f"(note) the merged runtime describes {merged_container!r}; the other "
            f"line's bring-up is in items/env/parts.json"
        )
    return ok


def check(content: Path, args: dict, reasons: list) -> bool:
    ok, parts = parts_ok(content, args, reasons)
    return all(
        [
            ok,
            one_deployment(parts, args, reasons),
            same_load(content, args, reasons),
            table_matches_trace(content, args, reasons),
            merged_env_agrees(content, parts, reasons),
        ]
    )


def main() -> int:
    args = zone.args()
    results = {}
    # **The reasons have to outlive stdout.** This one grades the merge across
    # four parts that arrived separately, so its refusal is usually about a
    # *relationship* — two lines disagreeing about an environment — and that is
    # the reason least reconstructible from the artefacts afterwards.
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
                # A crash is not a refusal, and `verdict.json` is `dict[str, bool]`
                # with no third state (todo.md T29). `False` because a check that
                # did not execute has established nothing; this text is the only
                # place the difference exists.
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
        print(f"check_profiling_evidence: {hid} {'PASS' if results[hid] else 'FAIL'}")
        for reason in reasons:
            print(f"  - {reason}")
    # Before the verdict, so a crash in the writer cannot take the reasons with it.
    W.write_report("check_profiling_evidence", findings, results)
    zone.write_verdict(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
