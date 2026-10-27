#!/usr/bin/env python3
"""Judge an A/B on its AgentX numbers by the project's rules, and write the verdict into the candidate's result files.

  ab_judge.py --baseline-files b1.json b2.json --candidate-files c1.json c2.json [--note TEXT] [--mechanism TEXT] --write
  ab_judge.py --baseline 12010 11950 --candidate 12650 12700 [--lower-is-better]    # a trial: printed, never written
  ab_judge.py --void failed-requests --candidate-files c2.json --write
  ab_judge.py --void other --reason "foreign load on GPU 3" --evidence log/rocm-smi.txt --candidate-files c2.json --write

Rules (agent_sys_v2.verdict.judge): the noise floor is the larger of the noise band and the run-to-run spread (1.96 x
the relative standard error of the difference). A difference inside the floor is noise, and noise never becomes an
accept. A worse result beyond the floor is a reject. A gain between the floor and the accept bar asks for another
run; so does a gain whose median disagrees with its mean, or whose X is worse beyond its own floor (compare at equal X
instead: the ledger's working point). Only a gain past the bar and the floor, with the medians agreeing and X not
worse, is an accept. The numbers alone decide: --mechanism is a remark kept with the verdict, not a condition.

The inputs are result files `ledger agentx` converted from AgentX output (they carry its raw file and sha256); no
other result, and no number typed in, is judged with --write. Every valid result of an input's series and
concurrency in its task (whatever series or role it says: one A/B per concurrency per task) is an input: void a
spoiled one rather than leave it out. The metric and X, and their
directions, are project.yaml's ledger.primary and ledger.x; the noise band and the accept bar are its
ledger.noise_pct and accept_pct (2 and 3 by default): --accept-pct can only raise the bar, and the band is not
given here (a wider band would turn a reject into noise too). The base arm is the project's ledger.baseline_series,
and both arms are at one concurrency. --write puts
the verdict into every candidate file with its inputs (files, raw digests, values, thresholds and where they came
from), which `ledger verify` recomputes from the raw output. --note adds a remark ("待集成" for a change whose expected
gain is below the noise).

A measurement is void for one of two reasons: failed-requests (AgentX counts more than 10% failed: computed here, and
refused when it does not) or other, with --reason and an --evidence file. --void writes into the --candidate-files
and --baseline-files given. Prints the verdict as JSON; exit 0, 2 when the input is wrong.

Run it with "$AGENT_SYS_PYTHON" (it imports agent_sys_v2) and $AGENT_SYS_PROJECT set (an agent's run has both).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

from agent_sys_v2 import ledger, verdict
from agent_sys_v2.model import write_json


def percent(text):
    """An --accept-pct: a finite number, not negative."""
    value = float(text)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError(f"{text} is not a percentage")
    return value


def threshold(name, given, rules, source):
    """(the value, where it came from) of a noise band or bar: the project's, or a stricter one given."""
    project = rules[name]
    if given is None:
        return project, source
    if given < project:
        raise ValueError(
            f"--{name.replace('_', '-')} {given:g} is looser than the project's {project:g} ({source}): "
            "the command line can only make the rules stricter"
        )
    return given, f"--{name.replace('_', '-')} {given:g} (stricter than {source}: {project:g})"


def root_of(files):
    """The project directory: the one the result files are in, else $AGENT_SYS_PROJECT."""
    for f in files:
        if root := verdict.project_root(f):
            return root
    env = os.environ.get("AGENT_SYS_PROJECT")
    return Path(env).resolve() if env else None


def void(args, root):
    """{file: its void verdict}. Raises ValueError when a file cannot be voided for that reason."""
    files = args.candidate_files + args.baseline_files
    if not files:
        raise ValueError(
            "--void names the files it voids: --candidate-files and/or --baseline-files"
        )
    out, why = {}, []
    for f in files:
        doc, errors = ledger.read(Path(f))
        if doc is None or errors:
            why.append(f"{f}: {'; '.join(errors) or 'not an infera.result file'}")
    if why:
        raise ValueError("\n".join(why))
    if args.void == "other":
        evidence = Path(args.evidence or "")
        if not str(args.reason or "").strip():
            why.append("--void other needs --reason: what spoiled the measurement")
        if not args.evidence or not evidence.is_file():
            why.append(
                f"--void other needs --evidence: a file that shows it ({args.evidence or 'none given'})"
            )
        if why:
            raise ValueError("\n".join(why))
        record = {
            "kind": "other",
            "reason": args.reason.strip(),
            "evidence": verdict.rel(evidence, root),
        }
        return {f: {"status": "void", "void": record, "notes": args.reason.strip()} for f in files}
    for f in files:
        doc, errors = ledger.read(Path(f))
        if doc is None or errors:
            why.append(f"{f}: {'; '.join(errors) or 'not an infera.result file'}")
            continue
        metrics, _conc, problems = verdict.reconvert(doc, root)
        if problems:
            why += [f"{f}: {p}" for _k, p, _fix in problems]
            continue
        pct = verdict.failed_pct(metrics)
        if pct is None or pct <= verdict.FAILED_MAX_PCT:
            why.append(
                f"{f}: AgentX counts {pct if pct is not None else 'no'}% failed requests, not more than "
                f"{verdict.FAILED_MAX_PCT:g}%: that is no void for failed requests"
            )
            continue
        out[f] = {
            "status": "void",
            "void": {"kind": "failed_requests", "failed_pct": pct},
            "notes": f"{pct}% of the requests failed",
        }
    if why:
        raise ValueError("\n".join(why))
    return out


def judged(args, root):
    """The verdict of the A/B the arguments name (with its inputs when it comes from result files)."""
    rules, source = verdict.project_rules(root)
    noise, noise_from = rules["noise_pct"], source
    accept, accept_from = threshold("accept_pct", args.accept_pct, rules, source)
    higher = False if args.lower_is_better else rules["higher_is_better"]
    x_higher = False if args.x_lower_is_better else rules["x_higher_is_better"]
    metric = args.metric or rules["primary"]
    x_metric = rules["x"] if args.x_metric is None else args.x_metric
    files = args.baseline_files or args.candidate_files
    if args.write:
        why = []
        if args.baseline or args.candidate or args.x_baseline or args.x_candidate:
            why.append(
                "--write judges result files only: numbers typed in are a trial (leave out --write)"
            )
        if not (args.baseline_files and args.candidate_files):
            why.append("--write needs --baseline-files and --candidate-files")
        # the project's metric, direction and X (when it names one) are the only ones a verdict is written with
        wanted = [("direction", higher, rules["higher_is_better"])]
        wanted += [("metric", metric, rules["primary"])] if rules["primary"] else []
        if rules["x"]:
            wanted += [("X metric", x_metric, rules["x"])]
            wanted += [("X direction", x_higher, rules["x_higher_is_better"])]
        for name, mine, theirs in wanted:
            if mine != theirs:
                why.append(
                    f"the {name} is {mine!r}, the project judges by {theirs!r} (project.yaml ledger): "
                    "another one is a trial (leave out --write)"
                )
        if why:
            raise ValueError("\n".join(why))
    if files and not metric:
        raise ValueError("--metric names the metric to read (project.yaml has no ledger.primary)")
    base_in = (
        verdict.arm(args.baseline_files, metric, x_metric, root) if args.baseline_files else []
    )
    cand_in = (
        verdict.arm(args.candidate_files, metric, x_metric, root) if args.candidate_files else []
    )
    why = []
    if args.write and rules["baseline_series"]:
        for f in args.baseline_files:
            if series_of(f) != rules["baseline_series"]:
                why.append(
                    f"{f}: the base arm is a repeat of the baseline series {rules['baseline_series']!r} "
                    f"(project.yaml ledger.baseline_series), not {series_of(f)!r}"
                )
        for f in args.candidate_files:
            if (ledger.read(Path(f))[0] or {}).get("role", "candidate") != "candidate":
                why.append(f"{f}: a verdict is written into candidates only (role candidate)")
            if series_of(f) == rules["baseline_series"]:
                why.append(f"{f}: a candidate is of its own series, not the baseline series")
    # the whole A/B: one concurrency, nothing left out, voids for a valid reason
    why += [
        f.problem for f in verdict.arm_findings(args.baseline_files + args.candidate_files, root)
    ]
    if why:
        raise ValueError("\n".join(why))
    base = args.baseline + [i["value"] for i in base_in]
    cand = args.candidate + [i["value"] for i in cand_in]
    x = None
    if x_metric or args.x_baseline or args.x_candidate:
        x = (
            args.x_baseline + [i["x"] for i in base_in if "x" in i],
            args.x_candidate + [i["x"] for i in cand_in if "x" in i],
            x_higher,
        )
        if not (x[0] and x[1]):
            x = None  # numbers without X: judged on Y alone
    out = verdict.judge(base, cand, higher, noise, accept, x)
    if args.mechanism:
        out["mechanism"] = args.mechanism
    if base_in and cand_in:
        out["inputs"] = {
            "metric": metric,
            "higher_is_better": higher,
            "x_metric": x_metric or "",
            "x_higher_is_better": x_higher,
            "baseline_series": ",".join(sorted({series_of(f) for f in args.baseline_files})),
            "baseline": base_in,
            "candidate": cand_in,
            "noise_pct": noise,
            "accept_pct": accept,
            "rules_from": {"noise_pct": noise_from, "accept_pct": accept_from},
        }
    return out


def series_of(f):
    doc, _errors = ledger.read(Path(f))
    return str((doc or {}).get("series") or "")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--baseline-files", nargs="+", default=[])
    ap.add_argument("--candidate-files", nargs="+", default=[])
    ap.add_argument("--baseline", nargs="+", type=float, default=[], help="numbers: a trial")
    ap.add_argument("--candidate", nargs="+", type=float, default=[], help="numbers: a trial")
    ap.add_argument("--metric", help="the metric to read (default: project.yaml ledger.primary)")
    ap.add_argument("--lower-is-better", action="store_true")
    ap.add_argument("--x-metric", help="the X metric (default: project.yaml ledger.x)")
    ap.add_argument("--x-baseline", nargs="+", type=float, default=[])
    ap.add_argument("--x-candidate", nargs="+", type=float, default=[])
    ap.add_argument("--x-lower-is-better", action="store_true")
    ap.add_argument("--accept-pct", type=percent, help="a higher accept bar than the project's")
    ap.add_argument(
        "--mechanism",
        default="",
        help="a remark on the mechanism metrics; not part of the decision",
    )
    ap.add_argument(
        "--void",
        choices=("failed-requests", "other"),
        help="the measurement is void: decide nothing",
    )
    ap.add_argument("--reason", default="", help="--void other: what spoiled the measurement")
    ap.add_argument("--evidence", help="--void other: a file that shows it")
    ap.add_argument("--note", default="", help="a remark added to the verdict's notes")
    ap.add_argument(
        "--write", action="store_true", help="write the verdict into every --candidate-files file"
    )
    args = ap.parse_args(argv)
    if args.write and not args.candidate_files and not (args.void and args.baseline_files):
        ap.error("--write writes into the --candidate-files")
    root = root_of(args.candidate_files + args.baseline_files)
    try:
        if args.write and root is None:
            raise ValueError(
                "no project: the result files are in no project directory and $AGENT_SYS_PROJECT is not set"
            )
        if args.void:
            verdicts = void(args, root)
        else:
            one = judged(args, root)
            verdicts = {f: one for f in args.candidate_files} or {"": one}
        distinct = list({id(v): v for v in verdicts.values()}.values())
        for v in distinct if args.note else ():
            v["notes"] = "; ".join(filter(None, (v["notes"], args.note)))
    except ValueError as e:
        print(f"ab_judge: {e}", file=sys.stderr)
        return 2
    for f, v in verdicts.items() if args.write else ():
        doc = json.loads(Path(f).read_text(encoding="utf-8"))
        write_json(Path(f), {**doc, "verdict": v})
    print(json.dumps(distinct[0] if len(distinct) == 1 else verdicts, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
