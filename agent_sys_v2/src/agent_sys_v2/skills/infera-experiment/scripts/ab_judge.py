#!/usr/bin/env python3
"""Judge an A/B the way the project rules say, and write the verdict into the candidate's result files.

  ab_judge.py --baseline 12010 11950 --candidate 12650 12700 --mechanism agree [--lower-is-better]
  ab_judge.py --metric input_tput_per_gpu --x-metric p90_intvty --baseline-files b0.json \
      --candidate-files c1.json c2.json --mechanism agree --write

Rules (infera-experiment SKILL.md): the noise floor is the larger of a fixed band (--noise-pct, 2%) and the
run-to-run spread (1.96 x the relative standard error of the difference). A difference inside the floor is noise,
and noise never becomes an accept. A worse result beyond the floor is a reject. A gain between the floor and the
accept bar (--accept-pct, 3%) asks for another run; so does a gain whose median disagrees with its mean, whose X
(--x-metric, or --x-baseline and --x-candidate: interactivity by default, --x-lower-is-better for a latency) is
worse beyond its own floor (compare at equal X instead: the ledger's working point), or whose mechanism metrics were
not shown to agree (--mechanism). Only a gain past the bar with the mechanism agreeing is an accept. --void REASON
records a voided measurement (foreign load, HSA out-of-resources, preemption, > 10% failed requests) without a
decision; --note adds a remark ("待集成" for a change whose expected gain is below the noise). --write writes the
verdict into every candidate file. Prints the verdict as JSON; exit 0, 2 when the input is wrong.

The numbers and the delta are the ledger's (agent_sys_v2.ledger): run it with "$AGENT_SYS_PYTHON".
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

from agent_sys_v2 import ledger
from agent_sys_v2.model import write_json


def rsd_pct(xs):
    """Relative standard deviation in percent; 0 for a single value or equal ones."""
    if len(set(xs)) < 2:
        return 0.0
    mean = statistics.mean(xs)
    if not mean:
        raise ValueError("an arm's values differ around a mean of 0: no relative spread")
    return statistics.stdev(xs) / abs(mean) * 100


def compare(base, cand, higher_is_better, noise_pct):
    """(gain in percent, its noise floor) of cand against base."""
    if not base or not cand:
        raise ValueError("both arms need at least one value")
    if not all(ledger.number(v) for v in base + cand):
        raise ValueError("every value must be a finite number")
    mb, mc = statistics.mean(base), statistics.mean(cand)
    if mb == 0:
        raise ValueError("the baseline's mean is 0: no relative difference")
    spread = 1.96 * math.sqrt(rsd_pct(base) ** 2 / len(base) + rsd_pct(cand) ** 2 / len(cand))
    return ledger.gain(ledger.pct(mc, mb), higher_is_better), max(noise_pct, spread)


def judge(
    base,
    cand,
    higher_is_better=True,
    mechanism="unknown",
    noise_pct=2.0,
    accept_pct=3.0,
    x=None,
):
    """The verdict of cand against base. x: (baseline X values, candidate X values, X higher is better) or None."""
    gain, floor = compare(base, cand, higher_is_better, noise_pct)
    x_gain, x_floor = compare(*x, noise_pct) if x else (0.0, 0.0)
    mb, mc = statistics.mean(base), statistics.mean(cand)
    median_agrees = (
        len(base) < 2
        or len(cand) < 2
        or (statistics.median(cand) - statistics.median(base)) * (mc - mb) > 0
    )
    if abs(gain) < floor:
        decision, note = "noise", f"|{gain:+.2f}%| is inside the noise floor {floor:.2f}%"
    elif gain < 0:
        decision, note = "reject", f"worse by {-gain:.2f}% (floor {floor:.2f}%)"
    elif gain < max(accept_pct, floor):
        decision = "rerun"
        note = f"{gain:+.2f}% is between the noise floor and the {accept_pct:g}% bar: run again"
    elif not median_agrees:
        decision, note = "rerun", "the medians disagree with the means: run again"
    elif x and x_gain <= -x_floor:
        decision = "rerun"
        note = f"X is worse by {-x_gain:.2f}% (floor {x_floor:.2f}%): compare at equal X (the working point)"
    elif mechanism != "agree":
        decision = "rerun"
        note = (
            "the mechanism metrics disagree: suspect the measurement"
            if mechanism == "disagree"
            else "show the mechanism metrics agree (queues, step time, hit rates ...) before accepting"
        )
    else:
        decision = "accept"
        note = f"{gain:+.2f}% past the {accept_pct:g}% bar and the floor {floor:.2f}%, mechanism agrees"
    return {
        "status": "valid",
        "decision": decision,
        "vs_baseline_pct": ledger.pct(mc, mb),
        "noise_floor_pct": round(floor, 2),
        **({"x_gain_pct": x_gain} if x else {}),
        "n_baseline": len(base),
        "n_candidate": len(cand),
        "mean_baseline": mb,
        "mean_candidate": mc,
        "notes": note,
    }


def percent(text):
    """A --noise-pct or --accept-pct: a finite number, not negative."""
    value = float(text)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError(f"{text} is not a percentage")
    return value


def values(files, metric):
    out = []
    for f in files:
        doc, errors = ledger.read(Path(f))
        if doc is None or errors:
            raise ValueError(f"{f}: {'; '.join(errors) or 'not an infera.result file'}")
        v = doc["metrics"].get(metric)
        if not ledger.number(v):
            raise ValueError(f"{f}: metrics.{metric} is not a number")
        out.append(float(v))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--baseline", nargs="+", type=float, default=[])
    ap.add_argument("--candidate", nargs="+", type=float, default=[])
    ap.add_argument("--baseline-files", nargs="+", default=[])
    ap.add_argument("--candidate-files", nargs="+", default=[])
    ap.add_argument("--metric", help="the metric to read from result files (metrics.<name>)")
    ap.add_argument("--lower-is-better", action="store_true")
    ap.add_argument("--x-baseline", nargs="+", type=float, default=[])
    ap.add_argument("--x-candidate", nargs="+", type=float, default=[])
    ap.add_argument("--x-metric", help="the X metric (interactivity) to read from the result files")
    ap.add_argument("--x-lower-is-better", action="store_true")
    ap.add_argument("--mechanism", choices=("agree", "disagree", "unknown"), default="unknown")
    ap.add_argument("--noise-pct", type=percent, default=2.0)
    ap.add_argument("--accept-pct", type=percent, default=3.0)
    ap.add_argument(
        "--void", metavar="REASON", help="the measurement is void: record why, decide nothing"
    )
    ap.add_argument("--note", default="", help="a remark added to the verdict's notes")
    ap.add_argument(
        "--write", action="store_true", help="write the verdict into every --candidate-files file"
    )
    args = ap.parse_args(argv)
    if args.write and not args.candidate_files:
        ap.error("--write writes into the --candidate-files")
    try:
        if args.void:
            verdict = {"status": "void", "notes": args.void}
        else:
            if (args.baseline_files or args.candidate_files) and not args.metric:
                ap.error("--metric names the metric to read from the result files")
            base = args.baseline + values(args.baseline_files, args.metric)
            cand = args.candidate + values(args.candidate_files, args.metric)
            x = None
            if args.x_metric or args.x_baseline or args.x_candidate:
                files = (args.baseline_files, args.candidate_files) if args.x_metric else ([], [])
                x = (
                    args.x_baseline + values(files[0], args.x_metric),
                    args.x_candidate + values(files[1], args.x_metric),
                    not args.x_lower_is_better,
                )
            verdict = judge(
                base,
                cand,
                not args.lower_is_better,
                args.mechanism,
                args.noise_pct,
                args.accept_pct,
                x,
            )
        if args.note:
            verdict["notes"] = "; ".join(filter(None, (verdict["notes"], args.note)))
    except ValueError as e:
        print(f"ab_judge: {e}", file=sys.stderr)
        return 2
    for f in args.candidate_files if args.write else []:
        doc = json.loads(Path(f).read_text(encoding="utf-8"))
        write_json(Path(f), {**doc, "verdict": verdict})
    print(json.dumps(verdict, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
