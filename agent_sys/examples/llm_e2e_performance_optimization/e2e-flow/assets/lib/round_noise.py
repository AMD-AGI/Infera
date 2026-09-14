#!/usr/bin/env python3
"""T7 window: the round-to-round distribution of one deployment, measured.

**Usable at every N, by construction.** The artefact is rewritten after every
round, so a hold cancelled at minute 12 leaves a smaller true statement rather
than nothing. Four holds died today, the shortest at 28 minutes; a protocol that
only pays out at the end would have yielded nothing on any of them.

WHAT IT MEASURES
    One deployment, no patch, no second arm. The replay is run N times back to
    back and each round's `output_token_throughput_tps` and
    `request_throughput_rps` are recorded. Those two carry the 5% bar and are
    the only metrics with NO within-round replicate at any price -- they are
    rates over the window, so only rounds sample them.

WHAT IT REPORTS, AND WHY IT IS NOT AN RSD
    Simulated over heavy-tailed round-to-round noise: at a true rsd of 8% the
    SAMPLE rsd from 20 rounds has median 4.45% and a 5th-95th range of
    2.78%-14.02%. It is **biased low**, because a rare tail is usually missed
    entirely -- which is m4's finding restated, and it is the dangerous
    direction: a short run reports a tightness that is not there. Deciding
    "is the rsd under 5%?" from 20 heavy-tailed rounds is right 43% of the time,
    worse than a coin.

    So this reports **counts, not moments**. "How many rounds deviated from the
    running median by more than the bar" is a direct observation with an exact
    binomial interval, unbiased at every N. With zero bad rounds seen the 95%
    upper bound is the rule of three, 3/N:

        N=12 -> <25%     N=20 -> <15%     N=30 -> <10%     N=60 -> <5%

    **Certifying the 5% bar therefore needs about 60 rounds**, roughly 90
    minutes plus bring-up, which does not fit the holds we can currently keep.
    That is a finding about the allocation, not a reason to report a smaller
    number as if it were the same one.

WINDOW CONDITION IS MEASURED PER ROUND, NOT ASSERTED ONCE
    m4's number is an upper bound because a neighbour sat at 90% throughout,
    and that only became clear afterwards. Here every round records the other
    tenants' GPU processes at the moment it ran, so "idle" is a property of the
    artefact rather than of somebody's recollection. A window that starts idle
    and does not stay idle is visible in the per-round record instead of being
    averaged into the result.

WHY THIS IS IN `assets/lib/` AND NOT AN m5 DIRECTORY
    It measures the instrument rather than this stage. m3's harness and m4's
    microbenchmark have the same question about their own rounds, and the answer
    — a count with an exact binomial bound, not a moment — is the same answer for
    all three. Placement ruled by the leader; `nodeprobe.sh`, `runprobe.py`,
    `read_events.py` and `check_agent_env.py` are the neighbours it belongs with.

USAGE
    round_noise.py --out <dir> --rounds N --label idle|contaminated
    Resumes: an existing rounds.jsonl is continued, so a cancelled window can be
    picked up if the same deployment is still reachable.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from math import comb
from pathlib import Path


def cp_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """Clopper-Pearson 95% upper bound on a rate, exact at any N.

    Exact and not normal: at the N and the rates this protocol lives at, a
    normal interval is wrong in the direction that flatters the measurement.
    """
    if n == 0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        tail = sum(comb(n, i) * mid ** i * (1 - mid) ** (n - i) for i in range(k + 1))
        if tail > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def summarise(rounds: list[dict], label: str, bars=(0.05, 0.10)) -> dict:
    """What the rounds so far support -- and, explicitly, what they do not."""
    out: dict = {
        "n_rounds": len(rounds),
        "window_label": label,
        "metrics": {},
        "conclusions": [],
        "not_yet_supported": [],
    }
    # A window is only "idle" if every round said so.
    conditions = {r.get("neighbour") for r in rounds}
    out["window_was_uniform"] = len(conditions) <= 1
    if not out["window_was_uniform"]:
        out["conclusions"].append(
            "the window did NOT hold one condition throughout; per-round neighbour "
            "records differ, so this is not a clean single-condition sample")

    for metric in ("output_token_throughput_tps", "request_throughput_rps"):
        vals = [r["metrics"].get(metric) for r in rounds]
        vals = [v for v in vals if isinstance(v, (int, float))]
        if len(vals) < 2:
            out["metrics"][metric] = {"n": len(vals), "note": "need at least 2 rounds"}
            continue
        med = st.median(vals)
        devs = [abs(v - med) / med for v in vals] if med else []
        entry = {
            "n": len(vals),
            "median": round(med, 6),
            "min": round(min(vals), 6),
            "max": round(max(vals), 6),
            "observed_max_deviation": round(max(devs), 6) if devs else None,
            # Reported because a reader will look for it, and labelled because
            # it is the statistic this protocol deliberately does not conclude
            # from. See the module docstring.
            "sample_rsd_BIASED_LOW": round(st.pstdev(vals) / med, 6) if med else None,
        }
        for bar in bars:
            k = sum(1 for d in devs if d > bar)
            entry[f"rounds_off_by_more_than_{int(bar*100)}pct"] = k
            entry[f"upper95_rate_off_by_more_than_{int(bar*100)}pct"] = round(
                cp_upper(k, len(vals)), 4)
        out["metrics"][metric] = entry

        bound5 = entry.get("upper95_rate_off_by_more_than_5pct")
        if bound5 is not None:
            if bound5 <= 0.05:
                out["conclusions"].append(
                    f"{metric}: at 95% confidence, fewer than 5% of rounds deviate from the "
                    f"median by more than the 5% bar (n={len(vals)}). This deployment can "
                    "resolve a 5% difference.")
            else:
                need = 60
                out["not_yet_supported"].append(
                    f"{metric}: the 95% upper bound on the rate of rounds off by >5% is "
                    f"{bound5:.0%} at n={len(vals)}. Certifying the 5% bar needs about "
                    f"n={need} with no bad round (rule of three). Do NOT read the sample rsd "
                    "as the answer -- it is biased low under a heavy tail.")
    return out


def write(out_dir: Path, rounds: list[dict], label: str) -> None:
    """Rewrite the artefact. Called after EVERY round; this is the whole design."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rounds.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rounds), encoding="utf-8")
    summary = summarise(rounds, label)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Round-to-round noise of one deployment",
        "",
        f"**Window: {label}.** Rounds completed: **{summary['n_rounds']}**.",
        "",
        "**This artefact is complete at every N.** It is rewritten after each round so a",
        "cancelled hold leaves a smaller true statement rather than nothing.",
        "",
    ]
    if not summary.get("window_was_uniform", True):
        lines += ["> **The window did not hold one condition throughout.** Per-round",
                  "> neighbour records differ. Treat this as a mixed sample, not as the",
                  "> labelled condition.", ""]
    lines += ["## What these rounds support", ""]
    lines += [f"- {c}" for c in summary["conclusions"]] or ["- (nothing yet)"]
    lines += ["", "## What they do NOT yet support", ""]
    lines += [f"- {c}" for c in summary["not_yet_supported"]] or ["- (nothing outstanding)"]
    lines += [
        "",
        "## Why no rsd is concluded here",
        "",
        "Simulated over heavy-tailed round-to-round noise, the sample rsd from 20 rounds",
        "at a true rsd of 8% has median 4.45% and a 5th-95th range of 2.78%-14.02%: it is",
        "**biased low**, because the tail is usually missed. Deciding whether the rsd is",
        "under 5% from 20 such rounds is right 43% of the time. Counts with an exact",
        "binomial bound are unbiased at every N, so that is what is reported above.",
        "",
        "## Standing",
        "",
        "This is **one window**. T7 asks for the neighbour's contribution, which is a",
        "difference between an idle window and a contaminated one. A single window of",
        "either kind is a bound, not that difference, and must not be quoted as the",
        "quiet-node figure.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--label", required=True, choices=("idle", "contaminated", "unknown"))
    ap.add_argument("--dry-run", action="store_true",
                    help="rebuild the artefact from an existing rounds.jsonl and stop")
    args = ap.parse_args()

    out_dir = Path(args.out)
    existing = out_dir / "rounds.jsonl"
    rounds: list[dict] = []
    if existing.is_file():
        rounds = [json.loads(l) for l in existing.read_text().splitlines() if l.strip()]
        print(f"resuming from {len(rounds)} round(s) already recorded")

    if args.dry_run:
        write(out_dir, rounds, args.label)
        print(json.dumps(summarise(rounds, args.label), indent=2))
        return 0

    raise SystemExit(
        "the measurement half is not wired yet: it needs a held node and the leader's\n"
        "scheduling. What is complete and testable today is the artefact half --\n"
        "run with --dry-run over a rounds.jsonl to see what any N would produce.")


if __name__ == "__main__":
    raise SystemExit(main())
