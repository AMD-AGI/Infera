###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Deployment mix: one configuration per model and latency tier.

An operator serving several models to several kinds of user does not want a
frontier, they want an answer per cell: which configuration runs model M for
users who need at least X tokens/s each and a first token within Y seconds.
This reads the points ``inferasim pareto`` projected and picks one per
(model, scenario, tier).

The rule, per cell:

1. A point qualifies if its interactivity meets the tier's floor and its
   projected TTFT is at most ``(1 - ttft_margin)`` of the tier's limit.
   If the margin leaves nothing, the plain limit is used and the pick says so.
2. Among qualifying points within ``tie`` of the best projected throughput per
   GPU, take the one with the most SLO headroom: the smaller of
   ``interactivity / floor`` and ``limit / TTFT``.

A point projected in both modes is taken from its anchored (``benchmark``)
projection. Simulate-mode points that use decode context parallelism are left
out, because the analytical path prices DCP attention poorly; an anchored DCP
point is kept.

``top`` lists more than the pick. After the pick come the points that meet the
plain limit, ranked the same way: those within ``tie`` of the best by
headroom, then the rest by throughput.

The picks are predictions. Treat the mix as a short list and confirm each pick
with one measured run before deploying it; ``--top 5`` gives the list to confirm.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from .pareto import _IDENTITY, METRICS, ParetoPoint, load_points


@dataclass(frozen=True)
class Tier:
    """A user class: an interactivity floor and a TTFT limit (``None`` = none)."""

    name: str
    min_interactivity: float = 0.0
    max_ttft_ms: float | None = None
    note: str = ""


DEFAULT_TIERS: tuple[Tier, ...] = (
    Tier("realtime", 150.0, 1_000.0, "a person watching the tokens arrive"),
    Tier("interactive", 70.0, 2_500.0, "an agent in a request/response loop with a user waiting"),
    Tier(
        "responsive", 25.0, 10_000.0, "a background agent fleet expected to finish a turn promptly"
    ),
    Tier("bulk", 0.0, None, "offline batch; only aggregate throughput matters"),
)

MODES = ("auto", "benchmark", "simulate")


@dataclass
class MixPick:
    model: str
    scenario: str
    tier: str
    min_interactivity: float
    max_ttft_ms: float | None
    point: ParetoPoint | None
    candidates: int
    qualifying: int
    margin_applied: bool
    headroom: float
    rank: int = 1

    def row(self) -> dict:
        p = self.point
        return dict(
            model=self.model,
            scenario=self.scenario,
            tier=self.tier,
            rank=self.rank,
            min_interactivity=self.min_interactivity,
            max_ttft_ms=self.max_ttft_ms,
            picked=p is not None,
            candidates=self.candidates,
            qualifying=self.qualifying,
            margin_applied=self.margin_applied,
            headroom=round(self.headroom, 4),
            engine=p.engine if p else "",
            precision=p.precision if p else "",
            mode=p.mode if p else "",
            parallelism=p.parallelism if p else "",
            disagg=p.disagg if p else "",
            spec=(f"{p.spec}:{p.spec_k}" if p.spec != "none" else "none") if p else "",
            knobs=p.knobs if p else "",
            concurrency=p.concurrency if p else 0,
            gpus=p.gpus if p else 0,
            interactivity=round(p.interactivity, 2) if p else 0.0,
            ttft_ms=round(p.ttft_ms, 1) if p else 0.0,
            total_tput_per_gpu=round(p.total_tput_per_gpu, 1) if p else 0.0,
            output_tput_per_gpu=round(p.output_tput_per_gpu, 1) if p else 0.0,
            anchor=p.anchor if p else "",
        )


def load_tiers(path: str) -> tuple[Tier, ...]:
    """Tiers from a JSON or YAML file: a list, or a mapping of name to fields.

    Each tier takes ``min_interactivity`` (tokens/s per user, 0 for none) and
    ``max_ttft_ms`` (null for none); ``intvty`` and ``ttft_ms`` are accepted as
    the InferenceX spellings.
    """
    with open(path) as f:
        text = f.read()
    if path.endswith((".yaml", ".yml")):
        import yaml

        doc = yaml.safe_load(text)
    else:
        doc = json.loads(text)
    if isinstance(doc, dict):
        doc = doc.get("tiers", doc)
    if isinstance(doc, dict):
        doc = [dict(name=k, **(v or {})) for k, v in doc.items()]
    tiers = []
    for t in doc:
        floor = t.get("min_interactivity", t.get("intvty", 0.0)) or 0.0
        limit = t.get("max_ttft_ms", t.get("ttft_ms"))
        tiers.append(
            Tier(
                name=str(t["name"]),
                min_interactivity=float(floor),
                max_ttft_ms=None if limit is None else float(limit),
                note=str(t.get("note", "")),
            )
        )
    if not tiers:
        raise ValueError(f"{path}: no tiers")
    return tuple(tiers)


def _config_key(p: ParetoPoint) -> tuple:
    return tuple(getattr(p, k) for k in _IDENTITY if k != "mode")


def candidates(
    points: Sequence[ParetoPoint], mode: str = "auto", metric: str = "total_tput_per_gpu"
):
    """The points a pick may come from.

    Infeasible and unpriced points are dropped, as are simulate-mode DCP
    points. ``auto`` keeps one point per configuration, the anchored one where
    both modes were projected.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    keep = [
        p
        for p in points
        if p.feasible
        and getattr(p, metric) > 0
        and p.interactivity > 0
        and not (p.dcp > 1 and p.mode != "benchmark")
        and (mode == "auto" or p.mode == mode)
    ]
    if mode != "auto":
        return keep
    best: dict[tuple, ParetoPoint] = {}
    for p in keep:
        k = _config_key(p)
        if k not in best or (p.mode == "benchmark" and best[k].mode != "benchmark"):
            best[k] = p
    return list(best.values())


def _ttft(p: ParetoPoint, ttft_field: str) -> float:
    return getattr(p, ttft_field) or p.ttft_ms


def meets(
    p: ParetoPoint, tier: Tier, ttft_margin: float = 0.0, ttft_field: str = "ttft_ms"
) -> bool:
    if tier.min_interactivity and p.interactivity < tier.min_interactivity:
        return False
    if tier.max_ttft_ms is not None and _ttft(p, ttft_field) > tier.max_ttft_ms * (1 - ttft_margin):
        return False
    return True


def headroom(p: ParetoPoint, tier: Tier, ttft_field: str = "ttft_ms") -> float:
    """The tighter of the two SLO ratios; 1.0 for a tier with no limits."""
    h = []
    if tier.min_interactivity:
        h.append(p.interactivity / tier.min_interactivity)
    if tier.max_ttft_ms is not None:
        h.append(tier.max_ttft_ms / max(_ttft(p, ttft_field), 1e-3))
    return min(h) if h else 1.0


def rank(
    cands: Sequence[ParetoPoint],
    tier: Tier,
    ttft_margin: float = 0.5,
    tie: float = 0.05,
    metric: str = "total_tput_per_gpu",
    ttft_field: str = "ttft_ms",
) -> tuple[list[ParetoPoint], bool]:
    """``(qualifying points best first, margin applied)`` for one tier.

    Points within ``tie`` of the best throughput come first, by SLO headroom;
    the rest follow by throughput. The first point is the pick.
    """
    ok = [p for p in cands if meets(p, tier, ttft_margin, ttft_field)]
    margin_applied = bool(ok) and ttft_margin > 0
    if not ok:
        ok = [p for p in cands if meets(p, tier, 0.0, ttft_field)]
    if not ok:
        return [], False
    top = max(getattr(p, metric) for p in ok)

    def key(p: ParetoPoint) -> tuple:
        v = getattr(p, metric)
        if v >= top * (1 - tie):
            return (1, headroom(p, tier, ttft_field), v)
        return (0, v, 0.0)

    return sorted(ok, key=key, reverse=True), margin_applied


def pick(
    cands: Sequence[ParetoPoint],
    tier: Tier,
    ttft_margin: float = 0.5,
    tie: float = 0.05,
    metric: str = "total_tput_per_gpu",
    ttft_field: str = "ttft_ms",
) -> tuple[ParetoPoint | None, int, bool]:
    """``(pick, qualifying count, margin applied)`` for one tier."""
    ranked, margin_applied = rank(cands, tier, ttft_margin, tie, metric, ttft_field)
    return (ranked[0] if ranked else None), len(ranked), margin_applied


def shortlist(
    cands: Sequence[ParetoPoint],
    tier: Tier,
    k: int,
    ttft_margin: float = 0.5,
    tie: float = 0.05,
    metric: str = "total_tput_per_gpu",
    ttft_field: str = "ttft_ms",
) -> list[ParetoPoint]:
    """The pick, then the next ``k - 1`` points ranked against the plain limit.

    The margin steers only the pick: a short list confirmed on hardware loses
    more by dropping fast points near the limit than it gains from the margin.
    """
    ranked, _ = rank(cands, tier, ttft_margin, tie, metric, ttft_field)
    if not ranked or k <= 1 or ttft_margin == 0:
        return ranked[:k]
    plain, _ = rank(cands, tier, 0.0, tie, metric, ttft_field)
    return [ranked[0]] + [p for p in plain if p is not ranked[0]][: k - 1]


def plan_mix(
    points: Sequence[ParetoPoint],
    tiers: Sequence[Tier] = DEFAULT_TIERS,
    ttft_margin: float = 0.5,
    tie: float = 0.05,
    metric: str = "total_tput_per_gpu",
    mode: str = "auto",
    ttft_field: str = "ttft_ms",
    top: int = 1,
) -> list[MixPick]:
    """Per (model, scenario, tier), in input order of models and scenarios, the
    pick and up to ``top - 1`` more points from :func:`shortlist`. A tier
    nothing meets gets one empty pick."""
    if not 0 <= ttft_margin < 1:
        raise ValueError("ttft_margin must be in [0, 1)")
    if tie < 0:
        raise ValueError("tie must be >= 0")
    if top < 1:
        raise ValueError("top must be >= 1")
    cands = candidates(points, mode, metric)
    groups: dict[tuple[str, str], list[ParetoPoint]] = {}
    for p in points:
        groups.setdefault((p.model, p.scenario), [])
    for p in cands:
        groups[(p.model, p.scenario)].append(p)
    out = []
    for (model, scenario), pts in groups.items():
        for t in tiers:
            best, n_ok, applied = pick(pts, t, ttft_margin, tie, metric, ttft_field)
            listed = shortlist(pts, t, top, ttft_margin, tie, metric, ttft_field)
            for i, p in enumerate(listed or [None], 1):
                out.append(
                    MixPick(
                        model=model,
                        scenario=scenario,
                        tier=t.name,
                        min_interactivity=t.min_interactivity,
                        max_ttft_ms=t.max_ttft_ms,
                        point=p,
                        candidates=len(pts),
                        qualifying=n_ok,
                        margin_applied=(
                            applied
                            if p is best
                            else ttft_margin > 0 and meets(p, t, ttft_margin, ttft_field)
                        ),
                        headroom=headroom(p, t, ttft_field) if p else 0.0,
                        rank=i if p else 0,
                    )
                )
    return out


def format_table(picks: Sequence[MixPick]) -> str:
    ranked = any(m.rank > 1 for m in picks)
    head = (
        "model",
        "scenario",
        "tier",
        "#",
        "configuration",
        "clients",
        "tok/s/user",
        "TTFT ms",
        "tok/s/GPU",
        "mode",
    )
    rows = [head]
    for m in picks:
        p = m.point
        if p is None:
            rows.append(
                (m.model, m.scenario, m.tier, "", "nothing meets this tier", "", "", "", "", "")
            )
            continue
        cfg = " ".join(
            x
            for x in (
                p.engine,
                p.precision,
                p.disagg or p.parallelism,
                "" if p.spec == "none" else f"{p.spec}:{p.spec_k}",
                p.knobs,
            )
            if x
        )
        if not m.margin_applied and m.max_ttft_ms is not None:
            cfg += " (no TTFT margin)"
        rows.append(
            (
                m.model,
                m.scenario,
                m.tier,
                str(m.rank),
                cfg,
                str(p.concurrency),
                f"{p.interactivity:.1f}",
                f"{p.ttft_ms:.0f}",
                f"{p.total_tput_per_gpu:.0f}",
                p.mode,
            )
        )
    if not ranked:
        rows = [r[:3] + r[4:] for r in rows]
    w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.ljust(w[i]) for i, c in enumerate(r)).rstrip() for r in rows)


def write_outputs(picks: Sequence[MixPick], out_dir: str, meta: dict) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    rows = [m.row() for m in picks]
    pj = os.path.join(out_dir, "mix.json")
    with open(pj, "w") as f:
        json.dump(dict(meta=meta, picks=rows), f, indent=1)
    pc = os.path.join(out_dir, "mix.csv")
    with open(pc, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["model"])
        w.writeheader()
        w.writerows(rows)
    return [pj, pc]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="inferasim mix",
        description="Pick one configuration per model and latency tier from an "
        "`inferasim pareto` sweep.",
    )
    ap.add_argument(
        "points",
        help="pareto_points.json written by `inferasim pareto`, or its --out-dir.",
    )
    ap.add_argument(
        "--tiers",
        default=None,
        help="JSON or YAML file of tiers (name, min_interactivity, max_ttft_ms). "
        "Default: realtime (>=150 tok/s/user, TTFT<=1 s), interactive (>=70, <=2.5 s), "
        "responsive (>=25, <=10 s) and bulk (no limits).",
    )
    ap.add_argument(
        "--ttft-margin",
        type=float,
        default=0.5,
        help="Require projected TTFT <= (1 - margin) x the tier's limit (default 0.5). "
        "Falls back to the plain limit when nothing qualifies.",
    )
    ap.add_argument(
        "--tie",
        type=float,
        default=0.05,
        help="Points within this fraction of the best throughput are ties, broken by "
        "SLO headroom (default 0.05).",
    )
    ap.add_argument(
        "--metric",
        default="total_tput_per_gpu",
        choices=[k for k, v in METRICS.items() if v[1] and k.endswith("per_gpu")],
        help="Throughput to maximize (default: total tokens/s per GPU, as InferenceX ranks).",
    )
    ap.add_argument(
        "--ttft-field",
        default="ttft_ms",
        choices=["ttft_ms", "ttft_p90_ms"],
        help="Projected TTFT the limit is checked against (default: ttft_ms).",
    )
    ap.add_argument(
        "--mode",
        default="auto",
        choices=MODES,
        help="Projection mode to pick from; auto prefers the anchored point of each "
        "configuration (default).",
    )
    ap.add_argument(
        "--top",
        type=int,
        default=1,
        help="List the pick and the next best points per tier, up to this many, to "
        "confirm on hardware; points after the pick need only meet the plain TTFT "
        "limit (default 1).",
    )
    ap.add_argument("--model", nargs="+", default=None, help="Only these models.")
    ap.add_argument("--scenario", nargs="+", default=None, help="Only these scenarios.")
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Write mix.json and mix.csv here (default: next to the points file).",
    )
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    path = a.points
    if os.path.isdir(path):
        path = os.path.join(path, "pareto_points.json")
    if not os.path.exists(path):
        print(f"[inferasim:mix] no points file at {path}; run `inferasim pareto` first.")
        return 2
    points = load_points(path)
    if a.model:
        points = [p for p in points if p.model in a.model]
    if a.scenario:
        points = [p for p in points if p.scenario in a.scenario]
    if not points:
        print(f"[inferasim:mix] {path} holds no matching points.")
        return 2
    tiers = load_tiers(a.tiers) if a.tiers else DEFAULT_TIERS
    picks = plan_mix(
        points,
        tiers,
        ttft_margin=a.ttft_margin,
        tie=a.tie,
        metric=a.metric,
        mode=a.mode,
        ttft_field=a.ttft_field,
        top=a.top,
    )
    print(format_table(picks))
    meta = dict(
        points=os.path.abspath(path),
        tiers=[asdict(t) for t in tiers],
        ttft_margin=a.ttft_margin,
        tie=a.tie,
        metric=a.metric,
        ttft_field=a.ttft_field,
        mode=a.mode,
        top=a.top,
    )
    out_dir = a.out_dir or os.path.dirname(os.path.abspath(path))
    for f in write_outputs(picks, out_dir, meta):
        print(f"[inferasim:mix] wrote {f}")
    print(
        "[inferasim:mix] these are predictions: confirm each pick with one measured run "
        "before deploying it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
