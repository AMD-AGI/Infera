"""KernelForge campaign state (Hyperloom AKLR branch; field paths per data2.md §B, pinned @7d51b4af1).

Campaign roots come from ledger records kind=campaign (`campaign_root`) plus configured globs. A root
may point at the campaign dir, its `engine/`, or the `forge_experiments/` dir itself.
Read directly (no Hyperloom import): run_state.json + events.jsonl (+ optional llm_usage.json,
dashboard.json for live.deadline_unix, forge_sources.json for revisions). `session_status` is
stale-prone (still `running` after the node is gone), so liveness uses the events.jsonl mtime.
"""

from __future__ import annotations

import glob
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from . import Problems

RUN_STATE_REQUIRED = [
    "campaign_id",
    "session_status",
    "termination_reason",
    "iteration",
    "cumulative.iterations",
    "cumulative.kept",
    "cumulative.reverted",
    "best.mean_case_speedup",
    "search_start_mean_case_speedup",
    "baseline_case_times",
    "best_case_times",
    "round_costs.campaign_sec",
    "git_branch",
]
_N = (int, float)
RUN_STATE_TYPES = {
    "campaign_id": (str,),
    "session_status": (str,),
    "termination_reason": (str, type(None)),
    "iteration": (int,),
    "cumulative.iterations": (int,),
    "cumulative.kept": (int,),
    "cumulative.reverted": (int,),
    "best.mean_case_speedup": (*_N, type(None)),
    "search_start_mean_case_speedup": (*_N, type(None)),
    "baseline_case_times": (dict,),
    "best_case_times": (dict,),
    "round_costs.campaign_sec": (*_N, type(None)),
    "git_branch": (str,),
}
EVENT_REQUIRED = ["type", "ts"]
ITER_REQUIRED = ["iter", "decision", "best_after_mean_case_speedup", "is_new_best"]
USAGE_REQUIRED = [
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "total_cost_usd",
]


MAX_SEARCH_DEPTH = 4  # levels below a root searched for forge_experiments/run_state.json


def resolve_forge_dir(root: str) -> str | None:
    """The `forge_experiments/` dir of a campaign root: the root itself, a known child, or one unique hit
    within MAX_SEARCH_DEPTH levels (bounded: roots can come from extracted, i.e. untrusted, records)."""
    p = Path(root)
    for cand in (
        p,
        p / "forge_experiments",
        p / "engine" / "forge_experiments",
        p / "workspace" / "forge_experiments",
    ):
        if (cand / "run_state.json").is_file():
            return str(cand)
    hits = []
    for depth in range(1, MAX_SEARCH_DEPTH + 1):
        pattern = str(p.joinpath(*(["*"] * (depth - 1)), "forge_experiments", "run_state.json"))
        hits += glob.glob(pattern)
        if len(hits) > 1:
            return None
    return str(Path(hits[0]).parent) if len(hits) == 1 else None


def campaign_root_problem(root: str, allowed_roots: list[str]) -> str | None:
    """Why an analyzer-extracted campaign root must not be trusted (None = acceptable): absolute, real
    directory at depth >= 3, containing KernelForge state, and inside an allowed root (configured campaign
    globs' base directories or roots that sessions themselves wrote to the ledger)."""
    if not root or not os.path.isabs(root):
        return "not an absolute path"
    real = os.path.realpath(root)
    if len(Path(real).parts) < 4:  # "/", "/a", "/a/b" are too shallow
        return "too shallow"
    if not os.path.isdir(real):
        return "not a directory"
    if resolve_forge_dir(real) is None:
        return "no forge_experiments/run_state.json under it"
    for a in allowed_roots:
        base = os.path.realpath(glob_base(a))
        if len(Path(base).parts) >= 3 and (real == base or real.startswith(base.rstrip("/") + "/")):
            return None
    return "outside the allowed campaign roots"


def glob_base(pattern: str) -> str:
    """The fixed directory prefix of a glob pattern."""
    parts = []
    for part in Path(pattern).parts:
        if any(c in part for c in "*?["):
            break
        parts.append(part)
    return str(Path(*parts)) if parts else "/"


def discover(ledger_roots: list[str], globs: list[str]) -> list[str]:
    dirs: list[str] = []
    for r in ledger_roots:
        d = resolve_forge_dir(r)
        if d:
            dirs.append(d)
    for g in (
        globs
    ):  # configured roots: every campaign up to MAX_SEARCH_DEPTH + 1 levels below each match
        for base in glob.glob(g):
            for depth in range(0, MAX_SEARCH_DEPTH + 2):
                pattern = os.path.join(
                    base, *(["*"] * depth), "forge_experiments", "run_state.json"
                )
                dirs += [str(Path(hit).parent) for hit in glob.glob(pattern)]
    seen, out = set(), []
    for d in dirs:
        rd = os.path.realpath(d)
        if rd not in seen:
            seen.add(rd)
            out.append(d)
    return out


def _forge_ts(s: str) -> float | None:
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
    except (TypeError, ValueError):
        return None


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_campaign(
    fdir: str, problems: Problems, now: float | None = None, stale_s: int = 1800
) -> dict | None:
    now = now or time.time()
    d = Path(fdir)
    rs_path = d / "run_state.json"
    try:
        rs = _load(rs_path)
    except (OSError, ValueError) as e:
        problems.add("forge", str(rs_path), "(file)", f"unreadable: {e}")
        return None
    if not isinstance(rs, dict) or not problems.require(
        "forge", str(rs_path), rs, RUN_STATE_REQUIRED, RUN_STATE_TYPES
    ):
        return None
    ev_path = d / "events.jsonl"
    first_start = None
    last_iter = None
    series = []
    bad = set()
    try:
        with open(ev_path, encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(e, dict):
                    continue
                miss = [k for k in EVENT_REQUIRED if k not in e]
                if e.get("type") == "iteration_result":
                    miss += [k for k in ITER_REQUIRED if k not in e]
                if miss:
                    for k in miss:
                        if k not in bad:
                            bad.add(k)
                            problems.add("forge", str(ev_path), k, "missing required field")
                    continue
                if e["type"] == "session_started" and first_start is None:
                    first_start = _forge_ts(e["ts"])
                if e["type"] == "iteration_result":
                    last_iter = e
                    series.append(
                        {
                            "ts": e["ts"],
                            "best": e["best_after_mean_case_speedup"],
                            "new_best": e["is_new_best"],
                            "decision": e["decision"],
                        }
                    )
        ev_mtime = os.path.getmtime(ev_path)
    except FileNotFoundError:
        problems.add("forge", str(ev_path), "(file)", "missing")
        ev_mtime = os.path.getmtime(rs_path)
    usage = None
    up = d / "llm_usage.json"
    if up.exists():
        try:
            u = _load(up)
            if problems.require("forge", str(up), u, USAGE_REQUIRED):
                usage = {k: u[k] for k in USAGE_REQUIRED}
        except (OSError, ValueError):
            pass
    deadline = None
    dp = d / "dashboard.json"
    if dp.exists():
        try:
            deadline = (_load(dp).get("live") or {}).get("deadline_unix")
        except (OSError, ValueError):
            pass
    best = rs["best"]["mean_case_speedup"]
    start = rs["search_start_mean_case_speedup"]
    start = 1.0 if start is None else start
    campaign_sec = rs["round_costs"]["campaign_sec"] or 0
    gain = (best - start) if best is not None else 0.0
    status = rs["session_status"]
    age_idle = now - ev_mtime
    live = status == "running" and age_idle <= stale_s
    branch = rs["git_branch"] or ""
    operator = (
        branch.split("/", 1)[1] if branch.startswith("forge/") else Path(fdir).parent.parent.name
    )
    return {
        "forge_dir": fdir,
        "campaign_id": rs["campaign_id"],
        "operator": operator,
        "status": status,
        "termination_reason": rs["termination_reason"],
        "stale": status == "running" and not live,
        "idle_s": round(age_idle),
        "started_at": first_start,
        "wall_age_s": (now - first_start) if first_start else None,
        "campaign_sec": campaign_sec,
        "iterations": rs["cumulative"]["iterations"],
        "kept": rs["cumulative"]["kept"],
        "reverted": rs["cumulative"]["reverted"],
        "best_speedup": best,
        "start_speedup": start,
        "gain": gain,
        "gain_per_hour": (gain / (campaign_sec / 3600)) if campaign_sec else None,
        "last_decision": (last_iter or {}).get("decision"),
        "last_validation_passed": (last_iter or {}).get("validation_passed"),
        "series": series[-50:],
        "llm_usage": usage,
        "deadline_unix": deadline,
    }
