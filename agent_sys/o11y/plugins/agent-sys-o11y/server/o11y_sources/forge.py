"""KernelForge campaign state (Hyperloom AKLR branch; field paths per data2.md §B, pinned @7d51b4af1).

Campaign roots come only from ledger records kind=campaign (`campaign_root`); a model-extracted root must lie
under a session-written root or a fleet launch dir. A root may point at the campaign
dir, its `engine/`, or the `forge_experiments/` dir itself.
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

from . import Problems, owned_json, owned_open, owned_problem

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


FORGE_FILES = ("run_state.json", "events.jsonl", "llm_usage.json", "dashboard.json")


def _under(real: str, bases: list[str]) -> bool:
    return any(real == b or real.startswith(b.rstrip("/") + "/") for b in bases)


def campaign_root_problem(root: str, allowed: list[str] | None = None) -> str | None:
    """Why a ledger campaign root must not be read (None = fine): absolute, depth >= 3, a real directory with
    KernelForge state under it, the root, its forge dir and the forge files ours and not group/world-writable.
    `allowed` (from `allowed_bases`) is given for a model-extracted root, which must lie under one of them (design §8.2)."""
    if not root or not os.path.isabs(root):
        return "not an absolute path"
    outside = (
        "extracted root outside the session-written campaign roots and the fleet's launch dirs"
    )
    if allowed is not None and not _under(os.path.normpath(root), allowed):
        return outside  # decided before touching the path (spec §6.7)
    real = os.path.realpath(root)
    if len(Path(real).parts) < 4:  # "/", "/a", "/a/b" are too shallow
        return "too shallow"
    if not os.path.isdir(real):
        return "not a directory"
    if allowed is not None and not _under(real, allowed):
        return outside  # a symlink inside an allowed base pointing elsewhere
    fdir = resolve_forge_dir(real)
    if fdir is None:
        return "no forge_experiments/run_state.json under it"
    for p in [real, fdir, *(os.path.join(fdir, f) for f in FORGE_FILES)]:
        why = os.path.lexists(p) and owned_problem(p, follow=True, external=True)
        if why:
            return why
    return None


def allowed_bases(records: list[dict], launch_dirs: list[str]) -> list[str]:
    """Where a model-extracted campaign root may lie: under a root a session itself recorded
    (writer.source=written) or under a fleet session's launch directory."""
    written = [
        r["campaign_root"] for r in records if (r.get("writer") or {}).get("source") == "written"
    ]
    bases = [p for p in [*written, *launch_dirs] if p and os.path.isabs(p)]
    return sorted({f(p) for p in bases for f in (os.path.normpath, os.path.realpath)})


def discover(
    records: list[dict], launch_dirs: list[str]
) -> tuple[list[str], list[tuple[str, str]]]:
    """(forge dirs, [(root, problem)]) for the ledger's campaign records (latest per root)."""
    allowed = allowed_bases(records, launch_dirs)
    dirs, bad, seen = [], [], set()
    for r in records:
        extracted = (r.get("writer") or {}).get("source") != "written"
        why = campaign_root_problem(r["campaign_root"], allowed if extracted else None)
        if why:
            bad.append((r["campaign_root"], why))
            continue
        d = resolve_forge_dir(os.path.realpath(r["campaign_root"]))
        if d and os.path.realpath(d) not in seen:
            seen.add(os.path.realpath(d))
            dirs.append(d)
    return dirs, bad


def _forge_ts(s: str) -> float | None:
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
    except (TypeError, ValueError):
        return None


def _load(path: Path):
    doc, why = owned_json(path, external=True)
    if why:
        raise ValueError(why)
    return doc


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
    f, why = owned_open(ev_path, external=True)
    if f is not None:
        with f:
            ev_mtime = os.fstat(f.fileno()).st_mtime  # the file that is read
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
    else:
        problems.add("forge", str(ev_path), "(file)", why)
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
    gain = (best - start) if best is not None else None  # no best yet: no gain, not a 0 % gain
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
        "gain_per_hour": (gain / (campaign_sec / 3600))
        if campaign_sec and gain is not None
        else None,
        "last_decision": (last_iter or {}).get("decision"),
        "last_validation_passed": (last_iter or {}).get("validation_passed"),
        "series": series[-50:],
        "llm_usage": usage,
        "deadline_unix": deadline,
    }
