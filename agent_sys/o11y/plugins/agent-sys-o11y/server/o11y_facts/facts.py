"""facts/{general,detail,kernelforge}.json (facts.<tab>.v1) from the ledger, the registry and the structured
sources; no model (design §8.3). Owns the usage totals (`general.spend`), which o11y_log diffs for
tokens.crossed. The ledger's latest record per identity wins; a recorded `best` is never overridden by a kept
`run_end` (review C3). Each document is validated before its atomic write; an invalid one is not written."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from o11y_job import door
from o11y_sources import e2eflow
from o11y_sources import pool as pool_src

STAGES = [
    "hot",
    "selected",
    "priced",
    "workset",
    "optimizing",
    "candidate",
    "handed_off",
    "e2e_testing",
    "e2e_accepted",
    "e2e_rejected",
    "e2e_inconclusive",
    "dropped",
]
CAMPAIGN_STATUS = {"running", "paused", "completed", "interrupted"}
TIMES = {"started_at", "ended_at", "since", "until", "at", "last_event_at", "last_at"}


def _t(s: str | None) -> float | None:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except (AttributeError, ValueError):
        return None


def _iso(epoch: float | None) -> str | None:
    return (
        datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if epoch
        else None
    )


def _r(x, n=6):
    return None if x is None else round(x, n)


def _clean(d: dict) -> dict:
    """Drop None-valued time keys (not nullable in the schemas)."""
    return {k: v for k, v in d.items() if not (v is None and k in TIMES)}


def _when(r: dict) -> str | None:
    return r.get("at") or r.get("ts")


# Fleet-level kinds: only the main session's records describe the fleet; a child's plan is its own next steps.
FLEET_LEVEL = ("task", "best", "plan")


def main_latest(records: list, kind: str, rows: list) -> list:
    """[the main's latest record of `kind`] (written beats extracted), or []."""
    mine = [r for r in records if r["kind"] == kind and door.api.record_role(r, rows) == "main"]
    written = [r for r in mine if r["writer"].get("source") == "written"]
    return (written or mine)[-1:]


def _next(plan: dict | None) -> list:
    keys = ("theme", "why", "point", "blocked_on")
    return [{k: p.get(k) for k in keys} for p in ((plan or {}).get("next") or [])[:3]]


def child_plans(records: list, rows: list) -> list:
    """Each child's own latest plan (written beats extracted), in registry order."""
    out = []
    for row in rows:
        if row.get("role") != "child":
            continue
        ids = {row["session_id"], *(row.get("lineage") or [])}
        mine = [
            r for r in records
            if r["kind"] == "plan" and (r.get("writer") or {}).get("session_id") in ids
            and door.api.record_role(r, rows) == "child"
        ]  # fmt: skip
        written = [r for r in mine if r["writer"].get("source") == "written"]
        if mine:
            plan = (written or mine)[-1]
            out.append(
                {"session_id": row["session_id"], "name": row.get("name"), "next": _next(plan)}
            )
    return out


def general(by_kind: dict, usage: dict, fleet: list, step: int, now: float) -> dict:
    task = (by_kind.get("task") or [None])[-1]
    task_doc = None
    if task:
        task_doc = {
            "task_id": task["task_id"],
            "title": task["title"],
            "metric": {k: task["metric"][k] for k in ("name", "unit", "direction")},
            "baseline": {k: v for k, v in task["baseline"].items() if k in ("value", "label")},
        }
        if isinstance((task.get("target") or {}).get("value"), (int, float)):
            task_doc["target"] = {k: v for k, v in task["target"].items() if k in ("value", "rule")}
        if task.get("phase"):
            task_doc["phase"] = task["phase"]
    sign = -1 if (task_doc or {}).get("metric", {}).get("direction") == "lower" else 1
    baseline = (task_doc or {}).get("baseline", {}).get("value")
    target = ((task_doc or {}).get("target") or {}).get("value")
    best, rec = None, (by_kind.get("best") or [None])[-1]
    if rec:
        best = {
            "value": rec["value"],
            "run_ids": list(rec["run_ids"]),
            "at": _when(rec),
            **{k: rec[k] for k in ("aggregate", "stack") if rec.get(k)},
        }
    else:  # no recorded best: the best kept run
        for r in by_kind.get("run_end", []):
            if r["status"] == "kept" and (best is None or sign * (r["value"] - best["value"]) > 0):
                best = {"value": r["value"], "run_ids": [r["run_id"]], "at": _when(r)}
    gain_abs = sign * (best["value"] - baseline) if best and baseline is not None else None
    gain_pct = gain_abs / abs(baseline) * 100 if gain_abs is not None and baseline else None
    if best:
        best["vs_baseline_pct"] = _r(gain_pct, 3)
        best["of_target_pct"] = (
            _r((best["value"] - baseline) / (target - baseline) * 100, 2)
            if None not in (baseline, target) and target != baseline
            else None
        )
        best = _clean(best)
    firsts = [x for x in (_t(u["first_at"]) for u in usage.values()) if x]
    # the task started at the earliest of its record and the fleet's first history line (an extracted task
    # record is written long after the work began)
    starts = [x for x in [_t(_when(task)) if task else None, *firsts] if x]
    t0 = min(starts) if starts else None
    wall_s = round(now - t0, 1) if t0 else 0.0
    by_id = {s.session_id: s for s in fleet}
    sessions, tot = [], {"new": 0, "cache_read": 0}
    for sid, u in usage.items():
        s = by_id.get(sid)
        tot["new"] += u["tokens"]["new"]
        tot["cache_read"] += u["tokens"]["cache_read"]
        sessions.append(
            _clean(
                {
                    "session_id": sid,
                    "name": s.name if s else None,
                    "role": s.role if s else None,
                    "tokens": u["tokens"],
                    "steps_crossed": u["tokens"]["new"] // step,
                    "last_at": u["last_at"],
                }
            )
        )
    hours, mtok = wall_s / 3600, tot["new"] / 1e6
    return {
        "task": task_doc,
        "best": best,
        "spend": _clean(
            {
                "tokens_new": tot["new"],
                "tokens_cache_read": tot["cache_read"],
                "steps_crossed": tot["new"] // step,
                "wall_s": wall_s,
                "started_at": _iso(t0),
                "sessions": sessions,
            }
        ),
        "progress": {
            "gain_abs": _r(gain_abs),
            "gain_pct": _r(gain_pct, 3),
            "per_hour_abs": _r(gain_abs / hours) if gain_abs is not None and hours else None,
            "per_hour_pct": _r(gain_pct / hours, 4) if gain_pct is not None and hours else None,
            "per_mtok_abs": _r(gain_abs / mtok) if gain_abs is not None and mtok else None,
            "per_mtok_pct": _r(gain_pct / mtok, 6) if gain_pct is not None and mtok else None,
        },
    }


def detail(by_kind: dict, grants: list, fleet: list, campaigns: list = (), rows: list = ()) -> dict:
    name_of = {s.session_id: s.name or s.session_id for s in fleet}
    for row in rows:  # a run opened before a /clear carries the session's earlier id
        for old_id in row.get("lineage") or []:
            name_of.setdefault(old_id, name_of.get(row.get("session_id")) or row.get("session_id"))

    def who(
        r: dict,
    ) -> str | None:  # its own `session`, else the writer's registry name, else as written
        w = r.get("writer") or {}
        return (
            r.get("session")
            or name_of.get(w.get("session_id"))
            or w.get("session_name")
            or w.get("session_id")
        )

    ends = {r["run_id"] for r in by_kind.get("run_end", [])}
    starts = {r["run_id"]: r for r in by_kind.get("run_start", [])}
    open_runs = [r for r in starts.values() if r["run_id"] not in ends]
    current = [
        _clean(
            {
                "run_id": r["run_id"],
                "theme": r["theme"],
                "hypothesis": r.get("hypothesis"),
                "point": r.get("point"),
                "machines": [m["node"] for m in r.get("machines") or []],
                "started_at": _when(r),
                "session": who(r),
            }
        )
        for r in open_runs[-5:]
    ]
    recent = []
    for r in sorted(by_kind.get("run_end", []), key=lambda r: _when(r) or "")[-3:][::-1]:
        st, delta = starts.get(r["run_id"], {}), r.get("delta") or {}
        recent.append(
            _clean(
                {
                    "run_id": r["run_id"],
                    "theme": r.get("theme") or st.get("theme") or "-",
                    "point": r.get("point") or st.get("point"),
                    "status": r["status"],
                    "value": r.get("value"),
                    "delta_pct": delta.get("pct"),
                    "delta_vs": delta.get("vs"),
                    "new_best": bool(r.get("new_best")),
                    "started_at": _when(st) if st else None,
                    "ended_at": _when(r),
                    "conclusion": r.get("conclusion"),
                    "session": who(st or r),
                }
            )
        )
    nxt = _next((by_kind.get("plan") or [None])[-1])
    # one entry per node: a held machine record, else an open run, else a running campaign, else a pool grant
    entries: dict[str, dict] = {}
    for m in by_kind.get("machine", []):
        if m["state"] == "held":
            entries[m["node"]] = {
                **{k: m[k] for k in ("node", "cluster", "jobid", "until") if m.get(k)},
                "task": m.get("task"),
                "session": who(m),
                "since": _when(m),
                "source": "machine",
            }
    for r in open_runs:
        for m in r.get("machines") or []:
            entries.setdefault(
                m["node"],
                {"node": m["node"], **({"cluster": m["cluster"]} if m.get("cluster") else {}),
                 "runs": f"{r['run_id']}: {r['theme']}", "task": r["theme"], "session": who(r), "source": "ledger"},
            )  # fmt: skip
    for c in campaigns:
        node = c.get("machine")
        if node and c.get("status") == "running":
            task = "kf: " + ", ".join(c.get("operator_ids") or [])
            entries.setdefault(
                node,
                {"node": node, "task": task, "session": c.get("session"), "source": "campaign"},
            )
    for g in grants:
        owner = pool_src.attribute(g.get("notify") or "", g["ours"], fleet)
        for n in g["nodes"] if owner else []:
            e = entries.setdefault(
                n["node"],
                {
                    "node": n["node"],
                    "task": g.get("purpose"),
                    "runs": g.get("purpose"),
                    "source": "pool",
                },
            )
            e.update(cluster=e.get("cluster") or g.get("cluster") or "", jobid=n["jobid"], since=n.get("since"),
                     until=g.get("until"), gpu_util=n.get("gpu_util"), session=e.get("session") or owner.label)  # fmt: skip
            if e["source"] == "ledger":
                e["source"] = "both"
    machines = [_clean(e) for e in entries.values()]
    return {
        "current_runs": current,
        "recent_runs": recent,
        "next": nxt,
        "machines": machines,
        "now": now_doing(open_runs, fleet, rows),
    }


def now_doing(open_runs: list, fleet: list, rows: list = ()) -> list:
    """What each session is doing: its latest open run (also one started before a /clear: registry lineage),
    else its registry status (main first)."""
    lineage = {r.get("session_id"): r.get("lineage") or [] for r in rows}
    out = []
    for s in sorted(
        (s for s in fleet if s.role in ("main", "child")), key=lambda s: s.role != "main"
    ):
        ids = {s.session_id, *lineage.get(s.session_id, [])}
        run = next(
            (r for r in reversed(open_runs) if (r.get("writer") or {}).get("session_id") in ids),
            None,
        )
        if run:
            activity = f"run {run['theme']}" + (f" ({run['point']})" if run.get("point") else "")
        elif s.status == "waiting":
            activity = f"waiting ({s.waiting_for or 'unknown'})"
        elif s.status == "busy":
            activity = "busy (no run recorded)"
        else:
            activity = s.status or "unknown"
        out.append(
            {
                "session_id": s.session_id,
                "name": s.name or None,
                "role": s.role,
                "activity": activity,
                "run_id": run["run_id"] if run else None,
                "theme": run["theme"] if run else None,
                "point": run.get("point") if run else None,
                "status": s.status,
                "since": s.status_since,
            }
        )
    return out


def reported_campaign(rec: dict, root: str, why: str) -> dict:
    """A ledger campaign whose root cannot be read here, from the progress the record reports."""
    p = rec.get("progress") or {}
    return {
        "campaign_root": root,
        "operator_ids": list(rec["operator_ids"]),
        "engine": rec.get("engine") or "hyperloom-forge",
        "status": p.get("status") or "unknown",
        "started_at": rec["started_at"],
        "iterations": p.get("iterations"),
        "kept": p.get("kept"),
        "start_speedup": p.get("start_speedup"),
        "best_speedup": p.get("best_speedup"),
        "machine": (rec.get("machine") or {}).get("node"),
        "session": (rec.get("writer") or {}).get("session_name")
        or (rec.get("writer") or {}).get("session_id"),
        "source": "reported",
        "as_of": p.get("as_of"),
        "problem": f"reported, not readable here: {why}",
    }


def kernelforge(by_kind: dict, campaigns: list, globs: list, problems, now: float) -> dict:
    hot = []
    ht = e2eflow.hot_ops(globs, problems) if globs else None
    if ht:
        hot.append(
            {
                "profile_id": Path(ht["source"]).parent.name or "kernel_table",
                "role": "e2e",
                "scope": ht["source"],
                "ops": [
                    {"name": o["name"], "share_pct": float(o["share_pct"]), "rank": i + 1}
                    for i, o in enumerate(ht["ops"])
                ],
            }
        )
    hot += [
        {
            "profile_id": h["profile_id"],
            "role": h["role"],
            "scope": h.get("scope"),
            "ops": [
                {k: o[k] for k in ("name", "share_pct", "rank", "op_id") if k in o}
                for o in h["ops"]
            ],
        }
        for h in by_kind.get("hotops", [])
    ]
    ops = {
        k["op_id"]: {
            "op_id": k["op_id"],
            "stage": k["stage"],
            "share_pct": dict(k.get("share_pct") or {}),
            "e2e_verdict": (k.get("e2e") or {}).get("verdict"),
            "e2e_delta_pct": (k.get("e2e") or {}).get("delta_pct"),
            "campaign_root": k.get("campaign_root"),
            "_src": (k.get("writer") or {}).get("source"),
        }
        for k in by_kind.get("kernel", [])
    }
    for op, r in (e2eflow.integration_reports(globs, problems) if globs else {}).items():
        o = ops.setdefault(op, {"op_id": op, "share_pct": {}, "campaign_root": None})
        o["stage"], o["_src"] = ("e2e_accepted" if r["accepted"] else "e2e_rejected"), "e2e"
        o["e2e_verdict"] = "accepted" if r["accepted"] else "rejected"
        sp = r.get("observed_e2e_speedup")
        o["e2e_delta_pct"] = _r((sp - 1) * 100, 3) if isinstance(sp, (int, float)) else None
    out = []
    for c in campaigns:
        led = c["ledger"] or {}
        fd = Path(c["forge_dir"])
        op_dir = next(
            (str(p) for p in fd.parents if p.name not in ("engine", "workspace")), str(fd)
        )
        led_ops = led.get("operator_ids") or []
        op_ids = led_ops if len(led_ops) == 1 else [c["operator"]]  # one forge dir = one operator
        root = led.get("campaign_root") if len(led_ops) == 1 else op_dir
        status = (
            "stale"
            if c["stale"]
            else (c["status"] if c["status"] in CAMPAIGN_STATUS else "unknown")
        )
        gain_pct = (
            c["gain"] / c["start_speedup"] * 100
            if c.get("start_speedup") and c["gain"] is not None
            else None
        )
        active_h = (c.get("campaign_sec") or 0) / 3600
        out.append(
            _clean(
                {
                    "campaign_root": root,
                    "operator_ids": op_ids,
                    "engine": led.get("engine") or "hyperloom-forge",
                    "status": status,
                    "started_at": _iso(c.get("started_at")),
                    "age_s": _r(c.get("wall_age_s"), 1),
                    "active_s": _r(c.get("campaign_sec"), 1),
                    "iterations": c.get("iterations"),
                    "kept": c.get("kept"),
                    "start_speedup": c.get("start_speedup"),
                    "best_speedup": c.get("best_speedup"),
                    "gain_pct": _r(gain_pct, 3),
                    "gain_pct_per_hour": _r(gain_pct / active_h, 4)
                    if gain_pct is not None and active_h
                    else None,
                    "last_event_at": _iso(now - c["idle_s"])
                    if c.get("idle_s") is not None
                    else None,
                    "termination_reason": c.get("termination_reason") or None,
                    "machine": (led.get("machine") or {}).get("node"),
                    "session": (led.get("writer") or {}).get("session_name"),
                    "source": "state",
                    "problem": f"events.jsonl unchanged for {c['idle_s'] // 60} min while status=running"
                    if c["stale"]
                    else None,
                }
            )
        )
        for op in op_ids:  # while a campaign exists its state beats a model-extracted kernel stage
            o = ops.setdefault(
                op, {"op_id": op, "share_pct": {}, "e2e_verdict": None, "e2e_delta_pct": None}
            )
            if "stage" not in o or o.get("_src") == "extracted":
                o["stage"] = "candidate" if status == "completed" else "optimizing"
                o["_src"] = "campaign"
            o["campaign_root"] = o.get("campaign_root") or root
    for h in hot:  # each op's share in the profiles, by role
        for x in h["ops"]:
            if x.get("op_id") in ops:
                ops[x["op_id"]]["share_pct"][h["role"]] = x["share_pct"]
    return {
        "hot_ops": hot,
        "campaigns": out,
        "ops": sorted(
            ({k: v for k, v in o.items() if k != "_src"} for o in ops.values()),
            key=lambda o: STAGES.index(o["stage"]) if o["stage"] in STAGES else 99,
        ),
    }


def write(root: Path, docs: dict, sources: dict, problems: list[str], stale_s: int) -> list[str]:
    """Validate and write each tab; returns the problems of documents not written."""
    gen, invalid = door.store.now(), []
    for name, body in docs.items():
        doc = {
            "v": 1,
            "generated_at": gen,
            "stale_after_s": stale_s,
            "sources": sources[name],
            **body,
            **({"problems": problems[:20]} if problems else {}),
        }
        probs = door.schemas.problems(f"facts.{name}.v1", doc)
        if probs:
            invalid.append(f"facts/{name}.json not written: {probs[0]}")
            continue
        door.store.write_json(root / "facts" / f"{name}.json", doc)
    return invalid
