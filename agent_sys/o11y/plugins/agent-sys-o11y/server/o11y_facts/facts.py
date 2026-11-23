"""Build facts/{general,detail,kernelforge,usage}.json (protocol facts.<tab>.v1) for the display.

Priority per fact: the run ledger (latest record per identity; written and extracted alike) >
structured sources (pool grants, forge dirs, e2e-flow files) > the display pass's LLM extraction
(listed in `sources` as `extracted`). Each document is validated before the atomic write; an invalid
document is not written and its problems are returned in the run report. Source problems seen while
building go into each document's `problems[]` (o11y_log reports them as events; o11y_facts writes no events).
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from o11y_core import api, schemas, store
from o11y_sources import Problems, e2eflow
from o11y_sources import pool as pool_src

STAGE_ORDER = [
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


def _t(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _iso(epoch: float | None) -> str | None:
    if not epoch:
        return None
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _when(r: dict) -> str | None:
    return r.get("at") or r.get("ts")


def _pct(a, b):
    if a is None or not b:
        return None
    return round((a - b) / abs(b) * 100.0, 3)


def _r(x, n=6):
    return None if x is None else round(x, n)


def _clean(d: dict) -> dict:
    """Drop None-valued optional keys whose schema type is not nullable (times)."""
    return {
        k: v
        for k, v in d.items()
        if not (
            v is None
            and k in ("started_at", "ended_at", "since", "until", "at", "last_event_at", "last_at")
        )
    }


def build(root: Path, cfg: dict, ctx: dict, now: float | None = None) -> dict:
    now = now or time.time()
    root = Path(root)
    gen = store.now()
    stale = cfg.get("facts_stale_after_s", 900)
    ledger, _ = api.read_ledger(root)  # latest per identity
    by_kind: dict[str, list[dict]] = {}
    for r in ledger:
        by_kind.setdefault(r.get("kind"), []).append(r)
    try:
        extract = json.loads(
            (root / "state" / "facts" / "display_extract.json").read_text(encoding="utf-8")
        )
    except (FileNotFoundError, ValueError):
        extract = {}
    usage, fleet = ctx["usage"], ctx["fleet"]
    fleet_tokens = usage["fleet"]["tokens"][
        "new"
    ]  # progress per 1M tokens uses new tokens (spec §6.2)

    # ---------------------------------------------------------------- general
    g_sources = ["ledger", "transcripts"]
    task = (by_kind.get("task") or [None])[-1]
    task_doc = None
    if task is not None:
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
    elif (
        extract.get("task")
        and extract["task"].get("title")
        and isinstance(extract["task"].get("baseline"), (int, float))
    ):
        t = extract["task"]
        task_doc = {
            "task_id": "extracted",
            "title": t["title"],
            "metric": {
                "name": t.get("metric_name") or "metric",
                "unit": t.get("unit") or "",
                "direction": t.get("direction") or "higher",
            },
            "baseline": {"value": t["baseline"]},
        }
        if isinstance(t.get("target"), (int, float)):
            task_doc["target"] = {"value": t["target"]}
        g_sources.append("extracted")
    direction = (task_doc or {}).get("metric", {}).get("direction", "higher")
    sign = -1 if direction == "lower" else 1
    baseline = (task_doc or {}).get("baseline", {}).get("value")
    target = ((task_doc or {}).get("target") or {}).get("value")

    best_doc = None
    best_rec = (by_kind.get("best") or [None])[-1]
    if best_rec is not None:
        best_doc = {"value": best_rec["value"], "run_ids": list(best_rec["run_ids"])}
        for k in ("aggregate", "stack"):
            if best_rec.get(k):
                best_doc[k] = best_rec[k]
        best_doc["at"] = _when(best_rec)
    for r in by_kind.get("run_end", []):  # a later kept run that beats the recorded best
        if (
            r.get("status") == "kept"
            and isinstance(r.get("value"), (int, float))
            and (best_doc is None or sign * (r["value"] - best_doc["value"]) > 0)
        ):
            best_doc = {"value": r["value"], "run_ids": [r["run_id"]], "at": _when(r)}
    if best_doc is not None:
        best_doc["vs_baseline_pct"] = (
            _r(sign * _pct(best_doc["value"], baseline), 3) if baseline else None
        )
        best_doc["of_target_pct"] = (
            _r((best_doc["value"] - baseline) / (target - baseline) * 100, 2)
            if None not in (baseline, target) and target != baseline
            else None
        )
        best_doc = _clean(best_doc)

    t0 = _t(_when(task)) if task else None
    firsts = [
        _t(u.get("first_at"))
        for sid, u in usage["sessions"].items()
        if u.get("role") in ("main", "child")
    ]
    firsts = [x for x in firsts if x]
    if t0 is None and firsts:
        t0 = min(firsts)
    wall_s = round(now - t0, 1) if t0 else 0.0
    gain_abs = sign * (best_doc["value"] - baseline) if best_doc and baseline is not None else None
    gain_pct = (gain_abs / abs(baseline) * 100) if gain_abs is not None and baseline else None
    hours = wall_s / 3600 if wall_s else None
    mtok = fleet_tokens / 1e6 if fleet_tokens else None
    general = {
        "v": 1,
        "generated_at": gen,
        "stale_after_s": stale,
        "sources": g_sources,
        "task": task_doc,
        "best": best_doc,
        "spend": _clean(
            {
                "tokens_new": int(fleet_tokens),
                "tokens_cache_read": int(usage["fleet"]["tokens"]["cache_read"]),
                "wall_s": wall_s,
                "started_at": _iso(t0),
                "sessions": len(fleet),
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

    # ---------------------------------------------------------------- detail
    d_sources = ["ledger", "pool"]
    ended = {r["run_id"] for r in by_kind.get("run_end", [])}
    open_runs = [r for r in by_kind.get("run_start", []) if r["run_id"] not in ended]
    current = [
        _clean(
            {
                "run_id": r["run_id"],
                "theme": r["theme"],
                "hypothesis": r.get("hypothesis"),
                "point": r.get("point"),
                "machines": [m["node"] for m in r.get("machines") or [] if m.get("node")],
                "started_at": _when(r),
                "session": (r.get("writer") or {}).get("session_name"),
            }
        )
        for r in open_runs[-5:]
    ]
    if not current and (extract.get("current_run") or {}).get("theme"):
        c = extract["current_run"]
        current = [
            _clean(
                {
                    "run_id": "(extracted)",
                    "theme": c["theme"],
                    "point": c.get("point"),
                    "machines": [m for m in c.get("machines") or [] if m],
                    "started_at": c.get("started_at") if _t(c.get("started_at")) else None,
                }
            )
        ]
        d_sources.append("extracted") if "extracted" not in d_sources else None
    starts = {r["run_id"]: r for r in by_kind.get("run_start", [])}
    recent = []
    for r in sorted(by_kind.get("run_end", []), key=lambda r: _when(r) or "")[-3:][::-1]:
        st = starts.get(r["run_id"], {})
        delta = r.get("delta") or {}
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
                }
            )
        )
    if not recent and extract.get("recent_runs"):
        for x in extract["recent_runs"][:3]:
            if x.get("theme"):
                recent.append(
                    {
                        "run_id": x.get("run_id") or "(extracted)",
                        "theme": x["theme"],
                        "status": x.get("status") or "unknown",
                        "conclusion": x.get("result"),
                    }
                )
        if recent:
            d_sources.append("extracted") if "extracted" not in d_sources else None
    plan = (by_kind.get("plan") or [None])[-1]
    nxt = [
        {
            "theme": p["theme"],
            "why": p.get("why"),
            "point": p.get("point"),
            "blocked_on": p.get("blocked_on"),
        }
        for p in ((plan or {}).get("next") or [])[:3]
    ]
    if not nxt and extract.get("next_plan"):
        nxt = [
            {"theme": p["theme"], "why": p.get("why")}
            for p in extract["next_plan"][:3]
            if p.get("theme")
        ]
        if nxt and "extracted" not in d_sources:
            d_sources.append("extracted") if "extracted" not in d_sources else None
    run_on_node = {}
    for r in open_runs:
        for m in r.get("machines") or []:
            if m.get("node"):
                run_on_node[m["node"]] = f"{r['run_id']}: {r['theme']}"
    machines, seen_nodes = [], set()
    for g in ctx.get("grants") or []:
        owner = pool_src.attribute(g.get("notify") or "", g.get("requester_user"), fleet)
        if owner is None:
            continue
        for n in g["nodes"]:
            seen_nodes.add(n["node"])
            machines.append(
                _clean(
                    {
                        "node": n["node"],
                        "cluster": g.get("cluster") or "",
                        "jobid": n["jobid"],
                        "runs": run_on_node.get(n["node"]) or g.get("purpose"),
                        "session": owner.label,
                        "since": n.get("since"),
                        "until": g.get("until"),
                        "gpu_util": n.get("gpu_util"),
                        "source": "both" if n["node"] in run_on_node else "pool",
                    }
                )
            )
    for node, what in run_on_node.items():
        if node not in seen_nodes:
            machines.append({"node": node, "runs": what, "source": "ledger"})
    detail = {
        "v": 1,
        "generated_at": gen,
        "stale_after_s": stale,
        "sources": d_sources,
        "current_runs": current,
        "recent_runs": recent,
        "next": nxt,
        "machines": machines,
    }

    # ---------------------------------------------------------------- kernelforge
    problems = Problems()
    k_sources = ["forge", "ledger"]
    globs = cfg["sources"].get("e2e_flow_globs") or []
    hot_docs = []
    ht = e2eflow.hot_ops(globs, problems) if globs else None
    if ht:
        hot_docs.append(
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
        k_sources.append("e2e_flow")
    for h in by_kind.get("hotops", []):
        hot_docs.append(
            {
                "profile_id": h["profile_id"],
                "role": h["role"],
                "scope": h.get("scope"),
                "ops": [
                    {k: o[k] for k in ("name", "share_pct", "rank", "op_id") if k in o}
                    for o in h["ops"]
                ],
            }
        )
    reports = e2eflow.integration_reports(globs, problems) if globs else {}
    ops: dict[str, dict] = {}
    for k in by_kind.get("kernel", []):
        e2e = k.get("e2e") or {}
        ops[k["op_id"]] = {
            "op_id": k["op_id"],
            "stage": k["stage"],
            "share_pct": {"total": k["share_pct"]}
            if isinstance(k.get("share_pct"), (int, float))
            else {},
            "e2e_verdict": e2e.get("verdict"),
            "e2e_delta_pct": e2e.get("delta_pct"),
            "campaign_root": k.get("campaign_root"),
        }
    for op, r in reports.items():
        o = ops.setdefault(op, {"op_id": op, "share_pct": {}, "campaign_root": None})
        o["stage"] = "e2e_accepted" if r["accepted"] else "e2e_rejected"
        o["e2e_verdict"] = "accepted" if r["accepted"] else "rejected"
        o["e2e_delta_pct"] = (
            _r((r["observed_e2e_speedup"] - 1) * 100, 3)
            if isinstance(r.get("observed_e2e_speedup"), (int, float))
            else None
        )
    campaigns = []
    for c in ctx.get("campaigns") or []:
        led = c.get("ledger") or {}
        root_path = led.get("campaign_root") or str(Path(c["forge_dir"]).parent)
        status = (
            "stale"
            if c["stale"]
            else (c["status"] if c["status"] in CAMPAIGN_STATUS else "unknown")
        )
        gain_pct = c["gain"] / c["start_speedup"] * 100 if c.get("start_speedup") else None
        active_h = (c.get("campaign_sec") or 0) / 3600
        campaigns.append(
            _clean(
                {
                    "campaign_root": root_path,
                    "operator_ids": led.get("operator_ids") or [c["operator"]],
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
                    "problem": f"events.jsonl unchanged for {c['idle_s'] // 60} min while status=running"
                    if c["stale"]
                    else None,
                }
            )
        )
        for op in led.get("operator_ids") or [c["operator"]]:
            o = ops.setdefault(
                op, {"op_id": op, "share_pct": {}, "e2e_verdict": None, "e2e_delta_pct": None}
            )
            o.setdefault("stage", "optimizing" if status == "running" else "candidate")
            o["campaign_root"] = o.get("campaign_root") or root_path
    for h in hot_docs:  # share of each op in the profiles, by role
        for x in h["ops"]:
            op_id = x.get("op_id")
            if op_id and op_id in ops:
                ops[op_id]["share_pct"][h["role"]] = x["share_pct"]
    kernelforge = {
        "v": 1,
        "generated_at": gen,
        "stale_after_s": stale,
        "sources": k_sources,
        "hot_ops": hot_docs,
        "campaigns": campaigns,
        "ops": sorted(
            ops.values(),
            key=lambda o: STAGE_ORDER.index(o["stage"]) if o["stage"] in STAGE_ORDER else 99,
        ),
    }

    # ---------------------------------------------------------------- usage
    sessions = []
    for sid, u in usage["sessions"].items():
        if u.get("role") not in ("main", "child"):
            continue
        sessions.append(
            _clean(
                {
                    "session_id": sid,
                    "name": u.get("name") or None,
                    "role": u.get("role"),
                    "tokens": u["tokens"],
                    "messages": u.get("messages", 0),
                    "subagent_files": u.get("subagent_files", 0),
                    "last_at": u.get("last_at"),
                    "steps_crossed": u.get("steps_crossed", 0),
                }
            )
        )
    usage_doc = {
        "v": 1,
        "generated_at": gen,
        "stale_after_s": stale,
        "sources": ["transcripts"],
        "fleet": {
            "tokens": usage["fleet"]["tokens"],
            "messages": usage["fleet"]["messages"],
            "steps_crossed": usage["fleet"]["steps_crossed"],
        },
        "sessions": sessions,
    }
    cost = (ctx.get("otel") or {}).get("cost_usd")
    if cost:
        usage_doc["cost_usd"] = {k: round(v, 4) for k, v in cost.items()}
        usage_doc["sources"].append("otel")

    docs = {"general": general, "detail": detail, "kernelforge": kernelforge, "usage": usage_doc}
    lines = [
        f"{p.source}: {p.path}: {p.field} {p.detail}".strip()
        for p in [*(ctx.get("problems").items if ctx.get("problems") else []), *problems.items]
    ][:20]
    invalid = []
    for name, doc in docs.items():
        if lines:
            doc["problems"] = lines
        probs = schemas.problems(f"facts.{name}.v1", doc)
        if probs:
            invalid.append(f"facts/{name}.json not written: {probs[0]}")
            continue
        store.write_json(root / "facts" / f"{name}.json", doc)
    return {"problems": problems, "invalid": invalid, "facts": docs}
