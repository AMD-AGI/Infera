"""Collect what the facts builder needs from the sources (o11y_facts' own reading, own cursors).

Usage totals come from the same pure rule as o11y_log's (`o11y_sources.transcripts.scan_sessions`) over
o11y_facts' own cursors in `state/facts/cursors`; nothing is taken from o11y_log's state.
"""

from __future__ import annotations

import os
from pathlib import Path

from o11y_core import api
from o11y_sources import Problems
from o11y_sources import forge as forge_src
from o11y_sources import otel as otel_src
from o11y_sources import pool as pool_src
from o11y_sources import registry as reg_src
from o11y_sources import transcripts as tr_src
from o11y_sources.cursor import CursorStore, read_new


def gather(root: Path, cfg: dict, state_dir: Path, now: float | None = None) -> tuple[dict, list]:
    """Returns (ctx for facts.build, cursors to commit after the facts are written)."""
    problems = Problems()
    store = CursorStore(state_dir)
    commit: list = []
    fleet = reg_src.read_fleet(root, cfg, problems)
    per, fleet_usage, _, c = tr_src.scan_sessions(store, fleet, problems, "usage")
    commit += c
    step = int(cfg["thresholds"]["token_step"])
    for u in per.values():
        u["steps_crossed"] = u["tokens"]["new"] // step
    fleet_usage["steps_crossed"] = fleet_usage["tokens"]["new"] // step
    pdir = cfg["sources"]["pool_dir"]
    grants = pool_src.read_grants(pdir, problems) if pdir and Path(pdir).is_dir() else []
    ledger, _ = api.read_ledger(root)
    led_roots = {r["campaign_root"]: r for r in ledger if r.get("kind") == "campaign"}
    campaigns = []
    for d in forge_src.discover(list(led_roots), cfg["sources"].get("campaign_globs") or []):
        camp = forge_src.read_campaign(
            d, problems, now=now, stale_s=cfg["thresholds"]["forge_stale_s"]
        )
        if camp:
            camp["ledger"] = next(
                (r for rt, r in led_roots.items() if d.startswith(rt.rstrip("/"))), None
            )
            campaigns.append(camp)
    cost: dict[str, float] = {}
    odir = cfg["sources"].get("otel_dir")
    if odir:
        files, unsafe = otel_src.private_jsonl_files(odir)
        for path, why in unsafe:
            problems.add("otel", path, "(file)", f"not trusted: {why}")
        sids = {s.session_id for s in fleet}
        for path in files:
            key = f"otel:{os.stat(path).st_ino}"
            chunk = read_new(store.get(key, path, follow_rename=True))
            acc = chunk.cursor.extra.setdefault("cost_usd", {})
            for r in otel_src.records(path, chunk.lines, problems):
                sid = r["attrs"].get("session.id")
                if r["event"] == "api_request" and sid in sids:
                    acc[sid] = acc.get(sid, 0.0) + float(r["attrs"].get("cost_usd") or 0)
            commit.append((key, chunk.cursor))
            for sid, v in acc.items():
                cost[sid] = cost.get(sid, 0.0) + v
    ctx = {
        "fleet": fleet,
        "usage": {"sessions": per, "fleet": fleet_usage},
        "grants": grants,
        "campaigns": campaigns,
        "otel": {"cost_usd": cost},
        "problems": problems,
    }
    return ctx, commit
