"""o11y_facts — the display's facts, model-free (design §8.3). Reads the ledger, the registry and the
structured sources with its own cursors; writes facts/{general,detail,kernelforge}.json, including the usage
totals (`general.spend`), then commits its cursors."""

from __future__ import annotations

import os
import time

from o11y_job import CursorStore, config, door
from o11y_sources import Problems
from o11y_sources import forge as forge_src
from o11y_sources import pool as pool_src
from o11y_sources import transcripts as tr_src

from . import facts


def handle(job) -> None:
    cfg, cfg_problems, _ = config.load(job.root, job.state_dir)
    sessions, reg_problem = config.fleet(job.root, cfg)
    problems, store, now = Problems(), CursorStore(job.state_dir), time.time()
    per, _, _, commit = tr_src.scan_sessions(store, sessions, problems, "usage")
    pdir = cfg["sources"]["pool_dir"]
    grants = pool_src.read_grants(pdir, problems) if pool_src.trusted_dir(pdir, problems) else []
    rows, _ = door.api.read_registry(job.root)
    gone = door.api.not_in_fleet(job.root)  # sessions the main declared not part of the fleet

    def ours(recs):
        return [r for r in recs if (r.get("writer") or {}).get("session_id") not in gone]

    ledger = ours(door.api.read_ledger(job.root)[0])
    by_kind: dict[str, list] = {}
    for r in ledger:
        by_kind.setdefault(r["kind"], []).append(r)
    every = ours(door.api.read_ledger(job.root, latest_only=False)[0])
    for kind in facts.FLEET_LEVEL:  # the fleet's task, best and plan are the main's
        by_kind[kind] = facts.main_latest(every, kind, rows)
    cmap = cfg["sources"]["campaign_roots"]
    # the display shows the mapped (host) path of a root recorded as seen inside a container
    led_roots = {
        forge_src.map_root(r["campaign_root"], cmap): {
            **r,
            "campaign_root": forge_src.map_root(r["campaign_root"], cmap),
        }
        for r in by_kind.get("campaign", [])
    }
    campaigns = []
    dirs, bad, notes = forge_src.discover(
        by_kind.get("campaign", []), [s.launch_dir for s in sessions], cmap
    )
    for root, why in bad:
        problems.add("forge", root, "campaign_root", why)
    notes_lines = [f"forge: {root}: {note}" for root, note in notes]
    for d in dirs:
        c = forge_src.read_campaign(d, problems, now=now, stale_s=cfg["forge_stale_s"])
        if c:
            c["ledger"] = next(
                (
                    r
                    for rt, r in led_roots.items()
                    if d.startswith(os.path.realpath(rt).rstrip("/") + "/")
                ),
                None,
            )
            campaigns.append(c)
    globs = cfg["sources"]["e2e_flow_globs"]
    kf = facts.kernelforge(by_kind, campaigns, globs, problems, now)
    # a campaign whose root is not readable here (e.g. another cluster's file system) is listed from the
    # progress its ledger record reports, not dropped
    kf["campaigns"] += [
        facts.reported_campaign(r, root, why)
        for root, why in bad
        for r in led_roots.values()
        if r["campaign_root"] == root
    ]
    detail = facts.detail(by_kind, grants, sessions, kf["campaigns"], rows)
    general = facts.general(by_kind, per, sessions, int(cfg["token_step"]), now)
    general["machines"] = [
        {k: m.get(k) for k in ("node", "cluster", "task", "session", "source")}
        for m in detail["machines"]
    ]
    docs = {
        "general": general,
        "detail": {**detail, "child_next": facts.child_plans(every, rows)},
        "kernelforge": kf,
    }
    sources = {
        "general": ["ledger", "transcripts"],
        "detail": ["ledger", "pool"],
        "kernelforge": ["ledger", "forge", *(["e2e_flow"] if globs else [])],
    }
    lines = [
        *cfg_problems,
        *([reg_problem] if reg_problem else []),
        *(f"{p.source}: {p.path}: {p.field} {p.detail}".strip() for p in problems.items),
        *notes_lines,
    ]
    invalid = facts.write(job.root, docs, sources, lines, cfg["facts_stale_after_s"])
    if invalid:
        job.report.setdefault("invalid", []).extend(invalid)
    for key, cur in commit:
        store.put(key, cur)
