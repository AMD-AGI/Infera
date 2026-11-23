"""o11y_facts — the display's facts, model-free (design §8.3). Reads the ledger, the registry and the
structured sources with its own cursors; writes facts/{general,detail,kernelforge}.json, including the usage
totals (`general.spend`), then commits its cursors."""

from __future__ import annotations

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
    ledger, _ = door.api.read_ledger(job.root)
    by_kind: dict[str, list] = {}
    for r in ledger:
        by_kind.setdefault(r["kind"], []).append(r)
    led_roots = {r["campaign_root"]: r for r in by_kind.get("campaign", [])}
    campaigns = []
    dirs, bad = forge_src.discover(list(led_roots.values()), [s.launch_dir for s in sessions])
    for root, why in bad:
        problems.add("forge", root, "campaign_root", why)
    for d in dirs:
        c = forge_src.read_campaign(d, problems, now=now, stale_s=cfg["forge_stale_s"])
        if c:
            c["ledger"] = next(
                (r for rt, r in led_roots.items() if d.startswith(rt.rstrip("/"))), None
            )
            campaigns.append(c)
    globs = cfg["sources"]["e2e_flow_globs"]
    docs = {
        "general": facts.general(by_kind, per, sessions, int(cfg["token_step"]), now),
        "detail": facts.detail(by_kind, grants, sessions),
        "kernelforge": facts.kernelforge(by_kind, campaigns, globs, problems, now),
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
    ]
    invalid = facts.write(job.root, docs, sources, lines, cfg["facts_stale_after_s"])
    if invalid:
        job.report.setdefault("invalid", []).extend(invalid)
    for key, cur in commit:
        store.put(key, cur)
