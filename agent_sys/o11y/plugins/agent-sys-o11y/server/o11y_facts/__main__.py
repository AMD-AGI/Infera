"""CLI: `python -m o11y_facts [--fleet-dir D] [--no-llm] [--no-sweep] [--report PATH]` — queue kind `display`.

Spawned by the trigger with a fixed argv and `AGENT_SYS_O11Y_FLEET_DIR` / `CLAUDE_CONFIG_DIR` in its env
(design §7). One invocation per design §8.1. Exit code 0 also when there was nothing to do.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from o11y_core import fleet as fleet_mod
from o11y_sources import config, oneshot
from o11y_sources.cursor import CursorStore

from . import facts
from .display_pass import display_pass
from .gather import gather

KIND = "display"


def run(root: Path, use_llm: bool = True, now: float | None = None, sweep: bool = True) -> dict:
    root = Path(root)
    cfg, cfg_problems = config.load(root, launch_dir=oneshot.main_launch_dir(root))
    state_dir = root / "state" / "facts"

    def one_pass(deadline: float, report: dict) -> bool:  # runs under the `display` kind lock
        ctx, commit = gather(root, cfg, state_dir, now=now)
        report["fleet"] = [s.session_id for s in ctx["fleet"]]
        mains = [s for s in ctx["fleet"] if s.role == "main"]
        if mains and all(s.alive is False for s in mains):
            report["main_gone"] = True
            return False
        f = facts.build(root, cfg, ctx, now=now)
        g, d = f["facts"]["general"], f["facts"]["detail"]
        dc = cfg["llm"]["display"]
        need = (
            use_llm
            and dc["enabled"]
            and dc["llm_fill_gaps"]
            and (g["task"] is None or not d["current_runs"] or not d["next"])
        )
        dr = display_pass(root, cfg, state_dir, ctx["fleet"], deadline, need)
        report.setdefault("display_pass", []).append(dr)
        if dr.get("llm") and dr["llm"]["ok"]:
            f = facts.build(root, cfg, ctx, now=now)  # merge the fresh extraction
        cursors = CursorStore(state_dir)
        for key, cur in commit:
            cursors.put(key, cur)
        invalid = cfg_problems + f["invalid"]
        if invalid:
            report.setdefault("invalid", []).extend(invalid)
        return True

    report = oneshot.run(root, KIND, cfg["budget_s"], one_pass, do_sweep=sweep)
    report["cost_usd"] = round(
        sum(
            ((r or {}).get("llm") or {}).get("cost_usd") or 0.0
            for r in report.get("display_pass") or []
        ),
        6,
    )
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="o11y_facts")
    ap.add_argument("--fleet-dir", help=f"fleet data root (default ${fleet_mod.FLEET_DIR_ENV_VAR})")
    ap.add_argument("--no-llm", action="store_true", help="facts from structured sources only")
    ap.add_argument("--no-sweep", action="store_true", help="do not run the collector sweep")
    ap.add_argument("--report", help="also write the run report JSON here")
    a = ap.parse_args(argv)
    root = (Path(a.fleet_dir).expanduser() if a.fleet_dir else fleet_mod.fleet_dir()).resolve()
    report = run(root, use_llm=not a.no_llm, sweep=not a.no_sweep)
    oneshot.append_run_log(root / "state" / "facts", report)
    if a.report:
        Path(a.report).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(json.dumps(report, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
