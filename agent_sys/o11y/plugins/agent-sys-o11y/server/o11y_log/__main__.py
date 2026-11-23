"""CLI: `python -m o11y_log [--fleet-dir D] [--no-llm] [--no-sweep] [--report PATH]` — queue kind `log`.

Spawned by the trigger with a fixed argv and `AGENT_SYS_O11Y_FLEET_DIR` / `CLAUDE_CONFIG_DIR` in its env
(design §7). One invocation per design §8.1 (lock, claim, pass, re-scan, exit; early exit when the main
session is gone). Exit code 0 also when there was nothing to do.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from o11y_core import fleet as fleet_mod
from o11y_sources import config, oneshot

from .ingest import Detectors
from .passes import log_pass

KIND = "log"


def run(root: Path, use_llm: bool = True, now: float | None = None, sweep: bool = True) -> dict:
    root = Path(root)
    cfg, cfg_problems = config.load(root, launch_dir=oneshot.main_launch_dir(root))
    state_dir = root / "state" / "log"

    def one_pass(deadline: float, report: dict) -> bool:  # runs under the `log` kind lock
        det = Detectors(root, cfg, state_dir, now=now)
        fleet = det.run()
        report["fleet"] = [s.session_id for s in fleet]
        report["detector_events"] = report.get("detector_events", 0) + len(det.events)
        gone = det.main_alive is False
        if not gone and use_llm and cfg["llm"]["log"]["enabled"]:
            report.setdefault("log_pass", []).append(
                log_pass(root, cfg, state_dir, fleet, deadline)
            )
        det.finish(cfg_problems)
        if det.emit_failures:
            report.setdefault("emit_failures", []).extend(det.emit_failures)
        if gone:
            report["main_gone"] = True
        return not gone

    report = oneshot.run(root, KIND, cfg["budget_s"], one_pass, do_sweep=sweep)
    report["cost_usd"] = round(
        sum(
            ((r or {}).get("llm") or {}).get("cost_usd") or 0.0
            for r in report.get("log_pass") or []
        ),
        6,
    )
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="o11y_log")
    ap.add_argument("--fleet-dir", help=f"fleet data root (default ${fleet_mod.FLEET_DIR_ENV_VAR})")
    ap.add_argument("--no-llm", action="store_true", help="detectors only")
    ap.add_argument("--no-sweep", action="store_true", help="do not run the collector sweep")
    ap.add_argument("--report", help="also write the run report JSON here")
    a = ap.parse_args(argv)
    root = (Path(a.fleet_dir).expanduser() if a.fleet_dir else fleet_mod.fleet_dir()).resolve()
    report = run(root, use_llm=not a.no_llm, sweep=not a.no_sweep)
    oneshot.append_run_log(root / "state" / "log", report)
    if a.report:
        Path(a.report).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(json.dumps(report, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
