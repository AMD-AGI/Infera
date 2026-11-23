"""CLI: `python -m o11y_analyze --kind log|display|all [--fleet-dir D] [--no-llm] [--report PATH]`.

The trigger spawns it with a fixed argv and `AGENT_SYS_O11Y_FLEET_DIR` / `CLAUDE_CONFIG_DIR` in its
environment (design §7). One invocation (design §8.1): a non-blocking lock per kind (a second spawn exits at
once) → claim the queue items of its kind → run → re-scan the queue and loop while items of its kind exist,
within the budget → exit. It exits early when the main session it serves is gone. Exit code 0 also when
there was nothing to do.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from o11y_core import fleet as fleet_mod

from . import config, core, facts, passes
from .ingest import Ingestor

LAUNCHER = config.PLUGIN_DIR / "bin" / "o11y-launch"


@contextmanager
def lock(root: Path, name: str, blocking: bool, timeout_s: float = 60):
    p = root / "locks" / f"analyze-{name}.lock"
    p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(p, os.O_RDWR | os.O_CREAT, 0o600)
    t0 = time.time()
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if not blocking or time.time() - t0 > timeout_s:
                os.close(fd)
                yield False
                return
            time.sleep(0.2)
    try:
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()} {time.time():.0f}\n".encode())
        yield True
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _queue_dir(root: Path) -> Path:
    return (root / "queue").resolve()


def claim_queue(root: Path, kinds: set[str]) -> list[Path]:
    """Move this run's queue items to queue/claimed/ (an atomic claim a parallel analyzer cannot repeat)."""
    q = root / "queue"
    claimed = []
    if not q.is_dir():
        return claimed
    dst = q / "claimed"
    for f in sorted(q.glob("*.json")):
        kind = None
        try:
            kind = json.loads(f.read_text(encoding="utf-8")).get("kind")
        except (OSError, ValueError, AttributeError):
            pass
        if kind is None:  # file-name convention <ts>-<kind>.json
            kind = f.stem.rsplit("-", 1)[-1]
        if kind in kinds:
            dst.mkdir(exist_ok=True, mode=0o700)
            target = dst / f"{os.getpid()}-{f.name}"
            try:
                os.replace(f, target)
                claimed.append(target)
            except FileNotFoundError:
                pass  # another analyzer claimed it first
    return claimed


def drop_claimed(root: Path, claimed: list[Path]) -> int:
    """Remove drained items — only files whose resolved path lies inside the fleet root's queue/ (design §8.1)."""
    qdir = _queue_dir(root)
    n = 0
    for p in claimed:
        try:
            rp = p.resolve(strict=True)
        except FileNotFoundError:
            continue
        if rp.is_file() and qdir in rp.parents:
            rp.unlink()
            n += 1
    return n


def _sweep() -> None:
    """Design §7: the collector sweep also runs at every analyzer start (covers a SIGKILLed last session)."""
    if LAUNCHER.is_file() and os.access(LAUNCHER, os.X_OK):
        try:
            subprocess.Popen(
                [str(LAUNCHER), "--sweep"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError:
            pass


def _main_launch_dir(root: Path) -> str | None:
    """The main session's launch directory (relative task-book paths resolve against it)."""
    from o11y_core import store

    doc = store.read_json(root / store.SESSIONS) or {}
    for s in doc.get("sessions") or []:
        if isinstance(s, dict) and s.get("role") == "main":
            return s.get("launch_dir") or s.get("cwd")
    for rec in api_launches(root):
        if rec.get("role") == "main" and rec.get("cwd"):
            return rec["cwd"]
    return None


def api_launches(root: Path) -> list[dict]:
    from o11y_core import api

    try:
        return api.launches(root)
    except Exception:  # noqa: BLE001 - a missing or invalid launch log must not stop analysis
        return []


def _main_gone(ing: Ingestor) -> bool:
    return ing.main_alive is False


def _one_pass(
    root: Path,
    cfg: dict,
    kinds: set[str],
    use_llm: bool,
    deadline: float,
    now: float | None,
    report: dict,
    cfg_problems: list[str],
) -> bool:
    """One analysis pass. Returns False when the main session is gone (stop looping)."""
    with lock(root, "ingest", blocking=True) as got:
        if not got:
            report.setdefault("notes", []).append("ingest lock busy")
            return True
        ing = Ingestor(root, cfg, now=now)
        ctx = ing.run()
        report["fleet"] = [s.session_id for s in ctx["fleet"]]
        report["detector_events"] = report.get("detector_events", 0) + len(ing.events)
        gone = _main_gone(ing)
        f = None
        if not gone:
            if "log" in kinds and use_llm and cfg["llm"]["log"]["enabled"]:
                report.setdefault("log_pass", []).append(passes.log_pass(root, cfg, ctx, deadline))
            if "display" in kinds:
                f = facts.build(root, cfg, ctx, now=now)
                g, d = f["facts"]["general"], f["facts"]["detail"]
                need = (
                    use_llm
                    and cfg["llm"]["display"]["enabled"]
                    and cfg["llm"]["display"]["llm_fill_gaps"]
                    and (g["task"] is None or not d["current_runs"] or not d["next"])
                )
                dr = passes.display_pass(root, cfg, ctx, deadline, need)
                report.setdefault("display_pass", []).append(dr)
                if dr.get("llm") and dr["llm"]["ok"]:
                    f = facts.build(root, cfg, ctx, now=now)  # merge the fresh extraction
        ing.finish(f["problems"] if f else None, cfg_problems + (f["invalid"] if f else []))
        report["events_total"] = report.get("events_total", 0) + len(ing.events)
        if ing.emit_failures:
            report.setdefault("emit_failures", []).extend(ing.emit_failures)
        if gone:
            report["main_gone"] = True
        return not gone


def run(
    root: Path, kind: str, use_llm: bool = True, now: float | None = None, sweep: bool = True
) -> dict:
    root = Path(root)
    cfg, cfg_problems = config.load(root, launch_dir=_main_launch_dir(root))
    deadline = time.time() + cfg["budget_s"]
    kinds = {"log", "display"} if kind == "all" else {kind}
    report = {
        "kind": kind,
        "started": core.now_iso(),
        "pid": os.getpid(),
        "fleet_dir": str(root),
        "passes": 0,
    }
    t0 = time.time()
    if sweep:
        _sweep()
    with lock(root, kind, blocking=False) as got:
        if not got:
            report["skipped"] = "another analyzer of this kind is running"
            return report
        claimed: list[Path] = []
        while True:
            batch = claim_queue(root, kinds)
            if report["passes"] and not batch:
                break  # served everything that arrived while we ran (spec §4.5)
            claimed += batch
            report["passes"] += 1
            if not _one_pass(root, cfg, kinds, use_llm, deadline, now, report, cfg_problems):
                break
            drop_claimed(root, batch)
            if time.time() >= deadline:
                report.setdefault("notes", []).append("budget reached")
                break
        report["claimed"] = [p.name for p in claimed]
    cost = 0.0
    for p in ("log_pass", "display_pass"):
        for r in report.get(p) or []:
            cost += ((r or {}).get("llm") or {}).get("cost_usd") or 0.0
    report["cost_usd"] = round(cost, 6)
    report["duration_s"] = round(time.time() - t0, 2)
    return report


def _append_run_log(root: Path, report: dict) -> None:
    runs = root / "state" / "analyze" / "runs.jsonl"
    runs.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(
        runs, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600
    )  # A-private run log (cost per run)
    try:
        os.write(fd, (json.dumps(report, default=str) + "\n").encode())
    finally:
        os.close(fd)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="o11y_analyze")
    ap.add_argument("--kind", choices=["log", "display", "all"], default="all")
    ap.add_argument("--fleet-dir", help=f"fleet data root (default ${fleet_mod.FLEET_DIR_ENV_VAR})")
    ap.add_argument("--no-llm", action="store_true", help="deterministic detectors and facts only")
    ap.add_argument("--no-sweep", action="store_true", help="do not run the collector sweep")
    ap.add_argument("--report", help="also write the run report JSON here")
    a = ap.parse_args(argv)
    root = Path(a.fleet_dir).expanduser() if a.fleet_dir else fleet_mod.fleet_dir()
    root = root.resolve()
    report = run(root, a.kind, use_llm=not a.no_llm, sweep=not a.no_sweep)
    _append_run_log(root, report)
    if a.report:
        Path(a.report).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(json.dumps(report, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
