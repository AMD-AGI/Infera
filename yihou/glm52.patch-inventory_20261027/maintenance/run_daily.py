#!/usr/bin/env python3
"""Refresh and publish only generated PR ledgers from a dedicated worktree."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

GENERATED = (
    "PR_HANDOFF.md",
    "OWNED_PR_INDEX.md",
    "PR_STATUS.md",
    "repro/PR_MAINTENANCE_SNAPSHOT.json",
)


def run(args, cwd, env, capture=False):
    print("Running:", " ".join(map(str, args)), flush=True)
    return subprocess.run(
        list(map(str, args)),
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        timeout=1800,
    ).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    runtime, worktree = Path(config["runtime"]), Path(config["worktree"])
    relative = Path(config["inventory_relative"])
    for path in (runtime, worktree):
        if "yihou" not in str(path.resolve()):
            raise SystemExit("Runtime and checkout must be under a yihou path")
    runtime.mkdir(parents=True, exist_ok=True)
    for name in ("tmp", "cache", "logs"):
        (runtime / name).mkdir(exist_ok=True)
    with (runtime / "daily.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another maintenance run owns the lock; leaving it untouched.")
            return
        started = datetime.now(timezone.utc).isoformat()
        report = {"started_at_utc": started, "state": "RUNNING"}
        status_path = runtime / "run_status.json"
        status_path.write_text(json.dumps(report, indent=2) + "\n")
        log_path = (
            runtime / "logs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".log")
        )
        with log_path.open("a", buffering=1) as log:
            previous_out, previous_err = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = log
            # Subprocess output must share the persistent log, including failures.
            old_stdout, old_stderr = os.dup(1), os.dup(2)
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            try:
                env = dict(
                    os.environ,
                    TMPDIR=str(runtime / "tmp"),
                    XDG_CACHE_HOME=str(runtime / "cache"),
                    PRE_COMMIT_HOME=str(runtime / "cache/pre-commit"),
                    PYTHONPYCACHEPREFIX=str(runtime / "cache/pycache"),
                    GIT_TERMINAL_PROMPT="0",
                    GIT_SSH_COMMAND="ssh -o BatchMode=yes",
                )
                if config.get("pre_commit_home"):
                    env["PRE_COMMIT_HOME"] = config["pre_commit_home"]
                git = ["git", "-c", "gc.auto=0"]

                def g(*parts, capture=False):
                    return run([*git, *parts], worktree, env, capture=capture)

                if g("branch", "--show-current", capture=True).strip() != config["local_branch"]:
                    raise RuntimeError("Dedicated publisher branch changed; refusing to touch it")
                if g("status", "--porcelain", capture=True).strip():
                    raise RuntimeError(
                        "Dedicated publisher checkout is dirty; preserve it and inspect manually"
                    )
                for field in ("user.name", "user.email"):
                    actual = g("config", "--get", field, capture=True).strip()
                    if actual != config[field]:
                        raise RuntimeError(f"Configured commit identity changed: {field}")
                g("pull", "--rebase", "origin", config["remote_branch"])
                inventory = worktree / relative
                run(
                    [
                        config["python"],
                        inventory / "maintenance/update_pr_ledger.py",
                        "--inventory",
                        inventory,
                        "--runtime",
                        runtime,
                        "--author",
                        config["author"],
                    ],
                    worktree,
                    env,
                )
                paths = [str(relative / name) for name in GENERATED]
                changed = set(g("diff", "--name-only", capture=True).splitlines())
                untracked = set(
                    g("ls-files", "--others", "--exclude-standard", capture=True).splitlines()
                )
                if not (changed | untracked).issubset(set(paths)):
                    raise RuntimeError(
                        "Unexpected changes in publisher; refusing to stage unrelated files"
                    )
                g("diff", "--check")
                g("add", "--", *paths)
                if g("diff", "--cached", "--name-only", capture=True).strip():
                    run([config["pre_commit"], "run", "--files", *paths], worktree, env)
                    g("add", "--", *paths)
                    g("commit", "-s", "-m", "docs: refresh daily PR handoff status")
                # Incorporate a concurrent remote update without force pushing.
                g("pull", "--rebase", "origin", config["remote_branch"])
                g("push", "origin", f"HEAD:refs/heads/{config['remote_branch']}")
                report.update(state="SUCCESS", commit=g("rev-parse", "HEAD", capture=True).strip())
                (runtime / "last_publication.json").write_text(json.dumps(report, indent=2) + "\n")
            except Exception as exc:
                report.update(state="FAILED", error=str(exc))
                traceback.print_exc()
            finally:
                report.update(
                    finished_at_utc=datetime.now(timezone.utc).isoformat(), log=str(log_path)
                )
                status_path.write_text(json.dumps(report, indent=2) + "\n")
                sys.stdout.flush()
                sys.stderr.flush()
                os.dup2(old_stdout, 1)
                os.dup2(old_stderr, 2)
                os.close(old_stdout)
                os.close(old_stderr)
                sys.stdout, sys.stderr = previous_out, previous_err
        print(json.dumps(report, indent=2))
        if report["state"] != "SUCCESS":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
