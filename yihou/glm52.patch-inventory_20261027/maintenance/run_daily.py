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

from git_transport import github_git_command

GENERATED = (
    "PR_HANDOFF.md",
    "OWNED_PR_INDEX.md",
    "PR_STATUS.md",
    "repro/PR_MAINTENANCE_SNAPSHOT.json",
)


def run(args, cwd, env, capture=False):
    args = github_git_command(args)
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


def ensure_team_tracker_draft(config, cwd, env):
    """Enforce the owner's permanent-draft policy for the configured tracker only."""
    team = config.get("team_tracker")
    if not team or not team.get("pr_url"):
        return None
    command = ["gh", "pr", "view", team["pr_url"], "--json", "state,headRefName,isDraft"]
    pr = json.loads(run(command, cwd, env, capture=True))
    if pr["headRefName"] != team["branch"]:
        raise RuntimeError("Configured tracker branch no longer matches the PR")
    if pr["state"] == "OPEN" and team.get("permanent_draft") and not pr["isDraft"]:
        run(["gh", "pr", "ready", team["pr_url"], "--undo"], cwd, env)
        pr = json.loads(run(command, cwd, env, capture=True))
        if pr["state"] != "OPEN" or not pr["isDraft"]:
            raise RuntimeError("Unable to restore the tracker to its required Draft state")
    return pr


def publish_team_tracker(config, runtime, inventory, env):
    """Publish the same audit to the team's permanent draft documentation PR."""
    team = config.get("team_tracker")
    if not team:
        return {"state": "NOT_CONFIGURED"}
    if not team.get("pr_url"):
        return {"state": "AWAITING_PR_CREATION"}
    pr = ensure_team_tracker_draft(config, inventory, env)
    if pr["state"] != "OPEN":
        return {
            "state": "TRACKER_CLOSED",
            "previous_pr_state": pr["state"],
            "action": "This tracker is intended to remain an open Draft. Preserve artifacts and ask its owner how to resume; never merge or push to main.",
        }
    worktree = Path(team["worktree"])
    if "yihou" not in str(worktree.resolve()):
        raise RuntimeError("Team publisher checkout must be under the yihou task directory")
    git = ["git", "-c", "gc.auto=0"]

    def g(*parts, capture=False):
        return run([*git, *parts], worktree, env, capture=capture)

    if (
        g("branch", "--show-current", capture=True).strip() != team["branch"]
        or pr["headRefName"] != team["branch"]
    ):
        raise RuntimeError("Team tracker branch no longer matches the owned open PR")
    if g("status", "--porcelain", capture=True).strip():
        raise RuntimeError(
            "Team tracker checkout is dirty; preserving it for manual reconciliation"
        )
    for field in ("user.name", "user.email"):
        if g("config", "--get", field, capture=True).strip() != config[field]:
            raise RuntimeError(f"Team publisher identity changed: {field}")
    g("pull", "--rebase", "origin", team["branch"])
    destination = worktree / team["directory"]
    run(
        [
            config["python"],
            inventory / "maintenance/export_team_tracker.py",
            "--inventory",
            inventory,
            "--runtime",
            runtime,
            "--destination",
            destination,
        ],
        worktree,
        env,
    )
    files = [
        str(Path(team["directory"]) / name)
        for name in ("README.md", "ACTIVE_PRS.md", "HISTORY.md", "PACKUP_MAP.md", "REFERENCES.md")
    ]
    changed = set(g("diff", "--name-only", capture=True).splitlines())
    untracked = set(g("ls-files", "--others", "--exclude-standard", capture=True).splitlines())
    if not (changed | untracked).issubset(set(files)):
        raise RuntimeError("Unexpected team worktree changes; refusing to stage them")
    g("diff", "--check")
    run([config["pre_commit"], "run", "--files", *files], worktree, env)
    g("add", "--", *files)
    if g("diff", "--cached", "--name-only", capture=True).strip():
        g("commit", "-s", "-m", "docs: refresh GLM-5.2 team PR tracking")
    g("pull", "--rebase", "origin", team["branch"])
    # Recheck lifecycle and the permanent-Draft policy before publishing.
    pr = ensure_team_tracker_draft(config, worktree, env)
    if pr["state"] != "OPEN":
        raise RuntimeError(
            "Tracking PR closed during refresh; preserve the generated commit for owner review"
        )
    g("push", "origin", f"HEAD:refs/heads/{team['branch']}")
    return {
        "state": "SUCCESS",
        "pr_url": team["pr_url"],
        "commit": g("rev-parse", "HEAD", capture=True).strip(),
    }


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
                ensure_team_tracker_draft(config, inventory, env)
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
                report["team_tracker"] = publish_team_tracker(config, runtime, inventory, env)
                if report["team_tracker"]["state"] == "TRACKER_CLOSED":
                    report["state"] = "SUCCESS_WITH_TRACKER_CLOSED"
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
        if report["state"] not in {"SUCCESS", "SUCCESS_WITH_TRACKER_CLOSED"}:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
