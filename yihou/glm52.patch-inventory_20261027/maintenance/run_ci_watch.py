#!/usr/bin/env python3
"""Run one bounded, unattended 48k PR maintenance cycle and publish its handoff."""

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

NUMBERS = [33059, 46820, 46830, 45524, 46918, 47276, 47278, 47897, 47872]
REPORTS = ("CI_48K_WATCH.md", "CI_48K_PENDING.md")


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".new")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def yihou_path(value):
    path = Path(value).resolve()
    if "yihou" not in str(path):
        raise ValueError(f"Task paths must contain yihou: {path}")
    return path


def run(command, cwd, env, timeout=180):
    result = subprocess.run(
        list(map(str, command)),
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if result.returncode:
        raise RuntimeError(
            f"{command[0]} exited {result.returncode}: {result.stderr[-2000:].strip()}"
        )
    return result.stdout


def environment(config):
    runtime = yihou_path(config["runtime"])
    env = dict(os.environ)
    for name in ("CODEX_THREAD_ID", "CODEX_SESSION_ID", "CODEX_TOOL_CALL_ID"):
        env.pop(name, None)
    env.update(
        PATH=config["path"],
        TMPDIR=str(runtime / "tmp"),
        XDG_CACHE_HOME=str(runtime / "cache"),
        PRE_COMMIT_HOME=config["pre_commit_home"],
        PYTHONPYCACHEPREFIX=str(runtime / "cache/pycache"),
        GIT_TERMINAL_PROMPT="0",
        GIT_SSH_COMMAND="ssh -o BatchMode=yes",
        YIHOU_CI_WATCH_WORKER="1",
    )
    return env


def validate_result(result):
    if not isinstance(result, dict):
        raise ValueError("Worker result must be an object")
    if set(result) != {"summary", "prs", "actions", "unresolved", "tests"}:
        raise ValueError("Unexpected worker result fields")
    if [row["number"] for row in result["prs"]] != NUMBERS:
        raise ValueError("Worker must report each of the nine PRs once, in scope order")
    for row in result["prs"]:
        if not isinstance(row["needs_user"], bool) or not isinstance(row["base_attempt"], int):
            raise ValueError("Malformed PR row")
        url = row["base_run_url"]
        if url and not url.startswith("https://github.com/sgl-project/sglang/actions/runs/"):
            raise ValueError("Base link must reference SGLang Actions")
    for item in result["actions"] + result["unresolved"]:
        if item["pr"] not in NUMBERS:
            raise ValueError("Out-of-scope action or unresolved item")
    keys = [item["key"] for item in result["unresolved"]]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate unresolved keys")


def carry_unresolved(result, previous):
    """Require affirmative evidence before removing a previously open concern."""
    current = {item["key"]: item for item in result["unresolved"]}
    states = {row["number"]: row["state"] for row in result["prs"]}
    for item in (previous or {}).get("unresolved", []):
        if item["key"] in current:
            continue
        resolved = states.get(item["pr"]) in {"CLOSED", "MERGED"} or any(
            action["kind"] == "resolved"
            and action["pr"] == item["pr"]
            and item["key"] in action["description"]
            and action["url"].startswith("https://github.com/")
            for action in result["actions"]
        )
        if not resolved:
            current[item["key"]] = item
    result["unresolved"] = list(current.values())
    for row in result["prs"]:
        if row["needs_user"] and not any(x["pr"] == row["number"] for x in current.values()):
            item = {
                "key": f"pr-{row['number']}-attention",
                "pr": row["number"],
                "summary": row["base_summary"],
                "next_action": row["next_action"],
                "evidence_urls": row["evidence_urls"],
            }
            result["unresolved"].append(item)
    return result


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_reports(result, checked):
    lines = [
        "# 48k-related Base CI watch",
        "",
        f"Last completed agent audit: **{checked}**.",
        "",
        "Cadence: every two hours, at even UTC hours. This is a local-host cron task; "
        "the host, network and saved credentials must remain available.",
        "",
        result["summary"],
        "",
        "Passing gates, skipped suites and retained passes are distinct from newly executed tests. "
        "#47276 is optional for frozen llr37; #45524 and #46918 have external authors.",
        "",
        "| PR | Head / state | Base attempt / status | Test progress | Last action | Next action |",
        "|---|---|---|---|---|---|",
    ]
    for row in result["prs"]:
        url = f"https://github.com/sgl-project/sglang/pull/{row['number']}"
        attempt = f"{row['base_attempt']} / {row['base_status']}"
        if row["base_run_url"]:
            attempt = f"[{attempt}]({row['base_run_url']})"
        values = [
            f"[#{row['number']}]({url})",
            f"`{row['head'][:12]}` / {row['state']}",
            attempt,
            row["base_summary"],
            row["action"],
            row["next_action"],
        ]
        lines.append("| " + " | ".join(cell(v) for v in values) + " |")
    lines += ["", "## Actions and validation", ""]
    lines += [
        f"- #{a['pr']} — {a['kind']}: {a['description']}"
        + (f" [Evidence]({a['url']})." if a["url"] else "")
        for a in result["actions"]
    ]
    if not result["actions"]:
        lines.append("No mutation was needed in this cycle.")
    lines += [
        "",
        *[f"- {test}" for test in result["tests"]],
        "",
        "Unresolved items: [CI_48K_PENDING.md](CI_48K_PENDING.md).",
        "",
    ]
    pending = [
        "# Pending handoff for the next user session",
        "",
        f"Updated: **{checked}**.",
        "",
        "These items remain open until supported by resolution evidence. Reporting them "
        "to the user does not resolve them.",
        "",
    ]
    if not result["unresolved"]:
        pending.append("No unresolved items were reported by the latest successful cycle.")
    for item in result["unresolved"]:
        pending += [
            f"## #{item['pr']}: {item['summary']}",
            "",
            f"Tracking key: `{item['key']}`.",
            "",
            f"Next action: {item['next_action']}",
            "",
        ]
        pending += [f"- [Evidence]({url})" for url in item["evidence_urls"]]
        pending.append("")
    return "\n".join(lines), "\n".join(pending).rstrip() + "\n"


def notify_next_session(runtime, checked, result=None, error=None):
    unresolved = (result or {}).get("unresolved", [])
    payload = {"unresolved": unresolved, "error": error}
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    notice = {
        "updated_at": checked,
        "pending": bool(unresolved or error),
        "fingerprint": fingerprint,
        "summary": (result or {}).get("summary", error or ""),
        "unresolved": unresolved,
        "error": error,
        "report": str(runtime / "NEXT_SESSION.md"),
    }
    write_json(runtime / "next_session.json", notice)
    if result:
        _, report = render_reports(result, checked)
        if error:
            report = f"# Maintenance runner failure\n\n{error}\n\n" + report
    else:
        report = f"# CI watch requires attention\n\nUpdated: {checked}\n\n{error}\n"
    (runtime / "NEXT_SESSION.md").write_text(report)


def run_worker(config, config_path, inventory, runtime, cycle, env):
    context = {
        "config_path": str(config_path),
        "runtime": str(runtime),
        "worktree": config["worktree"],
        "publisher_inventory": str(inventory),
        "experiment_inventory": config["experiment_inventory"],
        "cycle_directory": str(cycle),
        "previous_result_path": str(runtime / "latest_result.json"),
        "pending_path": str(runtime / "NEXT_SESSION.md"),
    }
    prompt = (inventory / "maintenance/ci_watch_prompt.md").read_text()
    prompt += "\n\nRUN_CONTEXT\n" + json.dumps(context, indent=2) + "\n"
    (cycle / "prompt.md").write_text(prompt)
    result_path = cycle / "result.json"
    command = [
        config["codex"],
        "--no-daemon",
        "-a",
        "never",
        "exec",
        "--sandbox",
        "danger-full-access",
        "-C",
        config["worktree"],
        "--json",
        "--output-schema",
        str(inventory / "maintenance/ci_watch_output.schema.json"),
        "-o",
        str(result_path),
        "-",
    ]
    write_json(cycle / "command.json", command)
    with (cycle / "events.jsonl").open("w") as out, (cycle / "stderr.log").open("w") as err:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=out,
            stderr=err,
            env=env,
            text=True,
            start_new_session=True,
        )
        write_json(cycle / "process.json", {"pid": process.pid, "started_at": now()})
        try:
            process.communicate(prompt, timeout=config.get("worker_timeout_seconds", 5400))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
            raise RuntimeError(
                "Worker reached its time limit; inspect preserved worktrees and logs"
            ) from None
    if process.returncode:
        raise RuntimeError(f"Codex worker exited {process.returncode}; see {cycle / 'stderr.log'}")
    result = read_json(result_path)
    validate_result(result)
    return carry_unresolved(result, read_json(runtime / "latest_result.json", {}))


def publish_reports(config, inventory, result, checked, env):
    worktree = yihou_path(config["worktree"])

    def git(*args):
        return run(["git", "-c", "gc.auto=0", *args], worktree, env)

    if git("status", "--porcelain").strip():
        raise RuntimeError("Worker changed its documentation checkout; preserving it for review")
    report, pending = render_reports(result, checked)
    for name, body in zip(REPORTS, (report, pending), strict=True):
        (inventory / name).write_text(body)
    path = inventory / "maintenance/handoff_notes.json"
    notes = read_json(path)
    for row in result["prs"]:
        note = notes["prs"][f"sgl-project/sglang#{row['number']}"]
        note["assessed_at"] = checked[:10]
        note["merge_action"] = f"Base audit {checked}: {row['base_summary']} {row['next_action']}"
        note["last_action"] = f"Automated two-hour audit {checked}: {row['action']}"
        note["ci_watch"] = {
            "checked_at": checked,
            "head": row["head"],
            "base_status": row["base_status"],
            "run_url": row["base_run_url"],
        }
    write_json(path, notes)
    files = [
        str(Path(config["inventory_relative"]) / name)
        for name in (*REPORTS, "maintenance/handoff_notes.json")
    ]
    changed = set(git("diff", "--name-only").splitlines())
    untracked = set(git("ls-files", "--others", "--exclude-standard").splitlines())
    if not (changed | untracked).issubset(set(files)):
        raise RuntimeError("Unexpected documentation worktree changes; refusing to stage")
    git("diff", "--check")
    run([config["pre_commit"], "run", "--files", *files], worktree, env, timeout=600)
    git("add", "--", *files)
    if git("diff", "--cached", "--name-only").strip():
        git("commit", "-s", "-m", "docs: refresh two-hour 48k CI maintenance")
    git("pull", "--rebase", "origin", config["remote_branch"])
    git("push", "origin", f"HEAD:refs/heads/{config['remote_branch']}")
    return git("rev-parse", "HEAD").strip()


def refresh_team_tracker(config, inventory, env, started):
    daily_config = yihou_path(config["daily_config"])
    daily = read_json(daily_config)
    report_path = Path(daily["runtime"]) / "run_status.json"
    # run_daily owns its own lock. A busy publisher returns zero without publishing.
    for attempt in range(3):
        run(
            [config["python"], inventory / "maintenance/run_daily.py", "--config", daily_config],
            config["worktree"],
            env,
            timeout=1800,
        )
        report = read_json(report_path, {})
        if report.get("state") == "SUCCESS" and report.get("started_at_utc", "") >= started:
            return report
        if attempt < 2:
            time.sleep(15)
    raise RuntimeError(
        "Reports were pushed, but the shared team publisher did not complete a fresh run"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--acknowledge", action="store_true")
    args = parser.parse_args()
    config_path = yihou_path(args.config)
    config = read_json(config_path)
    runtime, worktree = yihou_path(config["runtime"]), yihou_path(config["worktree"])
    runtime.mkdir(parents=True, exist_ok=True)
    if args.acknowledge:
        notice = read_json(runtime / "next_session.json", {})
        write_json(
            runtime / "acknowledged.json",
            {"fingerprint": notice.get("fingerprint"), "reported_at": now()},
        )
        print("Acknowledged the current notification; unresolved items remain open.")
        return
    if not config.get("enabled", False):
        print("CI watch is disabled in its configuration.")
        return
    for name in ("tmp", "cache", "logs", "cycles", "prs"):
        (runtime / name).mkdir(exist_ok=True)
    with (runtime / "ci-watch.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Previous CI watch cycle remains active; no duplicate worker started.")
            return
        started = now()
        cycle = runtime / "cycles" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        cycle.mkdir()
        status = {"state": "RUNNING", "started_at": started, "cycle": str(cycle)}
        write_json(runtime / "run_status.json", status)
        env = environment(config)
        inventory = worktree / config["inventory_relative"]
        result = None
        try:
            filesystem = run(
                ["findmnt", "-n", "-T", runtime, "-o", "FSTYPE"], worktree, env
            ).strip()
            if not filesystem.startswith("nfs"):
                raise RuntimeError("Configured artifacts are no longer on the verified NFS mount")
            stat = os.statvfs(runtime)
            if stat.f_bavail * stat.f_frsize < config.get("minimum_free_bytes", 2 * 1024**3):
                raise RuntimeError("Insufficient free artifact space; nothing was deleted")
            git = ["git", "-c", "gc.auto=0"]
            if (
                run([*git, "branch", "--show-current"], worktree, env).strip()
                != config["local_branch"]
            ):
                raise RuntimeError("Worker checkout is on an unexpected branch")
            if run([*git, "status", "--porcelain"], worktree, env).strip():
                raise RuntimeError("Worker checkout is dirty; preserve and reconcile its files")
            for name in ("user.name", "user.email"):
                if run([*git, "config", "--get", name], worktree, env).strip() != config[name]:
                    raise RuntimeError(f"Commit identity mismatch: {name}")
            login = run(["gh", "api", "user", "--jq", ".login"], worktree, env).strip()
            if login.lower() != config["author"].lower():
                raise RuntimeError("Unexpected GitHub login; refusing automated mutations")
            run([*git, "pull", "--rebase", "origin", config["remote_branch"]], worktree, env)
            result = run_worker(config, config_path, inventory, runtime, cycle, env)
            checked = now()
            write_json(runtime / "latest_result.json", result)
            write_json(cycle / "reconciled_result.json", result)
            notify_next_session(runtime, checked, result=result)
            status["report_commit"] = publish_reports(config, inventory, result, checked, env)
            status["publication"] = refresh_team_tracker(config, inventory, env, checked)
            status.update(
                state="SUCCESS",
                unresolved_count=len(result["unresolved"]),
                action_count=len(result["actions"]),
            )
        except Exception as error:
            status.update(state="FAILED", error=str(error))
            notify_next_session(
                runtime,
                now(),
                result=result or read_json(runtime / "latest_result.json"),
                error=str(error),
            )
        finally:
            status["finished_at"] = now()
            write_json(runtime / "run_status.json", status)
            write_json(cycle / "run_status.json", status)
        print(json.dumps(status, indent=2))
        if status["state"] != "SUCCESS":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
