#!/usr/bin/env python3
"""Bounded, author-only CI retries for the GLM-5.2 watchdog.

The official /rerun-failed-ci command retries unsuccessful workflows across one
PR, not just Base. Green jobs are normally retained; the upstream handler can
fall back to a full rerun. No external author's PR is modified by this helper.
Reservations are durable before posting. An uncertain send consumes its retry
slot: inspect GitHub and the retained body file instead of sending it again.
"""

import argparse
import base64
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import unicodedata
import uuid
from pathlib import Path
from urllib.parse import urlencode

REPO = "sgl-project/sglang"
AUTHOR = "dorado269"
OWNED = {33059, 46820, 46830, 47276, 47278, 47897, 47872}
EXTERNAL = {45524, 46918}
BASE_PATH = ".github/workflows/pr-test.yml"
UNSUCCESSFUL = {"failure", "cancelled", "timed_out", "skipped"}
AGGREGATES = {
    "pr-test-finish",
    "pr-test-extra-finish",
    "pr-test-amd-finish",
    "pr-test-amd-extra-finish",
    "pr-test-npu-finish",
    "pr-test-mlx-finish",
    "pr-test-musa-finish",
    "wait-for-base-a",
    "wait-for-base-b",
    "wait-for-stage-b-amd",
}
GATES = {"call-gate / pr-gate", "pr-gate / pr-gate", "pr-gate", "call-gate", "gate"}
# Audited upstream main fbe54087d6743291a0e6ba076d00eb2305fe4c97.
# A changed policy requires review; never infer new authorization from it.
COOLDOWN_GATE_SHA256 = "63e133892c2ef42609c0fdd33f3ae524afad21fb792c9eb3bbf8518916915af1"
COOLDOWN_GATE_JOBS = ("check-changes", "pr-gate", "call-gate", "pr-test-finish")
RESOURCE_CONCLUSIONS = {"success", "failure", "timed_out", "action_required"}
COOLDOWN_ERROR = re.compile(
    r"^\d{4}-\d\d-\d\dT\S+\s+##\[error\]User '([^']+)' already triggered "
    r"'PR Test Base' via '(pull_request(?:_target)?)' at "
    r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?Z)\. "
    r"Please wait ([1-9]\d*) minutes before triggering again\.$"
)


def utc(timestamp):
    return dt.datetime.fromtimestamp(timestamp, dt.timezone.utc).isoformat()


def parse_time(value):
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def validate_config(config):
    if config.get("repo") != REPO or config.get("author") != AUTHOR:
        raise ValueError("unsupported repository or author")
    for key, allowed in (("owned_prs", OWNED), ("external_prs", EXTERNAL)):
        values = config.get(key)
        if not isinstance(values, list) or any(type(n) is not int for n in values):
            raise ValueError("invalid PR allowlist")
        if not set(values) <= allowed:
            raise ValueError("PR outside the authorized allowlist")
    interval = config.get("min_retry_interval_seconds", 7200)
    quota = config.get("max_retries_per_head", 2)
    if type(interval) is not int or interval < 7200:
        raise ValueError("retry interval must be at least 7200 seconds")
    if type(quota) is not int or not 1 <= quota <= 2:
        raise ValueError("retry quota must be between one and two")
    config = dict(config)
    config.update(min_retry_interval_seconds=interval, max_retries_per_head=quota)
    return config


def prepare_runtime(value):
    path = Path(value).expanduser().resolve()
    if "yihou" not in str(path):
        raise ValueError("runtime must be in a yihou directory")
    existing = path
    while not existing.exists():
        existing = existing.parent
    mount = json.loads(
        subprocess.check_output(
            ["findmnt", "--json", "--target", str(existing), "--output", "TARGET,FSTYPE"],
            text=True,
            timeout=15,
        )
    )["filesystems"][0]
    fstype = mount["fstype"]
    if not (
        fstype in {"nfs", "nfs4"}
        or (mount["target"] != "/" and fstype in {"ext4", "xfs", "btrfs", "zfs"})
    ):
        raise ValueError("runtime must use NFS or a separately mounted data disk")
    if shutil.disk_usage(existing).free < 64 * 1024 * 1024:
        raise ValueError("insufficient runtime space")
    path.mkdir(parents=True, exist_ok=True)
    return path


class GitHub:
    def command(self, args):
        return subprocess.check_output(["gh", *args], text=True, stderr=subprocess.PIPE, timeout=45)

    def api(self, endpoint):
        return json.loads(self.command(["api", endpoint]))

    def pages(self, endpoint, key=None):
        pages = json.loads(self.command(["api", "--paginate", "--slurp", endpoint]))
        return [item for page in pages for item in (page[key] if key else page)]

    def actor(self):
        return self.api("user")["login"]

    def pr(self, number):
        return self.api(f"repos/{REPO}/pulls/{number}")

    def run(self, run_id):
        return self.api(f"repos/{REPO}/actions/runs/{run_id}")

    def runs(self, head):
        return self.pages(
            f"repos/{REPO}/actions/runs?head_sha={head}&per_page=100", "workflow_runs"
        )

    def jobs(self, run_id, attempt):
        return self.pages(
            f"repos/{REPO}/actions/runs/{run_id}/attempts/{attempt}/jobs?per_page=100",
            "jobs",
        )

    def comments(self, number, since):
        query = urlencode({"since": utc(since), "per_page": 100})
        return self.pages(f"repos/{REPO}/issues/{number}/comments?{query}")

    def job_log(self, job_id):
        return self.command(
            [
                "api",
                "--allow-escape-sequences",
                f"repos/{REPO}/actions/jobs/{job_id}/logs",
            ]
        )

    def source(self, path, revision):
        data = self.api(f"repos/{REPO}/contents/{path}?ref={revision}")
        if data.get("encoding") != "base64":
            raise ValueError("upstream source unavailable")
        return base64.b64decode(data["content"]).decode("utf-8")

    def cooldown_policy(self):
        revision = self.api(f"repos/{REPO}/commits/main")["sha"]
        gate = self.source(".github/workflows/pr-gate.yml", revision)
        if hashlib.sha256(gate.encode()).hexdigest() != COOLDOWN_GATE_SHA256:
            raise ValueError("upstream cooldown policy changed")
        base = self.source(BASE_PATH, revision)
        # The audited Base caller uses the gate's default 120 minutes. Fail
        # closed on caller overrides; unrelated Base jobs may evolve normally.
        section = re.search(r"(?ms)^  call-gate:\n(.*?)(?=^  [\w-]+:|\Z)", base)
        if (
            section is None
            or "    uses: ./.github/workflows/pr-gate.yml\n" not in section[1]
            or re.search(r"(?m)^    with:", section[1])
        ):
            raise ValueError("upstream Base gate caller changed")
        permissions = json.loads(self.source(".github/CI_PERMISSIONS.json", revision))
        override = permissions.get(AUTHOR, {}).get("cooldown_interval_minutes", 120)
        if type(override) not in (int, float) or not 0 <= override <= 100000:
            raise ValueError("invalid upstream cooldown override")
        # Higher collaborator permissions could waive the gate. Conservatively
        # do not assume that exemption; explicit upstream per-user limits apply.
        return {"revision": revision, "minutes": min(120, override)}

    def cooldown_runs(self, since):
        lower = dt.datetime.fromtimestamp(since, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        query = urlencode({"event": "pull_request", "created": ">=" + lower, "per_page": 100})
        pages = json.loads(
            self.command(
                [
                    "api",
                    "--paginate",
                    "--slurp",
                    f"repos/{REPO}/actions/workflows/pr-test.yml/runs?{query}",
                ]
            )
        )
        totals = {page["total_count"] for page in pages}
        runs = [run for page in pages for run in page["workflow_runs"]]
        # Filtered Actions searches can cap at 1000 results. A changing page
        # count or duplicate/missing run makes this observation inconclusive.
        if (
            len(totals) != 1
            or next(iter(totals)) >= 1000
            or len({run["id"] for run in runs}) != len(runs)
            or len(runs) != next(iter(totals))
        ):
            raise ValueError("incomplete upstream cooldown history")
        return runs

    def cooldown_jobs(self, run_id):
        return self.pages(
            f"repos/{REPO}/actions/runs/{run_id}/jobs?filter=latest&per_page=100", "jobs"
        )

    def post(self, number, body_file):
        return self.command(
            [
                "pr",
                "comment",
                str(number),
                "--repo",
                REPO,
                "--body-file",
                str(body_file),
            ]
        ).strip()


def atomic_state(path, state):
    """Replace a complete state snapshot; retain all prior history and failed temps."""
    if path.is_symlink():
        raise ValueError("state file must not be a symlink")
    temporary = path.with_name(f"retry_state.{uuid.uuid4().hex}.pending.json")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(state, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def load_state(path):
    if path.is_symlink():
        raise ValueError("state file must not be a symlink")
    if not path.exists():
        return {"schema_version": 1, "retries": [], "decisions": []}
    state = json.loads(path.read_text())
    if state.get("schema_version") != 1:
        raise ValueError("unsupported retry state")
    if not isinstance(state.get("retries"), list) or not isinstance(state.get("decisions"), list):
        raise ValueError("invalid retry history")
    for entry in state["retries"]:
        if not all(key in entry for key in ("pr", "head", "run_id", "attempt", "reserved_at")):
            raise ValueError("incomplete retry reservation")
        parse_time(entry["reserved_at"])
        if (
            type(entry["pr"]) is not int
            or entry["pr"] not in OWNED
            or not re.fullmatch(r"[0-9a-f]{40}", entry["head"])
            or any(type(entry[key]) is not int or entry[key] < 1 for key in ("run_id", "attempt"))
        ):
            raise ValueError("invalid retry reservation")
    return state


def pr_blocker(pr, expected_head):
    if pr.get("state") != "open":
        return "pr_not_open"
    if pr.get("draft") is not False:
        return "pr_draft"
    if pr.get("head", {}).get("sha") != expected_head:
        return "head_changed"
    if pr.get("user", {}).get("login") != AUTHOR:
        return "not_our_author"
    if "run-ci" not in {label["name"] for label in pr.get("labels", [])}:
        return "missing_run_ci_label"
    return None


def workflow_path(run):
    return run.get("path", "").split("@", 1)[0]


def scoped_runs(runs, pr, head):
    latest = {}
    for run in runs:
        if run.get("head_sha") != head:
            continue
        if run.get("event") == "pull_request_target":
            if run.get("head_branch") != pr["head"]["ref"] or (
                run.get("head_repository") or {}
            ).get("id") != (pr["head"].get("repo") or {}).get("id"):
                continue
        path = workflow_path(run)
        if not path:
            raise ValueError("workflow path unavailable")
        if path not in latest or run["id"] > latest[path]["id"]:
            latest[path] = run
    return latest


def target_blocker(run, latest, head, run_id):
    if run.get("head_sha") != head:
        return "run_head_changed"
    if workflow_path(run) != BASE_PATH:
        return "not_base_workflow"
    newest = latest.get(BASE_PATH)
    if newest is None or newest["id"] != run_id:
        return "not_latest_base_run"
    if run.get("run_attempt") != newest.get("run_attempt"):
        return "run_attempt_changed"
    if run.get("status") != "completed" or newest.get("status") != "completed":
        return "base_run_active"
    if run.get("conclusion") not in UNSUCCESSFUL or newest.get("conclusion") not in UNSUCCESSFUL:
        return "base_run_not_unsuccessful"
    return None


def actual_failures(jobs):
    return [
        job["name"]
        for job in jobs
        if job.get("conclusion") in {"failure", "timed_out", "startup_failure", "action_required"}
        and job.get("name") not in AGGREGATES | GATES
    ]


def slash_request(comment):
    body = "".join(ch for ch in comment.get("body", "") if unicodedata.category(ch) != "Cf")
    return bool(re.match(r"^\s*/(?:run|rerun|tag)[\w-]*(?:\s|$)", body))


def stale_gate_log(text):
    """Accept executed old Draft/run-ci messages, not echoed shell source."""
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    return any(
        re.match(
            r"^\d{4}-\d\d-\d\dT\S+\s+(?:PR is draft\. Blocking CI\.|Missing required label 'run-ci'\.)",
            line,
        )
        for line in text.splitlines()
    )


def cooldown_error(text):
    """Read one executed author error, excluding echoed/source-only messages."""
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    matches = [match for line in text.splitlines() if (match := COOLDOWN_ERROR.fullmatch(line))]
    if len(matches) != 1 or matches[0][1] != AUTHOR:
        return None
    match = matches[0]
    try:
        minutes = int(match[4])
        if minutes > 120:
            return None
        deadline = parse_time(match[3]) + minutes * 60
    except (ValueError, OverflowError):
        return None
    return {"deadline": deadline, "event": match[2], "minutes": minutes}


def cooldown_blocker(gh, run_id, evidence, now):
    """Mirror the audited gate conservatively, including its five-run limit."""
    deadline = max(item["deadline"] for item in evidence)
    # Upstream includes created_at == cutoff, so equality is still blocked.
    if now <= deadline:
        return {"reason": "author_cooldown_active", "retry_after": utc(deadline + 1)}
    try:
        policy = gh.cooldown_policy()
        seconds = policy["minutes"] * 60
        if seconds == 0:
            return None
        candidates = []
        for run in gh.cooldown_runs(now - seconds):
            if run["id"] == run_id:
                continue
            if workflow_path(run) != BASE_PATH or run.get("event") != "pull_request":
                raise ValueError("unexpected cooldown history scope")
            if parse_time(run["created_at"]) < now - seconds:
                continue
            actors = (
                (run.get("actor") or {}).get("login"),
                (run.get("triggering_actor") or {}).get("login"),
            )
            if AUTHOR not in actors:
                if None in actors:
                    raise ValueError("cooldown actor evidence unavailable")
                continue
            candidates.append(run)
        candidates.sort(key=lambda item: item["id"], reverse=True)
        for index, run in enumerate(candidates):
            retry_after = utc(parse_time(run["created_at"]) + seconds + 1)
            # Upstream assumes the sixth matching run passed its gate. Keep
            # that conservative behavior even if its jobs might be gate-only.
            if index >= 5:
                return {"reason": "author_cooldown_history_ambiguous", "retry_after": retry_after}
            jobs = gh.cooldown_jobs(run["id"])
            for job in jobs:
                name, conclusion = job["name"], job["conclusion"]
                if any(name == gate or name.startswith(gate + " ") for gate in COOLDOWN_GATE_JOBS):
                    continue
                if conclusion in RESOURCE_CONCLUSIONS:
                    return {
                        "reason": "author_cooldown_active",
                        "retry_after": retry_after,
                        "blocking_run_id": run["id"],
                    }
    except Exception:
        # Unknown source, pagination, permissions or jobs must not spend quota
        # or cause a comment. The upstream gate remains the final authority.
        return {"reason": "cooldown_policy_or_evidence_unavailable"}
    return None


def eligible_runs(latest):
    return [
        run
        for run in latest.values()
        if run.get("status") == "completed" and run.get("conclusion") in UNSUCCESSFUL
    ]


def scope_fingerprint(runs):
    return sorted((run["id"], run["run_attempt"], run["conclusion"]) for run in runs)


def retry(config, number, run_id, expected_head, reason, dry_run=False, gh=None, now=None):
    config = validate_config(config)
    if number < 1 or run_id < 1:
        raise ValueError("PR and run identifiers must be positive")
    if not re.fullmatch(r"[0-9a-f]{40}", expected_head):
        raise ValueError("expected head must be a full commit SHA")
    if reason not in {"cancelled", "transient", "stale_gate"}:
        raise ValueError("unsupported retry reason")
    runtime = prepare_runtime(config["runtime"])
    timestamp = time.time() if now is None else now
    result = {
        "status": "blocked",
        "reason": "not_evaluated",
        "requested_reason": reason,
        "pr": number,
        "run_id": run_id,
        "head": expected_head,
        "attempt": None,
        "observed_at": utc(timestamp),
        "scope": "PR-wide unsuccessful workflows",
    }
    lock_path = runtime / "retry.lock"
    if lock_path.is_symlink():
        raise ValueError("retry lock must not be a symlink")
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return dict(result, status="busy", reason="retry_lock_held")
        state_path = runtime / "retry_state.json"
        state = load_state(state_path)

        def finish(status, explanation, **extra):
            outcome = dict(result, status=status, reason=explanation, **extra)
            if not dry_run:
                state["decisions"].append(outcome)
                atomic_state(state_path, state)
            return outcome

        if number in config["external_prs"]:
            return finish("permission_needed", "external_author_controls_retry")
        if number not in config["owned_prs"]:
            return finish("blocked", "pr_not_authorized")
        gh = gh or GitHub()
        pr = gh.pr(number)
        blocker = pr_blocker(pr, expected_head)
        if blocker:
            return finish("blocked", blocker)
        if gh.actor() != AUTHOR:
            return finish("blocked", "authenticated_actor_not_author")
        run = gh.run(run_id)
        latest = scoped_runs(gh.runs(expected_head), pr, expected_head)
        result["attempt"] = run.get("run_attempt")
        blocker = target_blocker(run, latest, expected_head, run_id)
        if blocker:
            return finish("blocked", blocker)
        if type(result["attempt"]) is not int or result["attempt"] < 1:
            return finish("blocked", "invalid_run_attempt")
        if reason == "cancelled" and run["conclusion"] != "cancelled":
            return finish("blocked", "cancelled_reason_mismatch")

        history = [
            entry
            for entry in state["retries"]
            if entry["pr"] == number and entry["head"] == expected_head
        ]
        if any(
            entry["run_id"] == run_id and entry["attempt"] == result["attempt"] for entry in history
        ):
            return finish("blocked", "run_attempt_already_reserved")
        if len(history) >= config["max_retries_per_head"]:
            return finish("blocked", "head_retry_quota_exhausted")
        interval = config["min_retry_interval_seconds"]
        if any(timestamp - parse_time(entry["reserved_at"]) < interval for entry in history):
            return finish("blocked", "retry_interval_not_elapsed")

        eligible = eligible_runs(latest)
        result["workflow_ids"] = sorted(item["id"] for item in eligible)
        base_jobs = []
        for item in eligible:
            # A transient failure is classified by the caller/model, not guessed
            # from a job title. For other reasons, never mask test/build failures.
            if reason != "transient":
                jobs = gh.jobs(item["id"], item["run_attempt"])
                if item["id"] == run_id:
                    base_jobs = jobs
                if not jobs and item["conclusion"] != "skipped":
                    return finish("blocked", "job_evidence_unavailable")
                failures = actual_failures(jobs)
                if failures:
                    return finish("blocked", "actual_failure_requires_triage", failed_jobs=failures)
        cooldown_evidence = []
        if reason == "stale_gate":
            failed_gates = [
                job
                for job in base_jobs
                if job.get("name") in GATES and job.get("conclusion") == "failure"
            ]
            if not failed_gates:
                return finish("blocked", "gate_not_proven_stale")
            for job in failed_gates:
                log = gh.job_log(job["id"])
                if stale_gate_log(log):
                    continue
                evidence = cooldown_error(log)
                if evidence is None:
                    return finish("blocked", "gate_not_proven_stale")
                cooldown_evidence.append(evidence)
            if cooldown_evidence:
                blocker = cooldown_blocker(
                    gh, run_id, cooldown_evidence, time.time() if now is None else timestamp
                )
                if blocker:
                    return finish(
                        "blocked",
                        blocker["reason"],
                        **{key: value for key, value in blocker.items() if key != "reason"},
                    )
        for comment in gh.comments(number, timestamp - interval):
            changed = max(parse_time(comment["created_at"]), parse_time(comment["updated_at"]))
            if changed >= timestamp - interval and slash_request(comment):
                return finish(
                    "blocked", "recent_pr_slash_request", comment_url=comment.get("html_url")
                )

        if cooldown_evidence:
            blocker = cooldown_blocker(
                gh, run_id, cooldown_evidence, time.time() if now is None else timestamp
            )
            if blocker:
                return finish(
                    "blocked",
                    blocker["reason"],
                    **{key: value for key, value in blocker.items() if key != "reason"},
                )

        # Recheck mutable state after potentially slow cooldown/job/comment reads.
        current_pr = gh.pr(number)
        blocker = pr_blocker(current_pr, expected_head)
        if blocker:
            return finish("blocked", blocker)
        current_run = gh.run(run_id)
        current_latest = scoped_runs(gh.runs(expected_head), current_pr, expected_head)
        blocker = target_blocker(current_run, current_latest, expected_head, run_id)
        if blocker or current_run.get("run_attempt") != result["attempt"]:
            return finish("blocked", blocker or "run_attempt_changed")
        if current_run.get("conclusion") != run.get("conclusion"):
            return finish("blocked", "run_conclusion_changed")
        if scope_fingerprint(eligible_runs(current_latest)) != scope_fingerprint(eligible):
            return finish("blocked", "workflow_scope_changed")
        for comment in gh.comments(number, timestamp - interval):
            changed = max(parse_time(comment["created_at"]), parse_time(comment["updated_at"]))
            if changed >= timestamp - interval and slash_request(comment):
                return finish(
                    "blocked", "recent_pr_slash_request", comment_url=comment.get("html_url")
                )
        if dry_run:
            return finish("dry_run", "eligible_no_comment_posted")

        reservation_id = uuid.uuid4().hex
        body_file = runtime / f"retry-comment-{reservation_id}.md"
        with body_file.open("x", encoding="utf-8") as stream:
            stream.write(
                "/rerun-failed-ci\n\n"
                f"<!-- glm52-ci-watch:{reservation_id} -->\n"
                f"Bounded watchdog retry: {reason}; Base run {run_id}, "
                f"attempt {result['attempt']}, head `{expected_head}`.\n\n"
                "This command retries unsuccessful workflows across this PR. "
                "The upstream handler normally retains passing jobs and may use "
                "a full rerun when its recovery rules require one.\n"
            )
        reservation = dict(
            result,
            reservation_id=reservation_id,
            reserved_at=utc(time.time() if now is None else timestamp),
            status="reserved",
            reason=reason,
            body_file=body_file.name,
        )
        state["retries"].append(reservation)
        # If this write fails, posting is never attempted. On restart a durable
        # reservation always counts, even if its final outcome was never saved.
        atomic_state(state_path, state)
        try:
            comment_url = gh.post(number, body_file)
            expected_url = rf"https://github\.com/sgl-project/sglang/pull/{number}#issuecomment-\d+"
            if not re.fullmatch(expected_url, comment_url):
                raise ValueError("unrecognized comment response")
        except Exception as exc:
            reservation.update(status="uncertain", error_type=type(exc).__name__)
            try:
                return finish(
                    "uncertain",
                    "send_outcome_requires_reconciliation",
                    reservation_id=reservation_id,
                )
            except Exception:
                return dict(
                    result,
                    status="uncertain",
                    reason="send_and_state_outcome_require_reconciliation",
                    reservation_id=reservation_id,
                )
        reservation.update(status="sent", comment_url=comment_url)
        try:
            return finish(
                "sent",
                "retry_comment_posted",
                reservation_id=reservation_id,
                comment_url=comment_url,
            )
        except Exception:
            # The original durable reservation still prevents a second send.
            return dict(
                result,
                status="uncertain",
                reason="comment_posted_state_update_failed",
                reservation_id=reservation_id,
                comment_url=comment_url,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--pr", required=True, type=int)
    parser.add_argument("--run-id", required=True, type=int)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--reason", required=True, choices=("cancelled", "transient", "stale_gate"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        result = retry(
            json.loads(args.config.read_text()),
            args.pr,
            args.run_id,
            args.expected_head,
            args.reason,
            args.dry_run,
        )
    except Exception as exc:
        # Do not print subprocess stderr, GitHub response bodies, or credentials.
        result = {
            "status": "error",
            "reason": "local_or_github_check_failed",
            "error_type": type(exc).__name__,
            "pr": args.pr,
            "run_id": args.run_id,
            "head": args.expected_head,
        }
    print(json.dumps(result, sort_keys=True))
    return 1 if result["status"] in {"error", "uncertain"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
