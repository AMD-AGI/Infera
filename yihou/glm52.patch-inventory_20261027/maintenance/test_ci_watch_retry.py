"""Mocked retry safety tests; retained artifacts require a yihou NFS/data root.

Set CI_WATCH_TEST_TMPDIR to a dedicated artifact directory. These tests never
contact GitHub, post comments, trigger workflows, or remove their directories.
"""

import copy
import fcntl
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import ci_watch_retry as retry

HEAD = "a" * 40
OTHER_HEAD = "b" * 40
NOW = 1793689200
PR = 47276
RUN = 41467493468


def job(name, conclusion, job_id=1):
    return {"name": name, "conclusion": conclusion, "id": job_id}


class FakeGitHub:
    def __init__(self, runtime):
        self.runtime = runtime
        self.pr_data = {
            "state": "open",
            "draft": False,
            "head": {"sha": HEAD, "ref": "topic", "repo": {"id": 9}},
            "user": {"login": retry.AUTHOR},
            "labels": [{"name": "run-ci"}],
        }
        self.run_data = {
            "id": RUN,
            "path": retry.BASE_PATH,
            "head_sha": HEAD,
            "status": "completed",
            "conclusion": "cancelled",
            "run_attempt": 1,
            "event": "pull_request",
            "head_branch": "topic",
            "head_repository": {"id": 9},
        }
        self.actor_name = retry.AUTHOR
        self.extra_runs = []
        self.job_data = {RUN: [job("base-a-test-cpu", "success"), job("pr-test-finish", "failure")]}
        self.comment_data = []
        self.gate_log = ""
        self.posts = []
        self.post_error = None
        self.pr_reads = 0
        self.race_head = False
        self.cooldown_history = []
        self.cooldown_race_history = None
        self.cooldown_job_data = {}
        self.cooldown_reads = 0
        self.policy_error = None

    def actor(self):
        return self.actor_name

    def pr(self, number):
        self.pr_reads += 1
        value = copy.deepcopy(self.pr_data)
        if self.race_head and self.pr_reads > 1:
            value["head"]["sha"] = OTHER_HEAD
        return value

    def run(self, run_id):
        return copy.deepcopy(self.run_data)

    def runs(self, head):
        return copy.deepcopy([self.run_data, *self.extra_runs])

    def jobs(self, run_id, attempt):
        return copy.deepcopy(self.job_data.get(run_id, []))

    def comments(self, number, since):
        return copy.deepcopy(self.comment_data)

    def job_log(self, job_id):
        return self.gate_log

    def cooldown_policy(self):
        if self.policy_error:
            raise self.policy_error
        return {"revision": "c" * 40, "minutes": 120}

    def cooldown_runs(self, since):
        self.cooldown_reads += 1
        if self.cooldown_reads > 1 and self.cooldown_race_history is not None:
            return copy.deepcopy(self.cooldown_race_history)
        return copy.deepcopy(self.cooldown_history)

    def cooldown_jobs(self, run_id):
        return copy.deepcopy(self.cooldown_job_data.get(run_id, []))

    def post(self, number, body_file):
        # Assert ordering at the boundary where a real network mutation occurs.
        state = json.loads((self.runtime / "retry_state.json").read_text())
        assert state["retries"][-1]["status"] == "reserved"
        assert state["retries"][-1]["run_id"] == RUN
        assert body_file.parent == self.runtime
        assert body_file.read_text().startswith("/rerun-failed-ci\n")
        self.posts.append((number, body_file))
        if self.post_error:
            raise self.post_error
        return f"https://github.com/{retry.REPO}/pull/{number}#issuecomment-12345"


class RetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        value = os.environ.get("CI_WATCH_TEST_TMPDIR")
        if not value:
            raise RuntimeError("Set CI_WATCH_TEST_TMPDIR to an explicit yihou NFS/data directory")
        cls.artifacts = retry.prepare_runtime(value)

    def setUp(self):
        # Intentionally retained; no TemporaryDirectory or recursive cleanup.
        self.runtime = Path(tempfile.mkdtemp(prefix="retry-case-", dir=self.artifacts))
        self.config = {
            "repo": retry.REPO,
            "author": retry.AUTHOR,
            "owned_prs": sorted(retry.OWNED),
            "external_prs": sorted(retry.EXTERNAL),
            "runtime": str(self.runtime),
            "min_retry_interval_seconds": 7200,
            "max_retries_per_head": 2,
        }
        self.gh = FakeGitHub(self.runtime)

    def call(self, **kwargs):
        values = dict(
            config=self.config,
            number=PR,
            run_id=RUN,
            expected_head=HEAD,
            reason="cancelled",
            gh=self.gh,
            now=NOW,
        )
        values.update(kwargs)
        return retry.retry(**values)

    def history(self, entries):
        state = {"schema_version": 1, "retries": entries, "decisions": []}
        (self.runtime / "retry_state.json").write_text(json.dumps(state))

    def entry(self, attempt=1, age=8000, run_id=RUN, status="reserved"):
        return {
            "pr": PR,
            "head": HEAD,
            "run_id": run_id,
            "attempt": attempt,
            "reserved_at": retry.utc(NOW - age),
            "status": status,
        }

    def assert_blocked(self, expected_reason, **kwargs):
        result = self.call(**kwargs)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["reason"], expected_reason)
        self.assertFalse(self.gh.posts)
        return result

    def test_active_base_is_not_retried(self):
        for status in ("queued", "in_progress", "waiting"):
            with self.subTest(status=status):
                self.gh.run_data.update(status=status, conclusion=None)
                self.assert_blocked("base_run_active")

    def test_wrong_pr_head(self):
        self.gh.pr_data["head"]["sha"] = OTHER_HEAD
        self.assert_blocked("head_changed")

    def test_wrong_run_head(self):
        self.gh.run_data["head_sha"] = OTHER_HEAD
        self.assert_blocked("run_head_changed")

    def test_owner_read_only_blocks_every_head_attempt_and_reason_without_github_access(self):
        self.config["read_only_prs"] = [33059, 46820]
        for number in self.config["read_only_prs"]:
            for head, attempt in ((HEAD, 1), (OTHER_HEAD, 2)):
                for reason in ("owner_requested", "cancelled", "transient", "stale_gate"):
                    with self.subTest(number=number, head=head, attempt=attempt, reason=reason):
                        self.gh.run_data.update(head_sha=head, run_attempt=attempt)
                        self.assert_blocked(
                            "owner_read_only", number=number, expected_head=head, reason=reason
                        )
        self.assertEqual(self.gh.pr_reads, 0)
        state = json.loads((self.runtime / "retry_state.json").read_text())
        self.assertEqual(state["retries"], [])
        self.assertTrue(all(item["reason"] == "owner_read_only" for item in state["decisions"]))

    def test_owner_read_only_does_not_block_other_prs_and_can_be_explicitly_lifted(self):
        self.config["read_only_prs"] = [33059, 46820]
        self.assertEqual(self.call(dry_run=True)["status"], "dry_run")
        self.config["read_only_prs"] = [PR]
        self.assert_blocked("owner_read_only", dry_run=True)
        self.assertFalse((self.runtime / "retry_state.json").exists())
        self.config["read_only_prs"] = []
        self.assertEqual(self.call()["status"], "sent")

    def test_read_only_config_requires_a_list_of_owned_pr_numbers(self):
        for value in (None, "33059", [True], ["33059"], [45524], [99999]):
            with self.subTest(value=value):
                self.config["read_only_prs"] = value
                with self.assertRaises(ValueError):
                    self.call()
        self.config["owned_prs"] = [PR]
        self.config["read_only_prs"] = [33059]
        with self.assertRaises(ValueError):
            self.call()
        self.assertFalse(self.gh.posts)

    def defer_current_attempt(self, **overrides):
        item = {
            "pr": PR,
            "head": HEAD,
            "run_id": RUN,
            "attempt": 1,
            "reason": "Prioritize other unfinished Base runs before integrating the upstream fix.",
            "evidence_urls": [f"https://github.com/{retry.REPO}/actions/runs/{RUN}"],
            "expires_at": retry.utc(NOW + 3600),
        }
        item.update(overrides)
        self.config["deferred_base_attempts"] = [item]
        return item

    def test_exact_deferred_attempt_records_reason_without_reserving_or_posting(self):
        item = self.defer_current_attempt()
        outcome = self.assert_blocked("user_priority_deferred", reason="owner_requested")
        self.assertEqual(outcome["deferral"], item)
        state = json.loads((self.runtime / "retry_state.json").read_text())
        self.assertEqual(state["retries"], [])
        self.assertEqual(state["decisions"][-1]["reason"], "user_priority_deferred")

    def test_deferred_dry_run_does_not_create_retry_state(self):
        self.defer_current_attempt()
        self.assert_blocked("user_priority_deferred", dry_run=True)
        self.assertFalse((self.runtime / "retry_state.json").exists())

    def test_deferral_requires_exact_pr_head_run_and_attempt(self):
        for change in ({"pr": 33059}, {"head": OTHER_HEAD}, {"run_id": RUN + 1}, {"attempt": 2}):
            with self.subTest(change=change):
                self.defer_current_attempt(**change)
                self.assertEqual(self.call(dry_run=True)["status"], "dry_run")

    def test_deferred_attempt_becomes_eligible_at_expiry_without_approval(self):
        self.defer_current_attempt(expires_at=retry.utc(NOW))
        self.assertEqual(self.call()["status"], "sent")

    def test_fresh_head_and_active_guards_precede_deferral(self):
        self.defer_current_attempt()
        self.gh.pr_data["head"]["sha"] = OTHER_HEAD
        self.assert_blocked("head_changed")
        self.gh.pr_data["head"]["sha"] = HEAD
        self.gh.run_data.update(status="queued", conclusion=None)
        self.assert_blocked("base_run_active")

    def test_deferral_rejects_missing_expiry_or_unbounded_or_external_scope(self):
        for change in (
            {"expires_at": None},
            {"expires_at": "never"},
            {"expires_at": "2026-11-05T12:00:00"},
            {"pr": 45524},
            {"attempt": 0},
            {"reason": ""},
            {"evidence_urls": []},
        ):
            with self.subTest(change=change):
                self.defer_current_attempt(**change)
                with self.assertRaises(ValueError):
                    self.call()
        self.defer_current_attempt().pop("expires_at")
        with self.assertRaises(ValueError):
            self.call()
        self.assertFalse(self.gh.posts)

    def test_external_records_permission_need_without_github_mutation(self):
        result = self.call(number=45524)
        self.assertEqual(result["status"], "permission_needed")
        self.assertEqual(self.gh.pr_reads, 0)
        self.assertFalse(self.gh.posts)
        state = json.loads((self.runtime / "retry_state.json").read_text())
        self.assertEqual(state["retries"], [])
        self.assertEqual(state["decisions"][-1]["pr"], 45524)

    def test_draft(self):
        self.gh.pr_data["draft"] = True
        self.assert_blocked("pr_draft")

    def test_missing_run_ci_label(self):
        self.gh.pr_data["labels"] = []
        self.assert_blocked("missing_run_ci_label")

    def test_closed_and_success_are_not_retried(self):
        self.gh.pr_data["state"] = "closed"
        self.assert_blocked("pr_not_open")
        self.gh.pr_data["state"] = "open"
        self.gh.run_data["conclusion"] = "success"
        self.assert_blocked("base_run_not_unsuccessful")

    def test_different_author_or_authenticated_actor(self):
        self.gh.pr_data["user"]["login"] = "someone-else"
        self.assert_blocked("not_our_author")
        self.gh.pr_data["user"]["login"] = retry.AUTHOR
        self.gh.actor_name = "someone-else"
        self.assert_blocked("authenticated_actor_not_author")

    def test_old_base_run_is_not_retried(self):
        newest = dict(self.gh.run_data, id=RUN + 1)
        self.gh.extra_runs = [newest]
        self.assert_blocked("not_latest_base_run")

    def test_quota_counts_uncertain_reservations(self):
        self.history([self.entry(1), self.entry(2, status="uncertain")])
        self.gh.run_data["run_attempt"] = 3
        self.assert_blocked("head_retry_quota_exhausted")

    def test_same_attempt_is_never_retried_twice(self):
        self.history([self.entry(status="uncertain")])
        self.assert_blocked("run_attempt_already_reserved")

    def test_interval_applies_across_attempts(self):
        self.history([self.entry(age=7199)])
        self.gh.run_data["run_attempt"] = 2
        self.assert_blocked("retry_interval_not_elapsed")

    def test_cancelled_with_real_test_failure_requires_triage(self):
        self.gh.job_data[RUN].append(job("base-b-test-1-gpu-large (1)", "failure"))
        self.assert_blocked("actual_failure_requires_triage")

    def test_pr_wide_real_failure_is_not_silently_retried(self):
        extra = dict(
            self.gh.run_data,
            id=RUN + 2,
            path=".github/workflows/pr-test-amd.yml",
            conclusion="failure",
        )
        self.gh.extra_runs = [extra]
        self.gh.job_data[extra["id"]] = [job("amd-kernel-test", "failure")]
        self.assert_blocked("actual_failure_requires_triage")

    def test_recent_slash_request_blocks_duplicate_human_retry(self):
        self.gh.comment_data = [
            {
                "body": " /rerun-failed-ci",
                "created_at": retry.utc(NOW - 40),
                "updated_at": retry.utc(NOW - 30),
                "html_url": "https://github.com/comment",
            }
        ]
        self.assert_blocked("recent_pr_slash_request")

    def test_concurrent_lock(self):
        with (self.runtime / "retry.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.call()
        self.assertEqual(result["status"], "busy")
        self.assertFalse(self.gh.posts)

    def test_head_changed_during_checks(self):
        self.gh.race_head = True
        self.assert_blocked("head_changed")

    def test_success_reserves_before_post_and_preserves_history(self):
        older = self.entry(run_id=RUN - 1)
        self.history([older])
        result = self.call()
        self.assertEqual(result["status"], "sent")
        self.assertEqual(len(self.gh.posts), 1)
        state = json.loads((self.runtime / "retry_state.json").read_text())
        self.assertEqual(state["retries"][0], older)
        self.assertEqual(state["retries"][1]["status"], "sent")
        self.assertEqual(state["retries"][1]["reason"], "cancelled")
        self.assertIn("comment_url", state["retries"][1])

    def test_uncertain_send_is_durable_and_not_resent(self):
        self.gh.post_error = TimeoutError("network outcome unknown")
        self.assertEqual(self.call()["status"], "uncertain")
        self.gh.post_error = None
        result = self.call(now=NOW + 8000)
        self.assertEqual(result["reason"], "run_attempt_already_reserved")
        self.assertEqual(len(self.gh.posts), 1)

    def test_failed_reservation_write_never_posts(self):
        with mock.patch.object(retry, "atomic_state", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                self.call()
        self.assertFalse(self.gh.posts)

    def test_post_send_state_failure_retains_reservation_and_url(self):
        real_write = retry.atomic_state
        writes = 0

        def fail_second_write(path, state):
            nonlocal writes
            writes += 1
            if writes == 2:
                raise OSError("post-send disk failure")
            real_write(path, state)

        with mock.patch.object(retry, "atomic_state", side_effect=fail_second_write):
            result = self.call()
        self.assertEqual(result["status"], "uncertain")
        self.assertEqual(result["reason"], "comment_posted_state_update_failed")
        self.assertIn("comment_url", result)
        state = json.loads((self.runtime / "retry_state.json").read_text())
        self.assertEqual(state["retries"][0]["status"], "reserved")
        self.assertEqual(self.call(now=NOW + 8000)["reason"], "run_attempt_already_reserved")
        self.assertEqual(len(self.gh.posts), 1)

    def test_config_cannot_relax_authorized_bounds(self):
        for key, value in (
            ("max_retries_per_head", 0),
            ("max_retries_per_head", True),
            ("min_retry_interval_seconds", 3599),
            ("owned_prs", [46117]),
        ):
            with self.subTest(key=key):
                config = dict(self.config, **{key: value})
                with self.assertRaises(ValueError):
                    self.call(config=config)
        self.assertFalse(self.gh.posts)

    def test_corrupt_state_is_not_overwritten(self):
        path = self.runtime / "retry_state.json"
        path.write_text("broken history")
        with self.assertRaises(json.JSONDecodeError):
            self.call()
        self.assertEqual(path.read_text(), "broken history")
        self.assertFalse(self.gh.posts)

    def test_dry_run_has_no_reservation_or_comment(self):
        result = self.call(dry_run=True)
        self.assertEqual(result["status"], "dry_run")
        self.assertFalse((self.runtime / "retry_state.json").exists())
        self.assertFalse(self.gh.posts)

    def test_stale_gate_requires_real_executed_old_gate_message(self):
        self.gh.run_data["conclusion"] = "failure"
        self.gh.job_data[RUN] = [job("call-gate / pr-gate", "failure")]
        self.gh.gate_log = '2026-11-03T00:00:00Z echo "PR is draft. Blocking CI."\n'
        self.assert_blocked("gate_not_proven_stale", reason="stale_gate")
        self.gh.gate_log = "2026-11-03T00:00:00Z PR is draft. Blocking CI.\n"
        self.assertEqual(self.call(reason="stale_gate")["status"], "sent")

    def test_model_triaged_transient_remains_bounded(self):
        self.gh.run_data["conclusion"] = "failure"
        self.gh.job_data[RUN] = [job("base-a-test-cpu", "failure")]
        self.assertEqual(self.call(reason="transient")["status"], "sent")
        self.assertEqual(self.call(reason="transient")["reason"], "run_attempt_already_reserved")

    def owner_recovery_case(self):
        self.config.update(min_retry_interval_seconds=3600, max_retries_per_head=None)
        self.gh.run_data["conclusion"] = "failure"
        self.gh.job_data[RUN] = [
            job("call-gate / pr-gate", "success"),
            job("base-b-test-1-gpu-large (1)", "failure"),
            job("pr-test-finish", "failure"),
        ]

    def test_owner_requested_failure_retains_passed_gate_without_new_cooldown_gate(self):
        self.owner_recovery_case()
        # A failed-job rerun retains the passing gate; do not add a new gate run.
        self.gh.policy_error = AssertionError("passed gate must not be re-evaluated locally")
        outcome = self.call(reason="owner_requested")
        self.assertEqual(outcome["status"], "sent")
        self.assertEqual(outcome["requested_reason"], "owner_requested")
        self.assertEqual(self.gh.cooldown_reads, 0)
        body = self.gh.posts[0][1].read_text()
        self.assertIn("owner_requested", body)
        self.assertIn("across this PR", body)

    def test_owner_requested_does_not_duplicate_active_base(self):
        self.owner_recovery_case()
        for status in ("queued", "in_progress", "waiting"):
            with self.subTest(status=status):
                self.gh.run_data.update(status=status, conclusion=None)
                self.assert_blocked("base_run_active", reason="owner_requested")

    def test_owner_requested_cannot_bypass_failed_gate(self):
        self.owner_recovery_case()
        self.gh.job_data[RUN][0]["conclusion"] = "failure"
        self.assert_blocked("use_stale_gate_reason", reason="owner_requested")
        self.assert_no_retry_reserved()

    def test_owner_requested_requires_present_gate_to_have_passed(self):
        self.owner_recovery_case()
        for conclusion in (None, "cancelled", "skipped"):
            with self.subTest(conclusion=conclusion):
                self.gh.job_data[RUN][0]["conclusion"] = conclusion
                self.assert_blocked("base_gate_not_successful", reason="owner_requested")

    def test_hourly_interval_applies_across_prs_heads_and_uncertain_sends(self):
        self.owner_recovery_case()
        for status in ("reserved", "sent", "uncertain"):
            with self.subTest(status=status):
                older = dict(self.entry(age=3599, status=status), pr=33059, head=OTHER_HEAD)
                self.history([older])
                self.assert_blocked("retry_interval_not_elapsed", reason="owner_requested")
        self.history([dict(self.entry(age=3600), pr=33059, head=OTHER_HEAD)])
        self.assertEqual(self.call(reason="owner_requested")["status"], "sent")

    def test_null_quota_allows_recovery_after_two_prior_head_attempts(self):
        self.owner_recovery_case()
        self.history([self.entry(1, age=8000), self.entry(2, age=4000, status="uncertain")])
        self.gh.run_data["run_attempt"] = 3
        self.assertEqual(self.call(reason="owner_requested")["status"], "sent")
        self.assertEqual(
            len(json.loads((self.runtime / "retry_state.json").read_text())["retries"]), 3
        )

    def test_positive_configured_quota_remains_enforced(self):
        self.owner_recovery_case()
        self.config["max_retries_per_head"] = 3
        self.history([self.entry(i, age=16000 - i * 4000) for i in range(1, 4)])
        self.gh.run_data["run_attempt"] = 4
        self.assert_blocked("head_retry_quota_exhausted", reason="owner_requested")

    def test_owner_requested_unknown_send_remains_reserved(self):
        self.owner_recovery_case()
        self.gh.post_error = TimeoutError("unknown delivery")
        self.assertEqual(self.call(reason="owner_requested")["status"], "uncertain")
        self.gh.post_error = None
        self.assertEqual(
            self.call(reason="owner_requested", now=NOW + 4000)["reason"],
            "run_attempt_already_reserved",
        )
        self.assertEqual(len(self.gh.posts), 1)

    def test_owner_requested_rechecks_head_and_scope_before_posting(self):
        self.owner_recovery_case()
        self.gh.race_head = True
        self.assert_blocked("head_changed", reason="owner_requested")
        self.gh.race_head = False
        original = self.gh.runs
        reads = 0

        def changed_scope(head):
            nonlocal reads
            reads += 1
            runs = original(head)
            if reads > 1:
                runs.append(
                    dict(self.gh.run_data, id=RUN + 1, path=".github/workflows/pr-test-extra.yml")
                )
            return runs

        with mock.patch.object(self.gh, "runs", side_effect=changed_scope):
            self.assert_blocked("workflow_scope_changed", reason="owner_requested")

    def cooldown_case(self, age=7201):
        self.gh.run_data["conclusion"] = "failure"
        self.gh.job_data[RUN] = [job("call-gate / pr-gate", "failure")]
        created = retry.utc(NOW - age).replace("+00:00", "Z")
        self.gh.gate_log = (
            f"2026-11-03T00:00:00Z ##[error]User '{retry.AUTHOR}' already triggered "
            f"'PR Test Base' via 'pull_request' at {created}. "
            "Please wait 120 minutes before triggering again.\n"
        )

    def global_run(
        self, run_id=RUN + 20, age=10, actor=retry.AUTHOR, triggering="github-actions[bot]"
    ):
        return {
            "id": run_id,
            "created_at": retry.utc(NOW - age),
            "path": retry.BASE_PATH,
            "event": "pull_request",
            "actor": {"login": actor},
            "triggering_actor": {"login": triggering},
        }

    def assert_no_retry_reserved(self):
        self.assertFalse(self.gh.posts)
        self.assertEqual(json.loads((self.runtime / "retry_state.json").read_text())["retries"], [])

    def test_future_cooldown_deadline_blocks_without_spending_quota(self):
        self.cooldown_case(age=7100)
        outcome = self.assert_blocked("author_cooldown_active", reason="stale_gate")
        self.assertEqual(outcome["retry_after"], retry.utc(NOW + 101))
        self.assertEqual(self.gh.cooldown_reads, 0)
        self.assert_no_retry_reserved()

    def test_exact_cooldown_boundary_is_still_blocked_then_expires(self):
        self.cooldown_case(age=7200)
        self.assert_blocked("author_cooldown_active", reason="stale_gate")
        self.assert_no_retry_reserved()
        result = self.call(reason="stale_gate", now=NOW + 1)
        self.assertEqual(result["status"], "sent")
        self.assertEqual(self.gh.cooldown_reads, 2)

    def test_cooldown_rejects_echo_malformed_and_wrong_author(self):
        self.cooldown_case()
        valid = self.gh.gate_log
        variants = [
            valid.replace("##[error]User", 'echo "##[error]User'),
            valid.replace(retry.AUTHOR, "another-author"),
            valid.replace("120 minutes", "unknown minutes"),
            valid.replace("via 'pull_request'", "via 'workflow_dispatch'"),
            valid.replace("PR Test Base", "Other Workflow"),
        ]
        for value in variants:
            with self.subTest(value=value):
                self.gh.gate_log = value
                self.assert_blocked("gate_not_proven_stale", reason="stale_gate")
        self.assert_no_retry_reserved()

    def test_new_countable_run_for_actor_or_triggering_actor_extends_cooldown(self):
        self.cooldown_case()
        for actor, triggering in ((retry.AUTHOR, "github-actions[bot]"), ("someone", retry.AUTHOR)):
            with self.subTest(actor=actor):
                run = self.global_run(actor=actor, triggering=triggering)
                self.gh.cooldown_history = [run]
                self.gh.cooldown_job_data[run["id"]] = [job("base-a-test-cpu / unit", "success")]
                outcome = self.assert_blocked("author_cooldown_active", reason="stale_gate")
                self.assertEqual(outcome["blocking_run_id"], run["id"])
                self.assertEqual(outcome["retry_after"], retry.utc(NOW + 7191))
        self.assert_no_retry_reserved()

    def test_global_screen_ignores_current_old_other_author_and_gate_only_runs(self):
        self.cooldown_case()
        current = self.global_run(run_id=RUN)
        old = self.global_run(run_id=RUN + 1, age=8000)
        other = self.global_run(run_id=RUN + 2, actor="other", triggering="other")
        gated = self.global_run(run_id=RUN + 3)
        self.gh.cooldown_history = [current, old, other, gated]
        for item in (current, old, other):
            self.gh.cooldown_job_data[item["id"]] = [job("actual-test", "success")]
        self.gh.cooldown_job_data[gated["id"]] = [
            job("check-changes / check-changes", "success"),
            job("call-gate / pr-gate", "failure"),
            job("pr-test-finish", "failure"),
            job("actual-test", "cancelled"),
            job("pending-test", None),
        ]
        self.assertEqual(self.call(reason="stale_gate")["status"], "sent")

    def test_global_screen_matches_upstream_conservative_sixth_run_limit(self):
        self.cooldown_case()
        self.gh.cooldown_history = [self.global_run(run_id=RUN + i + 1) for i in range(6)]
        outcome = self.assert_blocked("author_cooldown_history_ambiguous", reason="stale_gate")
        self.assertIn("retry_after", outcome)
        self.assert_no_retry_reserved()

    def test_unavailable_fresh_policy_blocks_without_consuming_quota(self):
        self.cooldown_case()
        self.gh.policy_error = ValueError("upstream policy changed")
        self.assert_blocked("cooldown_policy_or_evidence_unavailable", reason="stale_gate")
        self.assert_no_retry_reserved()

    def test_missing_global_job_evidence_fails_closed(self):
        self.cooldown_case()
        self.gh.cooldown_history = [self.global_run()]
        with mock.patch.object(self.gh, "cooldown_jobs", side_effect=OSError("API unavailable")):
            self.assert_blocked("cooldown_policy_or_evidence_unavailable", reason="stale_gate")
        self.assert_no_retry_reserved()

    def test_countable_run_appearing_before_send_is_caught(self):
        self.cooldown_case()
        run = self.global_run()
        self.gh.cooldown_race_history = [run]
        self.gh.cooldown_job_data[run["id"]] = [job("actual-test", "failure")]
        self.assert_blocked("author_cooldown_active", reason="stale_gate")
        self.assertEqual(self.gh.cooldown_reads, 2)
        self.assert_no_retry_reserved()

    def test_head_changed_during_second_global_scan_is_caught(self):
        self.cooldown_case()
        original = self.gh.cooldown_runs

        def change_head_on_second_scan(since):
            result = original(since)
            if self.gh.cooldown_reads == 2:
                self.gh.pr_data["head"]["sha"] = OTHER_HEAD
            return result

        with mock.patch.object(self.gh, "cooldown_runs", side_effect=change_head_on_second_scan):
            self.assert_blocked("head_changed", reason="stale_gate")
        self.assertEqual(self.gh.cooldown_reads, 2)
        self.assert_no_retry_reserved()

    def test_verified_mlx_musa_summaries_do_not_mask_expired_gate(self):
        self.cooldown_case()
        for index, backend in enumerate(("mlx", "musa"), start=1):
            extra = dict(
                self.gh.run_data,
                id=RUN + index,
                path=f".github/workflows/pr-test-{backend}.yml",
            )
            self.gh.extra_runs.append(extra)
            self.gh.job_data[extra["id"]] = [
                job("call-gate / pr-gate", "failure"),
                job(f"{backend}-functional-test", "skipped"),
                job(f"pr-test-{backend}-finish", "failure"),
            ]
        self.assertEqual(self.call(reason="stale_gate")["status"], "sent")

    def test_mlx_musa_actual_failures_still_require_triage(self):
        self.cooldown_case()
        for backend in ("mlx", "musa"):
            with self.subTest(backend=backend):
                extra = dict(
                    self.gh.run_data,
                    id=RUN + 1,
                    path=f".github/workflows/pr-test-{backend}.yml",
                )
                self.gh.extra_runs = [extra]
                self.gh.job_data[extra["id"]] = [
                    job(f"{backend}-functional-test", "failure"),
                    job(f"pr-test-{backend}-finish", "failure"),
                ]
                outcome = self.assert_blocked("actual_failure_requires_triage", reason="stale_gate")
                self.assertEqual(outcome["failed_jobs"], [f"{backend}-functional-test"])
                self.assert_no_retry_reserved()

    def test_cooldown_history_paginates_and_rejects_incomplete_results(self):
        gh = retry.GitHub()
        pages = [
            {"total_count": 2, "workflow_runs": [{"id": 1}]},
            {"total_count": 2, "workflow_runs": [{"id": 2}]},
        ]
        with mock.patch.object(gh, "command", return_value=json.dumps(pages)) as command:
            self.assertEqual([r["id"] for r in gh.cooldown_runs(NOW - 7200)], [1, 2])
            args = command.call_args.args[0]
            self.assertIn("--paginate", args)
            self.assertIn("event=pull_request", args[-1])
            self.assertIn("created=%3E%3D", args[-1])
        with mock.patch.object(gh, "command", return_value=json.dumps(pages[:1])):
            with self.assertRaises(ValueError):
                gh.cooldown_runs(NOW - 7200)


if __name__ == "__main__":
    unittest.main()
