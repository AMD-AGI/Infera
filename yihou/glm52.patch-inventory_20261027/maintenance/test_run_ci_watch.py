"""Offline CI watch contract tests; all temporary artifacts are retained.

Set CI_WATCH_TEST_TMPDIR to the task's explicit yihou NFS artifact directory.
No test launches an agent, invokes GitHub, publishes documents, or deletes files.
"""

import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import run_ci_watch as watch

EXPECTED_PRS = [33059, 46820, 46830, 45524, 46918, 47276, 47278, 47897, 47872]
CHECKED = "2026-11-03T08:00:00+00:00"
LATER = "2026-11-03T10:00:00+00:00"
RUN_URL = "https://github.com/sgl-project/sglang/actions/runs/41467026948"


def result():
    return {
        "summary": "Two H100 jobs remain queued; completed passes were retained.",
        "prs": [
            {
                "number": number,
                "head": "a" * 40,
                "state": "OPEN_READY",
                "base_run_url": RUN_URL,
                "base_attempt": 2,
                "base_status": "queued",
                "base_summary": "36 retained passes; two jobs queued without a runner.",
                "action": "Inspected the current head and latest attempt.",
                "next_action": "Await a runner; retain separate AMD validation constraints.",
                "evidence_urls": [RUN_URL],
                "needs_user": False,
            }
            for number in EXPECTED_PRS
        ],
        "actions": [],
        "unresolved": [],
        "tests": ["Read-only current-head workflow audit; no local test run."],
    }


def concern(key="pr-33059-native-amd", pr=33059):
    return {
        "key": key,
        "pr": pr,
        "summary": "Native AMD validation remains outstanding.",
        "next_action": "Obtain evidence from the matching source and target hardware.",
        "evidence_urls": [f"https://github.com/sgl-project/sglang/pull/{pr}"],
    }


def resolution(item, **overrides):
    action = {
        "pr": item["pr"],
        "kind": "resolved",
        "description": f"Resolved {item['key']}: matching validation is now available.",
        "url": RUN_URL,
    }
    action.update(overrides)
    return action


class CommandDiagnosticsTests(unittest.TestCase):
    def test_stdout_only_hook_failure_preserves_the_actionable_cause(self):
        command = ["pre-commit", "run", "--files", "CI_48K_WATCH.md"]
        failure = "Executable scripts/check-author-email.sh not found"
        completed = subprocess.CompletedProcess(
            command, 1, stdout="earlier output\n" * 200 + failure + "\n", stderr=""
        )
        with mock.patch.object(watch.subprocess, "run", return_value=completed):
            with self.assertRaises(RuntimeError) as raised:
                watch.run(command, Path("/home/yihou/unused-test-worktree"), {})
        message = str(raised.exception)
        self.assertIn("pre-commit exited 1", message)
        self.assertIn(failure, message)
        self.assertLessEqual(len(message), 2050)


class ValidationTests(unittest.TestCase):
    def test_exact_scope_and_missing_workflow_are_valid(self):
        value = result()
        value["prs"][0].update(base_run_url="", base_attempt=0, base_status="unknown")
        watch.validate_result(value)

    def test_scope_order_missing_duplicate_and_extra_rows_are_rejected(self):
        valid = result()["prs"]
        variations = {
            "missing": valid[:-1],
            "extra": [*valid, valid[0]],
            "duplicate": [valid[0], valid[0], *valid[2:]],
            "reordered": [valid[1], valid[0], *valid[2:]],
            "outside_scope": [dict(valid[0], number=99999), *valid[1:]],
        }
        for name, rows in variations.items():
            with self.subTest(name=name):
                value = result()
                value["prs"] = rows
                with self.assertRaises(ValueError):
                    watch.validate_result(value)

    def test_unknown_top_level_fields_and_nonobjects_are_rejected(self):
        for value in (None, [], dict(result(), instructions="expand scope")):
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises(ValueError):
                    watch.validate_result(value)

    def test_actions_and_new_concerns_cannot_expand_scope(self):
        for field, item in (
            ("actions", resolution(concern(pr=99999))),
            ("unresolved", concern(pr=99999)),
        ):
            with self.subTest(field=field):
                value = result()
                value[field] = [item]
                with self.assertRaises(ValueError):
                    watch.validate_result(value)

    def test_duplicate_unresolved_keys_cannot_hide_different_prs(self):
        value = result()
        value["unresolved"] = [concern(), concern(pr=46820)]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            watch.validate_result(value)

    def test_base_link_must_be_the_expected_repository_actions_path(self):
        for url in (
            "https://github.com.example/sgl-project/sglang/actions/runs/1",
            "https://github.com/other/project/actions/runs/1",
            "https://github.com/sgl-project/sglang/pull/33059",
            "file:///home/yihou/evidence",
        ):
            with self.subTest(url=url):
                value = result()
                value["prs"][0]["base_run_url"] = url
                with self.assertRaises(ValueError):
                    watch.validate_result(value)

    def test_string_booleans_and_attempts_are_rejected(self):
        for field, malformed in (("needs_user", "false"), ("base_attempt", "2")):
            with self.subTest(field=field):
                value = result()
                value["prs"][0][field] = malformed
                with self.assertRaises(ValueError):
                    watch.validate_result(value)


class CarryForwardTests(unittest.TestCase):
    def test_green_base_new_head_and_missing_row_attention_do_not_resolve_amd(self):
        previous = result()
        previous["unresolved"] = [concern()]
        original = copy.deepcopy(previous)
        current = result()
        current["prs"][0].update(
            head="b" * 40,
            base_status="success",
            base_summary="Base succeeded on the new head.",
            needs_user=False,
        )
        reconciled = watch.carry_unresolved(current, previous)
        self.assertEqual(reconciled["unresolved"], original["unresolved"])
        self.assertEqual(previous, original)

    def test_resolution_requires_same_pr_key_kind_and_public_evidence(self):
        item = concern()
        insufficient = (
            resolution(item, kind="retry"),
            resolution(item, pr=46820),
            resolution(item, description="Everything looks green."),
            resolution(item, url=""),
            resolution(item, url="https://github.com.example/evidence"),
        )
        for action in insufficient:
            with self.subTest(action=action):
                current = result()
                current["actions"] = [action]
                actual = watch.carry_unresolved(current, {"unresolved": [item]})
                self.assertEqual(actual["unresolved"], [item])

    def test_affirmative_resolution_removes_only_the_identified_concern(self):
        first, second = concern(), concern("pr-33059-review")
        current = result()
        current["actions"] = [resolution(first)]
        actual = watch.carry_unresolved(current, {"unresolved": [first, second]})
        self.assertEqual(actual["unresolved"], [second])

    def test_scope_exclusion_preserves_history_and_other_open_concerns(self):
        excluded, retained = concern(), concern("pr-33059-review")
        previous = {"unresolved": [excluded, retained]}
        original = copy.deepcopy(previous)
        current = result()
        current["actions"] = [
            resolution(
                excluded,
                kind="scope_excluded",
                description=(
                    f"Excluded {excluded['key']}: current diff is isolated; Base-only policy. "
                    "This does not mean the non-Base test was fixed or passed."
                ),
            )
        ]
        actual = watch.carry_unresolved(current, previous)
        self.assertEqual(actual["unresolved"], [retained])
        self.assertEqual(actual["actions"][0]["kind"], "scope_excluded")
        self.assertEqual(previous, original)

    def test_scope_exclusion_requires_matching_pr_key_and_public_evidence(self):
        item = concern()
        for overrides in (
            {"pr": 46820},
            {"description": "Unrelated checks are outside scope."},
            {"url": ""},
            {"url": "https://github.com.example/policy"},
        ):
            with self.subTest(overrides=overrides):
                current = result()
                current["actions"] = [resolution(item, kind="scope_excluded", **overrides)]
                actual = watch.carry_unresolved(current, {"unresolved": [item]})
                self.assertEqual(actual["unresolved"], [item])

    def test_closed_and_merged_resolve_but_unknown_preserves_previous_concern(self):
        for state in ("CLOSED", "MERGED", "UNKNOWN", "OPEN_DRAFT"):
            with self.subTest(state=state):
                current = result()
                current["prs"][0]["state"] = state
                actual = watch.carry_unresolved(current, {"unresolved": [concern()]})
                expected = [] if state in ("CLOSED", "MERGED") else [concern()]
                self.assertEqual(actual["unresolved"], expected)

    def test_refreshed_concern_wins_without_mutating_the_previous_evidence(self):
        old = concern()
        fresh = dict(old, next_action="Inspect the newly linked AMD validation run.")
        current = result()
        current["unresolved"] = [fresh]
        actual = watch.carry_unresolved(current, {"unresolved": [old]})
        self.assertEqual(actual["unresolved"], [fresh])
        self.assertNotEqual(old["next_action"], fresh["next_action"])

    def test_needs_user_synthesizes_a_stable_concern_without_duplicate_items(self):
        current = result()
        current["prs"][0]["needs_user"] = True
        first = watch.carry_unresolved(current, None)
        item = first["unresolved"][0]
        self.assertEqual(item["key"], "pr-33059-attention")
        self.assertEqual(item["evidence_urls"], [RUN_URL])
        next_cycle = result()
        next_cycle["prs"][0]["needs_user"] = True
        second = watch.carry_unresolved(next_cycle, first)
        self.assertEqual(second["unresolved"], [item])

    def test_existing_specific_concern_prevents_generic_attention_duplicate(self):
        current = result()
        current["prs"][0]["needs_user"] = True
        current["unresolved"] = [concern()]
        self.assertEqual(watch.carry_unresolved(current, None)["unresolved"], [concern()])


class RetainedArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        value = os.environ.get("CI_WATCH_TEST_TMPDIR")
        if not value:
            raise RuntimeError("Set CI_WATCH_TEST_TMPDIR to the task's yihou NFS directory")
        cls.artifacts = watch.yihou_path(value)
        if not cls.artifacts.is_dir():
            raise RuntimeError("Create and capacity-check the artifact directory before testing")
        mount = subprocess.run(
            ["findmnt", "-n", "-T", str(cls.artifacts), "-o", "FSTYPE"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        if not mount.startswith("nfs"):
            raise RuntimeError("These retained test artifacts require the verified NFS mount")
        stat = os.statvfs(cls.artifacts)
        if stat.f_bavail * stat.f_frsize < 32 * 1024**2:
            raise RuntimeError("Insufficient free space for retained test artifacts")

    def setUp(self):
        # Intentionally use mkdtemp without cleanup; preserve all case evidence.
        self.runtime = Path(tempfile.mkdtemp(prefix="runner-case-", dir=self.artifacts))
        self.config = {
            "runtime": str(self.runtime),
            "worktree": str(self.runtime / "worktree"),
            "path": "/usr/bin:/bin",
            "pre_commit_home": str(self.runtime / "cache/pre-commit"),
        }
        self.no_commands = mock.patch.object(
            watch, "run", side_effect=AssertionError("Unexpected command execution")
        ).start()
        self.addCleanup(mock.patch.stopall)


class RecoveryPostconditionTests(RetainedArtifacts):
    def setUp(self):
        super().setUp()
        self.config.update(python=sys.executable, min_retry_interval_seconds=3600)
        self.cycle = self.runtime / "cycle"
        self.cycle.mkdir()

    def recover(self, value=None):
        return watch.ensure_priority_recovery(
            self.config,
            self.runtime / "config.json",
            self.runtime / "inventory",
            self.runtime,
            self.cycle,
            {},
            value or result(),
        )

    def reply(self, status, reason, **extra):
        return subprocess.CompletedProcess(
            [], 0, json.dumps(dict(status=status, reason=reason, **extra)), ""
        )

    def test_missed_eligible_priority_retry_is_requested_after_active_pr_is_refused(self):
        url = "https://github.com/sgl-project/sglang/pull/46820#issuecomment-123"
        worker_result = result()
        worker_result["summary"] = "No retry was requested during the worker audit."
        worker_result["prs"][1]["action"] = "No retry or source change."
        with mock.patch.object(
            watch.subprocess,
            "run",
            side_effect=[
                self.reply("blocked", "base_run_active"),
                self.reply("dry_run", "eligible_no_comment_posted"),
                self.reply("sent", "retry_comment_posted", comment_url=url),
            ],
        ) as helper:
            value = self.recover(worker_result)
        commands = [call.args[0] for call in helper.call_args_list]
        self.assertEqual(
            [cmd[cmd.index("--pr") + 1] for cmd in commands], ["33059", "46820", "46820"]
        )
        self.assertIn("--dry-run", commands[1])
        self.assertNotIn("--dry-run", commands[2])
        self.assertEqual(value["actions"][-1]["url"], url)
        self.assertIn("execution remain unverified", value["prs"][1]["action"])
        self.assertIn("posted a recovery request for #46820", value["summary"])
        self.assertTrue(value["prs"][1]["action"].startswith("Worker audit: No retry"))
        self.assertIn("A recovery request has since been posted", value["prs"][1]["base_summary"])
        self.assertTrue(value["prs"][1]["next_action"].startswith("Verify handler acceptance"))
        self.assertEqual(value["prs"][1]["base_status"], "queued")
        self.assertEqual(len(list(self.cycle.glob("recovery-*.json"))), 3)
        watch.validate_result(value)

    def test_only_owned_prs_reach_helper_in_priority_order_and_keep_head_guards(self):
        with mock.patch.object(
            watch.subprocess, "run", return_value=self.reply("blocked", "head_changed")
        ) as helper:
            value = self.recover()
        commands = [call.args[0] for call in helper.call_args_list]
        self.assertEqual(
            [cmd[cmd.index("--pr") + 1] for cmd in commands],
            ["33059", "46820", "46830", "47276", "47278", "47897", "47872"],
        )
        self.assertTrue(all("--dry-run" in cmd for cmd in commands))
        self.assertTrue(all("head_changed" in item["description"] for item in value["actions"]))

    def test_deferred_priority_attempt_allows_owned_fallback_recovery(self):
        url = "https://github.com/sgl-project/sglang/pull/47276#issuecomment-123"
        with mock.patch.object(
            watch.subprocess,
            "run",
            side_effect=[
                self.reply("blocked", "base_run_active"),
                self.reply("blocked", "user_priority_deferred"),
                self.reply("blocked", "base_run_active"),
                self.reply("dry_run", "eligible_no_comment_posted"),
                self.reply("sent", "retry_comment_posted", comment_url=url),
            ],
        ) as helper:
            value = self.recover()
        commands = [call.args[0] for call in helper.call_args_list]
        self.assertEqual(
            [cmd[cmd.index("--pr") + 1] for cmd in commands],
            ["33059", "46820", "46830", "47276", "47276"],
        )
        self.assertIn("user_priority_deferred", value["prs"][1]["action"])
        self.assertIn("posted a recovery request for #47276", value["summary"])
        self.assertEqual(value["actions"][-1]["url"], url)
        watch.validate_result(value)

    def test_successful_closed_and_unconfigured_owned_rows_do_not_reach_helper(self):
        current = result()
        current["prs"][5]["base_status"] = "SUCCESS"
        current["prs"][6]["state"] = "CLOSED"
        self.config["owned_prs"] = [33059, 46820, 46830, 47276, 47278, 47872]
        with mock.patch.object(
            watch.subprocess, "run", return_value=self.reply("blocked", "base_run_active")
        ) as helper:
            self.recover(current)
        commands = [call.args[0] for call in helper.call_args_list]
        self.assertEqual(
            [cmd[cmd.index("--pr") + 1] for cmd in commands],
            ["33059", "46820", "46830", "47872"],
        )
        self.assertTrue(all("--dry-run" in cmd for cmd in commands))

    def test_failed_gate_uses_only_the_existing_stale_gate_path(self):
        with mock.patch.object(
            watch.subprocess,
            "run",
            side_effect=[
                self.reply("blocked", "use_stale_gate_reason"),
                self.reply("dry_run", "eligible_no_comment_posted"),
                self.reply("sent", "retry_comment_posted"),
            ],
        ) as helper:
            self.recover()
        commands = [call.args[0] for call in helper.call_args_list]
        self.assertEqual(
            [cmd[cmd.index("--reason") + 1] for cmd in commands],
            ["owner_requested", "stale_gate", "stale_gate"],
        )

    def test_recent_uncertain_reservation_records_next_slot_without_helper(self):
        watch.write_json(
            self.runtime / "retry_state.json",
            {"retries": [{"status": "uncertain", "reserved_at": watch.now()}]},
        )
        with mock.patch.object(
            watch.subprocess, "run", side_effect=AssertionError("slot is occupied")
        ):
            value = self.recover()
        self.assertEqual(value["actions"][0]["kind"], "recovery_wait")
        self.assertIn("next_eligible_at", watch.read_json(self.cycle / "recovery-slot.json"))

    def test_uncertain_request_stops_and_durable_reservation_prevents_repost(self):
        calls = 0

        def helper(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return self.reply("dry_run", "eligible_no_comment_posted")
            watch.write_json(
                self.runtime / "retry_state.json",
                {"retries": [{"status": "uncertain", "reserved_at": watch.now()}]},
            )
            response = self.reply("uncertain", "send_outcome_requires_reconciliation")
            response.returncode = 1
            return response

        with mock.patch.object(watch.subprocess, "run", side_effect=helper):
            value = self.recover()
            self.recover()
        self.assertEqual(calls, 2)
        self.assertTrue(value["prs"][0]["needs_user"])
        self.assertIn("Reconcile the durable reservation", value["prs"][0]["next_action"])
        self.assertEqual(
            watch.read_json(self.cycle / "recovery-33059-owner_requested-request.json")["status"],
            "uncertain",
        )


class NotificationTests(RetainedArtifacts):
    def notify(self, value=None, checked=CHECKED, error=None):
        watch.notify_next_session(self.runtime, checked, result=value, error=error)
        return watch.read_json(self.runtime / "next_session.json")

    def test_timestamp_and_routine_summary_changes_do_not_renotify(self):
        value = result()
        value["unresolved"] = [concern()]
        first = self.notify(value)
        second_value = copy.deepcopy(value)
        second_value["summary"] = "A later audit found the same outstanding requirement."
        second_value["unresolved"][0] = dict(reversed(list(concern().items())))
        second = self.notify(second_value, checked=LATER)
        self.assertEqual(first["fingerprint"], second["fingerprint"])
        self.assertNotEqual(first["updated_at"], second["updated_at"])
        self.assertTrue(second["pending"])

    def test_changed_concern_or_new_error_changes_the_fingerprint(self):
        value = result()
        value["unresolved"] = [concern()]
        first = self.notify(value)
        value["unresolved"][0]["next_action"] = "Maintainer confirmation is now required."
        changed = self.notify(value, checked=LATER)
        failed = self.notify(value, checked=LATER, error="Publisher remains busy.")
        self.assertNotEqual(first["fingerprint"], changed["fingerprint"])
        self.assertNotEqual(changed["fingerprint"], failed["fingerprint"])
        self.assertTrue(failed["pending"])
        self.assertEqual(failed["error"], "Publisher remains busy.")

    def test_failure_without_previous_result_has_a_readable_handoff(self):
        notice = self.notify(error="GitHub authentication failed.")
        self.assertTrue(notice["pending"])
        self.assertIn(
            "GitHub authentication failed.", (self.runtime / "NEXT_SESSION.md").read_text()
        )

    def test_current_failure_is_visible_even_when_previous_audit_had_no_concerns(self):
        previous = result()
        watch.write_json(self.runtime / "latest_result.json", previous)
        healthy = self.notify(previous)
        saved_result = (self.runtime / "latest_result.json").read_bytes()
        error = "Reports were pushed, but the shared team publisher did not complete a fresh run"

        failed = self.notify(previous, checked=LATER, error=error)

        self.assertTrue(failed["pending"])
        self.assertEqual(failed["unresolved"], [])
        self.assertEqual(failed["error"], error)
        self.assertNotEqual(failed["fingerprint"], healthy["fingerprint"])
        handoff = (self.runtime / "NEXT_SESSION.md").read_text()
        self.assertIn(error, handoff)
        self.assertLess(handoff.index(error), handoff.index("No unresolved items"))
        self.assertEqual((self.runtime / "latest_result.json").read_bytes(), saved_result)

    def test_no_concerns_and_no_error_clear_pending_notification(self):
        self.notify(error="A previous cycle failed.")
        notice = self.notify(result(), checked=LATER)
        self.assertFalse(notice["pending"])
        self.assertEqual(notice["unresolved"], [])
        self.assertIsNone(notice["error"])

    def test_acknowledging_a_notice_does_not_resolve_or_erase_findings(self):
        value = result()
        value["unresolved"] = [concern()]
        watch.write_json(self.runtime / "latest_result.json", value)
        notice = self.notify(value)
        before = {
            name: (self.runtime / name).read_bytes()
            for name in ("latest_result.json", "next_session.json", "NEXT_SESSION.md")
        }
        config_path = self.runtime / "config.json"
        watch.write_json(config_path, self.config)
        with (
            mock.patch.object(
                sys, "argv", ["run_ci_watch.py", "--config", str(config_path), "--acknowledge"]
            ),
            mock.patch.object(watch, "now", return_value=LATER),
            mock.patch("sys.stdout", new=io.StringIO()),
            mock.patch.object(watch, "run_worker") as worker,
            mock.patch.object(watch, "publish_reports") as publish,
        ):
            watch.main()
        acknowledged = watch.read_json(self.runtime / "acknowledged.json")
        self.assertEqual(acknowledged["fingerprint"], notice["fingerprint"])
        self.assertEqual(acknowledged["reported_at"], LATER)
        for name, content in before.items():
            self.assertEqual((self.runtime / name).read_bytes(), content)
        worker.assert_not_called()
        publish.assert_not_called()
        self.no_commands.assert_not_called()
        value["unresolved"][0]["next_action"] = "A new review action is required."
        new_notice = self.notify(value, checked=LATER)
        self.assertTrue(new_notice["pending"])
        self.assertNotEqual(new_notice["fingerprint"], acknowledged["fingerprint"])
        self.assertEqual(watch.read_json(self.runtime / "acknowledged.json"), acknowledged)


class EnvironmentTests(RetainedArtifacts):
    def test_new_worker_drops_parent_thread_ids_and_preserves_auth_configuration(self):
        parent = {
            "CODEX_THREAD_ID": "parent-thread",
            "CODEX_SESSION_ID": "parent-session",
            "CODEX_TOOL_CALL_ID": "parent-call",
            "CODEX_HOME": "/home/yihou/.codex",
            "HOME": "/home/yihou",
            "GH_HOST": "github.com",
            "GITHUB_TOKEN": "test-only-placeholder",
            "PATH": "/untrusted/parent/path",
        }
        with mock.patch.dict(os.environ, parent, clear=True):
            env = watch.environment(self.config)
            self.assertEqual(dict(os.environ), parent)
        for name in ("CODEX_THREAD_ID", "CODEX_SESSION_ID", "CODEX_TOOL_CALL_ID"):
            self.assertNotIn(name, env)
        for name in ("CODEX_HOME", "HOME", "GH_HOST", "GITHUB_TOKEN"):
            self.assertEqual(env[name], parent[name])
        for name in ("TMPDIR", "XDG_CACHE_HOME", "PYTHONPYCACHEPREFIX", "PRE_COMMIT_HOME"):
            self.assertTrue(Path(env[name]).is_relative_to(self.runtime))
        self.assertEqual(env["PATH"], self.config["path"])
        self.assertEqual(env["GIT_TERMINAL_PROMPT"], "0")
        self.assertEqual(env["YIHOU_CI_WATCH_WORKER"], "1")
        self.assertIn("BatchMode=yes", env["GIT_SSH_COMMAND"])

    def test_artifact_path_guard_checks_normalized_absolute_path(self):
        for path in ("/var/tmp/watch-artifacts", "/var/tmp/yihou/../watch-artifacts"):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    watch.yihou_path(path)


class PublisherFreshnessTests(RetainedArtifacts):
    def setUp(self):
        super().setUp()
        self.daily_runtime = self.runtime / "daily"
        self.daily_runtime.mkdir()
        self.daily_config = self.runtime / "daily-config.json"
        watch.write_json(self.daily_config, {"runtime": str(self.daily_runtime)})
        self.config.update(daily_config=str(self.daily_config), python=sys.executable)
        self.inventory = self.runtime / "inventory"
        self.status_path = self.daily_runtime / "run_status.json"

    def test_busy_publisher_exit_zero_cannot_reuse_old_success(self):
        watch.write_json(
            self.status_path, {"state": "SUCCESS", "started_at_utc": "2026-11-03T06:00:00+00:00"}
        )
        with (
            mock.patch.object(watch, "run", return_value="") as run,
            mock.patch.object(watch.time, "sleep") as sleep,
        ):
            with self.assertRaisesRegex(RuntimeError, "did not complete a fresh run"):
                watch.refresh_team_tracker(self.config, self.inventory, {}, CHECKED)
        self.assertEqual(run.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_missing_status_does_not_turn_zero_exit_into_publication(self):
        with (
            mock.patch.object(watch, "run", return_value="") as run,
            mock.patch.object(watch.time, "sleep"),
        ):
            with self.assertRaises(RuntimeError):
                watch.refresh_team_tracker(self.config, self.inventory, {}, CHECKED)
        self.assertEqual(run.call_count, 3)

    def test_a_later_fresh_completed_publication_is_accepted(self):
        reports = iter(
            (
                {"state": "SUCCESS", "started_at_utc": "2026-11-03T06:00:00+00:00"},
                {"state": "RUNNING", "started_at_utc": CHECKED},
                {"state": "SUCCESS", "started_at_utc": LATER, "commit": "c" * 40},
            )
        )

        def invoke(*args, **kwargs):
            watch.write_json(self.status_path, next(reports))
            return ""

        with (
            mock.patch.object(watch, "run", side_effect=invoke) as run,
            mock.patch.object(watch.time, "sleep"),
        ):
            actual = watch.refresh_team_tracker(self.config, self.inventory, {}, CHECKED)
        self.assertEqual(actual["commit"], "c" * 40)
        self.assertEqual(run.call_count, 3)
        self.assertTrue(all("run_daily.py" in str(call.args[0][1]) for call in run.call_args_list))

    def test_fresh_running_failed_or_closed_tracker_status_is_not_success(self):
        for state in ("RUNNING", "FAILED", "SUCCESS_WITH_TRACKER_CLOSED"):
            with self.subTest(state=state):
                watch.write_json(self.status_path, {"state": state, "started_at_utc": LATER})
                with (
                    mock.patch.object(watch, "run", return_value=""),
                    mock.patch.object(watch.time, "sleep"),
                ):
                    with self.assertRaises(RuntimeError):
                        watch.refresh_team_tracker(self.config, self.inventory, {}, CHECKED)

    def test_publisher_command_failure_is_not_suppressed_or_retried_blindly(self):
        with mock.patch.object(watch, "run", side_effect=RuntimeError("Publisher failed")) as run:
            with self.assertRaisesRegex(RuntimeError, "Publisher failed"):
                watch.refresh_team_tracker(self.config, self.inventory, {}, CHECKED)
        self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
