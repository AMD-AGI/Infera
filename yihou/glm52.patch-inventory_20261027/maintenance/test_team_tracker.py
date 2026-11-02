import json
import unittest
from pathlib import Path
from unittest.mock import patch

import export_team_tracker as exporter
import run_daily


class TrackerTests(unittest.TestCase):
    def test_packup_links_are_pinned_and_do_not_require_main_files(self):
        result = exporter.pinned("[run](../sample.packup/README.md)")
        self.assertIn(f"/blob/{exporter.ARCHIVE}/yihou/sample.packup/README.md", result)

    def test_live_upstream_pr_links_remain_live(self):
        source = "[PR](https://github.com/sgl-project/sglang/pull/33059)"
        self.assertEqual(exporter.pinned(source), source)

    def test_old_branch_source_link_is_immutable(self):
        result = exporter.pinned(
            "[evidence](https://github.com/AMD-AGI/Infera/blob/dev/pd_opt/glm_5.2_agentx/yihou/run/README.md)"
        )
        self.assertIn(f"/blob/{exporter.ARCHIVE}/", result)
        self.assertNotIn("dev/pd_opt", result)

    def config(self):
        return {
            "team_tracker": {
                "pr_url": "https://github.com/AMD-AGI/Infera/pull/1",
                "branch": "dev/yihou/tracker",
                "worktree": "/home/yihou/unused-test-worktree",
            }
        }

    def test_merged_tracker_stops_before_any_git_mutation(self):
        with patch.object(
            run_daily,
            "run",
            return_value=json.dumps({"state": "MERGED", "headRefName": "dev/yihou/tracker"}),
        ) as call:
            result = run_daily.publish_team_tracker(
                self.config(), Path("/home/yihou/runtime"), Path("/home/yihou/inventory"), {}
            )
        self.assertEqual(result["state"], "FOLLOWUP_PR_NEEDED")
        self.assertEqual(call.call_count, 1)
        self.assertEqual(call.call_args.args[0][:3], ["gh", "pr", "view"])

    def test_changed_branch_is_not_written(self):
        answers = [
            json.dumps({"state": "OPEN", "headRefName": "dev/yihou/tracker"}),
            "someone-else\n",
        ]
        with patch.object(run_daily, "run", side_effect=answers) as call:
            with self.assertRaisesRegex(RuntimeError, "branch"):
                run_daily.publish_team_tracker(
                    self.config(), Path("/home/yihou/runtime"), Path("/home/yihou/inventory"), {}
                )
        self.assertEqual(call.call_count, 2)

    def test_dirty_team_worktree_is_preserved(self):
        answers = [
            json.dumps({"state": "OPEN", "headRefName": "dev/yihou/tracker"}),
            "dev/yihou/tracker\n",
            " M notes.md\n",
        ]
        with patch.object(run_daily, "run", side_effect=answers) as call:
            with self.assertRaisesRegex(RuntimeError, "dirty"):
                run_daily.publish_team_tracker(
                    self.config(), Path("/home/yihou/runtime"), Path("/home/yihou/inventory"), {}
                )
        self.assertEqual(call.call_count, 3)


if __name__ == "__main__":
    unittest.main()
