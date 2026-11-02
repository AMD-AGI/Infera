import unittest

import update_pr_ledger as ledger


class HandoffTests(unittest.TestCase):
    def test_unknown_or_restricted_visibility_never_publishes(self):
        prs = [{"repo": "public/a"}, {"repo": "private/b"}, {"repo": "unknown/c"}]
        self.assertEqual(
            ledger.public_only(prs, {"public/a": "public", "private/b": "restricted"}), [prs[0]]
        )

    def test_changed_head_does_not_inherit_evidence(self):
        self.assertIn(
            "HEAD CHANGED", ledger.evidence_status({"headRefOid": "new"}, {"reviewed_head": "old"})
        )
        self.assertEqual(ledger.evidence_status({"headRefOid": "new"}, {}), "NOT AUDITED")

    def test_same_head_retains_scoped_evidence(self):
        self.assertIn(
            "stated limits",
            ledger.evidence_status({"headRefOid": "abcd1234"}, {"reviewed_head": "abcd"}),
        )

    def test_new_pr_is_unclassified_and_has_an_owner_action(self):
        note = ledger.note_for({"repo": "public/a", "number": 1, "title": "New work"}, {"prs": {}})
        self.assertEqual(note["priority"], "TRIAGE")
        self.assertIn("Unclassified", note["deployment"])
        self.assertIn("review diff", note["ready_action"])

    def test_nullable_historical_checks(self):
        self.assertEqual(ledger.check_counts({"statusCheckRollup": None}), {})

    def test_new_activity_is_flagged_even_without_a_head_change(self):
        self.assertIn(
            "NEW/UNAUDITED",
            ledger.activity_status({"updatedAt": "new"}, {"reviewed_activity_at": "old"}),
        )

    def test_skips_and_gates_are_not_successes(self):
        counts = ledger.check_counts(
            {
                "statusCheckRollup": [
                    {"conclusion": "SKIPPED"},
                    {"conclusion": "FAILURE"},
                    {"status": "QUEUED"},
                ]
            }
        )
        self.assertEqual(counts, {"FAILURE": 1, "QUEUED": 1, "SKIPPED": 1})
        self.assertNotIn("SUCCESS", counts)

    def test_closed_draft_is_closed_not_active(self):
        self.assertEqual(ledger.state({"state": "CLOSED", "isDraft": True}), "CLOSED")
        self.assertEqual(ledger.state({"state": "MERGED", "isDraft": False}), "MERGED")
        self.assertEqual(ledger.state({"state": "OPEN", "isDraft": True}), "DRAFT")
        self.assertEqual(ledger.state({"state": "OPEN", "isDraft": False}), "READY")


if __name__ == "__main__":
    unittest.main()
