"""CPU contracts for the standalone synthetic-prefix decode benchmark."""
import importlib.util
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("batch_state"), "accounting helper is not implemented")
        from batch_state import DecodeAccounting, required_token_capacity
        self.Accounting = DecodeAccounting
        self.capacity = required_token_capacity

    def test_bonus_is_included_and_context_advances(self):
        state = self.Accounting(2, 70000, 10)
        step = state.record([3, 4], 0.02)
        self.assertEqual(step["useful_accept_lens"], [3, 4])
        self.assertEqual(state.seq_lens, [70003, 70004])
        report = state.summary(0.02)
        self.assertEqual(report["useful_output_tokens"], 7)
        self.assertEqual(report["num_correct_drafts"], 5)
        self.assertEqual(report["realized_accept_length"], 3.5)
        self.assertEqual(report["output_tokens_per_second"], 350.0)

    def test_terminal_clipping_tracks_extra_compute(self):
        state = self.Accounting(2, 70000, 5)
        state.record([4, 3], 0.1)
        step = state.record([4, 4], 0.1)
        self.assertEqual(step["useful_accept_lens"], [1, 2])
        self.assertTrue(state.complete)
        self.assertEqual(state.seq_lens, [70005, 70005])
        report = state.summary(0.2)
        self.assertEqual(report["useful_output_tokens"], 10)
        self.assertEqual(report["raw_accept_tokens"], 15)
        self.assertEqual(report["terminal_extra_tokens"], 5)

    def test_invalid_acceptance_does_not_mutate_state(self):
        state = self.Accounting(2, 70, 10)
        for values in ([0, 3], [7, 3], [3], [3.5, 4]):
            with self.assertRaises(ValueError):
                state.record(values, 0.1)
        self.assertEqual(state.seq_lens, [70, 70])
        self.assertEqual(state.verify_ct, 0)

    def test_capacity_rounds_each_request_and_includes_reserve(self):
        self.assertEqual(self.capacity(16, 70000, 10000, 64, 12), 16 * 80064)
        self.assertEqual(self.capacity(2, 70, 10, 1, 12), 184)

    def test_full_workload_emits_exactly_160000_tokens(self):
        state = self.Accounting(16, 70000, 10000)
        lengths = [3] * 39 + [4] * 61
        index = 0
        while not state.complete:
            state.record([lengths[index % 100]] * 16, 0.02)
            index += 1
        report = state.summary(index * 0.02)
        self.assertEqual(report["useful_output_tokens"], 160000)
        self.assertEqual(report["final_seq_lens"], [80000] * 16)
        self.assertLess(report["terminal_extra_tokens"], 64)
        self.assertAlmostEqual(report["realized_accept_length"], 3.61, delta=0.01)

    def test_rejects_record_after_completion(self):
        state = self.Accounting(1, 70, 1)
        state.record([3], 0.02)
        with self.assertRaises(ValueError):
            state.record([3], 0.02)
        self.assertEqual(state.verify_ct, 1)

    def test_smoke_is_not_marked_complete(self):
        state = self.Accounting(16, 70000, 10000)
        state.record([4] * 16, 0.1)
        self.assertFalse(state.summary(0.1)["complete"])


class StateContractTests(unittest.TestCase):
    def helper(self, name):
        import batch_state
        self.assertTrue(hasattr(batch_state, name), f"missing contract helper: {name}")
        return getattr(batch_state, name)

    def test_worker_lengths_checked_before_terminal_clipping(self):
        check = self.helper("validate_worker_progress")
        check([70000, 70000], [3, 4], [70003, 70004])
        with self.assertRaises(ValueError):
            check([70000, 70000], [3, 4], [70004, 70004])
        with self.assertRaises(ValueError):
            check([70000, 70000], [3], [70003, 70004])

    def test_raw_fp8_layout_rejects_mixed_rope_storage(self):
        check = self.helper("validate_kv_layout")
        check(576, 512, 64, False)
        with self.assertRaises(ValueError):
            check(656, 512, 64, True)

    def test_graph_execution_counter_records_calls_not_availability(self):
        from types import SimpleNamespace
        install = self.helper("count_graph_executions")
        runner = SimpleNamespace(execute=lambda value, increment=1: value + increment)
        counts = install({"draft": runner, "draft_extend": None})
        self.assertEqual(counts["draft"], 0)
        self.assertEqual(runner.execute(7, increment=2), 9)
        self.assertEqual(counts["draft"], 1)
        self.assertEqual(counts["draft_extend"], 0)

    def test_module_provenance_uses_specific_modules(self):
        import json
        from types import ModuleType
        paths = self.helper("module_source_paths")([json])
        self.assertEqual(paths["json"], str(pathlib.Path(json.__file__).resolve()))
        with self.assertRaises(ValueError):
            self.helper("module_source_paths")([ModuleType("namespace_without_file")])


class CliTests(unittest.TestCase):
    def test_parallel_state_import_matches_pinned_fork(self):
        source = (ROOT / "bench" / "profile_decode.py").read_text()
        self.assertIn("from sglang.srt.distributed.parallel_state_wrapper import ParallelState", source)

    def test_help_requires_no_torch_or_sglang(self):
        driver = ROOT / "bench" / "profile_decode.py"
        self.assertTrue(driver.exists(), "CLI driver is not implemented")
        result = subprocess.run([sys.executable, "-S", str(driver), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag in ("--batch-size", "--input-len", "--output-len", "--accept-length", "--result-dir", "--max-steps"):
            self.assertIn(flag, result.stdout)


if __name__ == "__main__":
    unittest.main()
