"""CPU contracts of batch_state.py (accounting, acceptance bounds, block sizing, KV fill policy)."""
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

from batch_state import (DecodeAccounting, accept_bounds, blocks_for_request, count_graph_executions,  # noqa: E402
                         kv_fill_rule, length_to_rates, required_tokens_per_request,
                         unconditional_to_conditional, validate_worker_progress)


class AccountingTests(unittest.TestCase):
    def test_bonus_is_included_and_context_advances(self):
        state = DecodeAccounting(2, 70000, 10)
        step = state.record([3, 4], 0.02)
        self.assertEqual(step["useful_accept_lens"], [3, 4])
        self.assertEqual(state.seq_lens, [70003, 70004])
        report = state.summary(0.02)
        self.assertEqual(report["useful_output_tokens"], 7)
        self.assertEqual(report["num_correct_drafts"], 5)
        self.assertEqual(report["realized_accept_length"], 3.5)
        self.assertEqual(report["output_tokens_per_second"], 350.0)

    def test_finished_request_keeps_decoding_but_is_not_counted(self):
        state = DecodeAccounting(2, 100, 5)
        state.record([4, 3], 0.1)
        self.assertFalse(state.complete)
        state.record([4, 3], 0.1)  # request 0 done at step 2 (8 raw, 5 useful), request 1 at 6 raw
        self.assertTrue(state.complete)
        self.assertEqual(state.emitted, [5, 5])
        self.assertEqual(state.seq_lens, [108, 106])  # committed context keeps every accepted token
        report = state.summary(0.2)
        self.assertEqual(report["useful_output_tokens"], 10)
        self.assertEqual(report["terminal_extra_tokens"], 4)
        self.assertEqual(report["request_completion_iteration_max"], 2)

    def test_per_request_completion_iterations(self):
        state = DecodeAccounting(2, 10, 6)
        state.record([4, 3], 0.1)
        state.record([4, 3], 0.1)
        self.assertEqual(state.completed_at, [2, 2])
        state = DecodeAccounting(2, 10, 7)
        state.record([4, 3], 0.1)
        state.record([4, 3], 0.1)
        self.assertEqual(state.completed_at, [2, None])
        state.record([4, 3], 0.1)
        self.assertEqual(state.completed_at, [2, 3])

    def test_invalid_acceptance_does_not_mutate_state(self):
        state = DecodeAccounting(2, 70, 10)
        for values in ([0, 3], [7, 3], [3], [3.5, 4], [True, 3]):
            with self.assertRaises(ValueError):
                state.record(values, 0.1)
        self.assertEqual(state.seq_lens, [70, 70])
        self.assertEqual(state.verify_ct, 0)

    def test_heterogeneous_input_len_is_none(self):
        report = DecodeAccounting(2, [10, 20], 3).summary(1.0)
        self.assertIsNone(report["input_len"])
        self.assertFalse(report["input_len_uniform"])

    def test_cannot_record_after_completion(self):
        state = DecodeAccounting(1, 10, 3)
        state.record([3], 0.1)
        with self.assertRaises(ValueError):
            state.record([3], 0.1)


class AcceptanceTests(unittest.TestCase):
    def test_length_to_rates_matches_vllm_minimum_variance(self):
        # vllm/config/speculative.py::_acceptance_length_to_rates, 3.51 with K=5
        rates = length_to_rates(3.51, 5)
        self.assertEqual(rates[:2], [1.0, 1.0])
        self.assertAlmostEqual(rates[2], 0.51)
        self.assertEqual(rates[3:], [0.0, 0.0])
        self.assertEqual(length_to_rates(6, 5), [1.0] * 5)
        with self.assertRaises(ValueError):
            length_to_rates(6.5, 5)

    def test_conditional_and_bounds(self):
        conditional = unconditional_to_conditional(length_to_rates(3.51, 5))
        self.assertEqual(accept_bounds(conditional), (3, 4))
        self.assertEqual(accept_bounds([0.5, 0.5]), (1, 3))
        self.assertEqual(accept_bounds([1.0, 1.0]), (3, 3))
        self.assertEqual(accept_bounds([0.0, 0.0]), (1, 1))

    def test_required_tokens_bound(self):
        # OSL 1024 with accepts in [3, 4]: <= 342 steps x 4 + 2 * (1 + 5)
        self.assertEqual(required_tokens_per_request(1000, 1024, 3, 4, 5), 1000 + 4 * 342 + 12)
        with self.assertRaises(ValueError):
            required_tokens_per_request(10, 10, 4, 3, 5)


class BlockTests(unittest.TestCase):
    def test_full_and_sliding_and_rings(self):
        self.assertEqual(blocks_for_request("full", 128, 999, 1000), (0, 8))
        # SWA window 128: blocks wholly before the window at admission are the null block
        self.assertEqual(blocks_for_request("sliding", 128, 999, 1000, window=128), (6, 2))
        self.assertEqual(blocks_for_request("circular", 8, 999, 1000), (0, 1))
        self.assertEqual(blocks_for_request("mamba", 128, 999, 1000, num_spec_blocks=5), (0, 6))
        with self.assertRaises(ValueError):
            blocks_for_request("unknown", 128, 1, 2)
        with self.assertRaises(ValueError):
            blocks_for_request("sliding", 128, 1, 2)


class FillPolicyTests(unittest.TestCase):
    def test_rules(self):
        self.assertEqual(kv_fill_rule("torch.uint8", 1, False, False), "bytes_0_0x7c")
        self.assertEqual(kv_fill_rule("torch.float8_e4m3fn", 1, True, False), "bytes_0_0x7c")
        self.assertEqual(kv_fill_rule("torch.bfloat16", 2, True, False), "uniform_-1_1")
        self.assertEqual(kv_fill_rule("torch.float32", 4, True, True), "zeros")
        with self.assertRaises(ValueError):
            kv_fill_rule("torch.int32", 4, False, False)

    def test_byte_rule_is_finite_for_every_float_reading(self):
        import struct
        top = 0x7B  # largest allowed byte
        self.assertTrue(struct.unpack("<f", bytes([0xFF, 0xFF, 0xFF, top]))[0] < float("inf"))
        self.assertTrue(struct.unpack("<e", bytes([0xFF, top]))[0] < float("inf"))


class MiscTests(unittest.TestCase):
    def test_progress_validation(self):
        validate_worker_progress([10, 20], [3, 4], [13, 24])
        with self.assertRaises(ValueError):
            validate_worker_progress([10, 20], [3, 4], [13, 23])

    def test_graph_counter_wraps_and_preserves_result(self):
        class Manager:
            def run_fullgraph(self, x):
                return x * 2
        manager = Manager()
        counts = count_graph_executions({"target": manager, "draft": None})
        self.assertEqual(manager.run_fullgraph(3), 6)
        self.assertEqual(counts, {"target": 1, "draft": 0})

    def test_help_requires_no_torch_or_vllm(self):
        program = ("import sys, runpy; sys.argv=['profile_decode.py','--help']; sys.path.insert(0, %r)\n"
                   "try:\n    runpy.run_path(%r, run_name='__main__')\nexcept SystemExit:\n    pass\n"
                   "print([m for m in ('torch','vllm','numpy') if m in sys.modules])"
                   % (str(ROOT / "bench"), str(ROOT / "bench" / "profile_decode.py")))
        out = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, check=True).stdout
        self.assertTrue(out.strip().endswith("[]"), out[-200:])


if __name__ == "__main__":
    unittest.main()
