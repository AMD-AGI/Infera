"""CPU-only TP/EP/DPA contracts for the pinned scheduler-free benchmark."""
import importlib.util
import pathlib
import sys
import unittest
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))


class TopologyTests(unittest.TestCase):
    def helper(self, name):
        self.assertIsNotNone(importlib.util.find_spec("topology"), "topology helper missing")
        import topology
        self.assertTrue(hasattr(topology, name), f"missing topology helper: {name}")
        return getattr(topology, name)

    def topology(self, concurrency=16, enabled=False, tp=4, ep=4):
        return self.helper("DecodeTopology")(tp, ep, concurrency, enabled)

    def test_global_concurrency_maps_to_local_graph_batch(self):
        for concurrency in (4, 8, 16, 20, 24):
            for enabled in (False, True):
                topology = self.topology(concurrency, enabled)
                self.assertEqual(topology.dp_size, 4 if enabled else 1)
                self.assertEqual(topology.local_batch_size, concurrency // topology.dp_size)
                self.assertEqual(topology.representative_ranks, [0, 1, 2, 3] if enabled else [0])

    def test_parallel_state_matches_pinned_rank_contract(self):
        for enabled in (False, True):
            topology = self.topology(enabled=enabled)
            for rank in range(4):
                state = topology.parallel_state_kwargs(rank)
                self.assertEqual((state["tp_rank"], state["tp_size"]), (rank, 4))
                self.assertEqual((state["moe_ep_rank"], state["moe_ep_size"]), (rank, 4))
                self.assertEqual((state["moe_dp_rank"], state["moe_dp_size"]), (0, 1))
                self.assertEqual((state["dp_rank"], state["dp_size"]), (rank, 4) if enabled else (None, 1))
                self.assertEqual((state["attn_tp_rank"], state["attn_tp_size"]), (0, 1) if enabled else (rank, 4))
                self.assertEqual((state["attn_dp_rank"], state["attn_dp_size"]), (rank, 4) if enabled else (0, 1))
                for prefix in ("pp", "attn_cp", "attn_dcp"):
                    self.assertEqual((state[prefix + "_rank"], state[prefix + "_size"]), (0, 1))

    def test_tp8_default_and_shard_request_ids(self):
        legacy = self.topology(tp=8, ep=1)
        self.assertEqual(legacy.local_batch_size, 16)
        self.assertEqual(legacy.parallel_state_kwargs(7)["moe_ep_rank"], 0)
        self.assertEqual(legacy.request_id_offset(7), 0)
        on = self.topology(concurrency=20, enabled=True)
        self.assertEqual([on.request_id_offset(r) for r in range(4)], [0, 5, 10, 15])

    def test_invalid_topology_rejected(self):
        for values in ((4, 4, 5, True), (4, 3, 16, False), (0, 1, 16, False), (4, 0, 16, False)):
            with self.assertRaises(ValueError):
                self.helper("DecodeTopology")(*values)
        with self.assertRaises(ValueError):
            self.topology().parallel_state_kwargs(4)

    def test_forward_metadata_uses_base_counts_and_preserves_draft_input(self):
        prepare = self.helper("prepare_dp_metadata")
        for enabled in (False, True):
            topology = self.topology(enabled=enabled)
            info = object()
            batch = SimpleNamespace(spec_info=info)
            prepare(batch, topology, is_extend=True, disable_cuda_graph=False)
            self.assertEqual(batch.global_num_tokens, [topology.local_batch_size] * topology.dp_size)
            self.assertEqual(batch.global_num_tokens_for_logprob, batch.global_num_tokens)
            self.assertTrue(batch.is_extend_in_batch)
            self.assertFalse(batch.can_run_dp_cuda_graph)
            self.assertFalse(batch.can_run_dp_breakable_cuda_graph)
            prepare(batch, topology, is_extend=False, disable_cuda_graph=False)
            self.assertEqual(batch.global_num_tokens, [topology.local_batch_size] * topology.dp_size)
            self.assertFalse(batch.is_extend_in_batch)
            self.assertTrue(batch.can_run_dp_cuda_graph)
            self.assertIs(batch.spec_info, info)
            prepare(batch, topology, is_extend=False, disable_cuda_graph=True)
            self.assertFalse(batch.can_run_dp_cuda_graph)

    def test_rank_guard_detects_acceptance_progress_and_completion_divergence(self):
        signature = self.helper("rank_progress_signature")
        validate = self.helper("validate_rank_progress")
        good = signature(1, [70, 70], [4, 4], [74, 74], 75)
        validate([good] * 4)
        for changed in (
            signature(1, [70, 70], [3, 3], [73, 73], 75),
            signature(1, [70, 70], [4, 3], [74, 73], 75),
            signature(1, [70, 70], [4, 4], [75, 74], 75),
            signature(1, [70, 70], [4, 4], [74, 74], 74),
        ):
            with self.assertRaises(ValueError):
                validate([good, good, changed, good])

    def test_rank_guard_covers_every_position_of_a_heterogeneous_batch(self):
        signature = self.helper("rank_progress_signature")
        validate = self.helper("validate_rank_progress")
        mixed = signature(1, [70, 800, 90], [4, 4, 4], [74, 804, 94], [75, 805, 95])
        validate([mixed] * 4)  # identical vectors across ranks: Plan 1's normal case
        # profile_decode.py preallocates torch.empty(7) all-gather buffers.
        self.assertEqual(len(mixed), 7)
        self.assertTrue(all(isinstance(value, int) and abs(value) < 2 ** 63 for value in mixed))
        for changed in (
            # Divergence outside element 0 — invisible to the old previous_lens[0] comparison.
            signature(1, [70, 801, 90], [4, 4, 4], [74, 805, 94], 75),
            signature(1, [70, 800, 91], [4, 4, 4], [74, 804, 95], 75),
            # Same multiset, different order: Plan 1 guarantees order, so this is a real bug.
            signature(1, [90, 800, 70], [4, 4, 4], [94, 804, 74], 75),
        ):
            with self.assertRaises(ValueError):
                validate([mixed, mixed, changed, mixed])

    def test_completion_is_elementwise_against_each_request_own_final(self):
        signature = self.helper("rank_progress_signature")
        # ISL [70, 800] with output_len 5: request 0 finishes at 75, request 1 at 805.
        finals = [75, 805]
        done = signature(9, [71, 801], [4, 4], [75, 805], finals)
        self.assertEqual(done[6], 1, "every request at its own final is complete")
        partial = signature(9, [71, 96], [4, 4], [75, 100], finals)
        self.assertEqual(partial[6], 0, "request 1 is far from 805, so the batch is not complete")
        self.assertTrue(partial[0], "an incomplete batch is still a valid one")
        # A single scalar cannot express this: min(finals) would call `partial` complete and
        # max(finals) would call `done` incomplete. Both are wrong for a ragged batch.
        self.assertTrue(all(length >= min(finals) for length in [75, 100]))
        self.assertFalse(all(length >= max(finals) for length in [75, 805]))

    def test_final_lens_length_mismatch_is_encoded_not_raised(self):
        signature = self.helper("rank_progress_signature")
        validate = self.helper("validate_rank_progress")
        bad = signature(9, [71, 801], [4, 4], [75, 805], [75])
        self.assertEqual((bad[0], bad[6]), (0, 0), "a mismatch must travel as invalidity, not a hang")
        with self.assertRaises(ValueError):
            validate([bad] * 4)

    def test_scalar_final_len_still_broadcasts_for_uniform_callers(self):
        signature = self.helper("rank_progress_signature")
        self.assertEqual(signature(1, [70, 70], [4, 4], [74, 74], 75),
                         signature(1, [70, 70], [4, 4], [74, 74], [75, 75]))

    def test_rank_guard_still_rejects_non_uniform_acceptance_when_lengths_differ(self):
        signature = self.helper("rank_progress_signature")
        validate = self.helper("validate_rank_progress")
        bad = signature(1, [70, 800], [4, 3], [74, 803], 75)
        self.assertFalse(bad[0], "non-uniform accept_lens must be encoded as invalid, not raised")
        with self.assertRaises(ValueError):
            validate([bad] * 4)

    def test_aggregate_rejects_divergent_isl_vectors(self):
        from batch_state import DecodeAccounting
        aggregate = self.helper("aggregate_rank_summaries")
        topology = self.topology(concurrency=8, enabled=True)
        lengths = [70, 800]
        reports = []
        for rank in range(4):
            state = DecodeAccounting(topology.local_batch_size, lengths, 5)
            state.record([3] * topology.local_batch_size, 0.1)
            report = state.summary(1.0)
            report.update(rank=rank, graph_execution_counts={})
            reports.append(report)
        merged = aggregate(reports, topology)
        self.assertIsNone(merged["input_len"])
        self.assertFalse(merged["input_len_uniform"])
        self.assertEqual(merged["final_seq_lens"], [73, 803] * 4)
        # Global view: zipping these two must not silently truncate.
        self.assertEqual(merged["input_lens"], lengths * 4)
        self.assertEqual(len(merged["input_lens"]), len(merged["final_seq_lens"]))
        self.assertEqual(len(merged["input_lens"]), merged["batch_size"])
        reports[2]["input_lens"] = [800, 70]
        with self.assertRaises(ValueError):
            aggregate(reports, topology)

    def test_aggregate_counts_each_dp_shard_once_and_uses_max_elapsed(self):
        from batch_state import DecodeAccounting
        aggregate = self.helper("aggregate_rank_summaries")
        for enabled in (False, True):
            topology = self.topology(concurrency=20, enabled=enabled)
            reports = []
            for rank in range(4):
                state = DecodeAccounting(topology.local_batch_size, 70, 5)
                state.record([3] * topology.local_batch_size, 0.1)
                state.record([3] * topology.local_batch_size, 0.1)
                report = state.summary(rank + 1.0)
                report.update(rank=rank, graph_execution_counts={"target": 2, "draft": 2, "draft_extend": 2})
                reports.append(report)
            report = aggregate(reports, topology)
            self.assertEqual(report["batch_size"], 20)
            self.assertEqual(report["useful_output_tokens"], 100)
            self.assertEqual(report["raw_accept_tokens"], 120)
            self.assertEqual(report["num_correct_drafts"], 80)
            self.assertEqual(report["accept_histogram"], {3: 40})
            self.assertEqual(report["elapsed_seconds"], 4.0)
            self.assertEqual(report["output_tokens_per_second"], 25.0)
            self.assertEqual(report["output_tokens_per_second_per_gpu"], 6.25)
            self.assertEqual(report["effective_token_latency_ms_per_user"], 800.0)
            self.assertEqual(report["final_seq_lens"], [75] * 20)
            self.assertEqual((report["input_len"], report["input_len_uniform"]), (70, True))
            self.assertEqual(report["input_lens"], [70] * 20)  # global in both DPA and non-DPA
            self.assertEqual(report["graph_execution_counts_scope"], "rank_0")
            reports[-1]["verify_iterations"] += 1
            with self.assertRaises(ValueError):
                aggregate(reports, topology)


class TopologyCliTests(unittest.TestCase):
    def args(self, flags=()):
        import profile_decode
        args, extra = profile_decode.make_parser().parse_known_args([
            "--model-path", "/model", "--result-dir", "/result", *flags,
        ])
        return profile_decode, args, extra

    def test_cli_defaults_preserve_tp8_ep1_dp1(self):
        driver, args, extra = self.args()
        self.assertTrue(hasattr(args, "ep_size"), "missing EP CLI")
        self.assertEqual((args.tp_size, args.ep_size, args.enable_dp_attention), (8, 1, False))
        cli = driver.server_cli(args, extra)
        self.assertEqual(cli[cli.index("--dp-size") + 1], "1")
        self.assertEqual(cli[cli.index("--moe-a2a-backend") + 1], "none")

    def test_cli_dpa_graph_batch_is_local(self):
        driver, args, extra = self.args(["--tp-size", "4", "--ep-size", "4", "--batch-size", "20", "--enable-dp-attention"])
        cli = driver.server_cli(args, extra)
        self.assertEqual(cli[cli.index("--cuda-graph-bs-decode") + 1], "5")
        self.assertEqual(cli[cli.index("--cuda-graph-max-bs-decode") + 1], "5")
        self.assertEqual(cli[cli.index("--dp-size") + 1], "4")
        self.assertIn("--enable-dp-attention", cli)

    def test_cli_rejects_conflicting_forwarded_topology(self):
        driver, args, _ = self.args(["--tp-size", "4", "--ep-size", "4"])
        for extra in (["--dp-size", "4"], ["--moe-a2a-backend", "deepep"], ["--cuda-graph-bs-decode", "2"], ["--enable-dp-attention"]):
            with self.assertRaises(ValueError):
                driver.server_cli(args, extra)


if __name__ == "__main__":
    unittest.main()
