"""CPU contracts of topology.py and of profile_decode's CLI resolution (no torch, no vLLM)."""
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import profile_decode  # noqa: E402
from topology import (DecodeTopology, aggregate_rank_summaries, compare_resolved_parallel,  # noqa: E402
                      lens_digest, rank_progress_signature, validate_rank_progress)


class TopologyTests(unittest.TestCase):
    def test_llying_decode_leg(self):
        topo = DecodeTopology(1, 2, 1, 448)
        self.assertEqual((topo.world_size, topo.local_batch_size), (2, 224))
        self.assertEqual(topo.representative_ranks, [0, 1])
        self.assertEqual(topo.rank_state(1)["dp_rank"], 1)
        self.assertEqual(topo.rank_state(1)["moe_tp_size"], 2)
        self.assertEqual(topo.request_id_offset(1), 224)
        self.assertEqual(topo.engine_overrides()["distributed_executor_backend"], "external_launcher")

    def test_tp_inside_dp(self):
        topo = DecodeTopology(2, 2, 4, 64)
        self.assertEqual(topo.representative_ranks, [0, 2])
        self.assertEqual(topo.rank_state(3), dict(topo.rank_state(3), dp_rank=1, tp_rank=1))
        self.assertTrue(topo.enable_expert_parallel)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            DecodeTopology(1, 2, 1, 7)
        with self.assertRaises(ValueError):
            DecodeTopology(1, 4, 2, 8)  # EP must span the whole world
        with self.assertRaises(ValueError):
            DecodeTopology(1, 2, 1, 4).rank_state(2)

    def test_resolved_comparison(self):
        topo = DecodeTopology(1, 2, 1, 8)
        compare_resolved_parallel(topo, 1, dict(dp_rank=1, dp_size=2, tp_rank=0, tp_size=1, pp_size=1,
                                                enable_expert_parallel=False))
        with self.assertRaises(ValueError):
            compare_resolved_parallel(topo, 1, dict(dp_rank=0, dp_size=2))


class SignatureTests(unittest.TestCase):
    def test_per_request_accept_vector_digested(self):
        a = rank_progress_signature(0, [10, 20], [3, 4], [13, 24], [100, 100])
        b = rank_progress_signature(0, [10, 20], [4, 3], [14, 23], [100, 100])
        self.assertEqual(a[0], 1)
        self.assertNotEqual(a[4], b[4])
        validate_rank_progress([a, list(a)])
        with self.assertRaises(ValueError):
            validate_rank_progress([a, b])

    def test_invalid_state_is_encoded_not_raised(self):
        sig = rank_progress_signature(0, [10], [3], [12], [100])
        self.assertEqual(sig[0], 0)
        self.assertEqual(rank_progress_signature(0, [10], [7], [17], [100])[0], 0)
        with self.assertRaises(ValueError):
            validate_rank_progress([sig])

    def test_digest_is_order_sensitive(self):
        self.assertNotEqual(lens_digest([1, 2]), lens_digest([2, 1]))


def report(rank, useful=20):
    return dict(rank=rank, batch_size=2, complete=True, verify_iterations=3, input_lens=[10, 10],
                input_len_uniform=True, output_len=10, emitted_per_request=[10, 10], final_seq_lens=[21, 22],
                raw_accept_tokens=23, accept_histogram={"3": 1, "4": 5}, useful_output_tokens=useful,
                num_correct_drafts=17, terminal_extra_tokens=3, elapsed_seconds=0.5 + rank,
                graph_execution_counts={"target": 3, "draft": 3}, step_seconds_p50=0.1)


class AggregateTests(unittest.TestCase):
    def test_dp_replicas_are_summed_elapsed_is_max(self):
        topo = DecodeTopology(1, 2, 1, 4)
        result = aggregate_rank_summaries([report(1), report(0)], topo)
        self.assertEqual(result["useful_output_tokens"], 40)
        self.assertEqual(result["elapsed_seconds"], 1.5)
        self.assertEqual(result["batch_size"], 4)
        self.assertEqual(result["realized_accept_length"], 46 / 12)
        self.assertEqual(len(result["input_lens"]), 4)
        self.assertEqual(result["output_tokens_per_second_per_gpu"], 40 / 1.5 / 2)

    def test_disagreement_raises(self):
        topo = DecodeTopology(1, 2, 1, 4)
        bad = report(1)
        bad["raw_accept_tokens"] = 24
        with self.assertRaises(ValueError):
            aggregate_rank_summaries([report(0), bad], topo)


class CliTests(unittest.TestCase):
    def parse(self, *argv):
        args, extra = profile_decode.make_parser().parse_known_args(
            ["--model-path", "/m", "--result-dir", "/tmp/x", *argv])
        profile_decode.validate_args(args)
        profile_decode.resolve_input_lens(args)
        return args, extra

    def test_engine_cli_owns_topology_and_spec(self):
        args, extra = self.parse("--batch-size", "8", "--input-len", "100", "--moe-backend", "aiter")
        cli = profile_decode.engine_cli(args, extra)
        self.assertIn("external_launcher", cli)
        self.assertIn("--no-enable-prefix-caching", cli)
        spec = cli[cli.index("--speculative-config") + 1]
        self.assertIn('"rejection_sample_method":"synthetic"', spec)
        self.assertIn('"synthetic_acceptance_length":3.51', spec)
        for owned in ("--tensor-parallel-size", "--speculative-config", "-dp", "--enable-expert-parallel"):
            args, extra = self.parse("--batch-size", "8", "--input-len", "100", owned, "2")
            with self.assertRaises(ValueError):
                profile_decode.engine_cli(args, extra)

    def test_graph_sizes_local(self):
        args, extra = self.parse("--batch-size", "448", "--input-len", "100", "--graph-sizes", "local")
        cli = profile_decode.engine_cli(args, extra)
        self.assertEqual(cli[cli.index("--cudagraph-capture-sizes") + 1], str(224 * 6))

    def test_exclusive_isl_and_spec_is_per_dp_rank(self):
        with self.assertRaises(ValueError):
            self.parse("--input-len", "10", "--input-len-spec", "uniform:10")
        args, _ = self.parse("--batch-size", "8", "--input-len-spec", "bimodal:100,800,0.25")
        self.assertEqual(args.input_lens, [100, 800, 800, 800])

    def test_rates_method(self):
        args, _ = self.parse("--accept-method", "rates", "--accept-rates", "1,1,0.5,0,0")
        self.assertIn('"synthetic_acceptance_rates":[1.0,1.0,0.5,0.0,0.0]',
                      profile_decode.engine_cli(args, [])[profile_decode.engine_cli(args, []).index("--speculative-config") + 1])
        with self.assertRaises(ValueError):
            self.parse("--accept-method", "rates", "--accept-rates", "1,1")

    def test_profile_ranks_span_dp_world(self):
        self.parse("--profile", "--profile-ranks", "1")
        with self.assertRaises(ValueError):
            self.parse("--profile", "--profile-ranks", "2")


if __name__ == "__main__":
    unittest.main()
