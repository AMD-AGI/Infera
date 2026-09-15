from __future__ import annotations

import importlib.util
import io
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "bench" / "glm5p2_pd" / "projection_sweep.py"
SPEC = importlib.util.spec_from_file_location("glm52_pd_projection_sweep", SCRIPT)
assert SPEC and SPEC.loader
sweep = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sweep
SPEC.loader.exec_module(sweep)


class ProjectionSweepTests(unittest.TestCase):
    def test_pool_strategies_only_use_off_or_full_ep_and_dpa(self) -> None:
        strategies = sweep.pool_strategies(8)

        self.assertEqual(
            [(item.mode, item.tp, item.ep, item.attention_dp) for item in strategies],
            [
                ("tp", 8, 1, 1),
                ("tp_ep", 8, 8, 1),
                ("tp_ep_dpa", 8, 8, 8),
                ("tp_dpa", 8, 1, 8),
            ],
        )
        self.assertEqual({item.gpus for item in strategies}, {8})

    def test_default_matrix_uses_exact_8_12_16_gpu_splits(self) -> None:
        specs = sweep.enumerate_specs(
            budgets=(8, 12, 16),
            pool_widths=(4, 8),
            concurrencies=(1, 2, 4, 8, 16, 32, 64),
            draft_cost_factors=(0.05,),
            input_len=111787,
            output_len=911,
        )

        self.assertEqual(len(specs), 448)
        allocations = {
            (item.total_gpus, item.prefill.gpus, item.decode.gpus) for item in specs
        }
        self.assertEqual(
            allocations,
            {
                (8, 4, 4),
                (12, 4, 8),
                (12, 8, 4),
                (16, 8, 8),
            },
        )
        self.assertTrue(
            all(item.prefill.gpus + item.decode.gpus == item.total_gpus for item in specs)
        )

    def test_rejects_pool_width_outside_deployable_scan_tp(self) -> None:
        parser = sweep.build_argument_parser()
        args = parser.parse_args(["--pool-widths", "2,4"])

        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            sweep.validate_args(args, parser)

    def test_projection_argv_sets_each_pool_independently(self) -> None:
        args = sweep.build_argument_parser().parse_args([])
        point = sweep.SweepSpec(
            total_gpus=12,
            prefill=sweep.PoolStrategy(4, 4, 4, "tp_ep_dpa"),
            decode=sweep.PoolStrategy(8, 1, 8, "tp_dpa"),
            concurrency=16,
            draft_cost_factor=0.02,
            input_len=12345,
            output_len=678,
        )

        argv = sweep.build_projection_argv(point, args)

        self.assertEqual(argv[argv.index("--prefill-tp") + 1], "4")
        self.assertEqual(argv[argv.index("--prefill-ep") + 1], "4")
        self.assertEqual(argv[argv.index("--prefill-attention-dp") + 1], "4")
        self.assertEqual(argv[argv.index("--decode-tp") + 1], "8")
        self.assertEqual(argv[argv.index("--decode-ep") + 1], "1")
        self.assertEqual(argv[argv.index("--decode-attention-dp") + 1], "8")
        self.assertEqual(argv[argv.index("--input-len") + 1], "12345")
        self.assertEqual(argv[argv.index("--output-len") + 1], "678")
        self.assertEqual(argv[argv.index("--kv-transfer-bw-gbps") + 1], "37.85")
        self.assertNotIn("--serving-model", argv)
        self.assertIn("tensor_model_parallel_size=4", argv)
        self.assertIn("expert_model_parallel_size=4", argv)

    def test_replica_plan_builds_26_point_tp8_dpa_schedule(self) -> None:
        plan = sweep.replica_plan(
            "1x2:32,64,96,128,168;"
            "2x1:32,64,96,128,168;"
            "1x3:32,64,96,128,168;"
            "2x2:64,96,128,192,256,336;"
            "3x1:32,64,96,128,168"
        )
        specs = sweep.enumerate_specs(
            budgets=(24, 32),
            pool_widths=(8,),
            concurrencies=(),
            draft_cost_factors=(0.05,),
            input_len=218922,
            output_len=1077,
            modes=("tp_dpa",),
            replicas=plan,
        )

        self.assertEqual(len(specs), 26)
        self.assertTrue(
            all(
                item.prefill_pool_gpus + item.decode_pool_gpus
                == item.total_gpus
                for item in specs
            )
        )
        self.assertEqual(
            {
                (item.total_gpus, item.prefill_replicas, item.decode_replicas)
                for item in specs
            },
            {(24, 1, 2), (24, 2, 1), (32, 1, 3), (32, 2, 2), (32, 3, 1)},
        )
        point = next(
            item
            for item in specs
            if item.prefill_replicas == 2
            and item.decode_replicas == 2
            and item.concurrency == 336
        )
        argv = sweep.build_projection_argv(
            point,
            sweep.build_argument_parser().parse_args([]),
        )
        self.assertEqual(argv[argv.index("--prefill-replicas") + 1], "2")
        self.assertEqual(argv[argv.index("--decode-replicas") + 1], "2")
        self.assertIn("_np2_nd2_", point.point_id)
        self.assertEqual(sweep._per_replica_concurrency(336, 2), 168)
        self.assertEqual(sweep._per_replica_concurrency(169, 2), 85)

    def test_human_report_keeps_cli_output_and_pool_memory_limits(self) -> None:
        args = sweep.build_argument_parser().parse_args([])
        point = sweep.SweepSpec(
            total_gpus=8,
            prefill=sweep.PoolStrategy(4, 4, 1, "tp_ep"),
            decode=sweep.PoolStrategy(4, 4, 1, "tp_ep"),
            concurrency=64,
            draft_cost_factor=0.05,
            input_len=111787,
            output_len=911,
        )
        record = sweep.schedule_record(point, args)
        record.update(
            {
                "attempt": 1,
                "status": "infeasible",
                "started_at": "start",
                "finished_at": "finish",
                "elapsed_seconds": 1.25,
            }
        )
        record["projection"]["prefill_memory"] = {
            "fits": False,
            "total_bytes": 300 * 1024**3,
            "max_concurrent_sequences": 31,
        }
        record["projection"]["decode_memory"] = {
            "fits": True,
            "total_bytes": 200 * 1024**3,
            "max_concurrent_sequences": 47,
        }

        report = sweep.render_human_report(record, "CLI PERFORMANCE REPORT\n")

        self.assertIn("CLI PERFORMANCE REPORT", report)
        self.assertIn("Prefill: fits=False", report)
        self.assertIn("max concurrency=31", report)
        self.assertIn("Decode: fits=True", report)
        self.assertIn("max concurrency=47", report)
        self.assertIn("inferasim inference", report)

    def test_run_settings_are_json_stable_for_resume(self) -> None:
        args = sweep.build_argument_parser().parse_args([])
        settings = sweep.run_settings(args)

        self.assertEqual(settings["budgets"], [8, 12, 16])
        self.assertEqual(settings["pool_widths"], [4, 8])
        self.assertEqual(settings["draft_cost_factors"], [0.05])
        self.assertEqual(settings["kv_transfer_bw_gbps"], 37.85)
        self.assertIsInstance(settings["config"], str)


if __name__ == "__main__":
    unittest.main()
