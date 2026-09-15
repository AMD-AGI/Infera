from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "bench" / "glm5p2_pd" / "analyze_projection_sweep.py"
SPEC = importlib.util.spec_from_file_location("glm52_pd_projection_analysis", SCRIPT)
assert SPEC and SPEC.loader
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


class ProjectionAnalysisTests(unittest.TestCase):
    def test_pareto_front_drops_dominated_and_infeasible_points(self) -> None:
        rows = [
            {
                "point_id": "balanced",
                "status": "ok",
                "interactivity_tok_s_user": 150.0,
                "total_throughput_tps_per_gpu": 5000.0,
            },
            {
                "point_id": "interactive",
                "status": "ok",
                "interactivity_tok_s_user": 170.0,
                "total_throughput_tps_per_gpu": 4500.0,
            },
            {
                "point_id": "dominated",
                "status": "ok",
                "interactivity_tok_s_user": 140.0,
                "total_throughput_tps_per_gpu": 4900.0,
            },
            {
                "point_id": "infeasible",
                "status": "infeasible",
                "interactivity_tok_s_user": 999.0,
                "total_throughput_tps_per_gpu": 9999.0,
            },
        ]

        self.assertEqual(
            {row["point_id"] for row in analysis.pareto_front(rows)},
            {"balanced", "interactive"},
        )

    def test_latest_attempt_overlays_schedule_for_intermediate_analysis(self) -> None:
        schedule = [
            {"point_id": "one", "schema_version": 2},
            {"point_id": "two", "schema_version": 2},
        ]
        attempts = [
            {"point_id": "one", "schema_version": 2, "status": "error"},
            {"point_id": "one", "schema_version": 2, "status": "ok"},
        ]

        latest = analysis.latest_projection_records(schedule, attempts)

        self.assertEqual(latest[0]["status"], "ok")
        self.assertNotIn("status", latest[1])

    def test_memory_plan_cuts_over_limit_and_adds_boundary_points(self) -> None:
        rows = [
            self._memory_row("tp", "tp", concurrency, 24, 24)
            for concurrency in (1, 2, 4, 8, 16, 32, 64)
        ]
        rows.extend(
            self._memory_row("tp_dpa", "tp_dpa", concurrency, 248, 248)
            for concurrency in (1, 2, 4, 8, 16, 32, 64)
        )

        combinations, actions = analysis.memory_concurrency_plan(rows)

        self.assertEqual(len(combinations), 2)
        limited = next(
            row for row in combinations if row["memory_max_concurrency"] == 24
        )
        self.assertEqual(limited["cut_concurrencies"], "32,64")
        self.assertEqual(limited["supplement_concurrencies"], "24")
        scalable = next(
            row for row in combinations if row["memory_max_concurrency"] == 248
        )
        self.assertEqual(scalable["cut_concurrencies"], "")
        self.assertEqual(
            scalable["supplement_concurrencies"],
            "96,128,192,248",
        )
        self.assertEqual(
            [(row["action"], row["concurrency"]) for row in actions],
            [
                ("CUT", 32),
                ("CUT", 64),
                ("SUPPLEMENT", 24),
                ("SUPPLEMENT", 96),
                ("SUPPLEMENT", 128),
                ("SUPPLEMENT", 192),
                ("SUPPLEMENT", 248),
            ],
        )

    @staticmethod
    def _memory_row(
        prefill_mode: str,
        decode_mode: str,
        concurrency: int,
        prefill_max: int,
        decode_max: int,
    ) -> dict[str, object]:
        mode_label = {
            "tp": "TP",
            "tp_dpa": "TP+DPA",
        }
        dpa = 4 if prefill_mode == "tp_dpa" else 1
        decode_dpa = 4 if decode_mode == "tp_dpa" else 1
        return {
            "point_id": f"{prefill_mode}-{decode_mode}-c{concurrency}",
            "total_gpus": 8,
            "prefill_gpus": 4,
            "prefill_gpus_per_replica": 4,
            "prefill_replicas": 1,
            "decode_gpus": 4,
            "decode_gpus_per_replica": 4,
            "decode_replicas": 1,
            "prefill_mode": mode_label[prefill_mode],
            "prefill_tp": 4,
            "prefill_ep": 1,
            "prefill_attention_dp": dpa,
            "decode_mode": mode_label[decode_mode],
            "decode_tp": 4,
            "decode_ep": 1,
            "decode_attention_dp": decode_dpa,
            "input_len": 111787,
            "output_len": 911,
            "prefix_cache_hit_rate": 0.97369,
            "draft_cost_factor": 0.05,
            "concurrency": concurrency,
            "prefill_max_concurrent_sequences": prefill_max,
            "decode_max_concurrent_sequences": decode_max,
        }


if __name__ == "__main__":
    unittest.main()
