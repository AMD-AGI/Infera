"""Regression checks for the duration denominator and asynchronous attribution."""
import json
from pathlib import Path
import tempfile
import unittest

from analyze import analyze_trace, union_duration


class TraceAnalysisTest(unittest.TestCase):
    def test_overlap_and_asynchronous_graph_attribution(self):
        # GPU execution occurs after its launching CPU stage has ended. The ID,
        # not timestamp containment, establishes ownership of both graph nodes.
        events = [
            dict(ph="X", cat="user_annotation", name="step[TARGET_VERIFY bs=32]", ts=0, dur=10, pid=1, tid=1),
            dict(ph="X", cat="cuda_runtime", name="hipGraphLaunch", ts=2, dur=1, pid=1, tid=1,
                 args={"correlation": 7}),
            dict(ph="X", cat="kernel", name="gemm", ts=20, dur=10, pid=9, tid=1,
                 args={"correlation": 7, "device": 0}),
            dict(ph="X", cat="kernel", name="allreduce", ts=25, dur=10, pid=9, tid=2,
                 args={"correlation": 7, "device": 0}),
            dict(ph="X", cat="kernel", name="unknown", ts=40, dur=5, pid=9, tid=1, args={"device": 0}),
            dict(ph="X", cat="gpu_memcpy", name="Memcpy DtoH", ts=45, dur=100, pid=9, tid=1),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test-TP-0.trace.json"
            path.write_text(json.dumps({"traceEvents": events}))
            result, _ = analyze_trace(path)
        self.assertEqual(result["kernel_count"], 3)
        self.assertEqual(result["verify_batch_sizes"], {"32": 1})
        self.assertEqual(result["graph_launches_without_kernels"], 0)
        self.assertEqual(result["kernel_sum_us"], 25)
        self.assertEqual(result["devices"][0]["kernel_busy_us"], 20)
        self.assertEqual(result["devices"][0]["kernel_window_us"], 25)
        self.assertEqual({r["name"]: r["percent_kernel_time"] for r in result["stages"]},
                         {"verify": 80, "unattributed": 20})
        self.assertAlmostEqual(sum(r["percent_kernel_time"] for r in result["categories"]), 100)

    def test_empty_union(self):
        self.assertEqual(union_duration([]), 0)
        self.assertEqual(union_duration([(1, 5), (2, 3), (5, 8), (10, 12)]), 9)


if __name__ == "__main__":
    unittest.main()
