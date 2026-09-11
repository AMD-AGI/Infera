import unittest

from analyze import percentile, summarize


class GoodputTest(unittest.TestCase):
    def test_request_slo_is_not_aggregate_throughput(self):
        # Three successful requests at 100, 76.9 and 50 decode tokens/s.
        # Total system throughput is 150 tokens/s, but only two meet 70 and
        # only one meets 80. TTFT is excluded from decode TPOT.
        tpots = [10., 13., 20.]
        itls = [[value / 1000] * 499 for value in tpots]
        flat = [value * 1000 for row in itls for value in row]
        data = {"max_concurrency": 3, "completed": 3, "duration": 10.,
            "input_lens": [10000] * 3, "output_lens": [500] * 3,
            "ttfts": [.1] * 3, "latencies": [.1 + value / 1000 * 499 for value in tpots],
            "itls": itls, "raw_chunk_gaps": itls, "chunk_token_counts": [[1] * 499 for _ in tpots],
            "start_times": [0.] * 3, "successes": [True] * 3, "errors": [""] * 3,
            "output_throughput": 150., "request_throughput": .3, "concurrency": 3.}
        for key, q in (("median", .5), ("p90", .9), ("p99", .99)):
            data[key + "_tpot_ms"] = percentile(tpots, q)
            data[key + "_itl_ms"] = percentile(flat, q)
        meta = {"point": "test", "num_requests": 3, "concurrency": 3, "repeat": 1}
        row, requests = summarize(data, meta)
        self.assertEqual(row["slo_70_pass_requests"], 2)
        self.assertEqual(row["slo_80_pass_requests"], 1)
        self.assertEqual(row["slo_70_goodput_tokens_s"], 100)
        self.assertEqual(row["slo_80_goodput_tokens_s"], 50)
        self.assertAlmostEqual(requests[0]["decode_tokens_s"], 100)
        self.assertLess(requests[0]["e2e_tokens_s"], 100)
        self.assertFalse(row["slo_70_p90_pass"])


if __name__ == "__main__":
    unittest.main()
