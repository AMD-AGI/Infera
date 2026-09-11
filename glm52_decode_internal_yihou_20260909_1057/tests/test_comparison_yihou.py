import importlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))


class ComparisonTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module('compare_server')
        except ImportError:
            self.fail('compare_server module must exist')

    def test_wave_sizes(self):
        m = self.module()
        self.assertEqual(m.wave_sizes(128, 16), [16] * 8)
        self.assertEqual(m.wave_sizes(128, 32), [32] * 4)
        self.assertEqual(m.wave_sizes(16, 32), [16])
        with self.assertRaises(ValueError):
            m.wave_sizes(127, 16)

    def test_fake_first_output_budget(self):
        m = self.module()
        self.assertEqual(m.decode_budget(500, 'fake-server'), 499)
        self.assertEqual(m.decode_budget(500, 'sikl'), 500)
        with self.assertRaises(ValueError):
            m.decode_budget(1, 'fake-server')

    def test_aggregate_does_not_count_handoff_as_compute(self):
        m = self.module()
        waves = [{'batch_size': 16, 'output_tokens': 8000, 'decode_tokens': 7984,
                  'seconds': 4.0, 'verify_iterations': 140, 'raw_accept_tokens': 8000} for _ in range(8)]
        r = m.summarize_waves(waves)
        self.assertEqual(r['num_requests'], 128)
        self.assertEqual(r['output_tokens'], 64000)
        self.assertEqual(r['decode_tokens'], 63872)
        self.assertEqual(r['output_tokens_per_second'], 2000)
        self.assertAlmostEqual(r['decode_tpot_ms'], 32 * 1000 * 128 / 63872)


if __name__ == '__main__':
    unittest.main()
