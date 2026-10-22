"""CPU contracts for heterogeneous-ISL specs; no GPU, no torch, no sglang."""
import pathlib
import random
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

from isl_spec import IslSpec, MIN_ISL, parse_isl_spec


class ImportCostTests(unittest.TestCase):
    def test_importing_the_module_pulls_in_no_heavy_dependency(self):
        # profile_decode.py imports this before spawning TP ranks; it has to stay cheap.
        program = ("import sys; sys.path.insert(0, %r); import isl_spec; "
                   "print([name for name in ('torch', 'numpy', 'sglang') if name in sys.modules])"
                   % str(ROOT / "bench"))
        result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "[]")


class ParseTests(unittest.TestCase):
    def test_bare_integer_is_uniform_back_compat(self):
        spec = parse_isl_spec("70000")
        self.assertEqual(spec.mode, "uniform")
        self.assertEqual(spec.raw, "70000")
        self.assertEqual(spec.generate(4, 0), [70000] * 4)
        self.assertEqual(spec.generate(4, 0), parse_isl_spec("uniform:70000").generate(4, 0))

    def test_uniform_describe_round_trips_the_raw_string(self):
        self.assertEqual(parse_isl_spec("uniform:8192").describe(),
                         {"mode": "uniform", "raw": "uniform:8192", "value": 8192})

    def test_list_inline_and_file_agree(self):
        inline = parse_isl_spec("list:8192,70000,8192")
        self.assertEqual(inline.generate(3, 0), [8192, 70000, 8192])
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
            handle.write("# a comment\n\n8192\n  70000  # trailing comment\n\n8192\n")
            path = handle.name
        from_file = parse_isl_spec(f"list:@{path}")
        self.assertEqual(from_file.generate(3, 0), [8192, 70000, 8192])
        self.assertEqual(from_file.describe()["values"], [8192, 70000, 8192])

    def test_normal_optional_clamp_fields(self):
        spec = parse_isl_spec("normal:1000,50")
        self.assertEqual((spec.mean, spec.std, spec.lo, spec.hi), (1000.0, 50.0, MIN_ISL, None))
        clamped = parse_isl_spec("normal:1000,50,16,2048")
        self.assertEqual((clamped.lo, clamped.hi), (16, 2048))

    def test_malformed_specs_are_rejected(self):
        for spec in ("", "   ", "seventy", "uniform:", "uniform:abc", "uniform:1", "uniform:-3",
                     "uniform:3.5", "gaussian:1,2", "bimodal:8192,70000", "bimodal:8192,70000,0.1,0.2",
                     "bimodal:8192,70000,0", "bimodal:8192,70000,1", "bimodal:8192,70000,1.5",
                     "bimodal:8192,70000,-0.1", "bimodal:8192,70000,abc", "bimodal:1,70000,0.1",
                     "normal:1000", "normal:1000,50,16", "normal:1000,-1", "normal:abc,50",
                     "normal:1000,50,16,8", "normal:1000,50,1,2048", "normal:nan,50", "normal:inf,50",
                     "list:", "list:8192,,70000", "list:8192,abc", "list:1,70000", "list:@",
                     "list:@/nonexistent/path/that/should/not/exist_yihou.txt"):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                parse_isl_spec(spec)
        with self.assertRaises(ValueError):
            parse_isl_spec(70000)

    def test_list_file_with_no_values_is_rejected(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
            handle.write("# only comments\n\n\n")
            path = handle.name
        with self.assertRaises(ValueError):
            parse_isl_spec(f"list:@{path}")


class BimodalCeilTests(unittest.TestCase):
    def split(self, spec, count):
        generated = parse_isl_spec(spec).generate(count, 0)
        return generated.count(generated[0]), count - generated.count(generated[0])

    def test_minority_side_rounds_up(self):
        # 32 requests, 10 % short -> ceil(3.2) = 4 short, 28 long. Hand-computed, per the user's rule.
        self.assertEqual(parse_isl_spec("bimodal:8192,70000,0.1").generate(32, 0),
                         [8192] * 4 + [70000] * 28)

    def test_ratio_above_half_mirrors_onto_side_b(self):
        # ratio 0.9 -> side b is the minority -> n_b = ceil(32 * 0.1) = 4, n_a = 28.
        self.assertEqual(parse_isl_spec("bimodal:8192,70000,0.9").generate(32, 0),
                         [8192] * 28 + [70000] * 4)

    def test_exact_half_breaks_toward_a(self):
        self.assertEqual(self.split("bimodal:8192,70000,0.5", 32), (16, 16))
        # Odd count: ceil(33 / 2) = 17 on side a.
        self.assertEqual(parse_isl_spec("bimodal:8192,70000,0.5").generate(33, 0),
                         [8192] * 17 + [70000] * 16)

    def test_ratio_is_exact_not_binary_float(self):
        # 10 % of 70 is 7. In IEEE 70 * 0.1 == 7.000000000000001, which would ceil to 8.
        self.assertEqual(parse_isl_spec("bimodal:8192,70000,0.1").generate(70, 0).count(8192), 7)
        self.assertEqual(parse_isl_spec("bimodal:8192,70000,1/10").generate(70, 0).count(8192), 7)

    def test_degenerate_split_is_refused_with_the_numbers(self):
        # Because the minority side is ceil'd it is >= 1 for every count >= 2, so the only way to
        # land on a one-sided "bimodal" batch is a single request.
        with self.assertRaises(ValueError) as caught:
            parse_isl_spec("bimodal:8192,70000,0.01").generate(1, 0)
        message = str(caught.exception)
        self.assertIn("count=1", message)
        self.assertIn("1 request(s) of 8192", message)
        self.assertIn("0 of 70000", message)
        with self.assertRaises(ValueError):
            parse_isl_spec("bimodal:8192,70000,0.5").generate(1, 0)
        self.assertEqual(parse_isl_spec("bimodal:8192,70000,0.01").generate(2, 0), [8192, 70000])

    def test_bimodal_consumes_no_randomness_and_is_ordered(self):
        spec = parse_isl_spec("bimodal:8192,70000,0.25")
        self.assertEqual(spec.generate(8, 0), spec.generate(8, 999))
        self.assertEqual(spec.generate(8, 0), [8192] * 2 + [70000] * 6)


class NormalTests(unittest.TestCase):
    def test_same_seed_same_list_across_calls_and_across_objects(self):
        first = parse_isl_spec("normal:4096,512")
        second = parse_isl_spec("normal:4096,512")
        self.assertEqual(first.generate(64, 7), first.generate(64, 7))
        self.assertEqual(first.generate(64, 7), second.generate(64, 7))
        self.assertNotEqual(first.generate(64, 7), first.generate(64, 8))

    def test_values_are_plain_ints_in_draw_order(self):
        drawn = parse_isl_spec("normal:4096,512").generate(64, 7)
        self.assertEqual(len(drawn), 64)
        self.assertTrue(all(type(value) is int for value in drawn))
        self.assertNotEqual(drawn, sorted(drawn), "normal must emit in draw order, not sorted")

    def test_zero_stddev_degenerates_to_uniform(self):
        self.assertEqual(parse_isl_spec("normal:4096,0").generate(16, 3), [4096] * 16)

    def test_clamp_saturates_and_counts_are_reported(self):
        # std = 0 pins every draw on the mean, so the clamp counts are exact by hand.
        high = parse_isl_spec("normal:100000,0,16,2048")
        self.assertEqual(high.generate(10, 1), [2048] * 10)
        self.assertEqual(high.describe()["clamped_high"], 10)
        self.assertEqual(high.describe()["clamped_low"], 0)
        low = parse_isl_spec("normal:3,0,16,2048")
        self.assertEqual(low.generate(10, 1), [16] * 10)
        self.assertEqual((low.describe()["clamped_low"], low.describe()["clamped_high"]), (10, 0))

    def test_clamp_bites_both_tails_of_a_wide_draw(self):
        spec = parse_isl_spec("normal:1000,5000,16,2048")
        drawn = spec.generate(256, 11)
        self.assertTrue(all(16 <= value <= 2048 for value in drawn))
        described = spec.describe()
        self.assertGreater(described["clamped_low"], 0)
        self.assertGreater(described["clamped_high"], 0)
        self.assertEqual(described["mean"], 1000.0)
        self.assertEqual(described["std"], 5000.0)
        self.assertEqual(spec.generate(256, 11), drawn)

    def test_no_value_falls_below_the_bootstrap_minimum(self):
        self.assertTrue(all(value >= MIN_ISL for value in parse_isl_spec("normal:3,2").generate(512, 5)))


class GlobalRngIsolationTests(unittest.TestCase):
    def test_generate_leaves_the_global_stream_untouched(self):
        # The benchmark's acceptance coins come off the global stream; shifting it by even one
        # draw would change realized_accept_length, which is the harness's correctness gate.
        random.seed(1234)
        before_state = random.getstate()
        before_sequence = [random.random() for _ in range(5)]
        random.seed(1234)
        for spec in ("70000", "bimodal:8192,70000,0.1", "normal:4096,512", "list:8192,70000"):
            parse_isl_spec(spec).generate(2 if spec.startswith("list") else 32, 42)
        self.assertEqual(random.getstate(), before_state)
        self.assertEqual([random.random() for _ in range(5)], before_sequence)


class ListCountTests(unittest.TestCase):
    def test_length_mismatch_names_both_numbers_and_the_arithmetic(self):
        with self.assertRaises(ValueError) as caught:
            parse_isl_spec("list:8192,8192,70000,70000").generate(8, 0)
        message = str(caught.exception)
        self.assertIn("4 values", message)
        self.assertIn("needs 8", message)
        self.assertIn("batch_size // dp_size", message)

    def test_values_are_used_verbatim_and_seed_is_irrelevant(self):
        spec = parse_isl_spec("list:70000,8192,70000,2")
        self.assertEqual(spec.generate(4, 0), [70000, 8192, 70000, 2])
        self.assertEqual(spec.generate(4, 7), [70000, 8192, 70000, 2])


class GenerateContractTests(unittest.TestCase):
    def test_count_must_be_a_positive_int(self):
        spec = parse_isl_spec("70000")
        for count in (0, -1, 3.0, "4", True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                spec.generate(count, 0)

    def test_every_mode_returns_count_plain_ints(self):
        for spec, count in (("70000", 16), ("bimodal:8192,70000,0.1", 32),
                            ("normal:4096,512", 16), ("list:2,3,4", 3)):
            with self.subTest(spec=spec):
                drawn = parse_isl_spec(spec).generate(count, 0)
                self.assertEqual(len(drawn), count)
                self.assertTrue(all(type(value) is int and value >= MIN_ISL for value in drawn))

    def test_direct_construction_validates(self):
        with self.assertRaises(ValueError):
            IslSpec(mode="wat", raw="wat")
        with self.assertRaises(ValueError):
            IslSpec(mode="uniform", raw="uniform:1", value=1)
        with self.assertRaises(ValueError):
            IslSpec(mode="list", raw="list:", values=())


if __name__ == "__main__":
    unittest.main()
