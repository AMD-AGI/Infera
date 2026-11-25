"""CPU contracts of the scripts: verify_point gates, collect_sweep import (GLM trap #17), ARG CHECK (trap #10),
run_profile MAX_STEPS guard (trap #6), compare_server wave arithmetic, profiling stage timer."""
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "bench"))

import arg_check  # noqa: E402
import verify_point  # noqa: E402


def good_result(iterations=300):
    return dict(complete=True, batch_size=448, output_len=1024, useful_output_tokens=448 * 1024,
                input_len=173000, input_len_uniform=True, input_lens=[173000] * 448, tp_size=1, dp_size=2,
                ep_size=1, enable_expert_parallel=False, local_batch_size=224, realized_accept_length=3.509,
                verify_iterations=iterations, target_graph_iterations=iterations,
                graph_execution_counts_by_rank={"0": {"target": iterations, "draft": iterations},
                                                "1": {"target": iterations, "draft": iterations}})


class VerifyPointTests(unittest.TestCase):
    def write(self, result, status=0):
        directory = pathlib.Path(tempfile.mkdtemp(prefix="yihou_v3_test_"))
        (directory / "result_yihou.json").write_text(json.dumps(result))
        (directory / "launch_status.json").write_text(json.dumps({"exit_code": status}))
        return directory

    def test_pass(self):
        verdict, summary = verify_point.check(self.write(good_result()))
        self.assertEqual(verdict, "pass", summary["problems"])

    def test_gates(self):
        cases = [("complete", False), ("useful_output_tokens", 1), ("realized_accept_length", 3.6),
                 ("target_graph_iterations", 1), ("ep_size", 2), ("profiled", True)]
        for key, value in cases:
            result = good_result()
            result[key] = value
            verdict, _ = verify_point.check(self.write(result))
            self.assertEqual(verdict, "fail", key)
        result = good_result()
        result["graph_execution_counts_by_rank"]["1"]["draft"] = 3
        self.assertEqual(verify_point.check(self.write(result))[0], "fail")
        self.assertEqual(verify_point.check(self.write(good_result(), status=1))[0], "fail")
        self.assertEqual(verify_point.check(self.write(good_result()), expect_accept=3.509)[0], "pass")
        self.assertEqual(verify_point.check(self.write(good_result()), expect_accept=3.51)[0], "fail")

    def test_heterogeneous(self):
        result = good_result()
        result.update(input_len=None, input_len_uniform=False)
        self.assertEqual(verify_point.check(self.write(result))[0], "fail")
        self.assertEqual(verify_point.check(self.write(result), expect_heterogeneous=True)[0], "pass")

    def test_collect_sweep_imports(self):
        out = subprocess.run([sys.executable, str(ROOT / "scripts" / "collect_sweep.py"), "--help"],
                             capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)


class ArgCheckTests(unittest.TestCase):
    def test_missing_flag_detected(self):
        payload = {"engine_cli": ["--moe-backend", "aiter"], "benchmark": {"batch_size": 8}}
        self.assertEqual(arg_check.missing_flags(payload, ["--moe-backend", "aiter", "--batch-size", "8"]), [])
        self.assertEqual(arg_check.missing_flags(payload, ["--block-size", "128"]), ["--block-size"])


class RunProfileGuardTests(unittest.TestCase):
    def run_profile(self, **env):
        import os
        full = dict(os.environ, **{k: str(v) for k, v in env.items()})
        return subprocess.run(["bash", str(ROOT / "scripts" / "run_profile.sh"), "--dry-run", "p_test"],
                              capture_output=True, text=True, env=full)

    def test_max_steps_zero_means_full_run(self):
        self.assertEqual(self.run_profile(MAX_STEPS=0).returncode, 0, self.run_profile(MAX_STEPS=0).stderr)
        self.assertNotEqual(self.run_profile(MAX_STEPS=10, PROFILE_START=20).returncode, 0)
        self.assertNotEqual(self.run_profile(CONC=7).returncode, 0)

    def test_dry_run_carries_profile_flags(self):
        out = self.run_profile(MAX_STEPS=40).stdout
        for flag in ("--profile", "--dp-size", "--gpu-memory-utilization", "--moe-backend"):
            self.assertIn(flag, out)


class CompareServerTests(unittest.TestCase):
    def test_waves(self):
        import compare_server
        self.assertEqual(compare_server.wave_sizes(8, 4), [4, 4])
        self.assertEqual(compare_server.wave_sizes(2, 4), [2])
        with self.assertRaises(ValueError):
            compare_server.wave_sizes(6, 4)
        self.assertEqual(compare_server.decode_budget(10, "server-like"), 9)
        self.assertEqual(compare_server.decode_budget(10, "synthetic"), 10)
        summary = compare_server.summarize_waves([dict(batch_size=2, output_tokens=20, decode_tokens=18, seconds=1.0,
                                                       verify_iterations=5, raw_accept_tokens=35)])
        self.assertAlmostEqual(summary["decode_tpot_ms"], 2000 / 18)
        self.assertEqual(summary["realized_accept_length"], 3.5)


class ProfilingTests(unittest.TestCase):
    def test_report_accumulates(self):
        from profiling_yihou import _DeviceTimerReport
        report = _DeviceTimerReport()
        fn = report.reporter_for("draft_propose")
        fn(0.5, category="draft_propose")
        fn(0.25, category="draft_propose")
        self.assertEqual(report.snapshot(), {"draft_propose": {"seconds": 0.75, "count": 2}})


class NoSecretsTests(unittest.TestCase):
    def test_no_token_shaped_strings(self):
        import re
        pattern = re.compile("(" + "|".join(["hf" + "_[A-Za-z0-9]{20,}", "ghp" + "_[A-Za-z0-9]{20,}",
                                              "github" + "_pat_[A-Za-z0-9]", "sk" + "-[A-Za-z0-9]{20,}",
                                              "AKIA" + "[0-9A-Z]{16}"]) + ")")
        for path in ROOT.rglob("*"):
            if path.is_file() and path.suffix in (".py", ".sh", ".md", ".args", ".env"):
                self.assertIsNone(pattern.search(path.read_text(errors="replace")), str(path))


if __name__ == "__main__":
    unittest.main()
