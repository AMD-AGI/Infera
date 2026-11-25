"""CPU contracts for the port's script helpers (gates, ARG CHECK, server summary). No GPU, no sglang."""
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def write_point(directory, **overrides):
    result = {
        "complete": True, "batch_size": 64, "output_len": 1024, "useful_output_tokens": 64 * 1024,
        "input_len": 8192, "input_len_uniform": True, "input_lens": [8192] * 64,
        "ep_size": 4, "tp_size": 4, "dp_size": 4, "enable_dp_attention": True, "moe_a2a_backend": "none",
        "local_batch_size": 16, "realized_accept_length": 3.5, "verify_iterations": 300,
        "target_graph_iterations": 300,
        "graph_execution_counts_by_rank": {str(r): {"target": 300, "draft": 300, "draft_extend": 0} for r in range(4)},
    }
    result.update(overrides)
    (directory / "result_yihou.json").write_text(json.dumps(result))
    (directory / "launch_status.json").write_text(json.dumps({"exit_code": 0}))


class VerifyPointTests(unittest.TestCase):
    def test_clean_point_passes_and_each_gate_trips(self):
        from verify_point import check
        with tempfile.TemporaryDirectory() as tmp:
            point = pathlib.Path(tmp)
            write_point(point)
            self.assertEqual(check(point, expect_accept=3.5)[0], "pass")
            for override in ({"complete": False}, {"useful_output_tokens": 1}, {"input_len": 4096},
                             {"dp_size": 1}, {"realized_accept_length": 3.51}, {"target_graph_iterations": 299},
                             {"graph_execution_counts_by_rank": {"0": {"target": 300, "draft": 12}}},
                             {"profiled": True}):
                write_point(point, **override)
                verdict, summary = check(point, expect_accept=3.5)
                self.assertEqual(verdict, "fail", override)
                self.assertTrue(summary["problems"])
            write_point(point, target_graph_iterations=0,
                        graph_execution_counts_by_rank={"0": {"target": 0, "draft": 0}})
            self.assertEqual(check(point, expect_accept=3.5, allow_eager=True)[0], "pass")

    def test_missing_point(self):
        from verify_point import check
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(check(pathlib.Path(tmp))[0], "missing")

    def test_collect_sweep_imports_the_current_module_name(self):
        # GLM-5.2 handoff trap #17: the consolidated copy imported verify_point_yihou.
        source = (ROOT / "scripts" / "collect_sweep.py").read_text()
        self.assertIn("from verify_point import", source)
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / "collect_sweep.py"), "--help"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class ArgCheckTests(unittest.TestCase):
    def test_flags_must_reach_the_config(self):
        from arg_check import missing_flags
        payload = {"server_cli": ["--mem-fraction-static", "0.85", "--enable-dp-lm-head"],
                   "benchmark": {"batch_size": 64, "warmup_steps": 10}}
        self.assertEqual(missing_flags(payload, ["--batch-size", "64", "--warmup-steps", "10", "--enable-dp-lm-head"]), [])
        self.assertEqual(missing_flags(payload, ["--enable-fused-qk-norm-rope"]), ["--enable-fused-qk-norm-rope"])


class ShellGuardTests(unittest.TestCase):
    def test_run_profile_accepts_max_steps_zero(self):
        # GLM-5.2 handoff trap #6: the shell guard rejected MAX_STEPS=0 (= full OSL).
        script = ROOT / "scripts" / "run_profile.sh"
        env = {"MAX_STEPS": "0", "PATH": "/usr/bin:/bin", "KIT_DIR": "/nonexistent-yihou", "WORK_DIR": "/nonexistent-yihou"}
        result = subprocess.run(["bash", str(script), "--dry-run", "p_test"], capture_output=True, text=True, env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--profile", result.stdout)
        env["MAX_STEPS"] = "10"
        result = subprocess.run(["bash", str(script), "--dry-run", "p_test"], capture_output=True, text=True, env=env)
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
