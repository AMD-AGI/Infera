import os
import pathlib
import subprocess
import unittest

SCRIPT = pathlib.Path(__file__).with_name("run_decode.sh")


class LauncherTests(unittest.TestCase):
    def run_script(self, *args):
        return subprocess.run(["bash", str(SCRIPT), *args], text=True, capture_output=True)

    def test_script_exists(self):
        self.assertTrue(SCRIPT.is_file(), "run_decode.sh must exist")

    def test_rejects_traversal(self):
        result = self.run_script("--dry-run", "../escape")
        self.assertEqual(result.returncode, 2)
        self.assertIn("iteration", result.stderr)

    def test_rejects_missing_iteration(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 2)
        self.assertIn("Usage", result.stderr)

    def test_dry_run_is_local_and_explicit(self):
        env = dict(os.environ, JOB_ID="130737", NODE="crsuse2-m2m-056", CONTAINER="yihou-sweep-test")
        result = subprocess.run(["bash", str(SCRIPT), "--dry-run", "002_smoke_yihou", "--max-steps", "2"], env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("130737", result.stdout)
        self.assertIn("crsuse2-m2m-056", result.stdout)
        self.assertIn("yihou-sweep-test", result.stdout)
        self.assertIn("profile_decode.py", result.stdout)
        self.assertIn("--max-steps 2", result.stdout)

    def test_rejects_forbidden_node_before_remote_call(self):
        env = dict(os.environ, JOB_ID="1", NODE="crsuse2-m2m-234", CONTAINER="yihou-test")
        result = subprocess.run(["bash", str(SCRIPT), "--dry-run", "test_yihou"], env=env, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("forbidden", result.stderr.lower())


if __name__ == "__main__":
    unittest.main()
