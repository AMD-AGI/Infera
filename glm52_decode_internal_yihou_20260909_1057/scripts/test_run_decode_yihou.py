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
        result = self.run_script("--dry-run", "002_smoke_yihou", "--max-steps", "2")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("126175", result.stdout)
        self.assertIn("crsuse2-m2m-055", result.stdout)
        self.assertIn("profile_decode.py", result.stdout)
        self.assertIn("--max-steps 2", result.stdout)


if __name__ == "__main__":
    unittest.main()
