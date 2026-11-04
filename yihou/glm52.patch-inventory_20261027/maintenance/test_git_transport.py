"""Exercise Git's URL and credential selection without network or file writes."""

import os
import subprocess
import unittest
from pathlib import Path

from git_transport import github_git_command


class GitTransportTests(unittest.TestCase):
    def git(self, *args):
        return subprocess.run(
            github_git_command(["git", *args]),
            cwd=Path(__file__).parent,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def test_both_github_ssh_urls_resolve_to_https(self):
        for url in (
            "git@github.com:AMD-AGI/Infera.git",
            "ssh://git@github.com/AMD-AGI/Infera.git",
        ):
            with self.subTest(url=url):
                self.assertEqual(
                    self.git(
                        "-c",
                        f"remote.transport-test.url={url}",
                        "ls-remote",
                        "--get-url",
                        "transport-test",
                    ),
                    "https://github.com/AMD-AGI/Infera.git",
                )

    def test_other_hosts_keep_their_transport(self):
        url = "git@example.invalid:team/repo.git"
        self.assertEqual(
            self.git(
                "-c", f"remote.transport-test.url={url}", "ls-remote", "--get-url", "transport-test"
            ),
            url,
        )

    def test_gh_helper_is_selected_only_for_github(self):
        for host, expected in (
            ("github.com", "!gh auth git-credential"),
            ("example.invalid", "other-host-helper"),
        ):
            with self.subTest(host=host):
                self.assertEqual(
                    self.git(
                        "-c",
                        "credential.helper=other-host-helper",
                        "config",
                        "--get-urlmatch",
                        "credential.helper",
                        f"https://{host}/team/repo.git",
                    ),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
