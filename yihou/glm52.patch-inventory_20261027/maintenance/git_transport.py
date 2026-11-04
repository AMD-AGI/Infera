#!/usr/bin/env python3
"""Use the existing gh identity for task-local GitHub Git operations."""

import subprocess
import sys


def github_git_command(command):
    """Select HTTPS per invocation without changing remotes or saved config."""
    command = list(map(str, command))
    if not command or command[0] != "git":
        return command
    return [
        "git",
        "-c",
        "url.https://github.com/.insteadOf=git@github.com:",
        "-c",
        "url.https://github.com/.insteadOf=ssh://git@github.com/",
        "-c",
        "credential.https://github.com.helper=",
        "-c",
        "credential.https://github.com.helper=!gh auth git-credential",
        *command[1:],
    ]


if __name__ == "__main__":
    raise SystemExit(subprocess.run(github_git_command(["git", *sys.argv[1:]])).returncode)
