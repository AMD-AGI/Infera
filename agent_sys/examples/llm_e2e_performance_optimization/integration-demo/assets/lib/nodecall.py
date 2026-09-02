#!/usr/bin/env python3
"""Call the compute node from a python body, through the same `on` every shell body uses.

`remote.sh` is the one definition of how this package reaches the node, and a
python re-implementation of `srun --overlap …` would be a second one to keep in
step. So this shells out to it: `bash -c '. remote.sh; on "$1"' _ <command>`,
which passes the command as a positional parameter and therefore needs no
quoting of its own.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REMOTE_SH = Path(__file__).resolve().parent / "remote.sh"


class NodeError(RuntimeError):
    """A command failed on the node. Carries what was run and what came back."""

    def __init__(self, command: str, returncode: int, output: str):
        self.command, self.returncode, self.output = command, returncode, output
        super().__init__(f"node command failed (rc={returncode}): {command}\n{output}")


def on(command: str, *, check: bool = True, text: bool = True) -> str:
    """Run `command` on the node and return its stdout.

    stderr is kept separate rather than folded in: several callers parse the
    output, and srun writes its own diagnostics to stderr.
    """
    proc = subprocess.run(
        ["bash", "-c", f'. "{REMOTE_SH}"; on "$1"', "_", command],
        capture_output=True,
        text=text,
    )
    if check and proc.returncode != 0:
        raise NodeError(command, proc.returncode, (proc.stdout or "") + (proc.stderr or ""))
    return proc.stdout


def visible_on_node(path: str | Path) -> bool:
    return subprocess.run(
        ["bash", "-c", f'. "{REMOTE_SH}"; on "$1"', "_", f"test -e '{path}'"],
        capture_output=True,
    ).returncode == 0
