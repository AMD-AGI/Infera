"""The program executor — design §9.

Spec §1.1: *"A program executor implements [level 1] directly and never touches
level 2."* Taken literally, which is what the two-protocol split makes possible.

**It raises nothing, because it declares nothing it cannot do.** `interrupt`,
`instruct` and `query` are level 2, a program has no level 2, and there is
therefore no method to raise from — which is better than a raising stub for the
reason spec §3.3.1 gives about Cursor: a raise should mean *this adapter is
incomplete*, and a program is not an incomplete AI harness.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

from agent.backend import (
    AgentResult,
    AgentStatus,
    Assignment,
    BackendUnsupported,
    ExecutorBase,
)

__all__ = ["ProgramExecutor"]

#: How long `wait` sleeps between polls of the child. Polling rather than
#: `Popen.wait()` so `mainloop` stays responsive to a `stop` from another
#: thread without a second waiter on the same process.
POLL = 0.01

#: How much of the body's output travels back on a failed `AgentResult`.
#:
#: A traceback is the case this exists for and fits many times over. The bound
#: is what stops a runaway body becoming a runaway record — the tail is kept
#: rather than the head, because the exception is at the end.
TAIL_BYTES = 8 * 1024

#: How long `_run` waits for the reader to finish after the child exits. It has
#: only the pipe's remaining buffer left to read at that point, so this is a
#: guard against a thread that will not end rather than a real wait.
DRAIN_GRACE = 5.0


class ProgramExecutor(ExecutorBase):
    """Satisfies `Executor`. **Not an `AgentBackend`.**

    The command comes from the assignment's `entry` — the task body's
    `entry.sh`, which is what makes a programmatic task programmatic (design
    §7.2.1) — or, for a backend declaration that carries its own, from
    `config["command"]`.
    """

    def __init__(
        self,
        key: str = "program",
        config: Mapping[str, Any] | None = None,
        assignment: Assignment | None = None,
    ) -> None:
        super().__init__(key, assignment)
        self.config = dict(config or {})
        self._process: subprocess.Popen[bytes] | None = None
        self._tail = bytearray()
        self._reader: threading.Thread | None = None
        self._command = self._resolve_command()

    # ---- the probe is the constructor (design §6.4) ---------------------- #

    def _resolve_command(self) -> list[str]:
        declared = self.config.get("command")
        if isinstance(declared, str):
            return shlex.split(declared)
        if isinstance(declared, Sequence) and declared:
            return [str(part) for part in declared]
        entry = self.assignment.entry
        if entry:
            return ["/bin/sh", entry]
        raise BackendUnsupported(
            self.key,
            "run here",
            "no command: the task body declares no entry.sh and the backend "
            "declaration carries no `command`",
        )

    def accept_confinement(self, spawn: Any) -> None:
        """A program *is* a command line, so this executor can be started
        confined — which is what makes the base's refusal a real distinction
        rather than a blanket ban."""
        self._spawn = spawn

    # ---- the asynchronous form ------------------------------------------- #

    def _deploy(self) -> None:
        """`on_started` fires once the process exists, which is the program's
        equivalent of the SDK's `connect()` handshake returning (§8.3).

        **The argv is wrapped before it is spawned**, because on rung 1
        bubblewrap *is* the exec: `env_mgr.isolation.apply.apply()` confines
        nothing and returns a `Confinement` describing what the argv will
        achieve. Skipping the wrap on a bwrap machine means `prepare` succeeded,
        the task ran, and there was no sandbox — the silent-no-op failure.
        """
        start = self._spawn or subprocess.Popen
        self._process = start(  # noqa: S603 — the command is a task-package path
            self._command,
            cwd=self.assignment.zone or None,
            env=dict(os.environ, **self.assignment.environment) or None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self._read_continuously()

    def _read_continuously(self) -> None:
        """Drain the pipe on a thread, **and the thread is not for the output**.

        Measured, `probe_program_output_is_lost.py`: with nobody reading, a body
        that writes 256 KiB to stdout never returns from `start()` — it blocks
        in `write()` once Linux's 64 KiB pipe buffer fills, `_run` polls a child
        that cannot exit, and the loop has no bound. A quiet body was fine, so
        every test and every demo run passed while the hazard sat in the same
        two lines.

        So the reader exists to keep the child *running*; keeping its output is
        the part that comes for free. It is a thread rather than a read after
        exit for exactly that reason — a read after exit is too late by the
        length of the deadlock.
        """
        stream = self._process.stdout if self._process else None
        if stream is None:  # pragma: no cover — `_deploy` always passes PIPE
            return
        self._reader = threading.Thread(target=self._drain, args=(stream,), daemon=True)
        self._reader.start()

    def _drain(self, stream: Any) -> None:
        """Read to EOF, keeping only the tail. The sole writer of `_tail`."""
        try:
            for chunk in iter(lambda: stream.read(4096), b""):
                self._tail += chunk
                if len(self._tail) > TAIL_BYTES:
                    del self._tail[:-TAIL_BYTES]
        except (ValueError, OSError):  # the stream closed under us; the child is gone
            pass
        finally:
            stream.close()

    def _run(self) -> AgentResult:
        process = self._process
        if process is None:  # pragma: no cover — _deploy always precedes _run
            return AgentResult(status=AgentStatus.FAILED, detail="no process")
        started = time.monotonic()
        while process.poll() is None:
            if self._stopping:
                break
            time.sleep(POLL)
        if process.poll() is None:
            return AgentResult(status=AgentStatus.INTERRUPTED, detail="stopped before exit")
        code = process.returncode
        if self._reader is not None:
            self._reader.join(DRAIN_GRACE)
        return AgentResult(
            status=AgentStatus.FINISHED if code == 0 else AgentStatus.FAILED,
            usage={"seconds": time.monotonic() - started},
            detail=self._detail(code),
        )

    def _detail(self, code: int) -> str:
        """`exit <code>`, and on a failure what the body said before it stopped.

        **`exit 1` alone is a true statement that ends the investigation where
        it starts.** `demo` measured the cost: a body died on a `KeyError` and
        the run reported only `output_absent`, so a crash on line 1, an exit 0
        that wrote to the wrong path, and a body never launched all presented
        identically. The exit code separates the third; its output separates the
        other two.

        **Only on failure**, because a successful body's stdout is its own
        business and belongs wherever the task's logs go, not in a status field.
        Where that is remains open: `<zone>/logs/` exists and `env_mgr` creates
        it, but `logs` is `env_mgr`'s name (`fs/layout.py:47`) and `Zone` carries
        only `root`, so writing there from here would duplicate a seam name —
        the mistake `AGENT_SYS_CLAUDE_CLI` just cost a fortnight of green tests.
        Asked rather than assumed.
        """
        if code == 0 or not self._tail:
            return f"exit {code}"
        said = self._tail.decode("utf-8", "replace").strip()
        return f"exit {code}: {said}" if said else f"exit {code}"

    def _terminate(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
