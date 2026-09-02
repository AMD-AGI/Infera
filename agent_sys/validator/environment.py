"""A validation environment is a rebuild, never a reuse.

Spec §8.2, and there is **no cost argument against it**. Measured on this
machine: `mkdir` a fresh zone 0.03 ms; `unshare --user --mount` + bind +
remount-ro 13.8 ms; `python3 -c pass` 61–66 ms; `docker run … /bin/true` 786 ms.
A full private, read-only, bind-mounted namespace costs about 4.4× *less* than
starting the Python interpreter that runs inside it.

**Two failure modes this file is built to avoid**, both first-hand:

*Freshness comes from allocation, never from cleanup.* pytest's `tmp_path` is a
new numbered directory and its cleanup is explicitly best-effort
(`rmtree(..., ignore_errors=True)`). A guarantee that depends on a teardown
succeeding is not a guarantee. `tempfile.mkdtemp` is the stdlib's allocation
primitive and is what this uses.

*A staleness check with a hidden off-switch is worse than none.* nox's, in full:
`if not os.environ.get("NOX_ENABLE_STALENESS_CHECK", ""): return True` — disabled
for a Python 2.7 bug and never re-enabled. Nothing here has an off-switch.

**What this layer is, and is not.** `env_mgr` owns the kernel layer — namespaces,
mounts, an allow-list — and it is specified and unbuilt (a grep for
landlock/bwrap/unshare/seccomp hits only its spec). This is the process-
perspective layer, and Nix's framing is the one to copy: *"what matters for
determinism is what the build process can observe… we therefore specify building
from the process's perspective, not Nix's."* Stating criterion 21 as what the
validation can observe is what lets `env_mgr` change mechanisms later without
invalidating it.

Measured, and it is why criterion 21 cannot be a directory check: a fresh zone
directory closes **one** of the channels a producer leaves state in. `/tmp`,
`os.environ`, an inherited `cwd`, `$HOME` and same-path reuse all still carry it.
`CHANNELS` below is that list, and each entry names what closes it.

**The zone's *placement* is not this module's, and the `mkdtemp` below is a
stand-in rather than the answer.** `env_mgr.fs.layout.validation_zone` puts a
validation's materials as a **sibling of the producing task's zone, never a
descendant** — their design D5, and their criterion 13 is *untrue* without it,
because anything under the producing task's directory is inside its subtree and
therefore reachable. A `mkdtemp` zone in `/tmp` is outside the granted set
entirely, so today the separation holds **by accident of location rather than by
placement**. That is worth less than it looks and is why it is written down
here: an accident is not a property, and the next person to move the zone root
would not know they were removing one. Two modules currently answer *where does
a validation go*; `env_mgr` and this one have raised it to `main` naming both
sides rather than either inventing a call site.
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from validator.protocols import PhaseKind, ValidatorInvalid

__all__ = [
    "CHANNELS",
    "ConfigSource",
    "EnvironmentConfig",
    "ValidationEnvironment",
    "assert_attributable",
    "assert_standard_unreachable",
    "build_environment",
    "choose_configuration",
]

#: The channels a producer leaves state in, and what closes each here. A fresh
#: directory closes exactly one of them, which is the whole reason criterion 21's
#: test enumerates channels instead of checking a directory.
#:
#: **`PATH` is deliberately not in this list, and not in the block either.**
#: `demo` F-D5 read that as a script body starting with an empty `PATH`; measured
#: (`scratch/impl-2026-08/validator/p2_env_path.py`), it does not — POSIX `sh`
#: substitutes a built-in default when none is inherited, so a body gets
#: `/usr/local/sbin:…:/bin` and finds `python3`. The residue is real but is the
#: opposite of the claim: the value comes from the **shell**, not from the
#: configuration, so it is neither recorded here nor something a `config` can
#: reason about, and it will differ on another platform. Choosing one is a policy
#: about which binaries a validation may reach, which is `env_mgr`'s allow-list
#: (§8.3) and not this layer's to invent.
CHANNELS: tuple[tuple[str, str], ...] = (
    ("zone", "a freshly allocated, never-reused path"),
    ("tmp", "TMPDIR points inside the zone"),
    ("home", "HOME points inside the zone"),
    ("cwd", "set explicitly to the zone, never inherited"),
    ("environ", "an explicit block; os.environ is not inherited"),
)


class ConfigSource(str, Enum):
    """Spec §8.2's four-row chain, as a value on the result.

    Recorded rather than inferred, so `test_configuration_chain_order` asserts
    which row applied instead of asserting the contents that row happened to
    produce.
    """

    BOUND = "bound"
    CONSUMER = "consumer"
    PRODUCER = "producer"
    GLOBAL = "global"


@dataclass(frozen=True)
class EnvironmentConfig:
    """A *resolved* configuration. Reusing one of these is fine; inheriting an
    environment or a conversation is not — that is spec §8.2 in one line."""

    source: ConfigSource
    values: Mapping[str, str] = field(default_factory=dict)


def choose_configuration(
    kind: PhaseKind,
    *,
    bound: Mapping[str, str] | None = None,
    consumer: Mapping[str, str] | None = None,
    producer: Mapping[str, str] | None = None,
    global_: Mapping[str, str] | None = None,
) -> EnvironmentConfig:
    """Spec §8.2's chain: bound env, else the consumer's for input validation,
    else the producer's for output validation, else a predefined global one.

    The middle two are why the phases sit inside the task (spec §3): the right
    configuration is the one already **resolved**, which is not the same as the
    one already running.

    `kind` is coerced because this is reachable from outside the phase runner and
    the enum is a `(str, Enum)`: a caller holding the *value* would otherwise fall
    through both `is` comparisons to the global row without any complaint.
    """
    kind = PhaseKind(kind)
    if bound is not None:
        return EnvironmentConfig(ConfigSource.BOUND, dict(bound))
    if kind is PhaseKind.INPUT and consumer is not None:
        return EnvironmentConfig(ConfigSource.CONSUMER, dict(consumer))
    if kind is PhaseKind.OUTPUT and producer is not None:
        return EnvironmentConfig(ConfigSource.PRODUCER, dict(producer))
    return EnvironmentConfig(ConfigSource.GLOBAL, dict(global_ or {}))


@dataclass(frozen=True)
class ValidationEnvironment:
    """One validator's rebuilt environment, from the process's perspective."""

    zone: Path
    cwd: Path
    env: Mapping[str, str]
    config: EnvironmentConfig
    agent_id: str

    @property
    def args_file(self) -> Path:
        """`args.json`, written before the body runs.

        Args reach a body as a **file, not as parameters** — the one shape that
        works for a script and for an agent without either learning about the
        other. Parameters would have worked for a callable and not for an agent,
        which is the same reason the callable went away.
        """
        return self.zone / "args.json"

    @property
    def verdict_file(self) -> Path:
        """What the body writes, and what makes the two kinds of validator
        substitutable at the phase runner's seam."""
        return self.zone / "verdict.json"


def assert_attributable(agent_id: str | None) -> str:
    """Criterion 10 says *"no read originating in a producer frame"*, and the SDK
    has no frame. Its only identity field is `agent_id`, present only inside a
    Task-spawned sub-agent, so **a phase running on the main thread is
    unattributable and criterion 10 is not testable for it.**

    That makes attribution a requirement this module states, and the *mechanism*
    — subagent, `fork_session`, `resume`, or a second client — `agent` design
    O6's, still open. So this fails loudly rather than assuming one.
    """
    if not agent_id:
        raise ValidatorInvalid(
            "a validation phase must be separately attributable: no agent_id. "
            "The backend mechanism is agent design O6's and is open; the "
            "requirement is not."
        )
    return agent_id


def assert_standard_unreachable(zone: Path, standards: Sequence[Path]) -> None:
    """**Absence is a property to assert, not merely to arrange.**

    SWE-bench's answer key was physically absent from the container for two years
    and still leaked: `git remote remove origin` leaves the fix commit reachable
    through `git cat-file --batch-all-objects`, and issue #465 names real cheating
    trajectories. Their fix is the lesson — `git_clone_timesafe` now ends with a
    count that must be zero or the clone fails. So the zone is checked, not
    trusted.
    """
    root = zone.resolve()
    for standard in standards:
        target = standard.resolve()
        if target == root or root in target.parents:
            raise ValidatorInvalid(
                f"the checking standard {target} is reachable from the validation "
                f"zone {root}; the producer's own zone is what this separates"
            )


def build_environment(
    root: Path,
    *,
    config: EnvironmentConfig,
    agent_id: str | None,
    standards: Sequence[Path] = (),
) -> ValidationEnvironment:
    """Allocate a fresh zone and return the environment a body will see.

    Never reuses a path, never cleans one up, and never inherits `os.environ`.
    The same absolute path being reused is not a cosmetic difference: any
    absolute path a producer baked into an artefact **still resolves** under it,
    which is exactly the locality dependence the handoff module exists to catch.
    """
    attributed = assert_attributable(agent_id)
    root.mkdir(parents=True, exist_ok=True)
    zone = Path(tempfile.mkdtemp(prefix="validation-", dir=root))
    for sub in ("tmp", "home"):
        (zone / sub).mkdir()
    assert_standard_unreachable(zone, standards)
    env = {
        **dict(config.values),
        "TMPDIR": str(zone / "tmp"),
        "HOME": str(zone / "home"),
        "PWD": str(zone),
    }
    return ValidationEnvironment(zone=zone, cwd=zone, env=env, config=config, agent_id=attributed)
