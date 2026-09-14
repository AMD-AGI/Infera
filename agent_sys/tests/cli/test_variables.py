"""`--var K=V`, and the one variable the CLI will not take from a user.

The flag replaced a hardcoded `outside=` keyword, so the property under test is
not that argparse can append to a list: it is that a value typed on the command
line **reaches a spec**, and that the one name the CLI supplies itself is
refused out loud rather than dropped.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from cli import main as cli_main
from cli import package
from cli.events import EventKind
from cli.stream import Stream

# --------------------------------------------------------------------------- #
# A package small enough to read, declaring one variable and nothing else.
#
# Built here rather than borrowed from `examples/`: the one variable
# `examples/demo` declares is `outside`, which is exactly the name this flag
# refuses, and `examples/demo2` is a large package whose contents are not this
# test's subject. Four files is the whole of what `YamlPackage` needs.

_MAIN = """\
module: task
name: main
description: One leaf, so there is somewhere for a variable to land.
handoffs: []
validators: []
task:
  goal: Carry one variable from the command line into an agent env block.
  version: '1'
  inputs: []
  outputs: []
  subgraph:
    - {closure: leaf, froms: [], is_end: true}
"""

_LEAF = """\
- module: agent
  name: worker
  kind: program
  version: '1'
  description: Runs a task's entry.sh. No model call.
  env:
    GREETING: '${greeting:-unset}'
- module: task
  name: leaf
  description: Does nothing, and says which greeting it was given.
  agent: worker
  handoffs: []
  validators: []
  task:
    goal: Echo the greeting.
    version: '1'
    inputs: []
    outputs: []
"""


@pytest.fixture()
def tiny(tmp_path: Path) -> Path:
    root = tmp_path / "tiny"
    (root / "steps").mkdir(parents=True)
    (root / "assets" / "leaf.task").mkdir(parents=True)
    (root / "assets" / "main.task").mkdir(parents=True)
    (root / "assets" / "main.task" / "readme.md").write_text("A root.\n", encoding="utf-8")
    (root / "main.yaml").write_text(_MAIN, encoding="utf-8")
    (root / "steps" / "leaf.yaml").write_text(_LEAF, encoding="utf-8")
    (root / "assets" / "leaf.task" / "readme.md").write_text("Nothing.\n", encoding="utf-8")
    (root / "assets" / "leaf.task" / "entry.sh").write_text(
        textwrap.dedent("""\
            #!/bin/sh
            echo "$GREETING"
        """),
        encoding="utf-8",
    )
    return root


def _greeting(root: Path, *argv: str) -> str:
    """Load `root` the way the CLI's `show` does, and read the value back."""
    args = cli_main.parser().parse_args(["show", "--package", str(root), *argv])
    args.variables = cli_main._parse_variables(cli_main.parser(), args.var)
    _, registry = cli_main._load(args, Stream())
    return dict(registry.get("agent_specs").get("worker").get("env") or {})["GREETING"]


def test_a_var_reaches_a_spec(tiny: Path) -> None:
    """The whole point of the flag: what was typed is what the spec holds."""
    assert _greeting(tiny, "--var", "greeting=hello") == "hello"


def test_without_the_flag_the_declared_default_stands(tiny: Path) -> None:
    """The other half, and the reason a package writes `${k:-default}` rather
    than a bare reference: an unfilled variable must be visibly unfilled."""
    assert _greeting(tiny) == "unset"


def test_the_last_of_two_values_for_one_name_wins(tiny: Path) -> None:
    """`--var` is repeatable **across names**; repeating one is a user error the
    CLI does not have to diagnose, and last-wins is what every `-D`-shaped flag
    does. Asserted so the behaviour is chosen rather than incidental."""
    assert _greeting(tiny, "--var", "greeting=a", "--var", "greeting=b") == "b"


def test_a_value_may_contain_an_equals_sign(tiny: Path) -> None:
    """`partition`, not `split`: a variable holding `a=b` is ordinary."""
    assert _greeting(tiny, "--var", "greeting=k=v") == "k=v"


def test_an_empty_value_is_a_value(tiny: Path) -> None:
    """`--var greeting=` sets it to the empty string, which is different from
    not passing it — and the difference is visible, because the default is
    what the other case renders."""
    assert _greeting(tiny, "--var", "greeting=") == ""


# --------------------------------------------------------------------------- #
# The two refusals


def test_outside_is_refused_by_name(capsys: pytest.CaptureFixture[str]) -> None:
    """**A silently ignored flag is worse than a rejected one.**

    `outside` is per-run and absolute and only the CLI knows it, so the CLI's
    value has to win. That leaves two behaviours for a user who passes it, and
    dropping it quietly hands them a flag they typed, no error, and a run that
    ignored them. So it is an argument error, and the message says which name
    and why rather than only that something was wrong.
    """
    with pytest.raises(SystemExit) as exit_info:
        cli_main.main(["show", "--var", "outside=/tmp/x"])
    assert exit_info.value.code == 2
    message = capsys.readouterr().err
    assert "outside" in message and "supplied by the CLI itself" in message


def test_a_var_without_an_equals_names_the_token(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli_main.main(["show", "--var", "n_problems"])
    assert exit_info.value.code == 2
    assert "'n_problems'" in capsys.readouterr().err


def test_a_var_with_an_empty_name_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    """`--var =2` parses as a name of `''`, which no package can reference."""
    with pytest.raises(SystemExit) as exit_info:
        cli_main.main(["show", "--var", "=2"])
    assert exit_info.value.code == 2
    assert "K may not be empty" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# The flag is on both verbs, which is what makes `show` a cheap check of one


@pytest.mark.parametrize("verb", ["show", "run"])
def test_both_verbs_take_it(verb: str) -> None:
    args = cli_main.parser().parse_args([verb, "--var", "a=1", "--var", "b=2"])
    assert cli_main._parse_variables(cli_main.parser(), args.var) == {"a": "1", "b": "2"}


def test_a_reserved_name_reaching_the_call_would_crash_rather_than_override() -> None:
    """The guarantee behind the refusal message, and it is the language's.

    `_registry` calls `task_package(root, **variables, outside=...)`. Python
    refuses a duplicate keyword with *got multiple values*, so an `outside` that
    somehow got past `_parse_variables` would be a `TypeError` at the call and
    never a run that quietly used the wrong path. Asserted because the refusal a
    user sees is a *message*, and a message can be deleted; this cannot.
    """
    with pytest.raises(TypeError, match="outside"):
        package.task_package(Path("/nonexistent"), **{"outside": "user"}, outside="cli")


def test_show_reports_the_package_it_loaded(tiny: Path) -> None:
    """A sanity check on the fixture itself: a package that loads nothing is
    indistinguishable from one that loaded (`interfaces.md` §4.11), so the test
    above would pass over an empty registry without this."""
    stream = Stream()
    assert cli_main.main(["show", "--package", str(tiny)]) == cli_main.OK, "the tiny package loads"
    args = cli_main.parser().parse_args(["show", "--package", str(tiny)])
    args.variables = {}
    _, registry = cli_main._load(args, stream)
    assert sorted(registry.get("closures").names()) == ["leaf", "main"]
    assert stream.count(EventKind.PACKAGE_LOADED) == 1
