# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""`check_packup_shape`'s shared-namespace rule.

The rule exists because its prose ancestor failed measurably: the task brief said
*"pick your ports; do not assume them"* before B5 ran, and B5's kit hardcoded all
four ports, both container names and the workdir. It would be a poor joke to
answer that with a check that nothing checks — so the rule has a test, and every
case here carries the case that must still pass beside the one that must fail.

The kits are synthetic and minimal on purpose. The real discriminating pair is
B5's own kit against a copy whose assignments were rewritten to `:=`; that lives
in `scratch/single-real-task-2026-08/` with the probe that runs it, because it is
evidence of a specific run and not a regression fixture. What is pinned here is
the rule's *shape*, which is what a later edit would break.

Loaded by path, not imported: the validator body is package data — it is not on
`sys.path` and has no reason to be.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(scope="module")
def check(package_root: Path) -> Any:
    body = (
        package_root.parent
        / "single_real_task"
        / "assets"
        / "check_packup_shape.validator"
        / "check.py"
    )
    spec = importlib.util.spec_from_file_location("_check_packup_shape", body)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def kit(root: Path, **scripts: str) -> Path:
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    for name, text in scripts.items():
        (root / "scripts" / f"{name}.sh").write_text(text)
    return root


FROZEN = """\
export CTR_NAME=srt_qwen36_mix
docker run -d --name "${CTR_NAME}" some/image
"""

PARAMETERISED = """\
: "${CTR_NAME:=srt_qwen36_mix}"
docker run -d --name "${CTR_NAME}" some/image
"""


def test_a_frozen_name_that_reaches_name_is_refused(check: Any, tmp_path: Path) -> None:
    faults = check.check_shared_identifiers(kit(tmp_path, env=FROZEN))
    assert len(faults) == 1, faults
    assert "CTR_NAME" in faults[0]


def test_the_same_kit_parameterised_passes(check: Any, tmp_path: Path) -> None:
    """The control. `: "${X:=v}"` is the only difference from the case above, and
    it is the whole of what the brief asks for — so if this failed, the rule
    would be refusing every kit rather than refusing the wrong ones."""
    assert check.check_shared_identifiers(kit(tmp_path, env=PARAMETERISED)) == []


def test_both_halves_are_required(check: Any, tmp_path: Path) -> None:
    """Frozen **and** bound. Either alone is not a fault, and saying so is what
    keeps the rule from becoming "no constants in shell scripts"."""
    frozen_only = 'export GREETING=hello\necho "${GREETING}"\n'
    bound_only = ': "${CTR_NAME:=x}"\ndocker run --name "${CTR_NAME}" i\n'
    assert check.check_shared_identifiers(kit(tmp_path / "a", env=frozen_only)) == []
    assert check.check_shared_identifiers(kit(tmp_path / "b", env=bound_only)) == []


def test_a_value_built_only_from_other_variables_is_not_frozen(check: Any, tmp_path: Path) -> None:
    """`"${HOST}:${PORT}"` is as parameterised as the variables it is made of, so
    the punctuation joining them must not count as a fixed value.

    Its control is the workdir shape on the next line, which differs only by
    having a literal leaf — and which must still be refused, because that leaf is
    exactly what two runs would collide on."""
    derived = """\
: "${HOST:=127.0.0.1}"
: "${PORT:=18000}"
export ENDPOINT="${HOST}:${PORT}"
docker run -v "${ENDPOINT}:/x" i
"""
    assert check.check_shared_identifiers(kit(tmp_path / "a", env=derived)) == []

    workdir = """\
: "${WORK_ROOT:=/var/tmp}"
export WORK="${WORK_ROOT}/srt_qwen36_mix"
docker run -v "${WORK}:/workdir" i
"""
    faults = check.check_shared_identifiers(kit(tmp_path / "b", env=workdir))
    assert len(faults) == 1 and "WORK" in faults[0], faults


def test_mkdir_p_is_not_a_published_port(check: Any, tmp_path: Path) -> None:
    """`-p` and `-v` mean something else outside docker, and `mkdir -p` appears
    in the very kit this rule was written against. Its control is the same flag
    inside a docker command, which must still be caught."""
    innocent = 'export LOGS=/var/tmp/kit/logs\nmkdir -p "${LOGS}"\n'
    assert check.check_shared_identifiers(kit(tmp_path / "a", env=innocent)) == []

    guilty = 'export PORT=18000\ndocker run -p "${PORT}:8000" i\n'
    assert len(check.check_shared_identifiers(kit(tmp_path / "b", env=guilty))) == 1


def test_a_bare_literal_at_name_is_refused_but_a_read_only_mount_is_not(
    check: Any, tmp_path: Path
) -> None:
    """A container name is never legitimately fixed. A host path mounted
    read-only very often is, and only its host side is shared at all — so
    `--volume` is deliberately outside the bare-literal rule."""
    literal_name = "docker run --name srt_qwen36_mix some/image\n"
    faults = check.check_shared_identifiers(kit(tmp_path / "a", env=literal_name))
    assert len(faults) == 1 and "srt_qwen36_mix" in faults[0], faults

    literal_mount = "docker run -v /data/models:/models some/image\n"
    assert check.check_shared_identifiers(kit(tmp_path / "b", env=literal_mount)) == []


def test_a_flag_split_over_continuations_is_still_seen(check: Any, tmp_path: Path) -> None:
    """Every `docker run` in a real kit spans a dozen backslash-continued lines.
    A scanner reading them one at a time sees a flag with no argument."""
    wrapped = """\
export CTR_NAME=srt_qwen36_mix
docker run -d \\
  --network=host \\
  --name "${CTR_NAME}" \\
  some/image sleep infinity
"""
    faults = check.check_shared_identifiers(kit(tmp_path, env=wrapped))
    assert len(faults) == 1 and "CTR_NAME" in faults[0], faults


def test_no_scripts_directory_reports_nothing_here(check: Any, tmp_path: Path) -> None:
    """Its absence is already a fault, reported by `check_packup` as
    `scripts/: missing`. Reporting it twice in two vocabularies helps nobody."""
    assert check.check_shared_identifiers(tmp_path) == []
