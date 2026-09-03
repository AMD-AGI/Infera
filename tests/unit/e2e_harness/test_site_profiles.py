###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Static checks on the checked-in cluster profiles under tests/sites/.

A profile is the one file in a run that nothing else validates: it is sourced by
tests/run_tests.sh before any tier starts, and a variable misspelled there is
simply never read. The run then proceeds with a default — the wrong partition, no
model directory, unpinned NICs — and fails half an hour later somewhere that says
nothing about the typo. Every check below exists to turn that into a failure here.

The vocabulary of legal names is not a list in this file; it is scraped from the
code that reads the variables, so a profile can only set something that is
actually consumed, and retiring a variable retires it from the profiles too.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SITES = REPO / "tests" / "sites"

# Where a variable becomes real: the launcher, and the harness the launcher runs.
READERS = [REPO / "tests" / "run_tests.sh", REPO / "tests" / "e2e"]

_VAR_RE = re.compile(r"INFERA_E2E_[A-Z0-9_]+")
# `site_default VAR value...`, the only statement a profile is allowed to make.
_CALL_RE = re.compile(r"^site_default\s+(\S+)\s+(.*)$", re.DOTALL)


@functools.cache
def _known_vars() -> frozenset[str]:
    """Every INFERA_E2E_* name mentioned by code that reads them."""
    names: set[str] = set()
    for root in READERS:
        files = [root] if root.is_file() else sorted(root.rglob("*.py"))
        for f in files:
            names |= set(_VAR_RE.findall(f.read_text(encoding="utf-8", errors="ignore")))
    return frozenset(names)


def _statements(text: str) -> list[str]:
    """Profile lines with comments and blanks dropped and continuations joined."""
    joined = re.sub(r"\\\n\s*", " ", text)
    out = []
    for line in joined.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def _settings(path: Path) -> list[tuple[str, str]]:
    """(variable, value) in file order, with the shell quoting stripped.

    Anything that is not a site_default call is skipped rather than rejected —
    test_profile_only_calls_site_default is what reports those, and asserting
    here too would make one stray line mask every other test's own finding.
    """
    pairs = []
    for stmt in _statements(path.read_text(encoding="utf-8")):
        m = _CALL_RE.match(stmt)
        if m:
            pairs.append((m.group(1), m.group(2).strip().strip("\"'")))
    return pairs


PROFILES = sorted(SITES.glob("*.env"))


def test_at_least_one_profile_ships():
    """Guards the glob itself: every test below passes vacuously on an empty dir,
    and the mechanism is only worth having if a profile uses it."""
    assert PROFILES, f"no site profiles found under {SITES}"


@pytest.mark.parametrize("path", PROFILES, ids=lambda p: p.stem)
def test_profile_only_calls_site_default(path):
    """A profile is data, not a program. It is sourced into the launcher's own
    shell, so a stray command runs with the launcher's privileges and ordering;
    keeping the grammar to one verb is what makes reviewing one cheap."""
    for stmt in _statements(path.read_text(encoding="utf-8")):
        assert _CALL_RE.match(stmt), (
            f"{path.name}: only `site_default VAR VALUE` is allowed, got: {stmt!r}"
        )


@pytest.mark.parametrize("path", PROFILES, ids=lambda p: p.stem)
def test_profile_sets_only_variables_something_reads(path):
    known = _known_vars()
    for var, _ in _settings(path):
        assert var in known, (
            f"{path.name} sets {var}, which nothing in tests/run_tests.sh or "
            f"tests/e2e reads — a typo here is silent at runtime"
        )


@pytest.mark.parametrize("path", PROFILES, ids=lambda p: p.stem)
def test_profile_sets_each_variable_once(path):
    """Two site_default lines for one variable is a merge artefact: the second is
    dead, because the first already exported it."""
    seen = [v for v, _ in _settings(path)]
    dupes = {v for v in seen if seen.count(v) > 1}
    assert not dupes, f"{path.name}: set more than once: {sorted(dupes)}"


@pytest.mark.parametrize("path", PROFILES, ids=lambda p: p.stem)
def test_kv_list_variables_parse(path):
    """BUILD_ARGS and WORKER_ENV are read as comma-separated K=V by build_image()
    and cluster._kv_list(). A stray space or a missing '=' drops the entry on the
    floor, and the build or the KV transport then just behaves as if it were never
    configured."""
    for var, value in _settings(path):
        if var not in ("INFERA_E2E_BUILD_ARGS", "INFERA_E2E_WORKER_ENV"):
            continue
        assert value, f"{path.name}: {var} is empty"
        for item in value.split(","):
            assert "=" in item, f"{path.name}: {var} entry {item!r} is not K=V"
            key = item.split("=", 1)[0]
            assert key and key == key.strip(), (
                f"{path.name}: {var} key {key!r} has stray whitespace"
            )


@pytest.mark.parametrize("path", PROFILES, ids=lambda p: p.stem)
def test_local_flag_is_zero_or_one(path):
    """_run_here() matches the literal 1 and 0 and treats anything else as "not
    specified", so a well-meant `true` would silently mean "decide by inspection"."""
    for var, value in _settings(path):
        if var == "INFERA_E2E_LOCAL":
            assert value in ("0", "1"), (
                f"{path.name}: INFERA_E2E_LOCAL must be 0 or 1, got {value!r}"
            )
