# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Exercise overlay application on real local Git repositories."""

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
APPLY = ROOT / "deploy/docker/scripts/apply_mooncake_overlays.sh"


def git(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture
def source(tmp_path):
    repo = tmp_path / "source"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "value").write_text("base\n")
    git(repo, "add", "value")
    git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "base")
    return repo


def patch(path, before, after):
    # Generate an independent hunk against the expected previous overlay.
    text = (
        f"diff --git a/value b/value\n--- a/value\n+++ b/value\n@@ -1 +1 @@\n-{before}\n+{after}\n"
    )
    path.write_text(text)
    return path


def apply(source, patches="", bundle="0"):
    env = dict(os.environ, MOONCAKE_PATCHES=patches, APPLY_MOONCAKE_RDMA_PATCHES=bundle)
    return subprocess.run(
        ["bash", str(APPLY), str(source)], env=env, capture_output=True, text=True
    )


def test_empty_overlay_preserves_source_and_records_revision(source):
    assert apply(source).returncode == 0
    assert git(source, "diff") == ""
    assert (
        source / "infera-build.txt"
    ).read_text() == f"revision {git(source, 'rev-parse', 'HEAD').strip()}\n"


def test_ordered_overlays_and_hashes(source, tmp_path):
    first = patch(tmp_path / "first.patch", "base", "first")
    second = patch(tmp_path / "second.patch", "first", "second")
    assert apply(source, f"{first} {second}").returncode == 0
    assert (source / "value").read_text() == "second\n"
    manifest = (source / "infera-build.txt").read_text().splitlines()
    assert manifest[1:] == [
        f"patch {hashlib.sha256(p.read_bytes()).hexdigest()}  {p}" for p in (first, second)
    ]


def test_conflict_stops_before_later_patch(source, tmp_path):
    conflict = patch(tmp_path / "bad.patch", "not-base", "wrong")
    valid = patch(tmp_path / "good.patch", "base", "later")
    assert apply(source, f"{conflict} {valid}").returncode != 0
    assert (source / "value").read_text() == "base\n"


@pytest.mark.parametrize("patches", ["relative.patch", "/no-such-mooncake.patch", "/one\n/two"])
def test_invalid_paths_fail(source, patches):
    assert apply(source, patches).returncode != 0
    assert git(source, "diff") == ""


def test_literal_glob_filename_is_not_expanded(source, tmp_path):
    literal = patch(tmp_path / "*.patch", "base", "literal")
    patch(tmp_path / "other.patch", "base", "other")
    assert apply(source, str(literal)).returncode == 0
    assert (source / "value").read_text() == "literal\n"


def test_builtin_ref_drift_and_invalid_switch_fail(source):
    assert "require Mooncake" in apply(source, bundle="1").stderr
    assert apply(source, bundle="enabled").returncode != 0
    assert git(source, "diff") == ""


@pytest.mark.parametrize(
    "dockerfile", ["Dockerfile.sglang", "Dockerfile.sglang.gfx942", "Dockerfile.sglang.glm53"]
)
@pytest.mark.parametrize("build,bundle", [("0", "0"), ("0", "1"), ("1", "0"), ("1", "1")])
def test_docker_build_switches_reach_builder(tmp_path, dockerfile, build, bundle):
    # Execute the actual RUN block with only the expensive compiler replaced.
    text = (ROOT / "deploy/docker" / dockerfile).read_text().replace("\\\n", "")
    command = next(
        line[4:]
        for line in text.splitlines()
        if line.startswith("RUN ") and "scripts/build_mooncake_sglang.sh" in line
    )
    build_dir = tmp_path / "build"
    scripts = build_dir / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "build_mooncake_sglang.sh").write_text(
        '#!/usr/bin/env bash\nset -eu\nprintf "%s" "$APPLY_MOONCAKE_RDMA_PATCHES" > "$CAPTURE"\n'
    )
    platform = tmp_path / "platform.env"
    platform.write_text("MC_GPU_ARCH=gfx950\n")
    command = command.replace("/tmp/mooncake-build", str(build_dir)).replace(
        "/etc/glm53-platform.env", str(platform)
    )
    capture = tmp_path / "captured"
    result = subprocess.run(
        ["bash", "-c", command],
        capture_output=True,
        text=True,
        env=dict(
            os.environ,
            BUILD_MOONCAKE=build,
            APPLY_MOONCAKE_RDMA_PATCHES=bundle,
            MOONCAKE_GIT_REF="test-ref",
            CAPTURE=str(capture),
        ),
    )
    if build == "0" and bundle == "1":
        assert result.returncode != 0
        assert "RDMA patches require BUILD_MOONCAKE=1" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
    if build == "1":
        assert capture.read_text() == bundle
    else:
        assert not capture.exists()
