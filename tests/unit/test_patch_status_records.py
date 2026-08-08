###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The patch status records under deploy/docker/patches/ hold, and the gate bites.

A patch that outlives its upstream fix is not obviously broken: it keeps applying
cleanly and the build log says nothing. The records exist so that state is written
down, and scripts/validate-patch-status.py exists so the writing-down cannot be
skipped. A validator that passes on a tree with a missing record would be worse
than none, so these tests check the REFUSALS as well as the happy path:

  * the real tree validates
  * a patch added without a record fails
  * a record whose date is not a date fails
  * an index whose totals disagree with the tree fails
  * an archived entry that claims `deleted` without a recovery commit fails
  * a record that reaches outside its own directory to cover a patch fails
  * a record left behind by a renamed or deleted patch fails
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = REPO_ROOT / "scripts" / "validate-patch-status.py"
PATCH_ROOT = Path("deploy/docker/patches")
INDEX = Path("deploy/docker/patch.upstream.status.yaml")
ARCHIVED = PATCH_ROOT / "archived" / "patch.archived.yaml"

pytestmark = pytest.mark.skipif(
    not VALIDATOR.is_file(), reason="validator script not present in this checkout"
)


def run(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VALIDATOR), "--repo-root", str(root)],
        capture_output=True,
        text=True,
    )


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A throwaway copy of just the parts the validator reads."""
    root = tmp_path / "repo"
    (root / PATCH_ROOT.parent).mkdir(parents=True)
    shutil.copytree(REPO_ROOT / PATCH_ROOT, root / PATCH_ROOT)
    shutil.copy(REPO_ROOT / INDEX, root / INDEX)
    (root / "scripts").mkdir()
    shutil.copy(VALIDATOR, root / "scripts" / VALIDATOR.name)
    return root


def test_repo_records_validate() -> None:
    """The tree as committed passes, so a later failure means a real regression."""
    done = run(REPO_ROOT)
    assert done.returncode == 0, done.stdout + done.stderr


def test_copied_tree_validates(tree: Path) -> None:
    """Guards the fixture itself — the mutations below only mean something from a clean base."""
    done = run(tree)
    assert done.returncode == 0, done.stdout + done.stderr


def test_patch_without_record_is_rejected(tree: Path) -> None:
    (tree / PATCH_ROOT / "vllm" / "patch_brand_new.py").write_text("# no record\n")
    done = run(tree)
    assert done.returncode == 1
    assert "no status record" in done.stdout


def test_record_with_non_date_is_rejected(tree: Path) -> None:
    record = tree / PATCH_ROOT / "vllm" / "patch_sched_guard.upstream.status.yaml"
    record.write_text(
        record.read_text().replace("status_updated: 2026-08-05", 'status_updated: "soon"')
    )
    done = run(tree)
    assert done.returncode == 1
    assert "status_updated" in done.stdout


def test_record_not_in_index_is_rejected(tree: Path) -> None:
    index = tree / INDEX
    text = index.read_text()
    # Drop the whole entry for one patch, keeping the file valid YAML.
    start = text.index("      - patch: deploy/docker/patches/vllm/patch_sched_guard.py")
    end = text.index("      - patch:", start + 1)
    index.write_text(text[:start] + text[end:])
    done = run(tree)
    assert done.returncode == 1
    assert "not listed in the index" in done.stdout


def test_wrong_totals_are_rejected(tree: Path) -> None:
    index = tree / INDEX
    index.write_text(index.read_text().replace("active_patches: 25", "active_patches: 24"))
    done = run(tree)
    assert done.returncode == 1
    assert "totals.active_patches" in done.stdout


def test_deleted_archive_entry_needs_a_recovery_commit(tree: Path) -> None:
    archived = tree / ARCHIVED
    archived.write_text(
        archived.read_text().replace(
            "last_commit_with_file: 89c86fb", "last_commit_with_file: null", 1
        )
    )
    done = run(tree)
    assert done.returncode == 1
    assert "cannot be recovered" in done.stdout


def test_record_pointing_at_the_wrong_patch_is_rejected(tree: Path) -> None:
    record = tree / PATCH_ROOT / "vllm" / "patch_sched_guard.upstream.status.yaml"
    record.write_text(
        record.read_text().replace(
            "path: deploy/docker/patches/vllm/patch_sched_guard.py",
            "path: deploy/docker/patches/vllm/patch_somewhere_else.py",
        )
    )
    done = run(tree)
    assert done.returncode == 1
    assert "patch.path is" in done.stdout


def test_extra_files_cannot_reach_into_another_directory(tree: Path) -> None:
    """extra_files is for the rest of one patch, not a way to vouch for someone else's.

    Left unbounded it is a hole straight through the gate: the new patch below needs a
    record, and one line in an unrelated record would otherwise be enough to excuse it.
    """
    (tree / PATCH_ROOT / "vllm" / "patch_brand_new.py").write_text("# no record\n")
    record = tree / PATCH_ROOT / "atom" / "patch_gdn_pd_state_transfer.upstream.status.yaml"
    record.write_text(
        record.read_text().replace(
            "patch:\n",
            "patch:\n  extra_files:\n    - deploy/docker/patches/vllm/patch_brand_new.py\n",
            1,
        )
    )
    done = run(tree)
    assert done.returncode == 1
    assert "is outside" in done.stdout
    # and the patch it tried to cover is still asked for a record of its own
    assert "no status record" in done.stdout


def test_orphan_record_is_rejected(tree: Path) -> None:
    """Nothing walks record -> patch, so an unclaimed record is never even parsed."""
    (tree / PATCH_ROOT / "vllm" / "patch_ghost.upstream.status.yaml").write_text(
        "status_updated: not-a-date\n"
    )
    done = run(tree)
    assert done.returncode == 1
    assert "does not correspond to any patch file" in done.stdout


def test_renamed_patch_may_not_leave_its_record_behind(tree: Path) -> None:
    """The realistic version of the above: `git mv` the patch, forget the record."""
    vllm = tree / PATCH_ROOT / "vllm"
    (vllm / "patch_sched_guard.py").rename(vllm / "patch_sched_guard_v2.py")
    done = run(tree)
    assert done.returncode == 1
    assert "does not correspond to any patch file" in done.stdout
