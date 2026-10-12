#!/usr/bin/env python3
"""Purpose: Prepare the exact InferenceX checkout used by AgentX and GSM8K.
Usage:
    python3 tools/ensure_inferencex.py --directory DIR --repository URL --ref COMMIT
Artifacts:
    A detached Git checkout and a serialization lock file.
Artifact paths:
    DIR and DIR.parent/.<DIR.name>.lock.

Prepare the pinned InferenceX checkout used by AgentX and GSM8K.

Usage: ensure_inferencex.py --directory DIR --repository URL --ref COMMIT
"""

import argparse
import fcntl
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def git(directory: Path, *args: str, capture: bool = False) -> str:
    command = ["git", "-C", str(directory), *args]
    try:
        result = subprocess.run(
            command, check=True, text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError):
            detail = (exc.stderr or exc.stdout or "").strip()
        raise SystemExit(f"{' '.join(command)} failed: {detail or exc}") from exc
    return result.stdout.strip() if capture else ""


def same_repository(actual: str, expected: str) -> bool:
    def normalize(value: str) -> str:
        return value.rstrip("/").removesuffix(".git")
    return normalize(actual) == normalize(expected)


def prepare_submodules(directory: Path) -> None:
    git(directory, "submodule", "sync", "--recursive")
    git(directory, "submodule", "update", "--init", "--recursive", "--depth", "1")
    git(
        directory,
        "submodule",
        "foreach",
        "--recursive",
        'test "$sha1" = "$(git rev-parse HEAD)"',
    )


def prepare_existing(directory: Path, repository: str, ref: str) -> None:
    if not (directory / ".git").is_dir():
        raise SystemExit(f"{directory} exists but is not a Git checkout")
    actual = git(directory, "remote", "get-url", "origin", capture=True)
    if not same_repository(actual, repository):
        raise SystemExit(
            f"{directory}: origin is {actual!r}, expected {repository!r}"
        )
    dirty = git(directory, "status", "--porcelain", capture=True)
    if dirty:
        raise SystemExit(f"{directory}: checkout is dirty; refusing to replace files")
    exists = subprocess.run(
        ["git", "-C", str(directory), "cat-file", "-e", f"{ref}^{{commit}}"],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0
    if not exists:
        git(directory, "fetch", "--depth", "1", "origin", ref)
        fetched = git(directory, "rev-parse", "FETCH_HEAD^{commit}", capture=True)
        if fetched != ref:
            raise SystemExit(f"origin resolved {ref} to unexpected commit {fetched}")
    git(directory, "checkout", "--detach", ref)
    actual_ref = git(directory, "rev-parse", "HEAD", capture=True)
    if actual_ref != ref:
        raise SystemExit(f"{directory}: checked out {actual_ref}, expected {ref}")
    prepare_submodules(directory)


def clone_atomic(directory: Path, repository: str, ref: str) -> None:
    temporary = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=directory.parent))
    try:
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", repository, str(temporary)],
            check=True,
        )
        git(temporary, "fetch", "--depth", "1", "origin", ref)
        fetched = git(temporary, "rev-parse", "FETCH_HEAD^{commit}", capture=True)
        if fetched != ref:
            raise SystemExit(f"origin resolved {ref} to unexpected commit {fetched}")
        git(temporary, "checkout", "--detach", ref)
        prepare_submodules(temporary)
        os.rename(temporary, directory)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"failed to clone {repository} at {ref}: {exc}") from exc
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--ref", required=True)
    args = parser.parse_args()
    if len(args.ref) != 40 or any(char not in "0123456789abcdef" for char in args.ref):
        raise SystemExit("--ref must be a full lowercase 40-character commit")
    directory = args.directory.expanduser().resolve()
    directory.parent.mkdir(parents=True, exist_ok=True)
    lock_path = directory.parent / f".{directory.name}.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if directory.exists():
            prepare_existing(directory, args.repository, args.ref)
        else:
            clone_atomic(directory, args.repository, args.ref)
    print(f"InferenceX ready: {directory} @ {args.ref}")


if __name__ == "__main__":
    main()
