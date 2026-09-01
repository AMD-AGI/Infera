#!/usr/bin/env python3
"""The patch manifest and the mount plan: one reader, one writer, one validator.

Four bodies touch these two documents -- `seed_patch` writes the manifest,
`apply_patch` reads it and writes the mount plan, `serve_patched` turns the plan
into docker arguments, and two validators check both. Writing the field names out
four times is how they drift, so they are written here once.

Nothing in this module talks to docker or to the network. It is import-safe from
a validator body, which runs on the login node with no allocation.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import yaml

SCHEMA_VERSION = 1

#: The only apply mode this stage implements. A patch that has to be compiled --
#: HIP, CK, assembly -- cannot be bind-mounted over a running image, and the
#: rebuild path is a 9m25s image build plus a second evidence chain for the build
#: itself. Declaring the mode makes an out-of-scope patch fail at the first
#: validator instead of being mounted and never executed.
APPLY_OVERLAY = "overlay_files"
APPLY_REBUILD = "rebuild"
APPLY_MODES = (APPLY_OVERLAY, APPLY_REBUILD)

#: Container roots a patch may name, as `@NAME@ -> /path/inside/the/image`.
#:
#: **Paths in a manifest are written in placeholder form and stay that way**, all
#: the way through the mount plan, and are expanded only at the moment something
#: talks to docker. That is not a style choice: `handoff/locality.py` refuses to
#: seal content naming an absolute path outside a small allow-list, and every one
#: of these roots is outside it. See `container_roots.yaml` for the whole
#: argument, including why `@NAME@` rather than `${NAME}`.
_ROOTS_FILE = Path(__file__).resolve().parent / "container_roots.yaml"
CONTAINER_ROOTS: dict[str, str] = {
    f"@{name}@": spec["path"]
    for name, spec in yaml.safe_load(_ROOTS_FILE.read_text(encoding="utf-8"))["roots"].items()
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PLACEHOLDER = re.compile(r"^(@[A-Z0-9_]+@)(/.*)?$")


def expand(path: str, roots: dict[str, str] | None = None) -> str:
    """`@SGLANG_ROOT@/srt/models/x.py` -> the path inside the image.

    Raises rather than passing an unknown placeholder through: a half-expanded
    path handed to `docker cp` fails as "no such file", which says nothing about
    the manifest being wrong.
    """
    roots = CONTAINER_ROOTS if roots is None else roots
    match = _PLACEHOLDER.match(path)
    if not match:
        raise ValueError(f"{path!r} does not start with a @PLACEHOLDER@")
    head, tail = match.group(1), match.group(2) or ""
    if head not in roots:
        raise ValueError(f"{head} is not a known container root ({', '.join(sorted(roots))})")
    return roots[head] + tail


def split_placeholder(path: str) -> tuple[str, str]:
    """`@SGLANG_ROOT@/srt/models/x.py` -> ('@SGLANG_ROOT@', 'srt/models/x.py')."""
    match = _PLACEHOLDER.match(path)
    if not match:
        raise ValueError(f"{path!r} does not start with a @PLACEHOLDER@")
    return match.group(1), (match.group(2) or "").lstrip("/")


def contract(container_path: str, roots: dict[str, str] | None = None) -> str:
    """The inverse: longest matching root becomes its placeholder."""
    roots = CONTAINER_ROOTS if roots is None else roots
    best = ""
    for name, root in roots.items():
        if container_path.startswith(root + "/") and len(root) > len(roots.get(best, "")):
            best = name
    if not best:
        raise ValueError(f"{container_path!r} is under no known container root")
    return best + container_path[len(roots[best]) :]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA256.match(value))


def looks_like_unified_diff(text: str) -> tuple[bool, str]:
    """Is this a diff a patcher could take? Returns (ok, why-not).

    Not a full parse. It checks the three markers whose absence means the file is
    something else entirely -- a commit message, an empty file, a diff that was
    truncated by a shell redirect -- because those are the failures worth a named
    error rather than `patch: **** malformed patch`.
    """
    if not text.strip():
        return False, "the diff is empty"
    if "\n--- " not in "\n" + text:
        return False, "no '---' source line"
    if "\n+++ " not in "\n" + text:
        return False, "no '+++' target line"
    if not re.search(r"^@@ -\d+(,\d+)? \+\d+(,\d+)? @@", text, re.M):
        return False, "no '@@' hunk header"
    return True, ""


def under_known_root(container_path: str, roots: dict[str, str] | None = None) -> bool:
    try:
        expand(container_path, roots)
    except ValueError:
        return False
    return True


# --------------------------------------------------------------------------- #
# manifest.json -- what the kernel-optimization stage hands over
# --------------------------------------------------------------------------- #

MANIFEST_REQUIRED = (
    "schema_version",
    "operator_id",
    "logical_operator",
    "image",
    "apply_mode",
    "files",
)

#: Per-file keys. `base_sha256` is what pins the patch, and it is a hash rather
#: than a git commit for a measured reason: both /sgl-workspace/sglang and
#: /sgl-workspace/aiter are git repositories, but the sglang working tree is
#: dirty relative to its own HEAD because the image build replaces python/sglang
#: wholesale with the PR overlay. A `git diff` against HEAD would carry changes
#: nobody in this pipeline made.
FILE_REQUIRED = ("container_path", "base_sha256", "patch", "change")

CHANGE_KINDS = ("modify", "add")


def read_manifest(codes_dir: str | Path) -> dict:
    return json.loads(Path(codes_dir, "manifest.json").read_text(encoding="utf-8"))


def check_manifest(manifest: dict, codes_dir: str | Path, *,
                   roots: dict[str, str] | None = None,
                   supported_modes=(APPLY_OVERLAY,)) -> list[str]:
    """Every reason this manifest cannot be applied. Empty list means it can."""
    codes = Path(codes_dir)
    bad: list[str] = []

    for key in MANIFEST_REQUIRED:
        if key not in manifest:
            bad.append(f"manifest.json has no {key!r}")
    if bad:
        return bad

    if manifest["schema_version"] != SCHEMA_VERSION:
        bad.append(
            f"schema_version is {manifest['schema_version']!r}, this stage reads {SCHEMA_VERSION}"
        )
    mode = manifest["apply_mode"]
    if mode not in APPLY_MODES:
        bad.append(f"apply_mode {mode!r} is not one of {list(APPLY_MODES)}")
    elif mode not in supported_modes:
        bad.append(
            f"apply_mode {mode!r} is declared but not implemented here. "
            "A patch that has to be compiled needs the image rebuilt; see DESIGN.md section 3.3."
        )

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        bad.append("files is empty -- a patch that touches nothing cannot be tested")
        return bad

    for i, entry in enumerate(files):
        where = f"files[{i}]"
        for key in FILE_REQUIRED:
            if key not in entry:
                bad.append(f"{where} has no {key!r}")
        if bad and any(where in b for b in bad):
            continue
        path = entry["container_path"]
        if path.startswith("/"):
            bad.append(
                f"{where}.container_path {path!r} is a bare absolute path. It has to name "
                "its container root as a @PLACEHOLDER@, or the handoff cannot be published "
                "at all -- see assets/lib/container_roots.yaml."
            )
        elif not under_known_root(path, roots):
            bad.append(f"{where}.container_path {path!r} is under no known container root")
        elif ".." in path.split("/"):
            bad.append(f"{where}.container_path {path!r} escapes its root with '..'")
        if not is_sha256(entry["base_sha256"]):
            bad.append(f"{where}.base_sha256 is not a sha256 hex digest")
        if entry["change"] not in CHANGE_KINDS:
            bad.append(f"{where}.change {entry['change']!r} is not one of {list(CHANGE_KINDS)}")
        diff = codes / "patches" / entry["patch"]
        if not diff.is_file():
            bad.append(f"{where}.patch names {entry['patch']!r}, which is not in patches/")
        else:
            ok, why = looks_like_unified_diff(diff.read_text(encoding="utf-8", errors="replace"))
            if not ok:
                bad.append(f"{where}.patch ({entry['patch']}): {why}")

    marker = manifest.get("runtime_marker")
    if marker is not None:
        if not isinstance(marker, dict):
            bad.append("runtime_marker is present but is not an object")
        else:
            for key in ("import", "first_call"):
                if key in marker:
                    try:
                        re.compile(marker[key])
                    except re.error as exc:
                        bad.append(f"runtime_marker.{key} is not a valid regex: {exc}")
    return bad


# --------------------------------------------------------------------------- #
# mounts.json -- what apply_patch produces and serve_patched consumes
# --------------------------------------------------------------------------- #

MOUNT_REQUIRED = ("container_path", "host_path", "sha256_stock", "sha256_patched", "change")

#: `host_path` travels the same way container paths do, for the same reason: the
#: node-local work area is `/data/...`, which the seal refuses. `apply_patch`
#: writes it as `@WORK_ROOT@/overlay/...` and every consumer expands it with the
#: work root it was itself configured with.
WORK_ROOT_PLACEHOLDER = "@WORK_ROOT@"


def expand_host(host_path: str, work_root: str) -> str:
    if host_path.startswith(WORK_ROOT_PLACEHOLDER):
        return work_root + host_path[len(WORK_ROOT_PLACEHOLDER) :]
    return host_path


def read_mounts(result_dir: str | Path) -> dict:
    return json.loads(Path(result_dir, "mounts.json").read_text(encoding="utf-8"))


def docker_mount_args(mounts: dict, work_root: str,
                      roots: dict[str, str] | None = None) -> list[str]:
    """The `-v host:container:ro` arguments, in the plan's own order.

    Read-only on purpose. A kernel under test has no business writing to its own
    source file, and a writable mount would let a run mutate the artefact that
    is supposed to describe it.
    """
    out: list[str] = []
    for entry in mounts["mounts"]:
        host = expand_host(entry["host_path"], work_root)
        out += ["-v", f"{host}:{expand(entry['container_path'], roots)}:ro"]
    return out


def check_mounts(mounts: dict, *, work_root: str | None = None,
                 require_difference: bool = True,
                 roots: dict[str, str] | None = None) -> list[str]:
    """Every reason this mount plan should not be trusted. Empty list means it can.

    `work_root` is optional: a validator running on the login node cannot see the
    node-local files at all, so it checks the plan's shape and leaves the bytes to
    `check_patch_live`, which reads what the running container actually holds.
    """
    bad: list[str] = []
    entries = mounts.get("mounts")
    if not isinstance(entries, list) or not entries:
        bad.append("mounts.json carries no mounts")
        return bad

    seen: set[str] = set()
    for i, entry in enumerate(entries):
        where = f"mounts[{i}]"
        for key in MOUNT_REQUIRED:
            if key not in entry:
                bad.append(f"{where} has no {key!r}")
        if any(where in b for b in bad):
            continue
        if not under_known_root(entry["container_path"], roots):
            bad.append(
                f"{where}.container_path {entry['container_path']!r} names no known container root"
            )
        if entry["container_path"] in seen:
            bad.append(f"{where}.container_path {entry['container_path']!r} is mounted twice")
        seen.add(entry["container_path"])
        if not is_sha256(entry["sha256_patched"]):
            bad.append(f"{where}.sha256_patched is not a sha256 hex digest")
        # A patch that applies cleanly and changes nothing gives two arms that
        # are byte-identical, and every check downstream then passes for the
        # wrong reason: the pipeline compares the stock deployment against itself
        # and reports no regression.
        if require_difference and entry["sha256_patched"] == entry["sha256_stock"]:
            bad.append(
                f"{where}: patched and stock hash the same, so this mount changes nothing"
            )
        if work_root is None:
            continue
        host = Path(expand_host(entry["host_path"], work_root))
        if not host.is_file():
            bad.append(f"{where}.host_path does not exist: {host}")
            continue
        if host.stat().st_size == 0:
            bad.append(f"{where}.host_path is empty: {host}")
            continue
        actual = sha256_file(host)
        if actual != entry["sha256_patched"]:
            bad.append(
                f"{where}: the file on disk hashes {actual[:12]}… but the plan says "
                f"{entry['sha256_patched'][:12]}…"
            )
    return bad


def write_mount_spec(mounts: dict, work_root: str, dest: str | Path) -> int:
    """The `host<TAB>container` file `mix_up.sh` turns into `-v` arguments.

    Expansion happens here and nowhere else, so there is one place that knows
    what a placeholder means and one place to get it wrong.
    """
    lines = [
        f"{expand_host(m['host_path'], work_root)}\t{expand(m['container_path'])}"
        for m in mounts["mounts"]
    ]
    Path(dest).write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return len(lines)


def check_published_files(mounts: dict, files_dir: str | Path) -> list[str]:
    """The copies the handoff carries hash as the plan says.

    This is what a validator on the login node can check: the node-local file is
    unreachable from there, but the handoff's own copy of the same bytes is not.
    """
    bad: list[str] = []
    for i, entry in enumerate(mounts.get("mounts", [])):
        _, rel = split_placeholder(entry["container_path"])
        copy = Path(files_dir) / str(i) / rel
        if not copy.is_file():
            bad.append(f"mounts[{i}]: the handoff carries no copy of {rel}")
            continue
        actual = sha256_file(copy)
        if actual != entry["sha256_patched"]:
            bad.append(
                f"mounts[{i}]: the published copy of {rel} hashes {actual[:12]}… but the "
                f"plan says {entry['sha256_patched'][:12]}…"
            )
    return bad


# --------------------------------------------------------------------------- #
# A shell-callable face, so `round.sh` does not have to know what a placeholder
# is. Everything above is importable; this is the only entry point a bash body
# uses, and it exists so that expansion lives in exactly one module.
# --------------------------------------------------------------------------- #

def _main(argv: list[str]) -> int:
    import sys

    if len(argv) == 4 and argv[0] == "mountspec":
        _, mounts_json, work_root, dest = argv
        plan = json.loads(Path(mounts_json).read_text(encoding="utf-8"))
        n = write_mount_spec(plan, work_root, dest)
        print(f"patchkit: {n} mount(s) -> {dest}")
        return 0
    if len(argv) == 1 and argv[0] == "redact-args":
        # `NAME=/path` for every container root, in the form redact.py takes.
        #
        # The patched arm's evidence names container paths that nothing else in
        # this package writes: `docker inspect`'s Mounts and the sha256 taken
        # inside the container both report the real destination, and the seal
        # refuses both. Feeding the same table that produced the placeholders in
        # `mounts.json` keeps the published evidence consistent with the plan it
        # is evidence for.
        for placeholder, path in sorted(CONTAINER_ROOTS.items()):
            print(f"{placeholder.strip('@')}={path}")
        return 0
    if len(argv) == 2 and argv[0] == "markers":
        plan = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
        marker = plan.get("runtime_marker") or {}
        for key in ("import", "first_call"):
            if marker.get(key):
                print(f"{key}\t{marker[key]}")
        return 0
    print(
        "usage: patchkit.py mountspec <mounts.json> <work_root> <dest.tsv>\n"
        "       patchkit.py markers <mounts.json>",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    import sys

    sys.exit(_main(sys.argv[1:]))
