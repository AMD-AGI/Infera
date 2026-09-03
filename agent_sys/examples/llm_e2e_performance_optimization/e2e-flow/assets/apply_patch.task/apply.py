#!/usr/bin/env python3
"""Apply m4's optimised kernel to files taken out of the engine image, and plan the mounts.

**Its input is m4's real `kernel_optimization`, not a hand-written patch**
(M5.1). `seed_patch` is gone; this reads the `apply/` block m4 writes against
the workset's declared integration point (M5.1.1), which is what lets this stay
a program rather than becoming judgement work — the decision of *which* file in
the engine the optimised kernel replaces was made by m3 when it built the
workset, and m4 only had to fill it in.

**The contract, in one paragraph.** `kernel_optimization` carries
`items/codes/<packup>/apply/manifest.json`. Its fields are patchkit's, the same
ones `integration-demo` proposed to stage 4 and then proved on the cluster:
`schema_version`, `operator_id`, `logical_operator`, `image`, `apply_mode`,
`files[]` and an optional `runtime_marker` and `expect`. Each `files[]` entry
names a `container_path` in `@ROOT@/...` form, the `base_sha256` of what it
replaces, a `change` of `modify` or `add`, and **exactly one of** `patch` (a
unified diff under `apply/patches/`) or `replacement` (a path, relative to the
packup, of the whole file that replaces it). `replacement` exists because m4's
actual artefact is a whole optimised kernel file rather than a diff; this body
generates the diff from stock to replacement so that everything downstream —
including `check_overlay_applies` — sees one shape.

Five things happen here, and the order is the argument:

1. **Refuse what this stage cannot do.** `apply_mode: rebuild` fails immediately
   and says why, rather than being quietly ignored into a deployment that runs
   stock code and reports no regression.
2. **Pin against the image, not against a commit.** Every file is extracted from
   the image the manifest names and hashed; a mismatch with `base_sha256` means
   the patch was cut against a different build, and that is the failure this
   check exists for. Both git repositories in the image are dirty relative to
   their own HEAD -- the build replaces the sglang python tree wholesale with a
   PR overlay -- so a commit id would pin nothing.
3. **Apply and compile.** A python file that does not compile takes the worker
   down during model import, fifteen minutes later, where it reads as a
   model-loading failure.
4. **Put the result on node-local disk.** The zone is discarded with the attempt
   and the deployment outlives it, so a mount pointing into the zone would break
   the first time the container restarted.
5. **Carry the environment forward** (mission G5). `patch_overlay` is a
   `reproducible` kind, so m1's record lands at `items/env/environment.yaml`,
   inherited rather than re-derived: a stage that rebuilt the record could
   differ from m1's and nothing would notice.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import nodecall  # noqa: E402
import patchkit  # noqa: E402


CONTRACT = """\
m4's `kernel_optimization` must carry an apply block:

    items/codes/<packup>/apply/manifest.json
    items/codes/<packup>/apply/patches/*.patch      (when an entry uses `patch`)

    {"schema_version": 1,
     "operator_id": "...", "logical_operator": "...",
     "image": "<the image the optimisation was measured against>",
     "apply_mode": "overlay_files",
     "files": [{"container_path": "@SGLANG_ROOT@/srt/...py",
                "base_sha256": "<64 hex>", "change": "modify",
                "replacement": "results/optimized_kernel.py"}],
     "runtime_marker": {"import": "<regex>", "first_call": "<regex>"},
     "expect": {"source": "forge", "speedup": 1.23}}

Each `files[]` entry names exactly one of `patch` (a unified diff under
`apply/patches/`) or `replacement` (a path relative to the packup directory).
`container_path` is written in `@ROOT@/...` form because the handoff seal refuses
an absolute container path; the roots are in `assets/lib/container_roots.yaml`.
"""


def env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"apply: {name} is unset")
    return value


def find_apply(kopt: Path) -> tuple[Path, Path]:
    """`(apply_dir, packup_dir)` inside m4's handoff.

    Searched rather than hard-coded, because the packup directory is named after
    the operator and the date. **Exactly one**: zero means m4 did not write the
    block, and two means a reader has to guess which optimisation is being
    integrated — neither is a case to resolve quietly.
    """
    codes = kopt / "items" / "codes"
    found = sorted(codes.glob("*/apply/manifest.json")) if codes.is_dir() else []
    if not found:
        listing = sorted(p.name for p in codes.iterdir()) if codes.is_dir() else []
        raise SystemExit(
            f"apply: no apply/manifest.json under {codes} (found: {listing})\n\n" + CONTRACT
        )
    if len(found) > 1:
        raise SystemExit(
            f"apply: {len(found)} apply blocks under {codes}: {[str(p) for p in found]}\n"
            "One kernel_optimization integrates one optimisation."
        )
    return found[0].parent, found[0].parent.parent


def check_apply_manifest(manifest: dict, apply_dir: Path, packup: Path) -> list[str]:
    """Every reason m4's apply block cannot be applied. Empty list means it can.

    **Not `patchkit.check_manifest`**, and the difference is the one field that
    matters: that function grades `kernel_patch`, whose every entry carries a
    `patch`, and this grades m4's block, whose entries may carry a `replacement`
    instead. Everything else — the placeholder rule, the sha256 rule, the
    apply-mode enumeration, the regex check on the markers — is patchkit's, by
    call rather than by copy.
    """
    bad: list[str] = []
    for key in ("schema_version", "operator_id", "image", "apply_mode", "files"):
        if key not in manifest:
            bad.append(f"apply/manifest.json has no {key!r}")
    if bad:
        return bad + ["", CONTRACT]

    if manifest["schema_version"] != patchkit.SCHEMA_VERSION:
        bad.append(f"schema_version is {manifest['schema_version']!r}, this stage reads {patchkit.SCHEMA_VERSION}")
    mode = manifest["apply_mode"]
    if mode not in patchkit.APPLY_MODES:
        bad.append(f"apply_mode {mode!r} is not one of {list(patchkit.APPLY_MODES)}")
    elif mode != patchkit.APPLY_OVERLAY:
        # Not a gap: a kernel that has to be compiled — HIP, CK, assembly —
        # needs the image rebuilt, which is a nine-minute build plus a second
        # evidence chain for the build itself. Failing here is what stops such a
        # patch being mounted, never executed, and reported as no regression.
        bad.append(
            f"apply_mode {mode!r} is declared and not implemented here. A patch that must be "
            "compiled needs the image rebuilt; this stage only does read-only bind mounts."
        )

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        return bad + ["files is empty — a patch that touches nothing cannot be tested"]

    for i, entry in enumerate(files):
        where = f"files[{i}]"
        for key in ("container_path", "base_sha256", "change"):
            if key not in entry:
                bad.append(f"{where} has no {key!r}")
        if any(where in b for b in bad):
            continue
        path = entry["container_path"]
        if path.startswith("/"):
            bad.append(
                f"{where}.container_path {path!r} is a bare absolute path. It has to name its "
                "container root as a @PLACEHOLDER@ or the handoff cannot be published at all "
                "— see assets/lib/container_roots.yaml."
            )
        elif not patchkit.under_known_root(path):
            bad.append(
                f"{where}.container_path {path!r} is under no known container root. A patch "
                "naming a host path has confused the machine it was cut on with the image it "
                "is to be applied to, and that error otherwise survives all the way to a "
                "deployment silently running stock code."
            )
        elif ".." in path.split("/"):
            bad.append(f"{where}.container_path {path!r} escapes its root with '..'")
        if not patchkit.is_sha256(entry["base_sha256"]):
            bad.append(f"{where}.base_sha256 is not a sha256 hex digest")
        if entry["change"] not in patchkit.CHANGE_KINDS:
            bad.append(f"{where}.change {entry['change']!r} is not one of {list(patchkit.CHANGE_KINDS)}")

        has_patch, has_replacement = bool(entry.get("patch")), bool(entry.get("replacement"))
        if has_patch == has_replacement:
            bad.append(
                f"{where} names {'both' if has_patch else 'neither'} `patch` and `replacement`; "
                "exactly one is required"
            )
        elif has_patch:
            diff = apply_dir / "patches" / entry["patch"]
            if not diff.is_file():
                bad.append(f"{where}.patch names {entry['patch']!r}, which is not in apply/patches/")
            else:
                ok, why = patchkit.looks_like_unified_diff(diff.read_text(encoding="utf-8", errors="replace"))
                if not ok:
                    bad.append(f"{where}.patch ({entry['patch']}): {why}")
        else:
            if not (packup / entry["replacement"]).is_file():
                bad.append(f"{where}.replacement {entry['replacement']!r} is not in the handoff")

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel-optimization", required=True, help="m4's staged content directory")
    ap.add_argument("--deploy-kit", required=True, help="m1's staged content directory, for the environment record")
    ap.add_argument("--out", required=True)
    ap.add_argument("--package", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    apply_dir, packup = find_apply(Path(args.kernel_optimization))
    manifest = json.loads((apply_dir / "manifest.json").read_text(encoding="utf-8"))

    node = env("E2E_NODE")
    image = env("E2E_IMAGE")
    work_root = env("E2E_WORK_ROOT")

    bad = check_apply_manifest(manifest, apply_dir, packup)
    if bad:
        for line in bad:
            print(f"apply: {line}", file=sys.stderr)
        raise SystemExit(1)

    # The patch names an image; this run was pointed at one. Different is not
    # automatically wrong -- a patch may legitimately be re-checked against a
    # rebuild -- but the hashes below will decide, and saying so here makes that
    # verdict readable when it comes.
    if manifest["image"] != image:
        print(
            f"apply: NOTE the patch was cut against {manifest['image']!r} and this run "
            f"serves {image!r}; the per-file hashes decide whether that matters"
        )

    stage = Path.cwd() / "stage"
    shutil.rmtree(stage, ignore_errors=True)
    (stage / "trees").mkdir(parents=True)
    logs = stage / "logs"
    logs.mkdir()
    # Every mount's diff, whether m4 shipped one or this body generated it from a
    # whole-file replacement. Published as `items/result/patches/`, so the record
    # says what changed and not only that something did.
    diffs = stage / "patches"
    diffs.mkdir()

    if not nodecall.visible_on_node(stage):
        raise SystemExit(
            f"apply: the attempt zone is not visible on {node}: {stage}\n"
            "This body has the node write extracted files here and reads them back, "
            "which needs the run root on a filesystem both hosts mount."
        )

    run_tag = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    overlay_root = f"{work_root}/overlay/{run_tag}"

    entries = manifest["files"]
    print(f"apply: {len(entries)} file(s), image {image}, overlay -> {overlay_root}")

    # One container handle for every extraction. `docker create` starts no
    # process and touches no GPU; the image's entrypoint never runs.
    extract = [
        "set -e",
        f"CID=$(docker create '{image}' true)",
        "trap 'docker rm -f $CID >/dev/null 2>&1' EXIT",
        f"mkdir -p '{overlay_root}'",
    ]
    for i, entry in enumerate(entries):
        dest = stage / "trees" / str(i) / "stock"
        dest.mkdir(parents=True)
        if entry["change"] == "modify":
            # Expanded here and only here: docker needs the real path, everything
            # written back out stays in placeholder form so it can be published.
            inside = patchkit.expand(entry["container_path"])
            extract.append(f"docker cp \"$CID:{inside}\" '{dest}/file'")
        else:
            # An added file has nothing to extract; the diff creates it whole.
            extract.append(f": > '{dest}/file'")
    nodecall.on("\n".join(extract))

    mounts = []
    for i, entry in enumerate(entries):
        base = stage / "trees" / str(i)
        stock = base / "stock" / "file"
        if not stock.exists():
            raise SystemExit(f"apply: {entry['container_path']} did not come out of the image")

        sha_stock = patchkit.sha256_file(stock) if stock.stat().st_size else ""
        if entry["change"] == "modify" and sha_stock != entry["base_sha256"]:
            raise SystemExit(
                f"apply: {entry['container_path']} hashes {sha_stock[:12]}… in {image} "
                f"but the patch was cut against {entry['base_sha256'][:12]}…\n"
                "The patch and the image do not match. Rebuild the image, or re-cut the patch."
            )

        root, rel = patchkit.split_placeholder(entry["container_path"])
        tree = base / "tree"
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(stock, tree / rel)
        # A second, untouched copy of the stock file, kept only so that a
        # whole-file replacement can be turned into a diff below. It costs a few
        # kilobytes and it is what lets m4 hand over a kernel file rather than a
        # patch without every consumer downstream learning about a second shape.
        (base / "a" / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(stock, base / "a" / rel)

        patch_name = entry.get("patch") or f"{i:04d}-{manifest['operator_id']}.patch"
        log = logs / f"{i}-{patch_name}.log"
        if entry.get("patch"):
            diff = apply_dir / "patches" / entry["patch"]
            # `patch` and not `git apply`: the staging tree is not a repository and
            # does not need to become one, and `patch --batch` never prompts, which
            # matters in a body whose stdin is closed.
            proc = subprocess.run(
                ["patch", "-p1", "--batch", "--forward", "-i", str(diff)],
                cwd=tree,
                capture_output=True,
                text=True,
            )
            log.write_text(proc.stdout + proc.stderr, encoding="utf-8")
            if proc.returncode != 0:
                print(proc.stdout + proc.stderr, file=sys.stderr)
                raise SystemExit(
                    f"apply: {entry['patch']} did not apply to {entry['container_path']} (rc={proc.returncode})"
                )
            shutil.copyfile(diff, diffs / patch_name)
        else:
            # **A whole-file replacement, which is what m4 actually produces.**
            # The diff is generated rather than demanded so that `patch_overlay`
            # has one shape downstream: `check_overlay_applies` parses a unified
            # diff per mount and does not need to know which of the two forms
            # m4 chose. `diff` exits 1 when the files differ, which is the normal
            # case here, so the return code is read for 2 (an error) rather than
            # for 0.
            shutil.copyfile(packup / entry["replacement"], tree / rel)
            proc = subprocess.run(
                ["diff", "-u", "--label", f"a/{rel}", "--label", f"b/{rel}",
                 f"a/{rel}", f"tree/{rel}"],
                cwd=base,
                capture_output=True,
                text=True,
            )
            if proc.returncode > 1:
                raise SystemExit(f"apply: could not diff {rel}: {proc.stderr}")
            (diffs / patch_name).write_text(proc.stdout, encoding="utf-8")
            log.write_text(
                f"generated {patch_name} from replacement {entry['replacement']}\n"
                f"{len(proc.stdout.splitlines())} diff line(s)\n",
                encoding="utf-8",
            )

        patched = tree / rel
        if patched.name.endswith(".py"):
            try:
                compile(patched.read_text(encoding="utf-8"), rel, "exec")
            except SyntaxError as exc:
                raise SystemExit(
                    f"apply: the patched {entry['container_path']} does not compile: {exc}"
                ) from exc

        sha_patched = patchkit.sha256_file(patched)
        if sha_patched == sha_stock:
            raise SystemExit(
                f"apply: {entry['patch']} applied but changed nothing in "
                f"{entry['container_path']}. Two identical arms would compare the stock "
                "deployment against itself and report no regression."
            )

        mounts.append(
            {
                "container_path": entry["container_path"],
                # Placeholder form for the same reason container paths use it:
                # the node-local work area is under /data, which the seal refuses.
                "host_path": f"{patchkit.WORK_ROOT_PLACEHOLDER}/overlay/{run_tag}/{i}/{rel}",
                "sha256_stock": sha_stock,
                "sha256_patched": sha_patched,
                "change": entry["change"],
                "container_root": root,
                "repo_relative": rel,
                "patch": patch_name,
            }
        )
        print(f"apply: {rel}  {sha_stock[:12]}… -> {sha_patched[:12]}…")

    # Node-local, because the deployment outlives this attempt's zone.
    push = ["set -e"]
    real_paths = [patchkit.expand_host(m["host_path"], work_root) for m in mounts]
    for i, (m, real) in enumerate(zip(mounts, real_paths)):
        src = stage / "trees" / str(i) / "tree" / m["repo_relative"]
        push.append(f"mkdir -p '{Path(real).parent}'")
        push.append(f"cp '{src}' '{real}'")
        push.append(f"chmod 0444 '{real}'")
    push.append("sha256sum " + " ".join(f"'{p}'" for p in real_paths))
    pushed = nodecall.on("\n".join(push))
    on_node = {
        line.split(None, 1)[1].strip(): line.split(None, 1)[0]
        for line in pushed.strip().splitlines()
        if line.strip()
    }
    for m, real in zip(mounts, real_paths):
        got = on_node.get(real)
        if got != m["sha256_patched"]:
            raise SystemExit(
                f"apply: {real} hashes {got} on {node} but {m['sha256_patched']} here"
            )
    print(f"apply: {len(mounts)} file(s) staged on {node} under {overlay_root}")

    # ---- the handoff ---------------------------------------------------------
    items = out / "items"
    for sub in ("result", "env", "logs"):
        (items / sub).mkdir(parents=True, exist_ok=True)

    plan = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "apply_mode": manifest["apply_mode"],
        "image": image,
        "operator_id": manifest["operator_id"],
        "overlay_root": f"{patchkit.WORK_ROOT_PLACEHOLDER}/overlay/{run_tag}",
        "runtime_marker": manifest.get("runtime_marker"),
        "mounts": mounts,
    }
    (items / "result" / "mounts.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    # The patched files themselves travel with the plan. The node-local copy is
    # what gets mounted, but it dies with the node; a handoff that only pointed
    # at it would stop describing anything the moment the allocation ended.
    files_dir = items / "result" / "files"
    for i, m in enumerate(mounts):
        dest = files_dir / str(i) / m["repo_relative"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(stage / "trees" / str(i) / "tree" / m["repo_relative"], dest)
    shutil.copytree(diffs, items / "result" / "patches")

    (items / "env" / "overlay.json").write_text(
        json.dumps(
            {
                "node": node,
                "slurm_jobid": env("E2E_JOBID"),
                "image": image,
                "work_root": work_root,
                "patch_manifest": {
                    k: manifest[k]
                    for k in ("schema_version", "operator_id", "logical_operator", "apply_mode")
                },
                "patcher": subprocess.run(
                    ["patch", "--version"], capture_output=True, text=True
                ).stdout.splitlines()[:1],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    command = items / "command"
    command.write_text(
        """#!/usr/bin/env bash
# Rebuild this overlay from the patch set, and print the docker arguments it
# becomes. `agent.gate` requires this item to be executable, so it is a script
# and not a transcript -- and writing it as one is what lets it survive
# publication, because every path it needs arrives as a shell variable and there
# is no absolute path here for the locality seal to reject.
#
# ROOTS is the one input that cannot travel in this handoff. It maps @NAME@ to
# the directory that placeholder stands for inside the image, and those are
# absolute paths, so the table itself would be refused publication for exactly
# the reason the placeholders exist. It lives in the package, at
# assets/lib/container_roots.yaml.
set -eu
: "${IMAGE:?export IMAGE=<the engine image the patch was cut against>}"
: "${OVERLAY_ROOT:?export OVERLAY_ROOT=<node-local directory to stage patched files in>}"
: "${PATCHES:?export PATCHES=<this handoff's items/result/patches directory>}"
: "${MANIFEST:?export MANIFEST=<the kernel_patch manifest.json>}"
: "${ROOTS:?export ROOTS=<the package's assets/lib/container_roots.yaml>}"

python3 - "$MANIFEST" "$PATCHES" "$IMAGE" "$OVERLAY_ROOT" "$ROOTS" <<'PY'
import hashlib, json, pathlib, re, shutil, subprocess, sys, tempfile, yaml
manifest, patches, image, overlay, roots_file = sys.argv[1:6]
roots = {f"@{k}@": v["path"] for k, v in yaml.safe_load(open(roots_file))["roots"].items()}
spec = json.load(open(manifest))
cid = subprocess.run(["docker", "create", image, "true"],
                     capture_output=True, text=True, check=True).stdout.strip()
try:
    for i, entry in enumerate(spec["files"]):
        head, rel = re.match(r"^(@[A-Z0-9_]+@)/(.*)$", entry["container_path"]).groups()
        inside = roots[head] + "/" + rel
        tree = pathlib.Path(tempfile.mkdtemp()) / "t"
        (tree / rel).parent.mkdir(parents=True)
        subprocess.run(["docker", "cp", f"{cid}:{inside}", str(tree / rel)], check=True)
        got = hashlib.sha256((tree / rel).read_bytes()).hexdigest()
        if got != entry["base_sha256"]:
            raise SystemExit(f"{rel}: the image has {got[:12]}, "
                             f"the patch expects {entry['base_sha256'][:12]}")
        subprocess.run(["patch", "-p1", "--batch", "--forward", "-i",
                        f"{patches}/{entry['patch']}"], cwd=tree, check=True)
        dest = pathlib.Path(overlay) / str(i) / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tree / rel, dest)
        dest.chmod(0o444)
        print(f"-v {dest}:{inside}:ro")
finally:
    subprocess.run(["docker", "rm", "-f", cid], capture_output=True)
PY
""",
        encoding="utf-8",
    )
    command.chmod(0o755)

    for log in sorted(logs.glob("*.log")):
        subprocess.run(
            ["gzip", "-9", "-c", str(log)],
            stdout=(items / "logs" / f"{log.name}.gz").open("wb"),
            check=True,
        )

    (items / "watchout").write_text(
        "host_path points at node-local storage on one machine. It is the path the\n"
        "engine container bind-mounts, and it does not exist anywhere else — when the\n"
        "allocation ends, so does it. items/result/files/ carries the same bytes so the\n"
        "record still says what was mounted after the node is gone.\n"
        "\n"
        "The mounts are read-only. A kernel under test has no business writing to its\n"
        "own source, and a writable mount would let a run mutate the artefact meant to\n"
        "describe it.\n"
        "\n"
        "A mount being present does not mean the code ran. That question belongs to\n"
        "check_patch_live, which re-hashes the file inside the running container and\n"
        "looks for the markers the patch declared. __file__ is no help: a bind mount\n"
        "leaves the path inside the container unchanged, so it reads identically on a\n"
        "patched and an unpatched deployment.\n",
        encoding="utf-8",
    )

    (out / "README.md").write_text(
        f"""# patch_overlay

## Purpose

Turn a patch set into something a container can be started with: one read-only
bind mount per touched file, each pinned by the hash of what it replaces and the
hash of what replaces it.

This works because sglang is installed into the image in editable mode, so the
interpreter reads the source tree in the image directly rather than a copy under
site-packages, and a file mounted over one of its members is compiled as-is. No
rebuild, no reinstall.

Per file rather than per tree, and that is a size decision with a hard edge: the
sglang python tree is 87 MB but the AITER tree is 6.9 GB, and a kernel patch
lands in AITER as often as not.

## How to run

`items/command` rebuilds the overlay from the patch set and prints the `-v`
arguments. It needs five variables: `IMAGE`, `OVERLAY_ROOT`, `PATCHES`,
`MANIFEST` and `ROOTS`.

`ROOTS` is the package's `assets/lib/container_roots.yaml`, and it is the one
input that could not travel in this handoff: it maps each `@NAME@` to the
directory inside the image it stands for, and those are absolute paths, which is
the reason the placeholders exist in the first place.

Site paths in this record are written as `@NAME@`. `@WORK_ROOT@` was the
node-local work area on the machine that produced it.

## Result

`items/result/mounts.json` is the plan: {len(mounts)} mount(s), each with its
`container_path`, its node-local `host_path`, and the stock and patched hashes.
`items/result/files/` holds the patched files themselves and
`items/result/patches/` the diffs they came from, so the record still describes
the change after the node it was built on is gone.

Applied to image `{image}`; the operator is `{manifest['operator_id']}`.

## Environment

`items/env/overlay.json` — the node, the allocation, the image, and the parts of
the patch manifest that decide how it was applied.

## Watch out

See `items/watchout`. In short: `host_path` is node-local and temporary; the
mounts are read-only; and a mount that is present is not yet a mount that ran.
""",
        encoding="utf-8",
    )

    # Mission G5: every handoff carries the environment record. Inherited from
    # m1 rather than rebuilt here — a stage that re-derived it could differ from
    # m1's and nothing in the flow would notice, which is the whole reason
    # `check_environment` compares the runtime block across its inputs.
    subprocess.run(
        [
            sys.executable,
            str(Path(args.package) / "assets" / "lib" / "env_render.py"),
            "--inherit", str(Path(args.deploy_kit) / "items" / "codes" / "environment.yaml"),
            "--content-type", "reproducible",
            "--out", str(out),
        ],
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            str(Path(args.package) / "assets" / "lib" / "redact.py"),
            str(out),
            f"WORK_ROOT={work_root}",
            f"TASK_PACKAGE={args.package}",
            f"ZONE={Path.cwd()}",
            "TMPDIR=/tmp",
            f"HOME={Path.home()}",
        ],
        check=True,
    )

    shutil.rmtree(stage, ignore_errors=True)
    print(f"apply: overlay ready, {len(mounts)} mount(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
