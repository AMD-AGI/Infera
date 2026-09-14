#!/usr/bin/env python3
"""Apply a `kernel_patch` to files taken out of the engine image, and plan the mounts.

Four things happen here, and the order is the argument:

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
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import nodecall  # noqa: E402
import patchkit  # noqa: E402


def env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"apply: {name} is unset")
    return value


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch-dir", required=True, help="the sealed kernel_patch content dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--package", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    codes = Path(args.patch_dir) / "items" / "codes"
    if not (codes / "manifest.json").is_file():
        raise SystemExit(f"apply: no manifest.json under {codes}")
    manifest = patchkit.read_manifest(codes)

    node = env("IT_NODE")
    image = env("IT_IMAGE")
    work_root = env("IT_WORK_ROOT")

    bad = patchkit.check_manifest(manifest, codes)
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

        diff = codes / "patches" / entry["patch"]
        log = logs / f"{i}-{entry['patch']}.log"
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
                "patch": entry["patch"],
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
    shutil.copytree(codes / "patches", items / "result" / "patches")

    (items / "env" / "overlay.json").write_text(
        json.dumps(
            {
                "node": node,
                "slurm_jobid": env("IT_JOBID"),
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
