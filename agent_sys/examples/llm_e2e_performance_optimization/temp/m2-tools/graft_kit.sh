#!/usr/bin/env bash
# Graft m1's real deploy_kit into m2's mock_root, refusing every way I know of
# to graft the wrong one.
#
#   usage: graft_kit.sh <path to the deploy_kit CONTENT directory>
#
# WHY A SCRIPT AND NOT A `cp`
# ---------------------------
# `mock.sh` wants `<mock_root>/stage1-deploy/deploy_kit/content/`, and a `cp`
# into it is one command. The reason this is thirty guards instead is that
# **every check below is one `line.sh` already performs at bring-up** — so a kit
# that would abort forty minutes in, after a real deployment, aborts here in a
# second instead. Nothing here is new policy; it is `line.sh`'s policy moved
# earlier.
#
# It also mechanises two rules that otherwise live in someone's memory:
#
#   * **the run directory is GIVEN, never found.** There is no glob, no `ls -t`,
#     no "latest run" anywhere in this file. A wrong-run kit is the failure that
#     reads like a producer defect, and the cheapest defence is that this script
#     cannot locate a kit on its own.
#   * **values come from the kit's record, never from a launch line.** It prints
#     `fixed.tp_size` / `fixed.image` / `fixed.gpu_devices` **read out of
#     environment.yaml**, because m1's own recorded `LAUNCH-LINE.txt` carries
#     `expect_ranks=2` (the mock value) and the record carries what the bring-up
#     actually did.
#
# DELETION: this script deletes nothing, ever. If the destination exists and is
# non-empty it ABORTS and says so. There is no recursive delete in this file and
# no `rm` with a variable target.
set -uo pipefail

# Overridable ONLY so the guards can be exercised against a throwaway target.
# The positive control must not land in the real mock_root: a well-formed fake
# kit sitting where the real one goes is a decoy, and "same name, different
# tree" has cost this project a wrong conclusion before.
DEST_ROOT="${GRAFT_DEST_ROOT:-/data/yihou/e2e_verify_20260906/m2/mock_root}"
DEST="$DEST_ROOT/stage1-deploy/deploy_kit/content"
THIS_NODE=smci355-ccs-aus-n04-25

die() { printf 'graft: ABORT: %s\n' "$*" >&2; exit 1; }
say() { printf 'graft: %s\n' "$*"; }

# ---- 0. the argument -------------------------------------------------------
SRC="${1:-}"
[ -n "$SRC" ] || die "no source given.
  usage: $0 <deploy_kit content dir>
  Get the path from the leader. Do not go looking for it: this script has no
  way to find a run on its own, deliberately."
[ -d "$SRC" ] || die "not a directory: $SRC"
case "$SRC" in
  /*) ;;
  *) die "give an absolute path, got: $SRC" ;;
esac

# ---- 1. every check line.sh makes, made now --------------------------------
ENV_IN="$SRC/items/codes/environment.yaml"
[ -r "$ENV_IN" ] || die "no environment record at $ENV_IN
  (line.sh:107 aborts on exactly this)"

SCRIPTS="$(python3 - "$SRC" <<'PY'
import sys
from pathlib import Path
codes = Path(sys.argv[1]) / "items" / "codes"
if not codes.is_dir():
    print("items/codes is not a directory", file=sys.stderr); raise SystemExit(1)
found = sorted(p for p in codes.iterdir() if p.is_dir() and (p / "scripts").is_dir())
if len(found) != 1:
    print(f"expected one packup directory with scripts/ under items/codes, found "
          f"{[p.name for p in found]}", file=sys.stderr)
    raise SystemExit(1)
print(found[0] / "scripts")
PY
)" || die "could not locate the kit's scripts (line.sh:110-121 does the same)"
say "scripts: $SCRIPTS"

for s in deploy.sh wait_ready.sh teardown.sh; do
  [ -r "$SCRIPTS/$s" ] || die "the kit has no $s"
done
say "deploy.sh, wait_ready.sh, teardown.sh all present"

# ---- 2. the record, and the node guard -------------------------------------
FACTS="$(python3 - "$ENV_IN" <<'PY'
import sys, yaml
doc = yaml.safe_load(open(sys.argv[1])) or {}
fixed, rt = doc.get("fixed") or {}, doc.get("runtime") or {}
for k in ("node", "image", "image_id", "model_name", "served_model_name", "tp_size"):
    print(f"KIT_{k.upper()}={fixed.get(k, '')}")
devs = fixed.get("gpu_devices") or []
if not isinstance(devs, (list, tuple)):
    devs = [devs]
print("KIT_GPU_DEVICES=" + ",".join(str(d) for d in devs))
print(f"KIT_REPLAYED_FROM={rt.get('replayed_from') or ''}")
PY
)" || die "environment.yaml is not readable YAML"

get() { printf '%s\n' "$FACTS" | sed -n "s/^$1=//p"; }
KIT_NODE="$(get KIT_NODE)"; KIT_TP="$(get KIT_TP_SIZE)"
KIT_IMAGE="$(get KIT_IMAGE)"; KIT_DEVS="$(get KIT_GPU_DEVICES)"
KIT_REPLAYED="$(get KIT_REPLAYED_FROM)"

[ -n "$KIT_NODE" ] || die "fixed.node is empty in the record"
[ "$KIT_NODE" = "$THIS_NODE" ] || die "this kit describes node '$KIT_NODE', we are on '$THIS_NODE'.
  line.sh refuses this too — measuring somewhere else makes it evidence about a
  different machine. This is the wrong-run failure; stop and check the path."
say "fixed.node = $KIT_NODE  (matches)"

[ -n "$KIT_TP" ] || die "fixed.tp_size is empty — expect_ranks would have no source"
[ -n "$KIT_IMAGE" ] || die "fixed.image is empty — --var image would have no source"

if [ -n "$KIT_REPLAYED" ]; then
  say "NOTE: runtime.replayed_from = '$KIT_REPLAYED' — this kit was REPLAYED, not"
  say "      produced by a bring-up here. Its scripts are real; its record was"
  say "      re-rendered. Numbers from it are not comparable to the deployment it"
  say "      originally described."
fi

# ---- 3. never overwrite, never delete --------------------------------------
# **NOT `$(ls -A "$DEST" 2>&1)`** — that was the first version, and it is the
# error-string-is-truthy shape: a failing `ls` returns its diagnostic on stdout,
# the string is non-empty, and the test reads "directory is not empty". It
# happened to fail SAFE here (refuse rather than overwrite), which is exactly
# how this idiom survives review. `find -mindepth 1` puts errors on stderr where
# they belong, so an unreadable directory is a visible failure and not a datum.
if [ -e "$DEST" ] && [ -n "$(find "$DEST" -mindepth 1 -print -quit)" ]; then
  die "$DEST already exists and is not empty.
  This script does not delete. Move it aside yourself, or graft elsewhere."
fi

mkdir -p "$DEST" || die "could not create $DEST"
cp -a "$SRC/." "$DEST/" || die "copy failed"
N="$(find "$DEST" -type f | wc -l)"
say "copied $N files -> $DEST"

# ---- 4. provenance, with a timestamp that is READ ---------------------------
# A copy's mtime describes the SOURCE, not the copy: `cp -a` preserves it. The
# only thing that describes this copy is the line below, and its timestamp comes
# out of `date`, not out of my head.
NOW="$(date -u +%FT%TZ)"
{
  echo "grafted_at_utc : $NOW"
  echo "source         : $SRC"
  echo "grafted_by     : $0"
  echo "files          : $N"
  echo "fixed.node     : $KIT_NODE"
  echo "fixed.tp_size  : $KIT_TP"
  echo "fixed.image    : $KIT_IMAGE"
  echo "fixed.gpu_devices : $KIT_DEVS"
  echo "runtime.replayed_from : ${KIT_REPLAYED:-<unset>}"
} > "$DEST_ROOT/GRAFTED.txt"

cat <<EOF

graft: done. The launch line's values, TAKEN FROM THE RECORD:

    --var tp=$KIT_TP
    --var expect_ranks=$KIT_TP        <- fixed.tp_size, NOT m1's LAUNCH-LINE.txt (which says 2)
    --var image=$KIT_IMAGE

  gpu_devices in the record: ${KIT_DEVS:-<none>}
  -> still OMIT --var gpu_devices; line.sh:181 will take this set itself and
     log 'cards: ... (the set the deploy_kit records taking)'.

  Provenance written to $DEST_ROOT/GRAFTED.txt
EOF
