#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Experiment 11 — kvd's L3 on a real block device, and BUG #2.
#
# Two independent paths, and you should run §A first because it is free:
#
#   MODE=desk   (default)  no cluster, no GPU, no container, ~1 second.
#                          Reproduces the bind-mount parsing bug against the
#                          pre-fix logic, confirms the fix, and runs the 47
#                          regression tests. This IS the root cause.
#
#   MODE=node              one node, ~2 min, no GPU. Bind-mounts a host
#                          directory into a container, runs the classifier
#                          A/B (with and without the patch hunk), and surveys
#                          the storage so you can see WHY the verdict is what
#                          it is.
#
# THE TRAP THIS EXPERIMENT EXISTS TO DOCUMENT, and it is a deployment-recipe
# issue rather than an infera defect:
#
#     Bind-mounting the PATH does not expose the DEVICE.
#
# A stock container has no /dev/md0 node, so lsblk cannot classify it even
# after the patch. Accurate L3 classification needs `--device=/dev/md0` (or
# --privileged) IN ADDITION TO the -v mount. MODE=node passes the device
# through, and reports honestly if it could not.
# ---------------------------------------------------------------------------
set -uo pipefail

MODE="${MODE:-desk}"
JUMP="${JUMP:-root@149.28.124.225}"
NODE="${NODE:-chi2867}"
IMAGE="${IMAGE:-infera/engine-sglang:pd-unified}"
HOST_L3="${HOST_L3:-/mnt/nvme-raid/kvd-long}"   # host dir to bind-mount
CTR_L3="${CTR_L3:-/kvd-long}"                   # where it lands in the container
CTR="${CTR:-l3cls11_$$}"                        # unique: never collide
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-$HERE/../results}"

# ===========================================================================
# §A — desk check. No cluster.
# ===========================================================================
if [ "$MODE" = "desk" ]; then
  echo "== 11 §A desk check: reproduce BUG #2, confirm the fix, run the tests"
  echo
  python3 "$HERE/mvp_bind_mount.py" 2>&1 | tee "$OUT/mvp_bind_mount.observed.txt"
  rc=${PIPESTATUS[0]}
  echo
  echo '== section 4 needs the `infera` package importable. If it said SKIPPED:'
  echo "     PYTHONPATH=<infera repo> bash scripts/run.sh"
  echo
  echo "== regression tests (install to tests/unit/kvd/ in the repo)"
  if python3 -c "import pytest" 2>/dev/null; then
    # The test module imports infera.kvd.storage_classify. If infera is not on
    # the path, stage the packed post-fix copy into a throwaway package so the
    # suite can run anywhere.
    if python3 -c "import infera.kvd.storage_classify" 2>/dev/null; then
      PP=""
    else
      TMP=$(mktemp -d); mkdir -p "$TMP/infera/kvd"
      : > "$TMP/infera/__init__.py"; : > "$TMP/infera/kvd/__init__.py"
      cp "$HERE/storage_classify_fixed.py" "$TMP/infera/kvd/storage_classify.py"
      PP="$TMP"
      echo "   (infera not importable — staging scripts/storage_classify_fixed.py in $TMP)"
    fi
    PYTHONPATH="${PP:-${PYTHONPATH:-}}" python3 -m pytest "$HERE/test_storage_classify.py" -q 2>&1 | tail -5
    echo "   expected: 47 passed  (2 of them are the new bind-mount cases)"
    [ -n "${TMP:-}" ] && rm -rf "$TMP"
  else
    echo "   pytest not installed — skipping. The 47 tests are in scripts/test_storage_classify.py"
  fi
  echo
  echo "== committed reference: results/step5_nvme_l3.txt, results/patch0002_note.md"
  exit "$rc"
fi

if [ "$MODE" != "node" ]; then echo "MODE must be desk|node"; exit 2; fi

# ===========================================================================
# §B — on a node, with a real bind mount.
# ===========================================================================
J(){ ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 "$JUMP" \
      "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 $NODE '$1'" 2>&1 \
      | grep -v "^Warning: Permanently"; }

cleanup(){ echo "[cleanup] removing $CTR"; J "docker rm -f $CTR >/dev/null 2>&1"; }
trap cleanup EXIT

echo "== 11 §B on $NODE: bind-mount $HOST_L3 -> $CTR_L3 and classify it"

# ---------------------------------------------------------------------------
# PREFLIGHT — and a hard hygiene rule.
# ---------------------------------------------------------------------------
echo
echo "===== PREFLIGHT ====="
img=$(J "docker image inspect $IMAGE --format '{{.Id}}' 2>/dev/null | head -c 24")
[ -n "$img" ] && echo "  [ok]   image present" || { echo "  [FAIL] $NODE missing $IMAGE"; exit 1; }

exists=$(J "test -d $HOST_L3 && echo yes")
if [ "$exists" != "yes" ]; then
  echo "  [info] $HOST_L3 does not exist — creating it"
  J "mkdir -p $HOST_L3 && echo created"
fi

# Which device backs it, and is that device SOMEONE ELSE'S?
DEV=$(J "findmnt -no SOURCE -T $HOST_L3" | sed 's/\[.*//' | tr -d ' ')
echo "  [info] $HOST_L3 is backed by: ${DEV:-<unknown>}"
echo
echo "  SHARED-CLUSTER HYGIENE: this script mounts NOTHING and formats NOTHING."
echo "  It only classifies a path that is already mounted. On the reference node"
echo "  the eight 7 TB NVMe drives are UNMOUNTED and /dev/nvme0n1 already holds"
echo "  another team's kvd-long/kvd-short (120 GB) — deliberately untouched."
echo "  If you want a genuine NVMe measurement, get owner sign-off first."

# ---------------------------------------------------------------------------
# STORAGE SURVEY — this is why the verdict is what it is
# ---------------------------------------------------------------------------
echo
echo "===== STORAGE SURVEY (host) ====="
{
  echo "# Experiment 11 §B — observed on $NODE, $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  echo "--- what backs $HOST_L3 ---"
  J "findmnt -no SOURCE,FSTYPE,SIZE,AVAIL -T $HOST_L3"
  echo
  echo "--- the device tree under it (this is what decides the io-mode verdict) ---"
  J "lsblk -no NAME,TRAN,ROTA,SIZE ${DEV:-/dev/md0} 2>&1"
  echo "# An md-raid node has NO TRAN of its own; resolving transport means"
  echo "# recursing to the members. If the members are SATA, 'buffered' is the"
  echo "# CORRECT verdict — the classifier prefers it for the readahead win."
  echo
  echo "--- NVMe drives on this box, and their mount state ---"
  J "lsblk -no NAME,TRAN,ROTA,SIZE,MOUNTPOINT | grep -E 'nvme|NAME' 2>&1 | head -20"
  echo "# On the reference node these are all UNMOUNTED, which is why true NVMe"
  echo "# O_DIRECT remains untested. That is a resource constraint, not a finding."
} | tee "$OUT/storage_survey.observed.txt"

# ---------------------------------------------------------------------------
# CONTAINER — bind-mount the PATH *and* pass the DEVICE through
# ---------------------------------------------------------------------------
echo
echo "===== container with the bind mount AND the device ====="
# --device is the half everyone forgets. Without it lsblk inside the container
# says 'not a block device' no matter how good the parser is.
DEVFLAG=""
[ -n "$DEV" ] && DEVFLAG="--device=$DEV"
J "docker rm -f $CTR >/dev/null 2>&1; docker run -d --name $CTR --network=host \
   -v $HOST_L3:$CTR_L3 $DEVFLAG --entrypoint '' $IMAGE sleep infinity >/dev/null && echo ok" \
  | grep -q ok || { echo "FATAL: container start failed (is $DEV a valid device?)"; exit 1; }

echo "== can the container see the device node?"
seen=$(J "docker exec $CTR bash -c 'ls -l ${DEV:-/dev/md0} 2>&1'")
echo "   $seen"
case "$seen" in
  *"No such file"*)
    echo
    echo "   !! THE TRAP: the container cannot see ${DEV:-the device}."
    echo "      Bind-mounting the PATH does not expose the DEVICE. lsblk will say"
    echo "      'not a block device' (rc=32) and the classifier will fall back to"
    echo "      buffered — NO MATTER what the parser does. Pass --device=$DEV"
    echo "      (or --privileged), or create the node by hand (mknod)."
    echo "      The A/B below will still run; read it knowing this."
    ;;
esac

# ---------------------------------------------------------------------------
# THE A/B — same path, same moment, only the patch hunk differs
# ---------------------------------------------------------------------------
echo
echo "===== A/B: classifier WITHOUT the fix vs WITH it ====="
ssh -o StrictHostKeyChecking=no "$JUMP" \
  "ssh -o StrictHostKeyChecking=no $NODE 'cat > /tmp/abtest.sh'" < "$HERE/abtest.sh"
J "docker cp /tmp/abtest.sh $CTR:/abtest.sh >/dev/null"

# The image may predate the fix; abtest.sh asserts the marker is present, so
# install the packed post-fix copy first if it is missing.
has=$(J "docker exec $CTR bash -c \"grep -c 'bracket = source.find' /opt/infera/infera/kvd/storage_classify.py 2>/dev/null\"" | tr -dc 0-9)
if [ "${has:-0}" = "0" ]; then
  echo "  (image predates the fix — installing scripts/storage_classify_fixed.py)"
  ssh -o StrictHostKeyChecking=no "$JUMP" \
    "ssh -o StrictHostKeyChecking=no $NODE 'cat > /tmp/sc_fixed.py'" < "$HERE/storage_classify_fixed.py"
  J "docker cp /tmp/sc_fixed.py $CTR:/opt/infera/infera/kvd/storage_classify.py >/dev/null && echo installed"
fi

{
  echo
  echo "# --- A/B, $(date -u +%Y-%m-%dT%H:%M:%SZ) ---"
  J "docker exec $CTR bash /abtest.sh 2>&1"
  echo
  echo "--- full classifier output, with the fix in place ---"
  J "docker exec $CTR python3 -m infera.kvd classify $CTR_L3 2>&1"
} | tee "$OUT/abtest.observed.txt"

# ---------------------------------------------------------------------------
# VERDICT
# ---------------------------------------------------------------------------
echo
echo "===== VERDICT ====="
obs="$OUT/abtest.observed.txt"
if grep -q 'devices *= *\[(none)\]' "$obs" && grep -qE 'devices *= *\[[a-z]' "$obs"; then
  echo "  ==> A/B WORKED: devices went [(none)] -> resolved. Patch 0002 confirmed."
elif grep -qc 'devices *= *\[(none)\]' "$obs" >/dev/null && ! grep -qE 'devices *= *\[[a-z]' "$obs"; then
  echo "  ==> BOTH arms show [(none)]. Almost certainly the DEVICE is not visible"
  echo "      inside the container (see the trap above), not a parser problem."
  echo "      Re-run with --device passed through."
else
  echo "  ==> Inspect $obs by hand and compare with results/step5_nvme_l3.txt."
fi
cat <<'EOF'

  Reference A/B on the node (chi2867, /mnt/nvme-raid/kvd-long -> /kvd-long):

    WITHOUT fix:  mount    = /dev/md0[/mnt/nvme-raid/kvd-long] (ext4)
                  devices  = [(none)]
                  rationale: unknown device, conservative buffered
                  WARN: lsblk returned no devices for source='/dev/md0[/mnt/...]'
    WITH fix:     mount    = /dev/md0 (ext4)
                  devices  = [md0 (?, ssd)]
                  rationale: unknown transport '' (md0), conservative buffered

  Device resolution fixed; the WARN is gone.

  TWO THINGS THE FIX DOES NOT SETTLE, and both are legitimate:

  (a) The verdict is STILL 'buffered', and that is CORRECT here. md0 is a raid1
      of two SATA SSDs, and the classifier deliberately prefers buffered for
      SATA (cold-read readahead win). md0 also has no TRAN of its own — an
      md-raid node's transport means recursing to sda2/sdb2 — so it reports
      "unknown transport '' (md0)" and stays conservative. Patch 0002 fixes
      DEVICE RESOLUTION, not the io-mode verdict on this hardware.

  (b) A stock container cannot see /dev/md0 at all. Bind-mounting the path is
      not enough; you also need --device=/dev/md0. For the reference A/B the
      node was created by hand (mknod /dev/md0 b 9 0). That is a
      DEPLOYMENT-RECIPE trap, not an infera defect — and everyone will hit it.

  STILL UNTESTED: O_DIRECT against a genuine NVMe device. The 8 real NVMe
  drives on the reference node are unmounted and one already holds another
  team's 120 GB kvd store. Untouched on purpose. That is a resource
  constraint, not a finding.
EOF
