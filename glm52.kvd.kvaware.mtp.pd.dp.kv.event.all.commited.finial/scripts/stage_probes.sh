#!/usr/bin/env bash
# Copy the leg launcher and the probes into a running container. Runs ON a node.
#
# The probes must come from THIS kit, not from the predecessor kit's on-node
# staging dir: theirs send `temperature: 0`, which manufactures the corruption
# signature under MTP (see ../notes.md §1). glm52_leg.sh is unchanged between
# the two and is taken from wherever it is available.
#
#   bash stage_probes.sh              # probes from this script's directory
#   PROBE_DIR=/tmp bash stage_probes.sh
set -eu
CTR="${CTR:-merged_run}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBE_DIR="${PROBE_DIR:-$HERE}"
KIT="${KIT:-/mnt/vast/c_huggingface/merge_20260731}"

leg="$PROBE_DIR/glm52_leg.sh"
[ -f "$leg" ] || leg="$KIT/scripts/glm52_leg.sh"
[ -f "$leg" ] || { echo "glm52_leg.sh not found in $PROBE_DIR or $KIT/scripts" >&2; exit 1; }
docker cp "$leg" "$CTR":/glm52_leg.sh >/dev/null
echo "  glm52_leg.sh <- $leg"

for f in probe.py prefix_reuse.py needle.py stress_capture.py; do
  src="$PROBE_DIR/$f"
  [ -f "$src" ] || { echo "  MISSING $src" >&2; exit 1; }
  docker cp "$src" "$CTR":/tmp/"$f" >/dev/null
  echo "  /tmp/$f <- $src"
done

# Assert the sampling fix is in the copies that will actually run. A probe with
# temperature 0 silently invalidates G2 and the stress gate.
for f in needle.py stress_capture.py; do
  n=$(docker exec "$CTR" grep -c "TEMPERATURE" /tmp/"$f" || true)
  [ "${n:-0}" -gt 0 ] || { echo "  $f in the container is the GREEDY version -- see notes.md §1" >&2; exit 1; }
done
echo "  sampling fix present in needle.py and stress_capture.py  OK"
