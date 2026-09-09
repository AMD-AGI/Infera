#!/usr/bin/env bash
# Kernel-level analysis of a captured trace directory. RUNS ON THE COMPUTE NODE.
# Equivalent to examples/sglang_1p1d_glm5.2/run_megapie_kernel_analyze.sh, with
# the trace directory passed in rather than assumed.
#
# Output: $TRACE_DIR/megapie/gap_analysis/gap_analysis.csv, whose columns are
#   Name, Calls, Self CUDA total (us), Avg time (us), % Total, Input Shapes
# The Input Shapes column only has content because capture.sh asked the profiler
# for record_shapes=true.
set -u
MAGPIE_ROOT="${MAGPIE_ROOT:?}"
TRACE_DIR="${1:?usage: 08_megapie.sh <dir holding *.trace.json.gz> [outdir]}"
# Separable from TRACE_DIR on purpose. run_megapie_kernel_analyze.sh writes into
# the trace directory, which works only when that directory is writable by the
# caller. Here the traces can land in a directory the engine container's root
# created, so the analysis needs somewhere of its own to go.
MEGAPIE_OUT="${2:-${MEGAPIE_OUT:-$TRACE_DIR/megapie}}"

[ -d "$TRACE_DIR" ] || { echo "no such directory: $TRACE_DIR"; exit 1; }
n=$(ls "$TRACE_DIR"/*.trace.json.gz 2>/dev/null | wc -l)
[ "$n" -gt 0 ] || { echo "no *.trace.json.gz under $TRACE_DIR"; exit 1; }

echo "[magpie] host=$(hostname -s) traces=$n"
echo "[magpie] trace-dir=$TRACE_DIR"
echo "[magpie] out=$MEGAPIE_OUT"
echo "[magpie] started $(date -Is)"

mkdir -p "$MEGAPIE_OUT" || { echo "cannot create $MEGAPIE_OUT"; exit 1; }
cd "$MAGPIE_ROOT"
python3 -m Magpie benchmark \
  --trace-dir "$TRACE_DIR" \
  --categories kernel \
  --top-k 10000 \
  --no-rank-csv \
  -o "$MEGAPIE_OUT"
rc=$?
echo "[magpie] finished $(date -Is) rc=$rc"

CSV="$MEGAPIE_OUT/gap_analysis/gap_analysis.csv"
if [ -s "$CSV" ]; then
  echo "--- $CSV ---"
  wc -l "$CSV"
  head -6 "$CSV"
  echo "MEGAPIE_OK $CSV"
else
  echo "MEGAPIE_FAIL: $CSV missing or empty"
  find "$MEGAPIE_OUT" -type f 2>/dev/null | head
  exit 1
fi
