#!/usr/bin/env bash
# Rank the kernels in a captured trace with Magpie.
#
# Runs on the LOGIN NODE and drives Magpie on the compute node. Magpie needs
# PyYAML and numpy and the login node is not guaranteed to have them, while the
# compute node was checked and does; both hosts see the same NFS home, so the
# staged input and the handoff output are reachable from either side.
#
# Equivalent to examples/sglang_1p1d_glm5.2/run_megapie_kernel_analyze.sh, with
# the output directory separated from the trace directory. That separation is not
# cosmetic: the reference script writes into the trace directory, which works only
# when the caller owns it, and a directory the engine container's root created
# on a bind mount is not owned by the user running the analysis.
set -uo pipefail

PKG="${AGENT_SYS_TASK_PACKAGE:?}"
OUT="${E2E_OUTPUT_KERNEL_TABLE:?}"
IN="${E2E_INPUT_TORCH_TRACE:?}"

. "$PKG/assets/lib/remote.sh"

ANALYZE="$PKG/assets/analyze"
WORKDIR="$(pwd)/scan"
rm -rf "$WORKDIR"; mkdir -p "$WORKDIR"

say() { printf '[kernel_scan] %s\n' "$*"; }

TRACES="$IN/items/result/traces"
STACKS="$IN/items/result/stacks"
say "traces=$TRACES"
say "stacks=$STACKS"

# ---- 1. preconditions -------------------------------------------------------
require_visible_on_node "$ANALYZE/megapie.sh" "staged task package" || exit 1
if ! on "test -d '$TRACES'" >/dev/null 2>&1; then
  say "ABORT: the staged trace directory is not visible on $E2E_NODE: $TRACES"
  exit 1
fi
n=$(on "ls '$TRACES'/*.trace.json.gz 2>/dev/null | wc -l" | tr -d ' \r\n')
say "$n trace file(s) staged"
[ "${n:-0}" -gt 0 ] || { say "ABORT: no traces to analyse"; exit 1; }

# ---- 2. Magpie ---------------------------------------------------------------
# Minutes, not seconds: 3m43s for eight ranks of a 15 s window, because every
# event in every rank is parsed.
MEGAPIE_OUT="$WORKDIR/megapie"
say "running Magpie (measured 3m43s for 8 ranks)"
on "MAGPIE_ROOT='${E2E_MAGPIE_ROOT:?}' bash '$ANALYZE/megapie.sh' '$TRACES' '$MEGAPIE_OUT'" \
  2>&1 | tee "$WORKDIR/megapie.log"
rc="${PIPESTATUS[0]}"
CSV="$MEGAPIE_OUT/gap_analysis/gap_analysis.csv"
if [ "$rc" != "0" ] || [ ! -s "$CSV" ]; then
  say "Magpie failed (rc=$rc). Last 30 lines:"
  tail -30 "$WORKDIR/megapie.log" >&2
  exit 1
fi
say "gap analysis -> $CSV"

# ---- 3. the handoff ----------------------------------------------------------
say "assembling $OUT"
ITEMS="$OUT/items"
rm -rf "$ITEMS"; mkdir -p "$ITEMS/result/gap_analysis" "$ITEMS/env" "$ITEMS/logs"

cp "$CSV" "$ITEMS/result/gap_analysis/"
python3 "$ANALYZE/top_kernels.py" "$CSV" "$ITEMS/result/top_kernels.json" \
  "${E2E_TOP_N:-25}" || { say "ABORT: could not summarise the gap analysis"; exit 1; }

# ---- 3b. which Python frame launched each ranked kernel ----------------------
# Runs on the NODE, like Magpie and for the same two reasons: the stack traces
# are large and the node is where the streaming passes should happen.
#
# **Not fatal when it resolves nothing.** A ranking with no launcher block is
# the state this package shipped in before the stack window existed, and it is
# still a usable ranking -- the consumer falls back to searching for the symbol
# name. What must not happen is the difference being invisible, which is why the
# outcome is written to `items/result/launchers.json` either way, including the
# reason when the answer is "there were no stack traces".
LAUNCHERS="$ITEMS/result/launchers.json"
if on "test -d '$STACKS'" >/dev/null 2>&1; then
  n_stacks=$(on "ls '$STACKS'/*.trace.json.gz 2>/dev/null | wc -l" | tr -d ' \r\n')
  say "${n_stacks:-0} stack trace file(s) staged; resolving launcher frames"
else
  n_stacks=0
  say "no stack window in this handoff; the ranking will carry no launcher frames"
fi

if [ "${n_stacks:-0}" -gt 0 ]; then
  on "python3 '$ANALYZE/launchers.py' --csv '$CSV' --stacks '$STACKS' \
      --out '$LAUNCHERS' --top-n '${E2E_LAUNCHER_TOP_N:-50}' \
      --max-files '${E2E_STACK_RANKS:-2}'" 2>&1 | tee "$WORKDIR/launchers.log"
  if [ "${PIPESTATUS[0]}" != "0" ] || [ ! -s "$LAUNCHERS" ]; then
    say "WARN: launcher resolution failed; continuing without launcher frames"
    tail -20 "$WORKDIR/launchers.log" >&2
    rm -f "$LAUNCHERS"
  fi
else
  python3 - "$LAUNCHERS" <<'PYEOF'
import json, sys
json.dump(
    {
        "available": False,
        "reason": "the torch_trace handoff carries no stack window",
        "wanted": 0,
        "resolved": 0,
        "launchers": {},
        "unmapped": [],
        "notes": [],
    },
    open(sys.argv[1], "w"),
    indent=2,
)
PYEOF
fi

# ---- 3c. the whole table, in the shape the next stage reads ------------------
# `top_kernels.json` is the head and is what this package's validator judges;
# this is every row, with the launcher frames merged in, under the consumer's own
# field names. See `kernel_doc.py`.
DOC_ARGS=(--csv "$CSV" --out "$ITEMS/result/text.json"
          --trace-manifest "$IN/items/result/manifest.json")
[ -s "$LAUNCHERS" ] && DOC_ARGS+=(--launchers "$LAUNCHERS")
python3 "$ANALYZE/kernel_doc.py" "${DOC_ARGS[@]}" \
  || { say "ABORT: could not build the kernel document"; exit 1; }

# Which trace this describes, by digest rather than by path -- the manifest the
# capture published is the join, and copying its totals here means the ranking
# and the traces it came from cannot drift apart unnoticed.
cp "$IN/items/result/manifest.json" "$ITEMS/env/trace_manifest.json" 2>/dev/null || true
cp "$IN/items/env/engine_argv.txt" "$ITEMS/env/" 2>/dev/null || true

on "cd '${E2E_MAGPIE_ROOT}' && git rev-parse HEAD 2>/dev/null || echo unknown" \
  > "$ITEMS/env/magpie_commit.txt" 2>&1

gzip -9 -c "$WORKDIR/megapie.log" > "$ITEMS/logs/megapie.log.gz"

cat > "$ITEMS/command" <<'EOF'
#!/usr/bin/env bash
# Reproduce this ranking. Executable because agent.gate requires it of a
# 'command' item; written with shell variables so it names no absolute path.
# (The seal does not check this -- redact.py does -- but a kit naming one
# machine's paths is a bad kit whether or not anything refuses it.)
set -eu
: "${MAGPIE_ROOT:?export MAGPIE_ROOT=<a checkout of Magpie>}"
: "${TRACES:?export TRACES=<directory of *.trace.json.gz>}"
: "${OUT:?export OUT=<where to write gap_analysis/>}"
: "${SCRIPTS:?export SCRIPTS=<the package's assets/analyze directory>}"

MAGPIE_ROOT="$MAGPIE_ROOT" bash "$SCRIPTS/megapie.sh" "$TRACES" "$OUT"
EOF
chmod +x "$ITEMS/command"

cat > "$ITEMS/watchout" <<'EOF'
This ranking is over a MIX deployment, so prefill and decode kernels are pooled.
A kernel's share here is its share of the whole engine under this load, not of
either phase; separating them needs a PD-disaggregated capture.

The engine ran with decode CUDA graphs off. That is what makes each kernel
appear individually -- with graphs on the profiler records one launch per step --
and it also means the absolute times are not the times a production deployment
would spend. Read the shares, not the seconds.

The Input Shapes column is populated only because the capture asked the profiler
for record_shapes. A trace taken without it produces the same CSV with that
column empty, and nothing downstream would notice until somebody tried to build a
roofline from it.

A kernels[].launcher block, where present, names the Python frame that launched
that kernel. Two things about it are not obvious.

It is the FRAMEWORK-LEVEL call site, not the kernel's own source. For a TileLang
or Triton kernel there is no kernel source file to name -- the device symbol is a
compilation artefact, which is why main_kernel is one name for every kernel
TileLang generates. The frame is the editable entry point that produced it, and
the JIT dispatch layers in between (triton/runtime, aiter/jit, flydsl/compiler,
tilelang/jit) are deliberately walked through rather than reported.

source_file is relative and path_form says to what. torch strips the longest
matching sys.path entry from a frame path, so the same capture yields both forms:
container_absolute means source_file is exactly relative to container_root, while
sys_path_relative means torch stripped an entry this package cannot identify and
the consumer has to bind the path to a file under the checkout it indexed. sglang
is installed editable in this image, so its frames arrive absolute and resolve
exactly; aiter's arrive relative because /sgl-workspace/aiter is on sys.path.
EOF

TOTAL_US=$(python3 -c "
import json,sys
d=json.load(open('$ITEMS/result/top_kernels.json'))
print(d.get('totals',{}).get('self_cuda_us',0))
" 2>/dev/null || echo 0)

cat > "$OUT/README.md" <<EOF
# kernel_table

## Purpose

Every GPU kernel in the captured window, ranked by the CUDA time it owns. This is
the input to the next stage of the pipeline: the list of operators worth
optimising is chosen from here.

## How to run

\`items/command\` reproduces the ranking from a directory of trace files; export
\`MAGPIE_ROOT\`, \`TRACES\`, \`OUT\` and \`SCRIPTS\` first.

## Result

\`items/result/gap_analysis/gap_analysis.csv\` is Magpie's own export:
\`Name, Calls, Self CUDA total (us), Avg time (us), % Total, Input Shapes\`.
\`items/result/top_kernels.json\` is the head of it as structured data, with the
totals, so a consumer does not have to parse the CSV. Total self CUDA time across
all ranks: ${TOTAL_US} us.

\`items/result/text.json\` is **every** row as structured data, and it is what the
next stage reads. The head is a judgement about this round; the operator-selection
stage classifies every row into a bucket before it sorts, so handing it only the
head would discard most of the routable candidates along with the noise.

Each kernel in it may carry a \`launcher\` block — \`source_file\`, \`line\`,
\`function\`, \`sample_count\`, \`launch_api\`, plus \`container_root\`, \`owner\`
and \`path_form\` — naming the Python frame that launched it. That is the evidence
that answers "which file do I edit" for a symbol like \`main_kernel\`, which no
amount of searching for the name can. \`items/result/launchers.json\` is the
resolution in full, including what it could not place and why.

## Environment

\`items/env/trace_manifest.json\` is the capture's own manifest, carried over so
this ranking and the traces behind it can be matched by digest.
\`items/env/engine_argv.txt\` is the engine command line that produced them, and
\`magpie_commit.txt\` the analyser's revision.

## Watch out

See \`items/watchout\`: shares rather than seconds, prefill and decode pooled, and
the Input Shapes column exists only because the capture asked for it.
EOF

# ---- 4. make it publishable --------------------------------------------------
say "redacting site-specific paths"
python3 "$PKG/assets/lib/redact.py" "$OUT" \
  "MAGPIE_ROOT=${E2E_MAGPIE_ROOT}" \
  "WORK_ROOT=${E2E_WORK_ROOT:?}" \
  "TASK_PACKAGE=$PKG" \
  "TMPDIR=/tmp" \
  "HOME=$HOME" || { say "ABORT: evidence still names local paths redact.py could not place"; exit 1; }

say "done"
exit 0
