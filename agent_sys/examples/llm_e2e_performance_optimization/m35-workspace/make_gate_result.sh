#!/usr/bin/env bash
# Produce M3-GATE-RESULT.md from a real workset, in one command.
#
#     bash make_gate_result.sh <run dir> <operator_workset content dir>
#
# **Why a generator and not a document I write when the moment comes.** The hold
# may end mid-sentence, and the gate answer is the last substantive result this
# hold can produce. Composing it under time pressure is how the launch line, the
# run id, or the "what this does NOT establish" section gets left out — and those
# are the parts that make it survive being read by somebody who was not here.
#
# It records the launch line from `/proc/<pid>/cmdline` rather than from any
# document: the document says what was intended, the process says what it got.
set -uo pipefail

RUN="${1:?usage: make_gate_result.sh <run dir> <workset content dir> [orchestrator pid]}"
WS="${2:?usage: make_gate_result.sh <run dir> <workset content dir> [orchestrator pid]}"
PID_ARG="${3:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/M3-GATE-RESULT.md"

[ -d "$RUN" ] || { echo "no such run dir: $RUN" >&2; exit 2; }
[ -d "$WS" ]  || { echo "no such workset dir: $WS" >&2; exit 2; }

NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)
HEAD_SHA=$(git -C /home/yihou/dev/git.16-19/infera rev-parse --short HEAD 2>/dev/null || echo unknown)

# **The orchestrator, and how it was associated with this run is recorded.**
#
# Measured 2026-09-06: the run id is NOT on the orchestrator's command line — it
# carries `--demo-root <parent>` and the framework names the run directory
# itself. So matching the run basename can never succeed, and inferring from
# `--demo-root` alone is an association rather than an identity: sequential runs
# share a demo-root. **Pass the pid explicitly when you have it** (the leader
# names it at every launch); the fallback is recorded as a fallback.
ORCH=""; ORCH_HOW=""
if [ -n "$PID_ARG" ] && [ -r "/proc/$PID_ARG/cmdline" ]; then
  ORCH="$PID_ARG"; ORCH_HOW="supplied explicitly"
else
  parent=$(dirname "$RUN")
  for p in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
    [ -r "/proc/$p/cmdline" ] || continue
    a=$(tr '\0' ' ' < "/proc/$p/cmdline") || continue
    case "$a" in
      *launch_chain*|*agent_sys.cli.main*)
        case "$a" in *"--demo-root $parent"*) ORCH="$p"; ORCH_HOW="inferred from --demo-root $parent (an association, not an identity: sequential runs share it)";; esac ;;
    esac
  done
fi

EXTRACT=$(bash -c "python3 '$HERE/m3_extract.py' '$WS' 2>&1"); EXRC=$?

{
  echo "# m3 gate result — the decision between the two m4 routes"
  echo
  echo "> ## READ THIS FIRST — is this a RESULT or a TEMPLATE?"
  echo ">"
  echo "> This file is generated. **Check the \`workset\` line under Provenance:**"
  echo ">"
  echo "> - a path under a **run directory** (\`.../runs/<id>/handoffs/...\`) → this is a"
  echo ">   **real result** from that run;"
  echo "> - a path under **\`known/\`** → this is a **known-answer FIXTURE**, not a result."
  echo ">   It says what the tool *would* report. **No real workset was read.**"
  echo ">"
  echo "> **As of 2026-09-06 the gate had never fired on a real workset.** Six chain runs"
  echo "> on this cluster; none reached m3. Every copy of this file produced that day is a"
  echo "> fixture rendering. **Do not quote it as a finding about any operator.**"
  echo
  echo "**Generated $NOW** by \`make_gate_result.sh\`, repo HEAD \`$HEAD_SHA\`."
  echo "Author: m35. **This is the artefact; any message about it is a summary of this.**"
  echo
  echo "## Provenance"
  echo
  echo '```'
  echo "run dir     : $RUN"
  echo "workset     : $WS"
  echo "orchestrator: ${ORCH:-<not found: the run may be terminal>}"
  echo "  how       : ${ORCH_HOW:-n/a}"
  echo '```'
  echo
  echo "**The launch line, read from the process rather than from any document**"
  echo "(a record says what was intended; the cmdline says what this run got):"
  echo
  echo '```'
  if [ -n "$ORCH" ]; then
    tr '\0' '\n' < "/proc/$ORCH/cmdline" | paste -sd' ' - | tr ' ' '\n' | grep -A1 -- '--var' | grep -vE '^--var$|^--$' | sort
  else
    echo "(orchestrator not running; launch line unrecoverable from a terminal run —"
    echo " the staged package keeps \${var:-default} unrendered. See RUNG5-CHECKLIST P6.)"
  fi
  echo '```'
  echo
  echo "## The extractor, verbatim"
  echo
  echo '```'
  printf '%s\n' "$EXTRACT"
  echo "exit status: $EXRC   (0 = usable, 1 = STOP, 2 = bad path)"
  echo '```'
  echo
  echo "## The decision this forces"
  echo
  if [ "$EXRC" -eq 0 ]; then
    echo "**Exit 0 — no stop condition.** The operator is \`module_symbol\`, it declares a"
    echo "\`public_symbol\`, \`target_files\` and an \`entry_function\`, and no \`build_step\`."
    echo
    echo "**Read the \`baseline\` line above, because it decides the route:**"
    echo
    echo "- **module-shaped** (defines the public symbol *and* \`run\`) → **m2's single line"
    echo "  with \`forge_mock=1\` survives.** \`30_run_forge.sh\` seeds from that baseline and"
    echo "  the overlay keeps the engine module's surface, so \`apply.py:828\` has nothing to"
    echo "  refuse. **My two-form split is unnecessary and should be retired.**"
    echo "- **harness-shaped** (defines \`run\` but not the public symbol) → **\`forge_mock=1\`"
    echo "  produces an overlay that drops the module's whole public surface and"
    echo "  \`apply.py:828\` refuses AFTER a bring-up.** The route is the reverse payload"
    echo "  (\`mk_reverse_payload.py --delegate-to <public_symbol>\`), which is the only"
    echo "  payload shape that has ever passed \`apply\` anywhere."
  else
    echo "**Exit $EXRC — STOP.** The reasons are listed verbatim above. **A stop is not"
    echo "automatically an m3 defect**; grade it against the three pre-registered"
    echo "explanations in \`PRE-REGISTER.md\` item D before charging it to a producer:"
    echo
    echo "1. \`stack_window_s=0\` — no launcher frames captured, so an empty"
    echo "   \`entry_function\` is *expected and declared*, visible in the launch line above;"
    echo "2. \`repos == []\` — \`identify\` had no source tree (\`E2E_SGLANG_SRC\` /"
    echo "   \`E2E_AITER_SRC\` unwired) and \`min_resolve_ratio: 0.0\` will not refuse for it;"
    echo "3. the workload's own prefix-hit profile."
  fi
  echo
  echo "## What this does NOT establish"
  echo
  echo "- **It is a read of a document, not a run of \`apply\`.** It predicts what"
  echo "  \`apply.py:828\` will do; it does not exercise it."
  echo "- **The delegation signature stays open.** The delegation is \`run(*args, **kwargs)\`"
  echo "  and restates nothing, but the public symbol's parameters must accept the"
  echo "  Definition's \`inputs\` keys. **A mismatch surfaces as an EMPTY MEASUREMENT from"
  echo "  \`check_speedup_substantiated\`, not as a signature error** — pre-registered as"
  echo "  \"the delegation did not match the Definition's inputs\", not a producer defect."
  echo "- **A weak or absent \`edit_target.entry_function\` is explanation 1, and we now"
  echo "  know exactly which resolution level went missing.** \`stack_window_s=0\` means no"
  echo "  launcher blocks, and a launcher block is what \`identify\` resolution **level 1**"
  echo "  (\`trace_python_stack\`, \`identify.py:9\`) reads. So level 1 is unavailable to this"
  echo "  run and resolution falls to **level 2, \`kernel_finder\`, which is what"
  echo "  \`magpie_root\` feeds**. On the first cluster level 1 always succeeded"
  echo "  (\`m3_analysis.yaml:122-126\`: all five operators at \`resolve_ratio: 1.0\`), **so the"
  echo "  Magpie fallback was never exercised there — and Magpie has never completed a real"
  echo "  scan on this cluster.** \`min_resolve_ratio\` defaults to \`0.0\`, so"
  echo "  \`check_identity_resolved\` **will not refuse for resolving nothing**: an"
  echo "  \`operator_identity\` is produced either way."
  echo "  **Measured, and over the whole validator set rather than a sample:** none of"
  echo "  \`check_worklist_shape\`, \`check_identity_resolved\`, \`check_workset_shape\`,"
  echo "  \`check_workset_runs\` or \`check_environment\` mentions"
  echo "  \`launcher\`/\`stacks_manifest\`/\`stack_window\` — 0 hits across 15 files. **So the"
  echo "  wall does NOT move to m3 validation; it moves m3 onto an untested route.**"
  echo "  (Question raised by the checkpoint writer; mechanism by the leader; file set"
  echo "  widened and re-measured here.)"
  echo "- **m4 cannot complete in a short hold.** \`check_workset_runs\` is \`cost: gpu_hours\`"
  echo "  and \`check_speedup_substantiated\` has a **60-minute hard ceiling** (two entrypoint"
  echo "  runs at \`timeout_seconds: 1800\`). See RUNG5-CHECKLIST P10."
} > "$OUT"

echo "wrote $OUT"
echo "extractor exit was $EXRC"
