#!/usr/bin/env bash
# Launch the chain from the vars recorded in LAUNCH-CHAIN.md.
# bash, not zsh: `mapfile -d ''` is bash. In zsh the fallback `read -r -d ''`
# has different semantics and SILENTLY DROPPED ONE of 30 --var pairs when this
# was attempted at the prompt. Counted, not assumed: the count is asserted below
# and the script refuses if it does not match.
set -uo pipefail
M=/data/yihou/e2e_verify_20260906/m2
awk '/^```sh$/{f=1;next} /^```$/{f=0} f' "$M/LAUNCH-CHAIN.md" > /tmp/yihou_lb.txt
python3 - <<'PY'
import shlex
txt = open('/tmp/yihou_lb.txt').read().replace('\\\n', ' ')
toks = shlex.split(txt.split('launch_chain.py', 1)[1])
pairs = [toks[i+1] for i, t in enumerate(toks) if t == '--var']
open('/tmp/yihou_lv.txt', 'w').write('\0'.join(pairs))
print(f"document pairs: {len(pairs)}")
PY
mapfile -d '' PAIRS < /tmp/yihou_lv.txt
ARGS=(); for v in "${PAIRS[@]}"; do ARGS+=(--var "$v"); done
N=$(( ${#ARGS[@]} / 2 ))
echo "argv --var flags: $N"
[ "$N" = "${EXPECT_VARS:-34}" ] || { echo "ABORT: expected ${EXPECT_VARS:-34} --var flags, argv carries $N" >&2; exit 1; }

# **Empty-default variables absent from the line.** A `${name:-}` whose consumer
# aborts on empty is invisible to every other check here: `show` accepts it, the
# count guard counts what IS there, and the failure surfaces an hour later as a
# validator refusing an artefact that was never produced. Run 8 lost m4 to
# exactly this -- `HIP_VISIBLE_DEVICES: '${gpu:-}'` (m4_kernel_opt.yaml:359) with
# `run_in_container.sh:136` aborting on empty -- while the line carried
# `measure_gpu=4`, a different variable for a different consumer, which made the
# missing one look covered.
#
# ABORTS only on the one with a DEMONSTRATED abort; the rest are printed. A guard
# that refuses everything it cannot classify turns an unknown into a dead run,
# and tonight already produced one of those.
PKG=/home/yihou/dev/git.16-19/infera/agent_sys/examples/llm_e2e_performance_optimization/e2e-flow
if [ -d "$PKG/steps" ]; then
  grep -rhoE '\$\{([a-z_0-9]+):-\}' "$PKG"/steps/*.yaml "$PKG"/shared.yaml 2>/dev/null \
    | sed 's/[${}]//g; s/:-//' | sort -u > /tmp/yihou_empty_defaults.txt
  printf '%s\n' "${PAIRS[@]}" | cut -d= -f1 | sort -u > /tmp/yihou_line_vars.txt
  MISSING=$(comm -23 /tmp/yihou_empty_defaults.txt /tmp/yihou_line_vars.txt | tr '\n' ' ')
  echo "empty-default vars not on the line: ${MISSING:-none}"
  case " $MISSING " in
    *" gpu "*) echo "ABORT: --var gpu is missing. HIP_VISIBLE_DEVICES='\${gpu:-}' and" >&2
               echo "  optimize_kernel/steps/run_in_container.sh:136 exits 1 on empty, so m4" >&2
               echo "  measures nothing and its validators refuse an artefact that was never" >&2
               echo "  produced. Run 8, 22:04:08. measure_gpu is a DIFFERENT variable." >&2
               exit 1 ;;
  esac
fi

MODE="${1:-show}"; shift || true

# **`--timeout 21600` is asserted, not hoped for.** It was `${TIMEOUT_ARG:-}` --
# optional, silently empty -- and on 2026-09-06 I launched a real run having
# dropped it, which nothing caught: the --var count guard says nothing about
# flags that are not --var. RUN-PLAN.md §2 is the only place --timeout appears,
# so an absent one is invisible against every other check we run.
if [ "$MODE" = "run" ]; then
  # **The jobid must name a RUNNING job, not merely be non-empty.** On
  # 2026-09-06 hold 29184 ended at 14:00:01 and 29313 took the same node 33
  # seconds later; a launch went out still carrying 29184, which is sealed into
  # every artefact as runtime.slurm_jobid. Nothing caught it: `show` sees a
  # valid string, and `_agree_or_die` compares the ambient value against the
  # record -- both from the same --var -- so a stale id AGREES WITH ITSELF and
  # the guard passes. A guarded field can be uniformly wrong.
  JID=$(printf '%s\n' "${PAIRS[@]}" | sed -n 's/^jobid=//p')
  [ -n "$JID" ] || { echo "ABORT: no --var jobid= in the document" >&2; exit 1; }
  if ! squeue -h -j "$JID" -o '%T' 2>/dev/null | grep -qx RUNNING; then
    echo "ABORT: jobid $JID is not a RUNNING job." >&2
    echo "  squeue says: $(squeue -h -j "$JID" -o '%T' 2>&1 | head -1)" >&2
    echo "  Live jobs:   $(squeue -h -u "$USER" -o '%i %T' | tr '\n' ' ')" >&2
    exit 1
  fi
  echo "jobid $JID: RUNNING"

  case "${TIMEOUT_ARG:-}" in
    *--timeout*) ;;
    *) echo "ABORT: a real run needs --timeout; set TIMEOUT_ARG=\"--timeout 21600\"" >&2
       echo "  (RUN-PLAN.md section 2 is the only place it appears; the --var" >&2
       echo "   count guard cannot see it)" >&2
       exit 1 ;;
  esac
  EXTRA="${TIMEOUT_ARG}"   # run-only: `show` rejects --timeout
fi

# **The stall threshold must not equal any downstream timeout.** AIPerf's request
# timeout is 900 s and this was 900 s: on 2026-09-06 the engine stopped generating
# at 18:35:24 and every 900 s AIPerf emitted a burst of timeouts that wrote into
# the run tree seconds after the stall deadline, re-satisfying the detector.
# Bursts measured at 18:50:24 and 19:05:25, 32 each. The run would have looked
# alive for about four more hours. Any value that is not 900 breaks the alias;
# 1200 is not claimed to be optimal.
STALL="${STALL_AFTER:-1200}"
case "$STALL" in
  ''|*[!0-9]*) echo "ABORT: STALL_AFTER must be an integer, got '$STALL'" >&2; exit 1 ;;
  900) echo "ABORT: stall 900 aliases AIPerf's 900 s request timeout; pick another" >&2; exit 1 ;;
esac
echo "stall_after: $STALL"

exec setsid env -u PYTHONPATH \
  CLAUDE_CONFIG_DIR=/data/yihou/e2e_verify_20260906/m1/claude-config \
  python3 "$M/launch_chain.py" --stall-after "$STALL" "$MODE" \
  --package agent_sys/examples/llm_e2e_performance_optimization/e2e-flow \
  --demo-root /data/yihou/agent_sys_runroot \
  ${EXTRA:-} "${ARGS[@]}"
