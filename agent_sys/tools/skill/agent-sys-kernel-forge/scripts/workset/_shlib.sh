# What the shell STEPs share. POSIX `/bin/sh`; sourced, never executed.
#
# Shared interpreter selection and the real campaign invocation.

# An interpreter that exists, or a refusal that names the variable.
#
# **`PY="${KFO_PYTHON:-python3}"` is not the same as having an interpreter.**
# `KFO_PYTHON` defaults to `/opt/venv/bin/python3` in the agent spec — a path
# inside the engine container — so on a host the `:-` never fires, the variable
# is set to something absent, and the script dies:
#
#     30_run_forge.sh: 35: /opt/venv/bin/python3: not found
#
# Exit 127, from the shell, naming neither the variable nor the reason. Measured
# on the login node, in `30_run_forge.sh` and `70_selfcheck.sh` both.
#
# The task adapter's `run_entrypoint.py::_interpreter()` has always done the
# right thing — probe each candidate, refuse with a sentence naming `KFO_PYTHON`.
# This is that shape for the shell steps. It deliberately does **not** probe for
# torch: these two scripts read JSON and run a validator body, and demanding
# torch of them would refuse a step that does not need it. The torch probe stays
# where the measurement is.
kfo_python() {
  for candidate in "${KFO_PYTHON:-}" "${AGENT_SYS_DEMO_PYTHON:-}" /opt/venv/bin/python3 python3; do
    [ -n "$candidate" ] || continue
    if command -v "$candidate" >/dev/null 2>&1; then
      printf '%s' "$candidate"
      return 0
    fi
  done
  echo "no usable python found. Tried KFO_PYTHON='${KFO_PYTHON:-}', " >&2
  echo "AGENT_SYS_DEMO_PYTHON='${AGENT_SYS_DEMO_PYTHON:-}', /opt/venv/bin/python3, python3." >&2
  echo "" >&2
  echo "KFO_PYTHON defaults to /opt/venv/bin/python3, which is a path INSIDE the engine" >&2
  echo "container. If this step is running on a host, that is the whole explanation:" >&2
  echo "modules 1-4 exec into m1's container (CONTRACT section 5) and this step did not." >&2
  echo "Either run it in the container, or set KFO_PYTHON to an interpreter that exists" >&2
  echo "here (--var kfo_python=...)." >&2
  return 1
}

# Run the workset launcher with explicit task-owned provider and round limits.
kfo_run_forge() {
  set -- --workspace "$WS" --max-hours "$KFO_MAX_HOURS" --lanes 1 \
    --agent-backend "${KFO_AGENT_BACKEND:-claude}" --max-rounds "${KFO_MAX_ROUNDS:-0}" \
    --driver "$WS/.infera_forge_workset/operators/$OPERATOR/scripts/forge_driver.py" \
    --result-json "$WORKDIR/forge_native_result.json" \
    --no-experience-kb --no-kb-warmstart --no-pr-kb
  [ -z "${KFO_SNR_THRESHOLD:-}" ] || set -- "$@" --snr-threshold "$KFO_SNR_THRESHOLD"
  [ -z "${KFO_FORGE_MODEL:-}" ] || set -- "$@" --model "$KFO_FORGE_MODEL"
  ( cd "$RUN" && sh "./$ONELINE" "$@" )
}
