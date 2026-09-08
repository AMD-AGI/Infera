# What the shell STEPs share. POSIX `/bin/sh`; sourced, never executed.
#
# One function so far, and it exists because two scripts could fail more quietly
# than a function in the same package already fails loudly.

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
# 2026-09-04 on the login node, in `30_run_forge.sh` and `70_selfcheck.sh` both.
#
# `run_entrypoint.py::_interpreter()` in this same directory has always done the
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
