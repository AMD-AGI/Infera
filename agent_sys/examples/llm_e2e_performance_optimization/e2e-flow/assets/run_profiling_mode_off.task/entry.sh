#!/bin/sh
# The profiler-detached line: CUDA graph ON, no profiler attached.
#
# **These are the numbers that mean something.** They are what stage 5's stock
# arm has to reproduce (M5.1.3.1), and they are the only throughput in this flow
# worth quoting — the profiler-attached line runs with graphs off and measured
# 8x slower on the sealed pair, which is the intent and not a regression.
#
# One task, not two: it brings its own service up and tears it down (M2.5).
# Everything else it shares byte for byte with the other line — see
# `../load/line.sh`, which is both of them.
#
# `#!/bin/sh` and `set -eu`, not bash and not `pipefail`: agent_sys never
# consults a body's shebang, it invokes one as `["/bin/sh", entry]`
# (`agent/backends/program.py:83`), and `/bin/sh` here is dash, which exits 2 on
# `set -o pipefail` before line 3 runs (CONTRACT §3.2a). `line.sh` is invoked
# with `bash` explicitly below, because it does use arrays.
set -eu
PKG="${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}"

# `|| rc=$?` and not `&& exit 0`: `mock.sh` exits **3** when this stage is not
# in `$E2E_MOCK_STAGES`, and 0 only when it actually wrote something. The `||`
# is what keeps `set -e` from killing the script before the assignment runs.
rc=0
bash "$PKG/assets/lib/mock.sh" stage2-profiling \
  profiling_mode_off.bench_result:aiperf_baseline || rc=$?
# The sealed `items/command` does not parse under the bash its own shebang names
# — an apostrophe inside `${VAR:?...}` runs a single quote to end of file — and
# `check_command_parses` grades exactly that. **`mock.sh` repairs it** for every
# mocked kind (MOCK-MAP J), so nothing is needed here; m2's own generators no
# longer emit the fault either.
if [ "$rc" = 0 ]; then
  # **MOCK-MAP (A), and it has to happen here.** `mock.sh` copies the sealed
  # `items/env/` faithfully — `engine_argv.txt`, `image.txt`, `load.json`,
  # `router_cmd.txt` — and those artefacts predate `environment.yaml`, so the
  # copy is right and incomplete. `check_environment` is `strong`, so without
  # this the handoff is invalid however good the bench is. Measured in run
  # 20260903T150156-33c6b8: `check_bench_result` PASS, `check_command_parses`
  # PASS, `check_environment` FAIL.
  #
  # **Inherited verbatim from m1, with no `--set`.** In mock mode there is no
  # bring-up of ours to describe, and m1's record is the deployment the sealed
  # bench would have run against. That also keeps every part agreeing, which is
  # what `check_profiling_evidence` compares across the two lines.
  #
  # Plain `python3`: a task body never gets `AGENT_SYS_DEMO_PYTHON`, and
  # `/usr/bin/python3` was measured to run `env_render.py` here — it has
  # `jsonschema` 4.10.3 and `PyYAML`, and `schema.py` inlines cross-file refs so
  # `referencing` is not needed.
  python3 "$PKG/assets/lib/env_render.py" \
    --inherit "${AGENT_SYS_INPUT_DEPLOY_KIT:?}/items/codes/environment.yaml" \
    --content-type reproducible \
    --out "${AGENT_SYS_OUTPUT_PROFILING_MODE_OFF_BENCH_RESULT:?}"
  exit 0
fi
if [ "$rc" != 3 ]; then exit "$rc"; fi

E2E_MODE=profiling_mode_off \
E2E_OUTPUT_AIPERF="${AGENT_SYS_OUTPUT_PROFILING_MODE_OFF_BENCH_RESULT:?}" \
exec bash "$PKG/assets/load/line.sh"
