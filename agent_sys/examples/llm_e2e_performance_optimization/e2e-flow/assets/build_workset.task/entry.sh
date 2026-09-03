#!/bin/sh
# **This closure runs under `kind: ai` (`workset_builder`), so this file is not
# the body — `readme.md` is.** Its STEPS section is the method; the AI sequences
# it and supplies the two judgements a program cannot make.
#
# It is kept, and it is not dead weight, for one case: a pure-mock run.
#
#     --var m3_agent=runner --var mock_stages=all
#
# swaps in the shared program agent, which does run this file, which mocks —
# producing `operator_workset` from the sealed evidence with no model call and
# no GPU.
#
# **Without that switch a mocked m3 is not mocked at all.** A `kind: ai` task
# never runs `entry.sh`, so promoting this closure to `ai` took it off the mock
# path; m5 hit exactly this and its first full mock run sat at
# `integrate_and_verify: running` while an agent prepared to do the real thing.
# The default is the real agent, so **the mock is the thing you have to ask
# for**, and this file exists so that asking is possible.
#
# It deliberately does **not** attempt the real work. A shell script cannot read
# a framework's test suite to find the correctness reference, and a scaffold with
# a sentinel where the reference should be is not a workset — it is a workset
# that fails `check_workset_shape` on a TODO marker, which is the correct
# outcome and a slow way to reach it. `scaffold.py` is a *step* of the method,
# not a substitute for it.
set -eu
PKG="${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}"

# `|| rc=$?` and not `; rc=$?`: under `set -e` a simple command exiting non-zero
# kills the script before the assignment runs, so the branch below is never
# reached and a real-mode task dies with the mock's status. A `||` puts it in a
# condition context, where `set -e` does not fire.
rc=0
# **`bash`, explicitly, and not `sh`.** This file's shebang is `/bin/sh` per
# CONTRACT §3.2a — the runner invokes `["/bin/sh", entry]` and `/bin/sh` here is
# dash — but `mock.sh` uses `${!var}` indirect expansion, which dash does not
# have. Running it under `sh` fails with a syntax error at parse time, before
# any branch below is reached. Every other owner's entry.sh does the same and
# says so; this comment is here because I wrote `sh` first.
bash "$PKG/assets/lib/mock.sh" stage3-analyze operator_workset || rc=$?
if [ "$rc" -eq 0 ]; then
  # MOCK-MAP (C). The sealed bytes alone do not satisfy this package's
  # `operator_workset`: the stage-3 half is `items/code/` where the merged kind
  # is `items/codes/`, and behind that first failure are five more. The
  # adaptation is a **step after the copy**, never a variant of it.
  #
  # It writes everything except `evidence/`, because evidence is a measurement
  # and not a document. The two entrypoints then produce it — the same STEP 7
  # and STEP 8 a real run does, which is the point: a mock that skipped them
  # would leave `check_workset_runs` grading an artefact nothing ever ran.
  OUT="${AGENT_SYS_OUTPUT_OPERATOR_WORKSET:?the runner exports this for an output slot}"
  # **No `.pyc` beside the artefact.** Measured on the first real GPU run: the
  # container runs as root — it has to, because a framework compiling kernels on
  # first call cannot write its cache as anyone else — so every `__pycache__`
  # the entrypoints leave is root-owned inside a handoff the runner then tries
  # to copy and clean as `yihou`. The failure only appears on the *second* run,
  # which is the worst kind. `analyze-demo` hit the same thing and recorded it;
  # this is the one-variable half of its fix.
  export PYTHONDONTWRITEBYTECODE=1
  python3 "$PKG/assets/build_workset.task/mock_adapt.py" "$OUT"
  cd "$OUT/items/codes" || exit 1
  ./run_correctness.sh --json evidence/correctness.json
  ./run_performance.sh --json evidence/performance.json
  python3 - <<'TRANSCRIBE'
import json, pathlib, yaml
floor = json.loads(pathlib.Path("evidence/performance.json").read_text())["noise_floor"]
p = pathlib.Path("workset.yaml"); d = yaml.safe_load(p.read_text())
d["evidence"] = {"correctness_report": "evidence/correctness.json",
                 "performance_report": "evidence/performance.json",
                 "measured_on": {"node": __import__("os").environ.get("E2E_NODE")
                                         or __import__("os").uname().nodename,
                                 "gpu_arch": d["ground_truth"]["environment"]["fixed"]["gpu_arch"],
                                 "container": d["ground_truth"]["environment"]["runtime"]["container"],
                                 "at": json.loads(pathlib.Path("evidence/performance.json").read_text())["started_at"]}}
for op in d["operators"]:
    op["noise_floor"] = floor
p.write_text(yaml.safe_dump(d, sort_keys=False))
print(f"mock_adapt: evidence recorded, noise_floor {floor}")
TRANSCRIBE
  exit 0
fi
if [ "$rc" -ne 3 ]; then exit "$rc"; fi

cat >&2 <<'WHY'
build_workset: this closure was run by the program agent with m3 NOT in
mock_stages, and there is nothing sensible for a shell script to do here.

Either:
  * drop `--var m3_agent=runner` so the `workset_builder` AI agent runs
    readme.md's STEPS, which is the real path; or
  * add m3 to `--var mock_stages=...` so this file mocks from the sealed
    evidence.

Failing rather than scaffolding-and-hoping: a scaffold with a TODO sentinel
where the correctness reference belongs is not a workset, and shipping one would
turn a clear failure here into an opaque one two validators later.
WHY
exit 1
