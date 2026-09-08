#!/usr/bin/env bash
# Mock one or more of a task's outputs from the 25 real sealed handoffs.
#
# Mission Brief item 2: "通过 mock ai agent，结合上轮单独运行的真实 handoff，构造
# 符合 validator 的" — the point of the exercise is that **the validators are
# unchanged**. A mock that writes something a validator would accept but a real
# producer would never emit proves nothing, so nothing here synthesises: it
# copies bytes that a real run on this cluster produced.
#
#   mock.sh <stage> <kind>[:<source-kind>] [<kind>[:<source-kind>] ...]
#
# `<stage>` is a directory under $E2E_MOCK_ROOT — stage1-deploy, stage2-profiling,
# stage3-analyze, stage4-kernel-opt, stage5-integration. `<source-kind>` is
# needed wherever this package renamed a kind: the sealed artefact is filed
# under the old name.
#
# Exits 0 having written nothing when the stage is not in $E2E_MOCK_STAGES, so a
# body can call this unconditionally as its first act and fall through to the
# real work.
# **This file is the one script in the package that requires bash**, for the
# `${!var}` indirect expansion that turns a kind name into its
# `AGENT_SYS_OUTPUT_*` variable. Every other body is `#!/bin/sh` + `set -eu`,
# because agent_sys invokes a body as `["/bin/sh", entry]`
# (`validator/phase.py:147`, `agent/backends/program.py:83`) — the shebang is
# never consulted — and `/bin/sh` here is dash, which exits 2 on
# `set -o pipefail` before line 2 runs.
#
# So callers must invoke this as `bash "$PKG/assets/lib/mock.sh" …`.
# **`. mock.sh` from a dash body is the natural thing to write and it fails**
# with an unhelpful `Bad substitution`; the guard below turns that into a
# sentence.
if [ -z "${BASH_VERSION:-}" ]; then
  echo "mock.sh needs bash for \${!var}; invoke it as: bash \"\$PKG/assets/lib/mock.sh\" ..." >&2
  exit 2
fi
set -euo pipefail

: "${E2E_MOCK_ROOT:=/shared_nfs/yihou/agent_sys/cheat_for_mock}"
: "${E2E_MOCK_STAGES:=all}"

stage="${1:?usage: mock.sh <stage> <kind>[:<source-kind>] ...}"
shift

# `m2` and `stage2-profiling` both select stage 2, because the variable is
# written by an operator and the directory name is not the obvious spelling.
short="${stage%%-*}"          # stage2-profiling -> stage2
short="m${short#stage}"       #               -> m2
case ",${E2E_MOCK_STAGES}," in
  *,all,*)            ;;
  *",${short},"*)     ;;
  *",${stage},"*)     ;;
  # **Exit 3, not 0.** Declining and having-mocked are different outcomes, and
  # a caller that cannot tell them apart writes `mock.sh ... && exit 0` — which
  # makes a task in REAL mode exit successfully having written nothing. That is
  # the "ten validators PASS over a run in which every result was zero" failure,
  # reproduced in four lines.
  *) echo "mock: ${stage} is not in E2E_MOCK_STAGES=${E2E_MOCK_STAGES}; running for real" >&2
     exit 3 ;;
esac

if [ ! -d "${E2E_MOCK_ROOT}/${stage}" ]; then
  echo "mock: no such stage ${E2E_MOCK_ROOT}/${stage}" >&2
  exit 1
fi

for spec in "$@"; do
  kind="${spec%%:*}"
  src="${spec#*:}"; [ "${src}" = "${spec}" ] && src="${kind}"

  # The output directory is named by the kind, uppercased with every
  # non-alphanumeric turned into `_` (`env_mgr/grants.py:450 _env_name`), so
  # `profiling_mode_on.bench_result` arrives as
  # `$AGENT_SYS_OUTPUT_PROFILING_MODE_ON_BENCH_RESULT`.
  var="AGENT_SYS_OUTPUT_$(printf '%s' "${kind}" | tr -c '[:alnum:]' '_' | tr '[:lower:]' '[:upper:]')"
  out="${!var:-}"
  if [ -z "${out}" ]; then
    echo "mock: ${var} is unset — the task does not declare ${kind} as an output" >&2
    exit 1
  fi

  from="${E2E_MOCK_ROOT}/${stage}/${src}/content"
  if [ ! -d "${from}" ]; then
    echo "mock: no sealed content at ${from}" >&2
    exit 1
  fi

  mkdir -p "${out}"
  # `cp -a` and not `cp -r`: the sealed digest is git-shaped and records each
  # file's executable bit, so a copy that normalises modes changes the digest
  # without changing a byte of content. (Every digest under $E2E_MOCK_ROOT is
  # already invalid for exactly that reason — a past `chmod -R 777` — which is
  # why nothing downstream may be written to depend on one.)
  cp -a "${from}/." "${out}/"

  # **Repair the one defect the sealed set carries in every stage** (MOCK-MAP
  # (J)): an apostrophe inside a `${VAR:?word}` message opens a single-quoted
  # string that runs to end of file, so 11 of the 14 sealed `items/command`
  # scripts do not parse under the shell their own shebang names. The sealed
  # bytes predate `check_command_parses`, exactly as they predate
  # `environment.yaml` -- so this is the same class of adaptation as (A), and
  # it is announced rather than silent.
  #
  # `the package's assets` -> `the package assets`. Nothing else is touched, and
  # the substitution is reported per file so a reader of the log can see which
  # artefacts needed it.
  for cmd in "${out}/items/command" "${out}/items/script"; do
    [ -f "${cmd}" ] || continue
    if grep -q ":?[^}]*'" "${cmd}" 2>/dev/null; then
      sed -i "s/\\(:?[^}]*\\)'s /\\1 /g; s/\\(:?[^}]*\\)'/\\1/g" "${cmd}"
      echo "mock: repaired an unterminated quote in $(basename "${cmd}") of ${kind} (MOCK-MAP J)" >&2
    fi
  done
  echo "mock: ${stage}/${src} -> ${kind} ($(find "${out}" -type f | wc -l) files)" >&2
done
