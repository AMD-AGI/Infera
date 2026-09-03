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
  *) echo "mock: ${stage} is not in E2E_MOCK_STAGES=${E2E_MOCK_STAGES}; running for real" >&2
     exit 0 ;;
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
  echo "mock: ${stage}/${src} -> ${kind} ($(find "${out}" -type f | wc -l) files)" >&2
done
