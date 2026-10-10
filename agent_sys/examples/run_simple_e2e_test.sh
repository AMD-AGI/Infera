#!/bin/bash
# Run the example packages that `as.e2e.test.yaml` calls end-to-end tests.
#
#   bash agent_sys/examples/run_simple_e2e_test.sh              # every case
#   bash agent_sys/examples/run_simple_e2e_test.sh dangling_handoff_kind
#   bash agent_sys/examples/run_simple_e2e_test.sh --list
#   bash agent_sys/examples/run_simple_e2e_test.sh --run-root /tmp/me/e2e
#
# **`bash`, not `sh`** — the shebang is load-bearing, not decoration: `set -o
# pipefail` below is a bash builtin and `dash` refuses it at the line, not at
# parse time, so `dash -n` on this file is silent and running it is not.
#
# Exit: 0 every case passed or was skipped; 1 a case failed; 2 the runner could
# not do its job (no yaml, no python3, bad argument). A SKIP is never a PASS,
# and the summary counts them separately for that reason.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
AGENT_SYS="$(cd "$HERE/.." && pwd)"
REPO="$(cd "$AGENT_SYS/.." && pwd)"
SPEC="$HERE/as.e2e.test.yaml"

[ -r "$SPEC" ] || { echo "no $SPEC" >&2; exit 2; }
command -v python3 >/dev/null 2>&1 || { echo "python3 not on PATH" >&2; exit 2; }

RUN_ROOT=""
WANTED=()
LIST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --list)     LIST=1; shift ;;
    --run-root) RUN_ROOT="${2:?--run-root needs a path}"; shift 2 ;;
    -h|--help)  sed -n '2,14p' "$0"; exit 0 ;;
    -*)         echo "unknown flag: $1" >&2; exit 2 ;;
    *)          WANTED+=("$1"); shift ;;
  esac
done

# The spec is YAML, so it is read by a YAML parser rather than by grep. One
# python3 call emits one \x1e-separated line per case; the shell does the rest.
plan() {
  RUN_ROOT_OVERRIDE="$RUN_ROOT" python3 - "$SPEC" <<'PY'
import os, secrets, sys, yaml
spec = yaml.safe_load(open(sys.argv[1]))
root = os.environ.get("RUN_ROOT_OVERRIDE") or spec.get("defaults", {}).get("run_root", "")
root = os.path.expandvars(root)
for case in spec.get("cases", []):
    argv = ["--package", "agent_sys/examples/" + case["package"]]
    argv += case.get("args") or []
    for name, how in (case.get("vars") or {}).items():
        if isinstance(how, dict) and how.get("generate") == "hex16":
            value = secrets.token_hex(16)
        elif isinstance(how, dict):
            value = os.environ.get(how.get("from_env", ""), "") or \
                    os.path.expandvars(how.get("default", ""))
        else:
            value = os.path.expandvars(str(how))
        argv += ["--var", f"{name}={value}"]
    expect = case.get("expect") or {}
    # \x1e between fields, never a TAB: TAB is whitespace, so bash `read`
    # collapses a run of them and an empty field silently shifts every field
    # after it by one.
    print("\x1e".join([
        case["name"],
        case.get("command", "run"),
        str(expect.get("exit_code", 0)),
        expect.get("stdout_contains", ""),
        ",".join(case.get("requires") or []),
        root.rstrip("/") + "/" + case["name"],
        "\x1f".join(argv),
    ]))
PY
}

if [ "$LIST" = 1 ]; then
  plan | while IFS=$'\x1e' read -r name cmd code want req dir argv; do
    printf '%-24s expects exit %s%s%s\n' "$name" "$code" \
      "${want:+, stdout containing \"$want\"}" "${req:+  [requires: $req]}"
  done
  exit 0
fi

pass=0; fail=0; skip=0; failed_names=""

while IFS=$'\x1e' read -r name cmd code want req dir argv; do
  if [ ${#WANTED[@]} -gt 0 ]; then
    case " ${WANTED[*]} " in *" $name "*) ;; *) continue ;; esac
  fi

  missing=""
  if [ -n "$req" ]; then
    IFS=, read -ra tools <<< "$req"
    for t in "${tools[@]}"; do
      command -v "$t" >/dev/null 2>&1 || missing="$missing $t"
    done
  fi
  if [ -n "$missing" ]; then
    printf 'SKIP %-24s not on PATH:%s\n' "$name" "$missing"
    skip=$((skip + 1)); continue
  fi

  mkdir -p "$dir"
  log="$dir/run.log"
  # `\x1f` because a --var value may contain spaces; word-splitting on it and
  # nothing else keeps one argument one argument.
  IFS=$'\x1f' read -ra args <<< "$argv"

  printf '.... %-24s ' "$name"
  set +e
  # `< /dev/null` is load-bearing: without it the child inherits the loop's
  # stdin -- the process substitution feeding `read` -- and swallows every
  # remaining case. The symptom is not an error; it is a suite that silently
  # runs one case and reports a total.
  ( cd "$REPO" && PYTHONPATH="$AGENT_SYS" python3 -m cli.main "$cmd" \
      "${args[@]}" --demo-root "$dir" ) > "$log" 2>&1 < /dev/null
  got=$?
  set -e

  why=""
  [ "$got" = "$code" ] || why="exit $got, expected $code"
  if [ -z "$why" ] && [ -n "$want" ] && ! grep -qF -- "$want" "$log"; then
    why="output does not contain \"$want\""
  fi

  if [ -z "$why" ]; then
    printf 'PASS\n'; pass=$((pass + 1))
  else
    printf 'FAIL  %s\n      log: %s\n' "$why" "$log"
    fail=$((fail + 1)); failed_names="$failed_names $name"
  fi
done < <(plan)

echo
echo "passed $pass   failed $fail   skipped $skip"
[ "$skip" = 0 ] || echo "a SKIP is not a PASS: those cases did not run."
if [ "$fail" != 0 ]; then
  echo "failed:$failed_names"
  echo "Read the log named above, not this summary — the exit code says a case"
  echo "did not meet its expectation, never why."
  exit 1
fi
exit 0
