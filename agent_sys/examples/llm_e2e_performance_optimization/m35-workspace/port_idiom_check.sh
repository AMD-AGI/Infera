#!/usr/bin/env bash
# Will this attempt's kit refuse on the port idiom? One read, before
# `check_deploy_kit` runs.
#
#     bash port_idiom_check.sh <run dir>
#
# **Checks for the PRESENCE of the good form, not the absence of the bad one.**
# That is the method that established the regression: `grep -n PORT` shows
# `DK_PORT_ROUTER` and `DK_ROUTER_PORT` looking like siblings, and only
# *is this overridable* separates them. A non-match is not evidence.
#
#   good:  : "${DK_PORT_ROUTER:=$(( E2E_KIT_PORT_BASE + 0 ))}"   overridable
#   bad:   DK_ROUTER_PORT=$(( E2E_KIT_PORT_BASE + 0 ))           unconditional
#
# **The path is narrow on purpose.** `find <run> -name env.sh` returns decoys:
# the staged package ships `workspace/examples/*/env.sh`, which are not the kit
# and would answer a different question. The kit is the handoff's own content:
#
#     <run>/handoffs/<id>/v*/content/items/codes/*/scripts/env.sh
#
# It exits 3 — not 0 — when the kit does not exist yet, so "not written" cannot
# be mistaken for "clean".
set -uo pipefail

RUN="${1:?usage: port_idiom_check.sh <run dir>}"
[ -d "$RUN" ] || { echo "port_idiom_check: no such run dir: $RUN" >&2; exit 2; }

# Newest first: a run can carry more than one version of the handoff.
mapfile -t KITS < <(find "$RUN" -path '*/handoffs/*/content/items/codes/*/scripts/env.sh' \
                    -printf '%T@ %p\n' 2>/dev/null | sort -rn | cut -d' ' -f2-)

if [ "${#KITS[@]}" -eq 0 ]; then
  echo "NOT-YET: no kit scripts/env.sh under $RUN"
  echo "  (the deployer has not written the kit; this is not a pass)"
  exit 3
fi

rc=0
for f in "${KITS[@]}"; do
  echo "kit: ${f##*/runs/}"
  good=$(grep -nE '^[[:space:]]*:[[:space:]]*"\$\{DK_PORT_ROUTER:=' "$f")
  bad=$(grep -nE '^[[:space:]]*DK_ROUTER_PORT=' "$f")
  any=$(grep -nE 'DK_(PORT_ROUTER|ROUTER_PORT)' "$f")

  echo "  --- every line mentioning either spelling ---"
  printf '%s\n' "${any:-  (none)}" | sed 's/^/    /'

  if [ -n "$good" ]; then
    echo "  VERDICT: OVERRIDABLE — the good form is present:"
    printf '%s\n' "$good" | sed 's/^/    /'
    [ -n "$bad" ] && { echo "  WARNING: the bad spelling is ALSO present:"; printf '%s\n' "$bad" | sed 's/^/    /'; }
  else
    echo "  VERDICT: WILL REFUSE — the good form is ABSENT."
    [ -n "$bad" ] && { echo "  the unconditional assignment is at:"; printf '%s\n' "$bad" | sed 's/^/    /'; }
    rc=1
  fi
done
exit "$rc"
