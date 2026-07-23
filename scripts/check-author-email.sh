#!/usr/bin/env bash
# pre-commit hook: refuse a commit whose author/committer email is a
# machine-generated local hostname — git's fallback when user.email is unset
# (e.g. user@crs-m2m-cpu-spur-009.us-east2-a.compute.internal). This keeps an
# internal build-host name from leaking into permanent (public) git history.
#
# Local commit-time gate only: skipped in CI (the runner has no real identity;
# enforce PR-author emails there with a separate commit-range check).
set -euo pipefail

# CI runners have no meaningful committer identity — don't fail the lint job.
[ -n "${CI:-}" ] && exit 0

fail=0
host_fqdn="$({ hostname -f 2>/dev/null || hostname 2>/dev/null || true; } | tr '[:upper:]' '[:lower:]')"

check() {
  local who="$1" email domain
  email="$(git var "GIT_${who}_IDENT" 2>/dev/null | sed -n 's/.*<\(.*\)>.*/\1/p' | tr '[:upper:]' '[:lower:]')"
  if [ -z "$email" ]; then echo "  ✖ ${who}: empty email"; fail=1; return; fi
  domain="${email#*@}"
  case "$email" in
    *@*.internal|*@*.local|*@*.localdomain|*@*.lan|*@localhost|*@localhost.*|*@*.localhost)
      echo "  ✖ ${who} email is internal/local: <$email>"; fail=1 ;;
    *@*.*) : ;;                                    # dotted, public-looking domain — OK
    *@?*)  echo "  ✖ ${who} email has no domain (bare hostname): <$email>"; fail=1 ;;
    *)     echo "  ✖ ${who} email malformed: <$email>"; fail=1 ;;
  esac
  # git's auto-generated identity uses the machine FQDN as the domain
  if [ -n "$host_fqdn" ] && [ "$domain" = "$host_fqdn" ]; then
    echo "  ✖ ${who} email domain == this machine's hostname ($host_fqdn): <$email>"; fail=1
  fi
}

check AUTHOR
check COMMITTER

if [ "$fail" -ne 0 ]; then
  cat >&2 <<'MSG'

Refusing the commit: the git email looks machine-generated, not a real address.
It would be baked into permanent history. Set a real email once, then re-commit:

  git config --global user.name  "Your Name"
  git config --global user.email you@amd.com

And make git refuse to auto-generate one from the hostname ever again:

  git config --global user.useConfigOnly true
MSG
  exit 1
fi
