#!/usr/bin/env bash
# pre-commit hook: refuse Go sources that gofmt would rewrite.
#
# Worth a script rather than an inline entry, because `gofmt -l` prints the
# offending files and *still exits 0*. A bare `entry: gofmt -l` therefore passes
# on every commit and enforces nothing — which is how this tree accumulated Go
# files main had never formatted, discovered only when something unrelated ran
# gofmt over the directory.
#
# Formatting is checked, never applied: a hook that rewrites files mid-commit
# leaves the staged and working copies disagreeing, and the diff you reviewed is
# not the diff you commit.
set -euo pipefail

if ! command -v gofmt >/dev/null 2>&1; then
  cat >&2 <<'MSG'
gofmt not found, but this commit touches Go sources.

Install the Go toolchain (https://go.dev/dl/), or skip this one check with:

  SKIP=gofmt git commit ...
MSG
  exit 1
fi

# pre-commit passes the staged files matching `files:`; nothing to do otherwise.
[ "$#" -eq 0 ] && exit 0

unformatted="$(gofmt -l "$@")"
[ -z "$unformatted" ] && exit 0

{
  echo
  echo "Refusing the commit: gofmt would rewrite these files."
  echo
  printf '  %s\n' $unformatted
  echo
  echo "Format them in place, then re-stage:"
  echo
  # Unquoted on purpose: word-splitting rejoins the newline-separated list into
  # a single space-separated command line.
  echo "  gofmt -w" $unformatted
  echo
} >&2
exit 1
