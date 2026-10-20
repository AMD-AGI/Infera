#!/bin/sh
# Install the agent_sys skills by symlink, so they track this checkout.
#
#   sh install.sh            -> ~/.claude/skills/<name>  (every project)
#   sh install.sh --repo     -> <repo root>/.claude/skills/<name>  (this repo only)
#
# Creates links only. An existing entry with the same name is left alone and
# reported — remove it yourself if you want this one instead.
set -eu
here=$(cd "$(dirname "$0")" && pwd -P)
if [ "${1:-}" = "--repo" ]; then
  dest="$(git -C "$here" rev-parse --show-toplevel)/.claude/skills"
else
  dest="${CLAUDE_CONFIG_DIR:-$HOME/.claude}/skills"
fi
mkdir -p "$dest"
for skill in "$here"/agent-sys-*/; do
  name=$(basename "$skill")
  link="$dest/$name"
  if [ -L "$link" ] && [ "$(readlink "$link")" = "${skill%/}" ]; then
    echo "ok       $link"
  elif [ -e "$link" ] || [ -L "$link" ]; then
    echo "skipped  $link exists and is not this skill (left untouched)"
  else
    ln -s "${skill%/}" "$link"
    echo "linked   $link -> ${skill%/}"
  fi
done
python3 "$here/lib/astk.py" --help >/dev/null && echo "astk     $here/lib/astk.py works with $(command -v python3)"
