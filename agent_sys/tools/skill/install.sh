#!/bin/sh
# Install the infera agent_sys skills for Claude Code (links, so they track this checkout).
#
#   sh install.sh                          link every skill into ~/.claude/skills/
#   sh install.sh --repo                   ... into <repo root>/.claude/skills/ instead
#   sh install.sh --hook                   also add the keyword hint hook (UserPromptSubmit)
#   sh install.sh --mode explicit          entry skills engage only when agent_sys is named
#   sh install.sh --remote <git url> [--user <name>] [--clone <dir>]
#                                          configure the shared knowledge repo now
#   sh install.sh --check                  report what is installed; change nothing
#
# Never deletes or overwrites: an existing entry with a skill's name is left
# alone and reported. settings.json is backed up before the hook is added.
set -eu
here=$(cd "$(dirname "$0")" && pwd -P)
astk="$here/lib/astk.py"
scope=user hook=0 mode="" remote="" ruser="" rclone="" check=0
while [ $# -gt 0 ]; do
  case "$1" in
    --repo) scope=repo ;;
    --hook) hook=1 ;;
    --mode) mode="$2"; shift ;;
    --remote) remote="$2"; shift ;;
    --user) ruser="$2"; shift ;;
    --clone) rclone="$2"; shift ;;
    --check) check=1 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done
if [ "$scope" = repo ]; then
  claude_dir="$(git -C "$here" rev-parse --show-toplevel)/.claude"
else
  claude_dir="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
fi
dest="$claude_dir/skills"
# Settings, rendering and the hook need only the standard library; astk's other
# commands need agent_sys's dependencies, checked below and reported, not assumed.
kit() { python3 -c "import sys; sys.path.insert(0, sys.argv[1]); from skillkit import settings as s; $1" "$here/lib"; }

echo "== python ($(command -v python3 || echo 'python3 NOT FOUND'))"
deps_ok=1
if python3 -c "import yaml, ruamel.yaml, jsonschema, markdown_it, pydantic, packaging" 2>/dev/null; then
  echo "  ok       pyyaml, ruamel.yaml, jsonschema, markdown-it-py, pydantic, packaging"
else
  deps_ok=0
  echo "  MISSING  agent_sys dependencies — astk's package/handoff/remote commands need them:"
  echo "           pip install pyyaml ruamel.yaml jsonschema markdown-it-py pydantic packaging"
  echo "           (or: pip install -e $(dirname "$(dirname "$here")"))"
fi

# The entry skills are rendered per trigger mode into the user's config dir (their
# description follows `astk mode`); every other skill is linked as it is.
case "$mode" in ""|auto|explicit|on|off|disable|enable) ;; *) echo "--mode: auto or explicit" >&2; exit 2 ;; esac
if [ -n "$mode" ] && [ "$check" = 0 ]; then kit "s.save(trigger='$mode')" >/dev/null; fi
rendered=$(kit "print(s.rendered_dir())")
previous=$(kit "print(s.rendered_from() or '')")
if [ -n "$previous" ] && [ "$previous" != "$here" ]; then
  echo "note: the entry skills were rendered from $previous; they now follow $here"
fi
[ "$check" = 1 ] || kit "s.render_gated(__import__('pathlib').Path(sys.argv[1]).parent)"
gated=" infera-agent-sys agent-sys-prior-work "

echo "== skills -> $dest"
[ "$check" = 1 ] || mkdir -p "$dest"
for skill in "$here"/*/; do
  [ -f "$skill/SKILL.md" ] || continue
  name=$(basename "$skill"); link="$dest/$name"; target="${skill%/}"
  case "$gated" in *" $name "*) target="$rendered/$name" ;; esac
  if [ -L "$link" ] && [ ! -e "$link" ]; then
    echo "  DANGLING $name: $link -> $(readlink "$link") (left untouched)"
  elif [ -L "$link" ] && [ "$(readlink "$link")" = "$target" ]; then
    echo "  ok       $name"
  elif [ -L "$link" ] && [ "$(readlink "$link")" = "${skill%/}" ]; then
    echo "  ok       $name (linked to the repo copy: explicit mode cannot narrow it; relink to $target to fix)"
  elif [ -e "$link" ] || [ -L "$link" ]; then
    echo "  skipped  $name: $link exists and is not this skill (left untouched)"
  elif [ "$check" = 1 ]; then
    echo "  missing  $name"
  else
    ln -s "$target" "$link"; echo "  linked   $name"
  fi
done

echo "== hook"
python3 - "$claude_dir/settings.json" "$here/hooks/prompt_hint.py" "$hook" "$check" <<'PY'
import json, shlex, shutil, sys, time
from pathlib import Path
path, script, want, check = Path(sys.argv[1]), sys.argv[2], sys.argv[3] == "1", sys.argv[4] == "1"
# Fail open: if the checkout moves or python breaks, the hook prints nothing and
# exits 0 — a hint must never block a prompt.
command = f"sh -c 'f=\"$1\"; [ -f \"$f\" ] || exit 0; python3 \"$f\" || exit 0' _ {shlex.quote(script)}"
data = json.loads(path.read_text()) if path.is_file() else {}
groups = data.get("hooks", {}).get("UserPromptSubmit", [])
present = any(script in h.get("command", "") for g in groups for h in g.get("hooks", []))
if present:
    print(f"  ok       keyword hint hook in {path}")
elif not want or check:
    print("  off      (sh install.sh --hook adds the keyword hint)")
else:
    if path.is_file():
        backup = path.with_name(f"{path.name}.bak-astk-{time.strftime('%Y%m%d-%H%M%S')}")
        shutil.copy2(path, backup)
        print(f"  backup   {backup}")
    data.setdefault("hooks", {}).setdefault("UserPromptSubmit", []).append(
        {"hooks": [{"type": "command", "command": command, "timeout": 5}]})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"  added    keyword hint hook to {path}")
PY

echo "== settings"
kit "d = s.load(); print(f\"  trigger: {d['trigger']}  ask_remote: {'yes' if d['ask_remote'] else 'no'}  ({s.path()})\")"

echo "== knowledge repo"
if [ "$deps_ok" = 0 ]; then
  echo "  (skipped: install the dependencies above first)"
  [ -z "$remote" ] || exit 1
elif [ -n "$remote" ] && [ "$check" = 0 ]; then
  set -- --url "$remote"
  [ -n "$ruser" ] && set -- "$@" --user "$ruser"
  [ -n "$rclone" ] && set -- "$@" --clone "$rclone"
  rc=0; out=$(python3 "$astk" remote init "$@" 2>&1) || rc=$?
  printf '%s\n' "$out" | sed 's/^/  /'
  [ "$rc" = 0 ] || exit "$rc"
else
  python3 "$astk" remote status --offline 2>&1 | sed 's/^/  /' || true
fi
