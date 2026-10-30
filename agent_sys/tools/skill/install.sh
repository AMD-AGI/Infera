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
# alone and reported. settings.json is backed up before the hook is added or
# re-pointed; other hooks in it are never touched.
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
kit() { code=$1; shift; python3 -c "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); from skillkit import settings as s; root = Path(sys.argv[1]).parent; $code" "$here/lib" "$@"; }

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

# The entry skills are rendered per trigger mode into the user's config dir, one
# copy per checkout (their description follows `astk mode`); every other skill
# is linked as it is.
case "$mode" in ""|auto|explicit|on|off|disable|enable) ;; *) echo "--mode: auto or explicit" >&2; exit 2 ;; esac
if [ -n "$mode" ] && [ "$check" = 0 ]; then kit "s.save(trigger='$mode')" >/dev/null; fi
rendered=$(kit "print(s.rendered_dir(root))")
if [ "$check" = 0 ]; then
  if [ -n "$mode" ]; then kit "s.rerender_all()"; fi  # the mode is per user: every checkout's copy
  kit "s.render_gated(root)"
fi
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
    stale=""
    case "$gated" in *" $name "*) stale=$(kit "print(s.render_status(root, sys.argv[2]) or '')" "$name") ;; esac
    if [ -n "$stale" ]; then echo "  STALE    $name: $stale (sh install.sh re-renders it)"; else echo "  ok       $name"; fi
  elif [ -e "$link" ] || [ -L "$link" ]; then
    echo "  skipped  $name: $link exists and is not this skill (left untouched)"
  elif [ "$check" = 1 ]; then
    echo "  missing  $name"
  else
    ln -s "$target" "$link"; echo "  linked   $name"
  fi
done

echo "== hook"
hook_rc=0
python3 - "$claude_dir/settings.json" "$here/hooks/prompt_hint.py" "$hook" "$check" <<'PY' || hook_rc=$?
import json, shlex, shutil, sys, time
from pathlib import Path
path, script, want, check = Path(sys.argv[1]), sys.argv[2], sys.argv[3] == "1", sys.argv[4] == "1"
# Fail open: if the checkout moves or python breaks, the hook prints nothing and
# exits 0 — a hint must never block a prompt.
command = f"sh -c 'f=\"$1\"; [ -f \"$f\" ] || exit 0; python3 \"$f\" || exit 0' _ {shlex.quote(script)}"
# Our hook in any checkout (or an older install): the script's path inside the repo.
MARKER = "/tools/skill/hooks/prompt_hint.py"
try:  # a shape other than {"hooks": {"UserPromptSubmit": [{"hooks": [{...}]}]}} fails here
    data = json.loads(path.read_text()) if path.is_file() else {}
    groups = data.setdefault("hooks", {}).setdefault("UserPromptSubmit", [])
    if not isinstance(groups, list) or not all(isinstance(g.get("hooks", []), list) for g in groups):
        raise ValueError("unexpected structure under hooks.UserPromptSubmit")
    ours = [h for g in groups for h in g.get("hooks", []) if MARKER in str(h.get("command", ""))]
except (OSError, ValueError, AttributeError) as exc:  # report, change nothing
    print(f"  ERROR    {path} is not a settings file this installer understands ({exc}); left untouched")
    sys.exit(1 if want and not check else 0)
if len(ours) == 1 and ours[0].get("command") == command:
    print(f"  ok       keyword hint hook in {path}")
elif not want or check:
    if any(h.get("command") == command for h in ours):
        print(f"  ok       keyword hint hook in {path}, but installed {len(ours)} times")
        print("           (sh install.sh --hook drops the duplicates)")
    elif ours:
        print(f"  ok       keyword hint hook in {path}, but not this checkout's: {ours[0].get('command')}")
        print("           (sh install.sh --hook re-points it here)")
    else:
        print("  off      (sh install.sh --hook adds the keyword hint)")
else:
    if path.is_file():
        backup = path.with_name(f"{path.name}.bak-astk-{time.strftime('%Y%m%d-%H%M%S')}")
        shutil.copy2(path, backup)
        print(f"  backup   {backup}")
    if ours:  # re-point our first entry, drop our duplicates; other hooks stay as they are
        ours[0]["command"] = command
        dup = {id(h) for h in ours[1:]}
        emptied = set()
        for g in groups:
            if any(id(h) in dup for h in g.get("hooks", [])):
                g["hooks"] = [h for h in g["hooks"] if id(h) not in dup]
                if not g["hooks"]:
                    emptied.add(id(g))
        groups[:] = [g for g in groups if id(g) not in emptied]
        print(f"  updated  keyword hint hook in {path} (now this checkout's)")
    else:
        groups.append({"hooks": [{"type": "command", "command": command, "timeout": 5}]})
        print(f"  added    keyword hint hook to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
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
[ "$hook_rc" = 0 ] || exit "$hook_rc"
