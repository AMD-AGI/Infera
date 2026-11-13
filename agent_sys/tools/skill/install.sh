#!/bin/sh
# Install the infera agent_sys skills for Claude Code (links, so they track this checkout):
# tools/skill/*/ and kernel_registry/skills/*/.
#
#   sh install.sh                          link every skill into ~/.claude/skills/
#   sh install.sh --repo                   ... into <repo root>/.claude/skills/ instead
#   sh install.sh --hook                   also add the keyword hint hook (UserPromptSubmit)
#   sh install.sh --mode explicit          entry skills engage only when agent_sys is named
#   sh install.sh --remote <git url> [--user <name>] [--clone <dir>]
#                                          configure the shared knowledge repo now
#   sh install.sh --check                  report what is installed; change nothing
#   sh install.sh --dashboard yes|no       install the third-party claude-dashboard status
#                                          line plugin without asking (default: ask on a tty)
#
# Never deletes or overwrites: an existing entry with a skill's name is left
# alone and reported. settings.json is backed up before the hook is added or
# re-pointed; other hooks in it are never touched.
set -eu
here=$(cd "$(dirname "$0")" && pwd -P)
astk="$here/lib/astk.py"
scope=user hook=0 mode="" remote="" ruser="" rclone="" check=0 dashboard=ask
while [ $# -gt 0 ]; do
  case "$1" in
    --repo) scope=repo ;;
    --hook) hook=1 ;;
    --mode) mode="$2"; shift ;;
    --remote) remote="$2"; shift ;;
    --user) ruser="$2"; shift ;;
    --clone) rclone="$2"; shift ;;
    --check) check=1 ;;
    --dashboard) [ $# -ge 2 ] || { echo "--dashboard: yes or no" >&2; exit 2; }; dashboard="$2"; shift ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
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
case "$dashboard" in ask|yes|no) ;; *) echo "--dashboard: yes or no" >&2; exit 2 ;; esac
if [ -n "$mode" ] && [ "$check" = 0 ]; then kit "s.save(trigger='$mode')" >/dev/null; fi
rendered=$(kit "print(s.rendered_dir(root))")
if [ "$check" = 0 ]; then
  if [ -n "$mode" ]; then kit "s.rerender_all()"; fi  # the mode is per user: every checkout's copy
  kit "s.render_gated(root)"
fi
gated=" infera-agent-sys agent-sys-prior-work "

echo "== skills -> $dest"
[ "$check" = 1 ] || mkdir -p "$dest"
# Skills that live beside the code they drive, outside tools/skill/.
if ! kernel_skills=$(cd "$here/../../kernel_registry/skills" 2>/dev/null && pwd -P); then
  echo "  missing  $here/../../kernel_registry/skills (its skills, e.g. kernel-artifacts, are not linked)"
  kernel_skills="$here/.none"
fi
for skill in "$here"/*/ "$kernel_skills"/*/; do
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

echo "== claude-dashboard (third-party status line plugin)"
# uppinote20/claude-dashboard: context, rate limits and cost in the status line,
# always installed from upstream. Its calibrate-pricing command (fits per-model
# token prices from a gateway's per-request cost header and produces Claude
# Code's modelPricing managed setting) comes from uppinote20/claude-dashboard#118;
# until an installed version ships it, the installer adds the PR's two runtime
# files (commands/calibrate-pricing.md, dist/calibrate-pricing.js) from the PR's
# reviewed commit, and only to an install from the upstream marketplace.
# Optional and never blocking: every failure is reported only.
dash_repo=uppinote20/claude-dashboard dash_pr=118
dash_sha=99a8a7bcc61c5fb9e5ff2d5eea742a14e441fe50  # #118's reviewed commit
dash_files="${ASTK_DASHBOARD_FILES:-https://raw.githubusercontent.com/$dash_repo/$dash_sha}"  # override: tests
dash_cfg="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"  # plugins are per user, also with --repo
dash_path() {  # path of the user-scope claude-dashboard install, or "none"
  python3 - "$dash_cfg/plugins/installed_plugins.json" <<'PY' 2>/dev/null || echo none
import json, os, sys
try:
    rows = json.load(open(sys.argv[1]))["plugins"]["claude-dashboard@claude-dashboard"]
    # project/local installs belong to one project and do not count for the user
    path = next(r["installPath"] for r in rows
                if r.get("scope", "user") == "user" and os.path.isdir(r["installPath"]))
except Exception:
    path = "none"
print(path)
PY
}
dash_source() {  # "upstream", the marketplace's other source, or "none"
  python3 - "$dash_cfg/plugins/known_marketplaces.json" "$dash_repo" <<'PY' 2>/dev/null || echo none
import json, re, sys
try:
    src = json.load(open(sys.argv[1]))["claude-dashboard"]["source"]
    repo = src.get("repo") if src.get("source") == "github" else None
    if repo is None:
        m = re.fullmatch(r"(?:https://|git@)github\.com[/:](.+?)(?:\.git)?/?", str(src.get("url", "")))
        repo = m.group(1) if m else None
    ref = src.get("ref")
    if repo and repo.lower() == sys.argv[2].lower() and not ref:
        print("upstream")
    else:
        print((src.get("repo") or src.get("url") or "?") + (f"#{ref}" if ref else ""))
except Exception:
    print("none")
PY
}
dash_add_pr_files() {  # add #118's runtime files to install path $1, never overwriting
  python3 - "$dash_files" "$1" <<'PY'
import os, sys, urllib.request
base, root = sys.argv[1].rstrip("/"), sys.argv[2]
for name in ("dist/calibrate-pricing.js", "commands/calibrate-pricing.md"):
    dest = os.path.join(root, name)
    if os.path.exists(dest):
        continue
    data = urllib.request.urlopen(f"{base}/{name}", timeout=30).read()
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(data)
PY
}
dash_calibrate_note() {
  echo "  calibrate: if your API gateway bills at contracted rates and returns each request's"
  echo "             cost in a response header, calibrate once (billed probes, a few cents per model):"
  echo "               /claude-dashboard:calibrate-pricing --models <model,...> --cost-header <header>"
  echo "             It prints /etc/claude-code/managed-settings.json with modelPricing merged into the"
  echo "             existing file; install that (root). Recommended: otherwise /cost and the status"
  echo "             line show list-price estimates."
}
path=$(dash_path) fresh=0
if [ "$path" = none ] && [ "$check" = 0 ]; then
  if [ "$dashboard" = ask ]; then
    if [ -t 0 ] && [ -r /dev/tty ]; then
      echo "  claude-dashboard is a THIRD-PARTY plugin (https://github.com/$dash_repo, MIT)."
      echo "  It is not written, reviewed or supported by infera / agent_sys; it is installed from"
      echo "  GitHub and runs as your status line with your user's permissions, reading Claude Code's"
      echo "  local session data. You install and use it at your own risk."
      echo "  It also offers price calibration (calibrate-pricing): recommended if your gateway bills"
      echo "  at contracted rates, so /cost and the status line show what you actually pay."
      printf "  Install claude-dashboard? (recommended) [Y/n] "
      read -r answer </dev/tty || answer=n
      case "$answer" in ""|y|Y|yes|YES) dashboard=yes ;; *) dashboard=no ;; esac
    else
      echo "  skipped  not a terminal; sh install.sh --dashboard yes installs it"
    fi
  fi
  if [ "$dashboard" = no ]; then
    echo "  skipped  (sh install.sh --dashboard yes installs it later)"
  elif [ "$dashboard" = yes ] && ! command -v claude >/dev/null 2>&1; then
    echo "  skipped  claude CLI not on PATH"
  elif [ "$dashboard" = yes ]; then
    # add fails harmlessly when the marketplace is already registered (an earlier
    # partial install); success is judged by the installed plugin alone.
    claude plugin marketplace add "$dash_repo" >/dev/null 2>&1 || true
    src=$(dash_source)
    if [ "$src" = upstream ]; then
      claude plugin install claude-dashboard@claude-dashboard >/dev/null 2>&1 || true
      path=$(dash_path)
    fi
    if [ "$src" != upstream ]; then
      echo "  skipped  marketplace claude-dashboard is $src, not $dash_repo; not installing from it"
    elif [ "$path" = none ]; then
      echo "  FAILED   try: claude plugin marketplace add $dash_repo; claude plugin install claude-dashboard@claude-dashboard"
    else
      fresh=1
      echo "  installed claude-dashboard from $dash_repo; run /claude-dashboard:setup in Claude Code to configure it"
    fi
  fi
fi
if [ "$path" = none ]; then
  [ "$check" = 0 ] || echo "  off      (sh install.sh --dashboard yes installs it)"
else
  src=$(dash_source)
  [ "$fresh" = 1 ] || echo "  ok       installed: $path"
  if [ -f "$path/dist/calibrate-pricing.js" ] && [ -f "$path/commands/calibrate-pricing.md" ]; then
    echo "  ok       calibrate-pricing available"
  elif [ "$src" != upstream ]; then
    echo "  missing  calibrate-pricing; not added: marketplace claude-dashboard is $src, not $dash_repo"
  elif [ "$check" = 1 ] || [ "$dashboard" = no ]; then
    echo "  missing  calibrate-pricing (sh install.sh adds it from $dash_repo#$dash_pr)"
  elif out=$(dash_add_pr_files "$path" 2>&1); then
    echo "  added    calibrate-pricing from $dash_repo#$dash_pr (not in this version yet)"
  else
    echo "  FAILED   adding calibrate-pricing from $dash_repo#$dash_pr: $(printf '%s' "$out" | tail -n 1)"
  fi
  dash_calibrate_note
fi

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
