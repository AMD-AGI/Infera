# Unified Claude Code install for agent_sys — requirements, findings, plan (2026-11-25)

Status: research done, NOT implemented; to be implemented on this branch in a later session.
Evidence (not committed; in the author's workspace `yihou/install_unify_20261125/`): `research/` (official
docs fetched 2026-11-25, Claude Code 2.1.325; `existing_install.md`, `o11y_install.md`, `cc_plugin_docs.md`,
`critic.md`) and `exp/` (isolated experiments, `exp/ns/log.md`, `exp/e1`, `exp/e3`, `exp/e4`). Tags: [doc] official docs, [exp] measured here,
[code] repo or binary. Anything without a tag is a proposal.

## 1. Requirements (user, 2026-11-24/25)
R1. One command (or one short script) installs every agent_sys Claude Code component: the agent_sys skills
    (`agent_sys/tools/skill/*`), `agent_sys/kernel_registry/skills/*`, the three o11y plugins, their runtimes.
R2. Use Claude Code's official ecosystem (plugins, marketplaces, dependencies, userConfig, CLI) wherever it
    covers the need; the result must be clean and formal, not an ad-hoc script.
R3. Keep EVERY feature of the old `tools/skill/install.sh` and of the o11y install (inventory §3).
R4. The claude-dashboard PR #118 file patch stays for now, but moves into a separate post-install script.
R5. Implement on a new branch in a new session; this document is the hand-over.
House rules that bind the design: never delete outside a `yihou/` dir without the user (so migration
reports, it does not remove); back up any user file before editing it; DCO sign-off.

## 2. What Claude Code offers (and does not) — verified facts
| # | Fact | Evidence |
|---|---|---|
| F1 | Plugin skills are namespaced `/<plugin>:<skill>`; the bare `/<skill>` also works when no other command owns it | [doc:skills.md][exp:ns E1, e1] |
| F2 | A user skill in `~/.claude/skills/x` and a plugin skill `p:x` both load side by side (duplicates) | [doc][exp:ns E2] |
| F3 | Plugin hooks (`hooks/hooks.json`) fire whenever the plugin is enabled; env has `CLAUDE_PLUGIN_ROOT/DATA`, `CLAUDE_PLUGIN_OPTION_<KEY>` | [doc][exp:ns E3] |
| F4 | A plugin CANNOT add `permissions.allow`, `statusLine` or any setting (plugin `settings.json` only honours the deprecated `agent` key) | [code:2.1.325][exp:e1 t_perm2] |
| F5 | A plugin CANNOT ship keybindings (only `<config>/keybindings.json`); mods have per-button hotkeys only | [doc:keybindings.md][code] |
| F6 | `userConfig` values reach skill BODIES (`${user_config.X}`), hook env, MCP config; NOT skill frontmatter/description; an unset option's `default` is NOT applied | [doc][exp:e1, ns E5] |
| F7 | `claude plugin install` never prompts; values via `--config K=V` or `claude plugin configure <id> --values-stdin` | [doc][exp:ns E5] |
| F8 | Directory marketplace: relative-path plugins run in place from the source tree (edits apply next session) | [doc][exp:ns E7] |
| F9 | Component paths must stay inside the plugin dir (`..` rejected); a symlink to elsewhere INSIDE the marketplace dir works (dereferenced into cache) | [doc][exp:ns E4, e1] |
| F10 | Bundle = plugin with only `name` + `dependencies`; one `claude plugin install` pulls all | [doc:deps.md] |
| F11 | A dependency is mandatory (plugin fails to load without it); cross-marketplace deps need `allowCrossMarketplaceDependenciesOn` AND the other marketplace added | [doc][exp:e1] |
| F12 | Repo `.claude/settings.json` `extraKnownMarketplaces` (directory `"./"`) + `enabledPlugins` loads relative-path plugins with no install step, after workspace trust | [doc][exp:e3] |
| F13 | Uninstall (last scope) removes enabledPlugins entry, pluginConfigs, data dir; cache orphaned (14 d); source and marketplace untouched. `marketplace remove` uninstalls all its plugins | [doc][exp:ns E6] |
| F14 | No post-install lifecycle hook exists; documented substitutes: SessionStart hook installing into `${CLAUDE_PLUGIN_DATA}`, `install --config`, `configure --values-stdin` | [doc] |
| F15 | `claude plugin list/details --json` give machine-readable status | [doc][exp] |

## 3. Features to keep (inventory; detail in research/existing_install.md, o11y_install.md)
Old install.sh: 8 skills linked (2 "gated" ones rendered per trigger mode `auto|explicit` (+aliases
on/off/enable/disable) into `$XDG_CONFIG_HOME/infera-agent-sys/rendered/<sha(checkout)>`); `--repo` scope;
`--hook` keyword-hint UserPromptSubmit hook (marker-found, re-pointed, de-duplicated, settings backup);
`--mode`; `--remote/--user/--clone` knowledge-repo setup (`astk remote init`, push dry-run) and offline
status; `--check` report-only with ok/linked/STALE/DANGLING/skipped/missing; python dep probe + pip hint;
claude-dashboard optional install (ask on tty / `--dashboard yes|no`, upstream-source check) + PR #118
two-file patch pinned to sha 99a8a7b… + calibrate note; never deletes/overwrites; runtime
`astk remote install-skill` links `pd-agentx-tuning` into `~/.claude/skills`.
o11y: directory marketplace `agent-sys-local` (3 plugins), env_mgr venv recipe `o11y.yaml` under
`AGENT_SYS_HOME`, permission allow rule `mcp__plugin_agent-sys-o11y-door_o11y__*`, keybindings snippet
(ctrl+q chord), `--plugin-dir` dev mode, optional `o11y-launch --otel` (docker + collector image).

## 4. Proposed design
### 4.1 One marketplace, plugins, a bundle
`agent_sys/.claude-plugin/marketplace.json` (root = `agent_sys/`, so it encloses `tools/skill`,
`kernel_registry` and `o11y`, which F9 requires for symlinks). Name: `agent-sys` (decide; see O8).
| Plugin | Contents | Notes |
|---|---|---|
| `agent-sys-skills` | `skills/<name>` symlinks → `../../tools/skill/<name>` (+ gated ones → rendered dir, 4.3); `hooks/hooks.json` hint hook (opt-in, 4.4) | skills become `agent-sys-skills:<name>` |
| `kernel-registry` | `skills/kernel-artifacts` → `../../kernel_registry/skills/kernel-artifacts` | |
| `agent-sys-o11y-door`, `agent-sys-o11y`, `agent-sys-o11y-dash` | unchanged, sources `./o11y/plugins/<p>` | the old `agent_sys/o11y` marketplace stays for dev, or migrates via `renames` (O8) |
| `agent-sys` (bundle) | `dependencies`: all of the above | the one install command |
NOT a dependency: claude-dashboard (F11 makes deps mandatory; the old behaviour is opt-in) → post script.

### 4.2 The command
`agent_sys/tools/install/install.sh` (thin, POSIX sh or a small Python CLI; the old path keeps a
forwarding stub so docs and habits still work):
1. `claude plugin marketplace add <agent_sys dir>` (`--scope project` for `--repo`, which also writes
   `extraKnownMarketplaces`/`enabledPlugins` in the repo's `.claude/settings.json`; F12, trust caveat).
2. `claude plugin install agent-sys@agent-sys [--config hint_hook=… --config trigger=…]` (F7).
3. Runtimes: env_mgr recipe(s) — the existing `o11y.yaml` plus a new one for astk's deps (pyyaml,
   ruamel.yaml, jsonschema, markdown-it-py, pydantic, packaging; and `kernel-registry` console script via
   `pip install -e agent_sys`) — replacing the old print-only pip hint. Keep the probe output.
4. Knowledge repo: unchanged `astk remote init|status` call, same flags and exit-code propagation.
5. `--check`: `claude plugin list --json` + `claude plugin details` + rendered-skill staleness + venv check +
   old-install detection (4.6), same status words as before; changes nothing.
6. Calls the post script (4.5) unless `--no-post`.
Uninstall: `install.sh --uninstall` = `claude plugin uninstall agent-sys@agent-sys --prune` (F13) and
reports (never deletes) what the post script added.

### 4.3 Trigger modes (gated skills) — the one real gap
F6: a plugin skill's description cannot depend on config. Options (decide by experiment, O1):
- (a) Keep the renderer; move its output INSIDE the marketplace, e.g. `agent_sys/.rendered/<mode>/` (git-
  ignored), and point the gated skill symlinks there. Directory marketplaces read in place (F8) and in-
  marketplace symlinks work (F9). `astk mode X` re-renders; takes effect next session. Smallest change.
- (b) Two plugins `agent-sys-entry-auto` / `agent-sys-entry-explicit` with committed SKILL.md variants;
  mode = which one is enabled (`claude plugin enable/disable`). Fully declarative, but duplicates two
  SKILL.md files (generate at build time).
Recommendation: (a) — keeps `astk mode/render` and the per-checkout model; (b) as fallback if a symlink
into a git-ignored dir misbehaves.

### 4.4 Keyword-hint hook (opt-in)
Plugin hook in `agent-sys-skills/hooks/hooks.json`, gated by userConfig boolean `hint_hook`: the script
exits 0 silently unless `CLAUDE_PLUGIN_OPTION_HINT_HOOK=true` (unset ⇒ off, because defaults are not
applied, F6). `install.sh --hook` = `--config hint_hook=true`. Migration: an old settings.json entry
(marker `/tools/skill/hooks/prompt_hint.py`) would run twice (doc) → report it; offer to remove only that
entry with backup and explicit consent.

### 4.5 Post-install script (R4) — things plugins cannot do (F4, F5)
`agent_sys/tools/install/post_install.sh`, each step asks (or `--yes`), backs up, never deletes:
1. Permission allow rule for the door tools (`mcp__plugin_agent-sys-o11y-door_o11y__*`) in user settings.
2. Keybindings: merge the o11y `ctrl+q` block into `keybindings.json` (skip on conflict, report).
3. claude-dashboard (optional, as before): `claude plugin marketplace add uppinote20/claude-dashboard` +
   `claude plugin install claude-dashboard@claude-dashboard` (official CLI), upstream-source check, then the
   PR #118 two-file patch from the pinned sha (unchanged logic, moved here), calibrate note. Re-runnable:
   a plugin update wipes the patch; `--check` reports it missing.
4. statusLine for the dashboard stays its own `/claude-dashboard:setup` (unchanged).

### 4.6 Migration from the symlink install
Old `~/.claude/skills/<name>` links (on this host they point at another checkout `…/infera/…`) would load
next to `agent-sys-skills:<name>` (F2). The installer detects links whose target is a `tools/skill/` or
`kernel_registry/skills/` dir of any checkout, reports them and prints the exact `rm` for the user; it
does not delete (house rule). Same for the old hint hook (4.4) and old rendered dirs.

### 4.7 Content changes the namespacing forces
- Cross-references by skill name (list in research/existing_install.md §2, ~40 sites): keep bare names
  (F1: bare works when unambiguous) or switch to `agent-sys-skills:<name>`; decide (O3) and regression-test
  triggering.
- `prompt_hint.py` SLASH regex `/(?:infera-agent-sys|agent-sys-[a-z-]+)\b` must also accept
  `/<plugin>:<skill>` (critic (a)3). o11y's trigger already strips the prefix [code:trigger/index.ts:29-32].
- `SKILL_ROOT="$(realpath "${CLAUDE_SKILL_DIR}/..")"` in five skills: verify what `CLAUDE_SKILL_DIR` holds
  for a symlinked plugin skill (O2); if it is the plugin path, compute the root differently.
- Permission rules `Skill(name)` vs `plugin:name`: verify (O4).

## 5. Open items for the implementing session (each with how to close it)
O1 Trigger-mode option (a): experiment — gated skill symlink → git-ignored rendered dir inside a directory
   marketplace; mode switch + next session shows the new description.
O2 `CLAUDE_SKILL_DIR` for a symlinked plugin skill (symlink path or resolved?) — one `-p` run.
O3 Cross-reference style (bare vs namespaced) — check bare-name resolution when the old user links are
   still present (they win the bare name, F2/doc precedence).
O4 `Skill(x)` permission rule vs `p:x`.
O5 Several checkouts on one host: a directory marketplace name points at one absolute path; define
   re-pointing (`marketplace remove` uninstalls its plugins, F13 — so prefer `marketplace update`/re-add
   semantics; test) and how `--check` shows which checkout is live.
O6 `--repo` (project scope): trust needed; `-p` loaded only on the second run [exp:e3]; check
   `CLAUDE_CODE_SYNC_PLUGIN_INSTALL=1`.
O7 Git-marketplace route (for users without a checkout): plugins are copied (F8 does not apply), symlinks
   dereferenced; the rendered-dir option (a) would not work there → (b) or render at build time.
O8 Marketplace naming / the existing `agent-sys-local` (o11y) marketplace: keep both, or migrate with the
   `renames` map [doc:plugins_host-marketplace.md].
O9 Tests: extend `agent_sys/tests/tools/` (runs in CI) with an isolated-`CLAUDE_CONFIG_DIR` install /
   check / uninstall round trip; old `--repo/--remote/--check` were never covered.

## 6. Defects found during this research (not install work)
- FIXED on the o11y branch (PR #252, 738788ee7): five skills named a door tool `event_report` that agents cannot call (internal
  since v3); now `ledger_append`, `session_register` like kernel-forge.
- `env_mgr/recipes/o11y.yaml` comment names a launcher `server/py.sh` that does not exist (real:
  `server/o11y-run`).
- install.sh `--mode` rejects the `disabled` alias that `settings.ALIASES` accepts.
- `agent-sys-handoff-packup` references `experiment-result-packup`, which is not in the repo (a user skill
  on this host only).
- The writer skill's CLI fallback path is repo-relative; breaks outside the source tree.
- Origin of `~/.config/infera-agent-sys/installations/` unknown (no writer on this branch).
