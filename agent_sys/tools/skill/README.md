# Claude Code skills for infera / agent_sys

Skills that let plain Claude Code find, create, fill and run agent_sys tasks,
using agent_sys's own definitions — task, handoff, validator, closure — without
starting the agent_sys runtime, and share them through a team knowledge repo.

| skill | use it when | it asks you |
|---|---|---|
| `infera-agent-sys` | **entry point** — any LLM deployment / serving / perf / kernel / profiling / bench / PD / agentic request (or agent_sys named). Looks for prior work first, routes, offers to capture the result | — (respects `astk mode explicit`) |
| `agent-sys-prior-work` | find tasks and handoffs in the shared knowledge repo relevant to a request; read, judge, report | the repo URL, once |
| `agent-sys-task-from-history` | work done interactively should become a re-runnable task — preferably ONE large decoupled task | the scope; where to store it (3 suggestions); back it up? |
| `agent-sys-handoff-packup` | a result should become a handoff agent_sys accepts; calls skill 1 if no task exists | where to store it (3 suggestions); big files and logs; back it up? |
| `agent-sys-task-runner` | try, debug or learn a task/package in Claude Code; recommends ultracode and runs the graph as a Workflow | the run dir and `--var`s; back up the handoffs? |

The shared knowledge repo (`release/`, `<user>/staging/`, `<user>/draft/`, each
with `tasks/` and `handoffs/`) is described in `docs/knowledge_repo.md`.

## Install

```bash
sh agent_sys/tools/skill/install.sh                    # link all skills into ~/.claude/skills/
sh agent_sys/tools/skill/install.sh --hook             # + keyword hint hook (UserPromptSubmit)
sh agent_sys/tools/skill/install.sh --mode explicit    # entry skill only when agent_sys is named
sh agent_sys/tools/skill/install.sh --remote git@github.com:<org>/<repo>.git   # configure the knowledge repo
sh agent_sys/tools/skill/install.sh --check            # report, change nothing
```

Links, not copies: the skills track the checkout. The two entry skills
(`infera-agent-sys`, `agent-sys-prior-work`) link to per-checkout copies in
`~/.config/infera-agent-sys/rendered/<checkout hash>/` whose description follows
`astk mode` (explicit: "only when agent_sys is named"); the hint hook follows the
same switch. `--repo` installs into `<repo>/.claude/`. The installer never deletes
or overwrites: `settings.json` is backed up first, `--hook` from another checkout
re-points the existing hint hook, and a `settings.json` it cannot parse is left alone.
Requirements: `python3` that can import `pyyaml`, `ruamel.yaml`, `jsonschema`,
`markdown-it-py` (agent_sys's own dependencies) and `git`.

## Layout

```
agent-sys-*/SKILL.md, infera-agent-sys/   the skills (+ references/)
docs/primer.md            agent_sys in one page (what the skills rely on, from the code)
docs/calibration.md       how validators are set: strict on what matters, silent on the rest
docs/knowledge_repo.md    the shared repo's layout and lifecycle
lib/astk.py               the toolkit's CLI: python3 lib/astk.py <command> -h
lib/skillkit/             history, pkg, scaffold, hstore, validate, runner, wf, audit, secretscan,
                          remote (knowledge repo), search, settings (trigger mode)
lib/skillkit/templates/handoff_check.py   generic validator copied into generated packages
hooks/prompt_hint.py      optional UserPromptSubmit hook: domain keywords -> hint to use the entry skill
```

`astk` resolves its own path before importing, so a symlinked skill always
runs this checkout's agent_sys code. Every check goes through agent_sys's own
functions (`spec_loader.YamlPackage`, `handoff.readme.check`,
`handoff.content.check_items`, `handoff.digest.tree_digest`) and `agent-sys
show`; a rule agent_sys does not enforce is not invented, except the two the
skills add on purpose and say so: the 4 MB / logs question, and the secret scan.

## astk commands

| command | does |
|---|---|
| `history list / digest` | sessions of a project; a redacted, reviewable digest of chosen sessions |
| `scaffold PLAN DEST` | plan.yaml → package, then agent_sys `show` |
| `show`, `inspect` | agent_sys's static check; leaves in order with waits-for, kinds, validators |
| `template`, `check`, `seal`, `audit`, `digest` | build and prove a handoff; publish it into an agent_sys-shaped store |
| `run-init`, `seed`, `prepare`, `run-program`, `finish`, `status`, `closure`, `workflow`, `export` | execute a package leaf by leaf; the generated Workflow drives the same commands |
| `remote init / status / sync / list / search / backup / promote / release-pr` | the shared knowledge repo |
| `mode [auto\|explicit] [--ask-remote yes\|no] [--prompt ...] [--prior-work]` | when the entry skill engages (standard library only) |

Exit codes are listed by `astk -h`; 4 always means astk itself failed, never a verdict.

## What differs from an agent_sys run

AI leaves run as Claude Code agents reading the same prompt text; the workspace is
not a git clone; a failed validation is shown to the executor and retried (≤ 3).
Handoff layout, seal and gate tests, validator contract and verdict rows are the
same. Full list: `agent-sys-task-runner/references/workflow_patterns.md` § divergences.

Tests: `agent_sys/tests/tools/test_skillkit*.py` (run in the agent_sys CI job).
