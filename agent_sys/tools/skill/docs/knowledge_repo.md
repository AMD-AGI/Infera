# The knowledge repo

A plain git repo shared by the team, holding agent_sys **tasks** (how to do a
kind of work) and **handoffs** (what someone achieved). Configured once per
user with `astk remote init --url <git url>`; the clone defaults to
`~/.cache/infera-agent-sys/knowledge`, the config to
`~/.config/infera-agent-sys/config.json`.

```
repo_root/
  release/                     reviewed; changed ONLY by a merged pull request
    tasks/<name>/
    handoffs/<name>/
  <user>/
    draft/                     where backups land (astk remote backup)
      tasks/<name>/
      handoffs/<name>/
    staging/                   promoted from draft after the user's own review
      tasks/<name>/
      handoffs/<name>/
```

An item:

| type | directory holds |
|---|---|
| task | the package as written (`main.yaml`, `steps/`, `assets/`, `plan.task.yaml.txt`), symlinks dereferenced |
| handoff | `content/` (`README.md` + `items/`), plus `manifest.yaml` and `validation.yaml` when it was sealed |

and `META.astk.json`: type, name, user, tier, one-line summary, tags, source
path and host, time, the agent_sys commit, and for a handoff its digest.

## Lifecycle

| step | command | who decides |
|---|---|---|
| back up | `astk remote backup <path> --summary "..." --tag ...` → `<user>/draft/` | the user, when a skill asks after producing a task or handoff |
| self-review, promote | `astk remote promote task\|handoff <name>` → `<user>/staging/` (one commit moving it) | the user, after reading it again |
| release | `astk remote release-pr task\|handoff <name>` → a branch + pull request adding `release/...` | reviewers, by merging the PR |

Every write is built as a commit on the remote's current tip with git
plumbing and pushed by hash; the local clone is only ever fast-forwarded. A
failure (push race, hook, bad input) therefore changes nothing locally or
remotely, two hosts backing up the same name get `name` and `name.v2`, and the
toolkit never deletes or overwrites. `release/` is written only on a
`release/<user>-<type>-<name>` branch proposed as a pull request.
`astk remote backup --dry-run` shows what would be published.

Before any backup it audits exactly what will be published — for a task,
including the targets of its symlinks: a literal secret or unfilled template
text refuses (exit 1); a file over 4 MB needs the user's yes (`--allow-big`).

## Finding things

`astk remote search <terms>` syncs and ranks every item in release and in every
user's staging and draft by term overlap with the request (Chinese domain words
are expanded to English), weighted release > staging > draft. The
`agent-sys-prior-work` skill reads the top hits and judges them.
