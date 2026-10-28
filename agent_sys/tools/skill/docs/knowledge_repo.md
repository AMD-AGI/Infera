# The knowledge repo

A plain git repo shared by the team, holding agent_sys **tasks** (how to do a
kind of work) and **handoffs** (what someone achieved). Configured once per
user with `astk remote init --url <git url>`; the clone defaults to
`~/.cache/infera-agent-sys/knowledge`, the config to
`~/.config/infera-agent-sys/config.json`.

```
release/{tasks,handoffs}/<name>/          reviewed; changed ONLY by a merged pull request
<user>/draft/{tasks,handoffs}/<name>/     where backups land (astk remote backup)
<user>/staging/{tasks,handoffs}/<name>/   promoted from draft after the user's own review
```

A task item is the package as written (`main.yaml`, `steps/`, `assets/`,
`plan.task.yaml.txt`), symlinks dereferenced; a handoff item is `content/`
(`README.md` + `items/`), plus `manifest.yaml` and `validation.yaml` when it
was sealed. Each carries `META.astk.json`: type, name, user, tier, summary,
tags, context, source dir name and host as placeholders (`<source-dir>/<name>`,
`<host>`; no local path or host name is published), time, the agent_sys commit, and for a
handoff its digest. A tier name cannot be a user name (user `release` would
write straight into `release/`).

## Lifecycle

| step | command | who decides |
|---|---|---|
| back up | `astk remote backup <path> --summary "..." --tag ...` → `<user>/draft/` | the user, when a skill asks after producing a task or handoff |
| promote | `astk remote promote task\|handoff <name>` → `<user>/staging/` | the user, after reading it again |
| release | `astk remote release-pr task\|handoff <name>` → branch `release/<user>-<type>-<name>` + pull request | reviewers, by merging the PR |

Every write is one commit built on the remote's current tip with git plumbing
and pushed by hash; the local clone only fast-forwards. A failure (push race,
hook, bad input) changes nothing, two hosts backing up the same name get
`name` and `name.v2`, and nothing is ever deleted or overwritten.
`astk remote backup --dry-run` shows what would be published.

## What a backup refuses

Every file to be published is read once, streaming, audited, and exactly those
bytes are written (for a task, the targets of its symlinks; for a sealed
handoff, its records too). A handoff that changes while it is read is refused.

- A literal secret or unfilled template text: refused (exit 1). Every file is
  scanned whatever its size; gzip / bzip2 / xz are decompressed first (up to 1 GB),
  UTF-16 is decoded, and in a binary (a NUL in its first 8000 bytes) only the
  printable strings are read.
- A file over 4 MB: needs the user's yes, `--allow-big` (it is still scanned).
- What the scan cannot vouch for — zip / zstd / 7z, a corrupt stream, a
  decompression bomb, or a secret-looking string inside a binary (a string
  table as often as a secret): needs the user's yes after checking it by hand,
  `--allow-unscanned`.

## Finding things

`astk remote search <terms>` syncs and ranks every item by term overlap with
the request (Chinese domain words are expanded to English), weighted
release > staging > draft; the `agent-sys-prior-work` skill reads the top hits
and judges them. Anybody can push, so search and list never follow a symlink
out of an item, and a malformed `META.astk.json` field is normalised rather
than failing the listing for everybody.
