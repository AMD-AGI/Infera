# Task — Make `dev.yihou.aiopt.0.1` mergeable into `main`

> Previous round's file: `/home/yihou/ws_cleanup_0.1_20260914T092025Z/CLAUDE.e2e-verify.20260914-0920.md.bak`
> — it is about running the e2e chain on a GPU cluster. **This round touches no
> GPU, no cluster, no container.** Read it only if you need mechanism.

Branch `dev.yihou.aiopt.0.1` currently carries **2 squashed commits, 1200 files
changed vs `main`**. It is the branch intended for a real merge. It still
contains run records, run artefacts, local host information and the whole
development process trail. Strip it down to the product.

**Workspace: `/home/yihou/ws_cleanup_0.1_20260914T092025Z/`.** All scratch,
audits and reports live there. Nothing temporary is written into the repo.

**The deliverable is a cleaned-up branch plus a report to the user, in Chinese,
naming every deleted file.**

## The six requirements, from the user (2026-09-14)

1. Delete run records, run artefacts, local information and process records.
   **Delete every change outside `agent_sys/` and `deploy/docker/`.**
2. Delete all development-process documents. Keep only
   **spec.md / design.md / readme / TODO / roadmap**-class documents.
3. Purge any local information and any process information from what remains.
4. Delete the **standalone pre-chaining llm-e2e stage packages**; keep only the
   refined version that runs all five stages in one chain (`e2e-flow/`).
5. Clean the demos hard: local information, process information, logs, test
   results, run artefacts.
6. Report the full scope of the edit, including every deleted file.

## What counts as what — the working definitions

These are the judgement calls. Apply them the same way everywhere.

| Class | Keep? | Recognisers |
|---|---|---|
| **Product doc** | keep | `spec*.md`, `design*.md`, `README.md`/`readme.md`, `TODO.md`/`todo.md`, `ROADMAP.md`, `interfaces.md`, JSON schemas |
| **Process doc** | delete | dated filenames (`bug.record.2026-09-05.md`), run/jobid in the name (`*-298750.md`), `PRE-REGISTER`, `*-CHECKLIST`, `*-HANDOVER`, `*-VERDICT`, `*-WATCH`, `*-REACHABILITY`, `LAUNCH-*`, `*.proposal.md`, `*.patch.md`, `*review*result*`, `work.checkpoint.summary.md`, anything whose subject is "what we tried and when" |
| **Run artefact** | delete | `launch-records/`, `*.LAUNCH-RECORD.txt`, `*.out`, `*.log`, run trees, copied `verdict.json`, captured traces, benchmark outputs, `*.bak` |
| **Local information** | scrub | host names (`crsuse2-m2m-*`, `smci355-*`), IPs, slurm jobids, `/shared_nfs/...`, `/data/yihou/...`, `/apps/...`, `/home/yihou/...`, container names, ports bound on a shared host, a named person, a dated measurement presented as a constant |
| **Process information inside a kept file** | scrub | "on 2026-09-05 we found...", "m3 discovered...", "this cost us a run", counts of launches, references to deleted packages (`analyze-demo`, `integration-demo`, ...), narrative of past failures |

**A comment that explains why the code is the way it is, stated as a property of
the code, stays.** A comment that narrates the day it was discovered goes.
This is the line: *keep the reason, drop the incident.*

## Core principles

1. **Every deletion is `git rm`, and the branch keeps a backup ref.**
   `bak.squash.20260914` holds the pre-cleanup tree; `origin/dev.yihou.aiopt.0.1`
   holds the pushed pre-cleanup state. Nothing here is unrecoverable — but say
   what you removed, do not remove quietly.
2. **Deletion rule (user standing rule 7):** nothing whose path lacks `yihou` or
   `/tmp`. Every repo path here is under `/home/yihou/dev/git/...`, so the
   substring holds; this is recorded, not assumed. Nothing outside the repo and
   the workspace is touched.
3. **Do not break the package.** `e2e-flow/` must still load:
   `PYTHONPATH=$PWD/agent_sys python3 -m cli.main show --package <dir> --var ...`
   type-checks every yaml in < 1 s. **Run it after every batch of deletions.**
   `PYTHONPATH` is mandatory — a bare `agent-sys` resolves to a different worktree.
4. **Do not rewrite code to make it prettier.** This is a removal task. A file is
   either removed, or edited only to remove local/process information.
5. **Scrubbing is an edit with a test:** after scrubbing, the file must still read
   as a complete statement. A dangling "as described above" is a defect.
6. **`git grep` the whole tree after deleting a directory** — a deleted package
   referenced by name from a kept file leaves process information behind.
7. **Read the artefact, not the exit code.** Every claim in the final report names
   a file and a condition.
8. Work in English. Report to the user in Chinese.

## Team

Leader polls every 20 min. A problem seen once is recorded; intervened on at the
second sighting. Owners are split by directory so no two agents edit one file.

## Verification before the report

- `git diff --name-only main HEAD | grep -vE '^(agent_sys/|deploy/docker/)'` -> empty.
- `git grep -nE 'crsuse2|smci355|shared_nfs|/data/yihou|/apps/|10\.245\.|spur' -- agent_sys deploy`
  -> only entries that are genuinely generic or deliberately parameterised.
- `git ls-files agent_sys/examples/llm_e2e_performance_optimization` -> `e2e-flow/`
  and product docs only.
- The package still loads.
