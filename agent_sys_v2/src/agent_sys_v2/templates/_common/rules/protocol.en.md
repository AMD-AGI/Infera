# Task conventions

I manage this project's tasks with agent_sys: each task has a prompt file `tasks/<type>/<name>.md` and a workspace `workspaces/<type>/<name>/`, and the current directory is the project root. Below is what I would like you to do in every task.

## Files in the workspace

- `progress.md`: where the task stands; I and the downstream tasks read it to know how far it got. Its form is fixed: two head lines, an empty line, then only an unordered list, one line per item, conclusions only:
  ```
  Status: In progress
  Next: <one sentence>

  - A key fact or decision (the branch, a confirmed cause, a choice made by default)
  - [x] A step that is done
  - [ ] A step still planned
  ```
  The status is one of "In progress", "Blocked" and "Done". When everything is done, set the status to "Done" and add a third line `Finished: YYYY-MM-DD HH:MM`; take the time from `date '+%F %H:%M'`, do not write it yourself. agent_sys then runs the task's acceptance checks (the first message lists them; when one fails, you get its output), and a task that needs my review waits for it; only then is the task done and the downstream tasks start.
  List the planned steps as `- [ ]` from the start, tick each one off when it is done, and change the list when the plan changes; a finished task has no unticked item left (for a step you dropped, say why and remove it). How you got there and the details go into `log/` or report.md, not into progress.md.
- `report.md`: the report for me and the downstream tasks, starting with "## Conclusion". After the conclusion, write a "## Handoff" section, one line per item, `` `path` — what it is in one sentence ``: what the downstream tasks will use (data, scripts, screenshot directories and so on; paths relative to the workspace) and what the main directories of the workspace hold; branches, commits and PRs go there too, but not starting with a backtick (e.g. "branch `dev/x`, PR #5"), so that they are not taken for paths. Downstream tasks read this list first instead of searching the whole workspace.
- `issue.md`: the problems you ran into and how they were fixed, one `##` section per problem.
- `log/`: write the output of long commands here with `tee`; detailed experiment records go here too, not into progress.md.

## Language and task references

- Work in the language of the task prompt I wrote: a prompt in English means progress.md, report.md and the other files and your replies are in English; a prompt in Chinese means Chinese. The language of these conventions does not decide it. The head of progress.md may use either `Status`/`Next`/`Finished` with "In progress"/"Blocked"/"Done", or "状态/下一步/完成时间" with "进行中/阻塞/已完成".
- `@<type>/<name>` in a prompt names another task of this project: its prompt is `tasks/<type>/<name>.md` and its workspace `workspaces/<type>/<name>/`. Read its report.md and progress.md when you need them.
- The environment has these variables, and the acceptance checks run in the same one: `$AGENT_SYS_PROJECT` (the project directory), `$AGENT_SYS_TASK` (this task, `<type>/<name>`), `$AGENT_SYS_WORKSPACE` (the absolute path of this task's workspace), `$AGENT_SYS_MACHINES` (the machines this task may use, comma separated), `$AGENT_SYS_SKILLS` (the skills that come with agent_sys), `$AGENT_SYS_PYTHON` (the Python that runs agent_sys's own tools).

## When I need to decide

- Write the question into progress.md (status "Blocked", and Next says what you need from me), then end the round; carry on once I reply.
- With more than 2 questions, write a `questions.md` table with the columns `# | Question | Suggestion | Your answer`, the last column left empty for me (write a `|` inside a cell as `\|`); for a row I leave empty, follow your suggestion. When asking again, write a new questions.md and keep the old one under another name (for example `questions_1022.md`).
- Do not stop for choices that do not matter: take the conservative default and go on, write "Chosen by default: …" in progress.md, and list them under "Decisions made for you" in report.md for me to confirm afterwards.

## Waiting for jobs, files or remote state

- Run a check script that exits once the condition holds as a background command (`run_in_background`) with a long enough `timeout`; write in progress.md what you are waiting for, then end the round. A new round starts by itself when the command ends; there is nothing for you to do meanwhile. Do not wait with a sleep loop in the foreground.
- On "reached its background time limit", start the check again with a longer timeout, as the message says.

## Boundaries

- Write only into this task's workspace (and the code repositories the task asks you to change), never into another task's workspace. Temporary files go into this task's workspace as well, not into `/tmp` or anywhere else.
- To split off a subtask, write a draft prompt under `tasks/<type>/` (frontmatter with `parent` and `summary`) and do not start on it yourself; I add it once I have looked at it.
- Before anything that cannot be undone (a push, submitting cluster jobs, deleting data), write in progress.md what you are about to do.

## Lessons

- Write what is worth keeping for later tasks into `lessons/<name>.md` in the workspace; I review them and take them into the knowledge library. Only what cannot be read straight from the code or the docs, holds for other tasks too and has been verified; write it by mechanism, not by machine.
- When I give a standing rule in the conversation ("from now on, always …"), follow it right away and also write a lesson with `type: rule`. Memory is off in this environment; this is the only way to keep it.
- Format:
  ```markdown
  ---
  name: <short English name, words joined with hyphens>
  description: <one line, shown in the knowledge index>
  type: pitfall   # pitfall | env-fact | method | negative-result | rule
  source: <type>/<name>
  ---
  **What to do**: …  **Why**: …  **Where it applies**: …
  ```
