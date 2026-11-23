---
name: agent-sys-o11y-writer
description: Use when agent_sys work in a fleet of Claude Code sessions should be recorded for observability — a task's goal, metric, baseline or target is set; a run or experiment starts or finishes (result, kept/rejected/failed); the current best changes; the next plan changes; a machine is granted, lost or released; a KernelForge campaign is launched; hot operators or a kernel's stage or end-to-end verdict are known; an incident or a finding happens; or a child session (e.g. KernelForge in tmux) is about to be launched. Also when an automated "[o11y automated …]" prompt asks to record runs, complete the run ledger, or register a session. 记录运行台账 / 登记子会话 / 上报事件. Records go through the o11y tools (ledger_append, event_report, session_register), never by editing files.
---

# agent-sys-o11y-writer — record the fleet's facts through the one door

The o11y package keeps one **run ledger** per fleet (one main Claude Code session plus the child sessions
it launches). The display, the log analysis and the next session all read it. It is written only through
the agent door, which validates every record and tells you exactly what is missing or misplaced.
Spec and design: `agent_sys/o11y/docs/`.

## 1. The door

| Tool (MCP server `o11y`) | Use it to |
|---|---|
| `ledger_append {kind, record}` | record one fact; the description lists each kind's required fields |
| `ledger_schema {kind}` | read every field of a kind (types, enums, optional fields) before an unfamiliar record |
| `ledger_query` | read what is recorded (latest version of each fact) |
| `event_report` | report something notable that is not a ledger fact: a service up or down, a conclusion, an incident |
| `session_register` | register this session as main, a child you are about to launch, or decline a stray session |
| `fleet_status` | see the fleet: sessions, task, best, open runs, next plan, recent warnings |

The tools are offered only in fleet sessions. If they are absent, the session is not in a fleet; fall
back to the command `<plugin>/server/py.sh -m o11y_core ledger append '<json>'` (same validation;
`--help` lists the others) only when the user wants the record anyway.

## 2. When to record — one record per fact, at the moment it happens

| Moment | kind | Essentials |
|---|---|---|
| goal set or changed | `task` | metric {name, unit, direction}, baseline {value}, target {value} |
| a user or leader decision changes the work | `decision` | by, text |
| a run starts | `run_start` | run_id, theme, point (explore / validate / replicate / …), machines[{node}] |
| a run ends | `run_end` | status; value and evidence {path, pointer} for kept / neutral / rejected; conclusion or failure for the rest |
| the best stack changes | `best` | value, run_ids, stack |
| the next steps change | `plan` | next[{theme, why, blocked_on}] |
| an idea is opened, scheduled, rejected | `idea` | idea_id, title, status, reason |
| a machine is granted, lost, released | `machine` | event, node, runs (what it is for) |
| a KernelForge campaign is launched or resumed | `campaign` | campaign_root (absolute), operator_ids, started_at, machine{node} |
| a profile ranks hot operators | `hotops` | profile_id, role, ops[{name, share_pct}] |
| a kernel moves stage or gets an e2e verdict | `kernel` | op_id, stage, e2e {verdict, run_ids, delta_pct} |
| something went wrong | `incident` | severity, category, summary, impact |
| a mechanism is established or refuted | `finding` | text, evidence_rank |

## 3. Rules

- **Never edit the ledger file.** To correct a record, append a new record with the same identity
  (`run_id`, `op_id`, `campaign_root`, …); the later one supersedes it. A direct edit is reported back at the
  next tool call.
- **Do not invent values.** If a required value (a result, an evidence path) is not known, ask or record
  the run as `failed` / `aborted` / `unscored` with a conclusion — never guess a number.
- **Evidence is a path the next reader can open**: the result file, and a JSON pointer to the number
  inside it when it is JSON.
- **Who wrote it is filled in for you** (session, role, time); do not pass `v`, `id`, `ts` or `writer`.
- **A child session is launched with the env `session_register` returns** (`AGENT_SYS_O11Y=child`,
  `AGENT_SYS_O11Y_FLEET_DIR`, `AGENT_SYS_O11Y_PARENT_SESSION_ID`), or the fleet cannot follow it. Children
  are reached only through the main session: when an automated prompt asks you to relay, tell each child.
- **A problem list means fix and call again**, with the whole record. Unknown fields are refused with a
  hint where they belong (e.g. `node -> machines[].node`).
