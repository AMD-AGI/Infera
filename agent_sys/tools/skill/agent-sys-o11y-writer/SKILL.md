---
name: agent-sys-o11y-writer
description: Use when agent_sys work in a fleet of Claude Code sessions should be recorded for observability — a task's goal, metric, baseline or target is set; a run or experiment starts or finishes (result, kept/rejected/failed); the current best changes; the next plan changes; a KernelForge campaign is launched; hot operators or a kernel's stage or end-to-end verdict are known; an incident or a finding happens; or a child session (e.g. KernelForge in tmux) is about to be launched. Also when an automated "[o11y automated …]" prompt asks to record runs, complete the run ledger, or register a session. 记录运行台账 / 登记子会话. Records go through the o11y tools (ledger_append, session_register), never by editing files.
---

# agent-sys-o11y-writer — record the fleet's facts through the one door

The o11y package keeps one **run ledger** per fleet (one main Claude Code session plus the child sessions
it launches). The display, the log analysis and the next session all read it. It is written only through
the door, which validates every record and answers a bad one with the exact problems and the kind's full
schema. Spec and design: `agent_sys/o11y/docs/`.

## 1. The door

| Tool (MCP server `o11y`) | Use it to |
|---|---|
| `ledger_append {kind, record}` | record one fact; `record` holds the kind's fields (§2) |
| `ledger_query {kind?, run_id?, task_id?, since?, limit?}` | read what is recorded (latest version of each fact) |
| `session_register {decision, purpose, …}` | `child`: a child you are about to launch (`name` or `tmux_target`; returns its env and launch argv) or a running session you adopt (`session_id`); `not_in_fleet`: a session that is not yours (`session_id`) |
| `fleet_status` | see the fleet: sessions, task, best, open runs, next plan, recent warnings |

The main session is registered by the package itself; never register it. The tools are offered only in
fleet sessions. Without them, and only when the user wants the record anyway, use the command
`<door plugin>/bin/py -m o11y_door ledger append '<json with kind and the fields>'` (same validation;
`<door plugin>` is `agent_sys/o11y/plugins/agent-sys-o11y-door`; `--help` lists the other commands).

## 2. When to record — one record per fact, at the moment it happens

Required fields first; `(…)` = useful optional fields.

| Moment | kind | Fields |
|---|---|---|
| goal set or changed | `task` | task_id, title, metric {name, unit, direction: higher\|lower}, baseline {value} (target {value}) |
| a user or leader decision changes the work | `decision` | decision_id, by: user\|leader\|child\|external, text |
| a run starts | `run_start` | run_id, theme, point: explore\|validate\|replicate\|baseline\|repro\|screen\|diagnostic\|calibration\|profile (hypothesis, machines [{node}]) |
| a run ends | `run_end` | run_id, status; kept\|neutral\|rejected need value and evidence {path (pointer)}; failed\|aborted\|cancelled\|unscored\|inconclusive\|done need a conclusion |
| the best changes | `best` | value, run_ids (stack, vs_baseline_pct) |
| the next steps change | `plan` | next [{theme (why, blocked_on)}] |
| an idea is opened, scheduled, rejected | `idea` | idea_id, title, status (reason) |
| a KernelForge campaign is launched or resumed | `campaign` | campaign_root (absolute), operator_ids, started_at (RFC 3339), machine {node} |
| a profile ranks hot operators | `hotops` | profile_id, role: prefill\|decode\|colocated\|verify\|extend\|mixed, ops [{name, share_pct}] |
| a kernel moves stage or gets an e2e verdict | `kernel` | op_id, stage (e2e {verdict, run_ids, delta_pct}) |
| something went wrong | `incident` | incident_id, severity: info\|warning\|error, category, summary (impact, node) |
| a mechanism is established or refuted | `finding` | finding_id, text, evidence_rank: first_hand\|second_hand\|inference |

Machines granted, lost or released are recorded by the package from the machine pool; put the node of a run
in its `machines`.

## 3. Rules

- **Never edit the ledger file.** To correct a record, append a new record with the same identity
  (`run_id`, `op_id`, `campaign_root`, …); the later one supersedes it. A direct edit or an invalid
  out-of-door line is reported back at the next tool call.
- **Do not invent values.** If a required value (a result, an evidence path) is not known, ask or record
  the run as `failed` / `aborted` / `unscored` with a conclusion — never guess a number.
- **Evidence is a path the next reader can open**: the result file, and a JSON pointer to the number
  inside it when it is JSON.
- **Who wrote it is filled in for you** (session, role, time); do not pass `v`, `id`, `ts` or `writer`.
- **A child session is launched with the env `session_register` returns** (`AGENT_SYS_O11Y=child`,
  `AGENT_SYS_O11Y_FLEET_DIR`, `AGENT_SYS_O11Y_PARENT_SESSION_ID`), or the fleet cannot follow it. Children
  are reached only through the main session: when an automated prompt asks you to relay, tell each child.
- **A refusal means fix and call again**, with the whole record: it lists the problems (a misspelt key is
  named next to the field it resembles) and the kind's full schema.
