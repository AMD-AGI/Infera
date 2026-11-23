// Schedule (design §7, spec §4.1–4.3, §3.5): which automated prompt is due and its text. Pure functions
// over plain values; index.ts submits and records.
import { quoted, sanitize } from '../../../protocol/ts/index.ts'
import type { Candidate } from './registry.ts'

export const MARK = { kick: '[o11y automated kick] ', packup: '[o11y automated packup] ', question: '[o11y automated question] ' }
export type PromptKind = keyof typeof MARK
export const LEDGER_TOOL = 'mcp__plugin_agent-sys-o11y-door_o11y__ledger_append'

export interface Schedule { next_kick_at: number; kicks: number; kicks_since_packup: number; last_packup_at: number }
export interface Pending { kind: PromptKind; submitted_at: number; subject?: string }

export function fresh(now: number, cfg: any): Schedule {
  return { next_kick_at: now + cfg.kick.interval_s * 1000, kicks: 0, kicks_since_packup: 0, last_packup_at: now }
}

// The kick due now, if any, and the schedule after it. An absolute schedule: missed intervals are not replayed.
export function dueKick(s: Schedule, cfg: any, now: number): { kind: PromptKind; next: Schedule } | null {
  if (now < s.next_kick_at) return null
  let next_kick_at = s.next_kick_at
  while (next_kick_at <= now) next_kick_at += cfg.kick.interval_s * 1000
  const packup = s.kicks_since_packup + 1 >= cfg.packup.every_n_kicks || now - s.last_packup_at >= cfg.packup.interval_s * 1000
  return packup
    ? { kind: 'packup', next: { next_kick_at, kicks: s.kicks + 1, kicks_since_packup: 0, last_packup_at: now } }
    : { kind: 'kick', next: { ...s, next_kick_at, kicks: s.kicks + 1, kicks_since_packup: s.kicks_since_packup + 1 } }
}

// One pending prompt at most (spec §4.3). A pending entry restored after a reload (its promise gone)
// blocks until the next turn starts; three intervals cap a lost turn.start.
export function isPending(p: Pending | null, live: boolean, now: number, cfg: any): boolean {
  if (!p) return false
  return live || now - p.submitted_at < 3 * cfg.kick.interval_s * 1000
}

function render(template: string, vars: Record<string, string>): string {
  return template.replace(/\{(\w+)\}/g, (m, k) => (k in vars ? vars[k] : m))
}

// Children are named as quoted data after the template's fixed words, never as free text.
export function childrenText(sessions: any[]): string {
  const kids = sessions.filter((s) => s.role === 'child' && s.alive)
  if (!kids.length) return 'none'
  return kids.map((s) => `[${quoted('session_id', s.session_id)} ${quoted('name', s.name)} ${quoted('tmux', s.tmux)}]`).join(' ')
}

export function kickText(kind: 'kick' | 'packup', cfg: any, taskBook: string, sessions: any[]): string {
  const vars = { task_book: quoted('path', sanitize(taskBook)), children: childrenText(sessions), ledger_tool: LEDGER_TOOL }
  return MARK[kind] + render(kind === 'packup' ? cfg.packup.template : cfg.kick.template, vars)
}

// The fixed question (spec §3.5); the session's own strings follow as quoted data.
export function questionText(c: Candidate): string {
  return (
    MARK.question +
    'A live session that is not registered in this fleet runs in your tmux server under the fleet directory. ' +
    'Either register it as your child with its purpose (session_register, decision child), or declare it not part ' +
    'of the fleet (session_register, decision not_in_fleet). Session data: ' +
    [quoted('session_id', c.sid), quoted('name', c.name), quoted('tmux', c.tmux), quoted('cwd', c.cwd)].join(' ')
  )
}

export function pickCandidate(candidates: Candidate[], asked: string[]): Candidate | null {
  return candidates.find((c) => !asked.includes(c.sid)) ?? null
}
