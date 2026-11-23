// The log tab (spec §7.3): the latest N records at or above a level. Pure.
import type { Line } from './fmt.ts'

export const LEVELS = ['INFO', 'WARN', 'ERROR'] as const
const NUM: Record<string, number> = { INFO: 9, WARN: 13, ERROR: 17 }
const COLOR: Record<string, string | undefined> = { WARN: 'yellow', ERROR: 'red' }

export function log(events: any[], wide: boolean, n: number, minLevel: string): Line[] {
  const min = NUM[minLevel] ?? 9
  const ev = events.filter((x) => (x.severity_number ?? NUM[x.severity_text] ?? 9) >= min).slice(-n)
  const head: Line = [`level ≥ ${minLevel} (f: change)`, 'dim']
  if (!ev.length) return [head, ['no events yet', 'dim']]
  return [
    head,
    ...ev.map((x): Line => {
      const who = wide && x.attributes?.['session.name'] ? ` ${x.attributes['session.name']}` : ''
      return [`${String(x.ts ?? '').slice(11, 19)} ${String(x.severity_text).padEnd(5)}${wide ? ` ${x.event_name}` : ''}${who} ${x.body}`, COLOR[x.severity_text]]
    }),
  ]
}
