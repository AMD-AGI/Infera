// Counters (design §7, spec §4.4): main-session tool calls and turns → the analysis kinds to request. Pure.
export type JobKind = 'extract' | 'facts' | 'log'

export interface Counts { tool_calls: number; turns: number }

// `refresh` (config) is the frequent cheap pass: ledger extraction and display facts (spec §6.5); `log` is
// the levelled-record pass.
export function count(c: Counts, what: 'tool' | 'turn', cfg: any): { counts: Counts; kinds: JobKind[]; reason: 'tool_calls' | 'turns' } {
  const counts = what === 'tool' ? { ...c, tool_calls: c.tool_calls + 1 } : { ...c, turns: c.turns + 1 }
  const n = what === 'tool' ? counts.tool_calls : counts.turns
  const key = what === 'tool' ? 'every_n_tool_calls' : 'every_n_turns'
  const kinds: JobKind[] = []
  if (n % cfg.refresh[key] === 0) kinds.push('extract', 'facts')
  if (n % cfg.log[key] === 0) kinds.push('log')
  return { counts, kinds, reason: what === 'tool' ? 'tool_calls' : 'turns' }
}
