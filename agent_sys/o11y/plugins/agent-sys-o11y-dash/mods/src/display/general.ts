// The general tab (spec §7.3): task metric, baseline, target, best, spend, progress. Pure.
import { dur, num, pct, tok, type Line } from './fmt.ts'

export function general(g: any, wide: boolean, sessions: any[]): Line[] {
  const out: Line[] = []
  const t = g.task
  if (t) {
    if (wide) out.push([`${t.title}${t.phase ? ` [${t.phase}]` : ''}`])
    out.push([`${t.metric.name} (${t.metric.unit}, ${t.metric.direction} is better)`])
    out.push([`base ${num(t.baseline.value)}${wide && t.baseline.label ? ` (${t.baseline.label})` : ''} · target ${t.target ? num(t.target.value) : '-'}`])
  } else out.push(['task: not recorded yet (ledger kind=task)', 'yellow'])
  const b = g.best
  if (b) {
    out.push([`best ${num(b.value)}  ${pct(b.vs_baseline_pct)} vs base · ${b.of_target_pct == null ? '-' : b.of_target_pct.toFixed(1) + '%'} of target`, 'green'])
    if (wide) out.push([`  runs ${b.run_ids.join(', ')}${b.stack ? ` · ${b.stack}` : ''}`])
  }
  const s = g.spend
  out.push([`tokens ${tok(s.tokens_new)} new + ${tok(s.tokens_cache_read)} cache reads · ${dur(s.wall_s)}`])
  const p = g.progress
  out.push([`progress ${pct(p.gain_pct)} · ${pct(p.per_hour_pct)}/h · ${pct(p.per_mtok_pct)}/Mtok`])
  if (!wide) return out
  out.push([`  abs ${num(p.gain_abs, 1)} · ${num(p.per_hour_abs, 1)}/h · ${num(p.per_mtok_abs, 1)}/Mtok`])
  out.push(['sessions:'])
  for (const x of sessions) {
    const u = (s.sessions ?? []).find((y: any) => y.session_id === x.session_id)?.tokens
    const state = x.alive ? `${x.status}${x.waiting_for ? ` (${x.waiting_for})` : ''}` : 'DEAD'
    out.push([`  ${x.active ? '*' : ' '} ${x.role} ${x.name ?? x.session_id.slice(0, 8)} ${state} · ${tok(u?.new)} new ${tok(u?.cache_read)} cached`, x.alive ? undefined : 'red'])
  }
  for (const pr of g.problems ?? []) out.push([`! ${pr}`, 'yellow'])
  return out
}
