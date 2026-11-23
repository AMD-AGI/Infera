// The detail tab (spec §7.3): current run, last n runs, next planned point, machines. Pure.
import { num, pct, type Line } from './fmt.ts'

export function detail(d: any, wide: boolean, recentN: number): Line[] {
  const out: Line[] = []
  for (const r of d.current_runs) {
    const m = (r.machines ?? []).map((x: any) => (typeof x === 'string' ? x : x.node)).join(',')
    out.push([`▶ ${r.run_id} ${r.theme}${r.point ? ` [${r.point}]` : ''}${m ? ` @${m}` : ''}`, 'cyan'])
    if (wide && r.hypothesis) out.push([`    hyp: ${r.hypothesis}`])
  }
  if (!d.current_runs.length) out.push(['no run in progress', 'dim'])
  out.push([`last ${Math.min(recentN, d.recent_runs.length)} runs:`])
  for (const r of d.recent_runs.slice(0, recentN)) {
    const c = r.new_best ? 'green' : r.status === 'failed' || r.status === 'rejected' ? 'red' : undefined
    out.push([`  ${r.status.padEnd(8)} ${num(r.value)} ${pct(r.delta_pct)} ${r.run_id} ${r.theme}`, c])
    if (wide && r.conclusion) out.push([`      ${r.conclusion}`])
  }
  out.push([`next: ${d.next.length ? d.next[0].theme : '-'}`])
  if (wide) for (const n of d.next.slice(1)) out.push([`      ${n.theme}${n.blocked_on ? ` (blocked: ${n.blocked_on})` : ''}`])
  out.push([`machines (${d.machines.length}):`])
  for (const m of wide ? d.machines : d.machines.slice(0, 4)) {
    out.push([`  ${m.node}${wide && m.cluster ? `/${m.cluster}` : ''} → ${m.runs ?? 'idle'}${wide && m.gpu_util != null ? ` · gpu ${m.gpu_util}%` : ''}`])
  }
  if (wide) for (const pr of d.problems ?? []) out.push([`! ${pr}`, 'yellow'])
  return out
}
