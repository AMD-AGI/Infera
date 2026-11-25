// The detail tab (spec §7.3): current run, last n runs, next planned point, machines. Pure.
import { num, pct, type Line } from './fmt.ts'

function runLine(r: any): Line {
  const m = (r.machines ?? []).map((x: any) => (typeof x === 'string' ? x : x.node)).join(',')
  return [`  ▶ ${r.run_id} ${r.theme}${r.point ? ` [${r.point}]` : ''}${m ? ` @${m}` : ''}`]
}

// What each session is doing now (detail.now, main first), each with its open runs under it, newest first:
// narrow shows the newest and "+N more", wide all with their hypotheses. The most important lines, so first.
export function nowLines(d: any, wide = false): Line[] {
  const runs = [...(d.current_runs ?? [])].sort((a, b) => String(b.started_at ?? '').localeCompare(String(a.started_at ?? '')))
  const mine = (n: any) => runs.filter((r) => r.session != null && (r.session === n.name || r.session === n.session_id))
  const groups: [Line, any[]][] = (d.now ?? []).map((n: any) => {
    const col = n.activity.startsWith('waiting') ? 'yellow' : n.activity === 'idle' ? 'dim' : undefined
    return [[`${n.name ?? n.session_id.slice(0, 8)} [${n.role}]: ${n.activity}`, col], mine(n)]
  })
  const placed = new Set(groups.flatMap(([, rs]) => rs))
  const rest = runs.filter((r) => !placed.has(r))
  if (rest.length) groups.push([['(session not recorded)', 'dim'], rest])
  if (!groups.length) return []
  return [['sessions:', 'bold'], ...groups.flatMap(([head, rs]): Line[] => [
    head,
    ...(wide ? rs.flatMap((r): Line[] => [runLine(r), ...(r.hypothesis ? [[`      hyp: ${r.hypothesis}`] as Line] : [])]) : rs.slice(0, 1).map(runLine)),
    ...(!wide && rs.length > 1 ? [[`    +${rs.length - 1} more`, 'dim'] as Line] : []),
  ])]
}

export function detail(d: any, wide: boolean, recentN: number): Line[] {
  const out: Line[] = [...nowLines(d, wide)]
  if (!d.current_runs.length) out.push(['no run in progress', 'dim'])
  out.push([`last ${Math.min(recentN, d.recent_runs.length)} runs:`])
  for (const r of d.recent_runs.slice(0, recentN)) {
    const c = r.new_best ? 'green' : r.status === 'failed' || r.status === 'rejected' ? 'red' : undefined
    out.push([`  ${r.status.padEnd(8)} ${num(r.value)} ${pct(r.delta_pct)} ${r.run_id} ${r.theme}`, c])
    if (wide && r.conclusion) out.push([`      ${r.conclusion}`])
  }
  out.push([`next: ${d.next.length ? d.next[0].theme : '-'}`])
  if (wide) for (const n of d.next.slice(1)) out.push([`      ${n.theme}${n.blocked_on ? ` (blocked: ${n.blocked_on})` : ''}`])
  for (const c of d.child_next ?? []) { // each child's own plan
    out.push([`  ${c.name ?? c.session_id.slice(0, 8)} next: ${c.next[0]?.theme ?? '-'}`])
    if (wide) for (const n of c.next.slice(1)) out.push([`      ${n.theme}${n.blocked_on ? ` (blocked: ${n.blocked_on})` : ''}`])
  }
  out.push([`machines (${d.machines.length}):`])
  for (const m of wide ? d.machines : d.machines.slice(0, 4)) {
    out.push([`  ${m.node}${wide && m.cluster ? `/${m.cluster}` : ''} → ${m.runs ?? 'idle'}${wide && m.gpu_util != null ? ` · gpu ${m.gpu_util}%` : ''}`])
  }
  if (wide) for (const pr of d.problems ?? []) out.push([`! ${pr}`, 'yellow'])
  return out
}
