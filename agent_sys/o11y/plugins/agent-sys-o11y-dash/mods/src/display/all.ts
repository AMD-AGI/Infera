// The all tab (tab 0, the default; spec §7.3): a brief of every other tab, then the configuration in effect,
// so the user need not switch tabs. Pure.
import { dur, num, pct, tok, type Line } from './fmt.ts'
import { byRole } from './kernelforge.ts'
import { log } from './log.ts'

export interface Brief { general: any; detail: any; kernelforge: any; problems: Record<string, string | null>; ages: Record<string, string | null>
  events: any[]; trigger: any; analysis: any; taskBook: string }

const head = (name: string, age: string | null, problem: string | null): Line => [`── ${name}${problem ? ` · ${problem}` : age ? ` · ${age}` : ''}`, 'dim']

export function all(b: Brief): Line[] {
  const out: Line[] = []
  const g = b.general
  out.push(head('general', b.ages.general, b.problems.general))
  if (g) {
    const t = g.task
    out.push(t ? [`${t.metric.name}: base ${num(t.baseline.value)} · best ${num(g.best?.value)} ${pct(g.best?.vs_baseline_pct)} · target ${t.target ? num(t.target.value) : '-'}`, g.best ? 'green' : undefined] : ['task: not recorded yet', 'yellow'])
    out.push([`tokens ${tok(g.spend.tokens_new)}+${tok(g.spend.tokens_cache_read)}c in ${dur(g.spend.wall_s)} · ${pct(g.progress.per_hour_pct)}/h`])
  }
  const d = b.detail
  out.push(head('detail', b.ages.detail, b.problems.detail))
  if (d) {
    for (const r of d.current_runs.slice(0, 2)) out.push([`▶ ${r.run_id} ${r.theme}`, 'cyan'])
    const r = d.recent_runs[0]
    if (r) out.push([`last ${r.status} ${num(r.value)} ${pct(r.delta_pct)} ${r.run_id} ${r.theme}`, r.new_best ? 'green' : undefined])
    out.push([`next: ${d.next[0]?.theme ?? '-'} · machines ${d.machines.length}`])
    for (const c of d.child_next ?? []) out.push([`${c.name ?? c.session_id.slice(0, 8)} next: ${c.next[0]?.theme ?? '-'}`])
  }
  const k = b.kernelforge
  out.push(head('KernelForge', b.ages.kernelforge, b.problems.kernelforge))
  if (k) {
    for (const [role, hs] of byRole(k.hot_ops)) {
      const ops = hs[0]?.ops.slice(0, 3).map((o: any) => `${o.name} ${o.share_pct.toFixed(0)}%`).join(' · ')
      out.push([`${role}: ${ops || 'not focused'}`, ops ? undefined : 'dim'])
    }
    const running = k.campaigns.filter((c: any) => c.status === 'running').length
    out.push([`campaigns ${k.campaigns.length} (${running} running) · ops ${k.ops.length}`])
  }
  out.push(head('log', null, null))
  out.push(...log(b.events, false, 3, 'INFO').slice(1))
  const tc = b.trigger
  const ac = b.analysis
  out.push(head('config in effect', null, null))
  out.push([`kick ${dur(tc.kick.interval_s)} · packup /${tc.packup.every_n_kicks} kicks or ${dur(tc.packup.interval_s)}`])
  out.push([`task book ${b.taskBook}`])
  out.push([`refresh /${tc.refresh.every_n_tool_calls} calls /${tc.refresh.every_n_turns} turns · rotate ${tc.rotate_mb} MB`])
  out.push([`log /${tc.log.every_n_tool_calls} calls /${tc.log.every_n_turns} turns · slow ${dur(tc.log.slow_interval_s)}`])
  out.push([`extract ${ac.extract.model}/${ac.extract.effort} · log ${ac.log.model}/${ac.log.effort}`])
  return out
}
