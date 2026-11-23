// Collector health (design §7, §10): only for a session launched through o11y-launch, which sets
// AGENT_SYS_O11Y_OTEL_HEALTH_URL. Each tick: sample, write the outage edges to otel/health.json (o11y_log turns
// them into events), and run the launcher's sweep.
import { P, nowIso, readDoc, writeDoc } from '../../../protocol/ts/index.ts'
import { launch, type SpawnCtx } from './spawn.ts'

let health: any = null

export async function sample($: any, c: SpawnCtx, url: string): Promise<void> {
  let healthy = false
  try {
    healthy = (await $.http.fetch(url)).ok === true
  } catch {}
  if (!health) health = (await readDoc($, `${c.fleet}/${P.otelHealth}`, 'otel_health')).value
  const changed = !health || health.healthy !== healthy
  const outages = (health?.outages ?? 0) + (health?.healthy !== false && !healthy ? 1 : 0)
  const last_change = changed ? nowIso() : health.last_change
  // down_since: set while down, absent while healthy
  const down_since = healthy ? undefined : changed ? last_change : health.down_since
  health = { v: 1, ts: nowIso(), healthy, outages, ...(last_change ? { last_change } : {}), ...(down_since ? { down_since } : {}) }
  await writeDoc($, `${c.fleet}/${P.otelHealth}`, health)
}

// The sweep stops the collector when no launched session is alive; detached, so it outlives a hook budget.
export async function sweep($: any, c: SpawnCtx): Promise<void> {
  const launcher = `${c.pluginRoot}/bin/o11y-launch`
  if (await $.fs.exists(launcher)) await launch($, c, [launcher, '--sweep'])
}
