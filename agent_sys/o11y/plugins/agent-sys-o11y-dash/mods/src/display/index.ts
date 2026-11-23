// D — the display mod of plugin agent-sys-o11y-dash (design §9): commands and event wiring. Dormant on its
// own AGENT_SYS_O11Y veto; otherwise active while the registry lists this session as an alive, active main.
// Reads only facts/*, logs/events.jsonl, triggers/*, registry/sessions.json and config/*; writes only
// config/*.json through the forms.
import { FORM, openForm, renderForm } from './forms.ts'
import { MENU, PANE, bandShown, fleetRoot, isActive, openMenu, renderBand, renderMenu, renderPane, setBand, setMode, showTab, start } from './frame.ts'

const COMMANDS = [
  { name: 'o11y', description: 'o11y: menu (g/d/k/l tabs, m mode, b band, t/a/v config)' },
  { name: 'o11y-general', description: 'o11y: general tab' },
  { name: 'o11y-detail', description: 'o11y: detail tab (runs, plan, machines)' },
  { name: 'o11y-kf', description: 'o11y: KernelForge tab' },
  { name: 'o11y-log', description: 'o11y: log tab' },
  { name: 'o11y-mode', description: 'o11y: switch NARROW / WIDE', argumentHint: '[narrow|wide]' },
  { name: 'o11y-band', description: 'o11y: band above the prompt on/off', argumentHint: '[on|off]' },
  { name: 'o11y-cfg-trigger', description: 'o11y: edit the trigger configuration' },
  { name: 'o11y-cfg-analysis', description: 'o11y: edit the analysis configuration' },
  { name: 'o11y-cfg-display', description: 'o11y: edit the display configuration' },
]

let vetoReason = ''

async function command($: any, name: string, args: string): Promise<any> {
  if (!isActive()) {
    const why = vetoReason || `this session is not the active main of fleet ${fleetRoot()} (AGENT_SYS_O11Y=true or a first agent_sys skill use makes it one; the display follows within one trigger tick)`
    return { text: `o11y display inactive: ${why}`, isTextDisplayOnly: true }
  }
  const a = (args ?? '').trim()
  if (name === 'o11y') await openMenu($)
  else if (name === 'o11y-general') await showTab($, 'general')
  else if (name === 'o11y-detail') await showTab($, 'detail')
  else if (name === 'o11y-kf') await showTab($, 'kernelforge')
  else if (name === 'o11y-log') await showTab($, 'log')
  else if (name === 'o11y-mode') await setMode($, a)
  else if (name === 'o11y-band') setBand($, a)
  else if (name.startsWith('o11y-cfg-')) await openForm($, fleetRoot()!, name.slice(9) as 'trigger' | 'analysis' | 'display')
  return {}
}

export function register(on: any) {
  on('session.start', async ($: any, e: any, next: any) => {
    for (const c of COMMANDS) {
      try {
        await $.command.register({ ...c, immediate: true })
      } catch (err: any) {
        $.ui.log(`o11y: /${c.name} not registered: ${err.message}`)
      }
    }
    const flag = await $.env.get('AGENT_SYS_O11Y')
    if (flag !== undefined && flag !== 'true') vetoReason = `AGENT_SYS_O11Y=${flag}; the display runs only in a fleet's main session`
    else await start($)
    return next(e)
  })

  for (const c of COMMANDS) {
    on('command.run', { command: c.name }, async ($: any, e: any) => command($, e.command, e.args))
  }

  on('ui.render', { component: 'Pane' }, async ($: any, e: any, next: any) => {
    if (e.requestId === FORM) return renderForm($, e)
    if (e.requestId === MENU) return renderMenu($, e)
    if (e.requestId === PANE && isActive()) return renderPane($, e)
    return next(e)
  })

  on('ui.render', { component: 'AbovePrompt' }, async ($: any, e: any, next: any) => (bandShown() ? renderBand($, e) : next(e)))
}
