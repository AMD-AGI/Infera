// D — the display mod of plugin agent-sys-o11y-dash (design §9): commands and event wiring. Dormant on its
// own AGENT_SYS_O11Y veto; otherwise active while the registry lists this session as an alive, active main.
// Reads only facts/*, logs/events.jsonl, triggers/*, registry/sessions.json and config/*; writes only
// config/*.json through the forms.
import { FORM, openForm, renderForm } from './forms.ts'
import { MENU, PANE, type Tab, bandShown, fleetRoot, isActive, openMenu, renderBand, renderMenu, renderPane, setBand, setMode, showTab, start } from './frame.ts'

const CMD = 'as_o11y' // the command prefix: /as_o11y (menu), /as_o11y-<sub>
const COMMANDS = [
  { name: CMD, description: 'agent_sys o11y: menu (0/g/d/k/l tabs, m mode, b band, t/a/v config)' },
  { name: `${CMD}-all`, description: 'agent_sys o11y: all tab (a brief of every tab and the configuration)' },
  { name: `${CMD}-general`, description: 'agent_sys o11y: general tab' },
  { name: `${CMD}-detail`, description: 'agent_sys o11y: detail tab (runs, plan, machines)' },
  { name: `${CMD}-kf`, description: 'agent_sys o11y: KernelForge tab' },
  { name: `${CMD}-log`, description: 'agent_sys o11y: log tab' },
  { name: `${CMD}-mode`, description: 'agent_sys o11y: switch NARROW / WIDE', argumentHint: '[narrow|wide]' },
  { name: `${CMD}-band`, description: 'agent_sys o11y: band above the prompt on/off', argumentHint: '[on|off]' },
  { name: `${CMD}-cfg-trigger`, description: 'agent_sys o11y: edit the trigger configuration' },
  { name: `${CMD}-cfg-analysis`, description: 'agent_sys o11y: edit the analysis configuration' },
  { name: `${CMD}-cfg-display`, description: 'agent_sys o11y: edit the display configuration' },
]
const TAB_OF: Record<string, Tab> = { all: 'all', general: 'general', detail: 'detail', kf: 'kernelforge', log: 'log' }

let vetoReason = ''

async function command($: any, name: string, args: string): Promise<any> {
  if (!isActive()) {
    const why = vetoReason || `this session is not the active main of fleet ${fleetRoot()} (AGENT_SYS_O11Y=true or a first agent_sys skill use makes it one; the display follows within one trigger tick)`
    return { text: `/${CMD}: the o11y display is inactive: ${why}`, isTextDisplayOnly: true }
  }
  const a = (args ?? '').trim()
  const sub = name.slice(CMD.length + 1)
  if (name === CMD) await openMenu($)
  else if (TAB_OF[sub]) await showTab($, TAB_OF[sub])
  else if (sub === 'mode') await setMode($, a)
  else if (sub === 'band') setBand($, a)
  else if (sub.startsWith('cfg-')) await openForm($, fleetRoot()!, sub.slice(4) as 'trigger' | 'analysis' | 'display')
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
