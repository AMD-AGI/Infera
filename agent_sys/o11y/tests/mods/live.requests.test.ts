// Live acceptance criteria 7 and 6 (spec §12; design §12 map), mods side: with N=10, about 200 main-session
// tool calls including failed calls, parallel batches and an interrupted turn → one analysis request per
// threshold crossed, every request served; and an idle stretch of trigger ticks makes no model request.
// Long (about 15-25 min, with analysis jobs calling the model): runs only with O11Y_LIVE_LONG=1 besides the
// variables of live.test.ts. The jobs' own clause (no history window given to the model twice) is tested
// with the jobs (tests/analysis).
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { existsSync, readdirSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { makeRun, readJson, skipReason, sleep, waitFor } from './live_harness.ts'

const skip = skipReason() || (process.env.O11Y_LIVE_LONG === '1' ? false : 'O11Y_LIVE_LONG=1 not set')
const ROUNDS = 10

// Main-session transcript lines (no subagents).
function lines(cfg: string, sid: string): any[] {
  const projects = join(cfg, 'projects')
  for (const d of existsSync(projects) ? readdirSync(projects) : []) {
    const p = join(projects, d, `${sid}.jsonl`)
    if (existsSync(p)) return readFileSync(p, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l)).filter((x) => !x.isSidechain)
  }
  return []
}

// Tool calls that ran: a tool_result for a tool_use, not a rejection by the interrupt.
function toolCallsRan(cfg: string, sid: string): number {
  return lines(cfg, sid)
    .flatMap((x) => (x.type === 'user' && Array.isArray(x.message?.content) ? x.message.content : []))
    .filter((b: any) => b.type === 'tool_result' && !/interrupted by user|doesn't want to proceed/i.test(JSON.stringify(b.content)))
    .length
}

test('live: criteria 7 and 6 (mods side) — requests per threshold over ~200 tool calls, all served; idle ticks make no model request', { skip, timeout: 45 * 60_000 }, async (t) => {
  const never = 1_000_000
  const run = makeRun({ trigger: { v: 1, kick: { interval_s: 86_400 }, refresh: { every_n_tool_calls: 10, every_n_turns: never }, log: { every_n_tool_calls: 20, every_n_turns: never, slow_interval_s: 86_400 } } })
  t.diagnostic(`run directory ${run.dir}`)
  t.after(run.kill)
  const s = run.start()
  const sid = await waitFor('the registry to list the main', () => run.registry()?.main_session_id, 90_000)

  await t.test('criterion 6: three idle registry ticks make no model request', async () => {
    const requests = () => lines(run.cfg, sid).filter((x) => x.type === 'assistant').length
    const before = requests()
    await sleep(100_000)
    assert.equal(requests(), before)
  })

  await t.test('criterion 7: one request per threshold crossed, every request served', async () => {
    for (let r = 1; r <= ROUNDS; r++) {
      await waitFor(`idle before round ${r}`, () => run.liveStatus(s.pid) === 'idle', 300_000)
      run.type(s.name, `Round ${r}. Make exactly 20 Bash tool calls in 4 messages of 5 parallel calls each; call k (k = 1..20) runs ` +
        '`python3 -c "import sys; print(K); sys.exit(K % 7 == 0)"` with K replaced by k (K = 7 and 14 fail on purpose: carry on). Then reply DONE.')
      await waitFor(`round ${r} to start`, () => run.liveStatus(s.pid) === 'busy', 60_000, 200)
      if (r === 3) {
        await sleep(6_000)
        run.keys(s.name, 'Escape') // the interrupted turn
      }
    }
    await waitFor('idle after the rounds', () => run.liveStatus(s.pid) === 'idle', 300_000)
    await sleep(35_000) // the trigger's last tick and the requests it schedules
    const st = readJson(join(run.fleet, 'state', 'trigger.json'))
    const ran = toolCallsRan(run.cfg, sid)
    t.diagnostic(`tool calls: counted by the trigger ${st.counts.tool_calls}, ran per transcript ${ran}`)
    assert.ok(ran >= 150, `enough tool calls ran (${ran})`)
    assert.equal(st.counts.tool_calls, ran, 'the trigger counted every call that ran')
    const req = readJson(join(run.fleet, 'requests', 'analysis.json'))
    assert.equal(req.extract.seq, Math.floor(ran / 10))
    assert.equal(req.facts.seq, Math.floor(ran / 10))
    assert.equal(req.log.seq, Math.floor(ran / 20))
    for (const k of ['extract', 'facts', 'log']) {
      await waitFor(`${k} served up to ${req[k].seq}`, () => readJson(join(run.fleet, 'state', k, 'served.json'))?.seq === req[k].seq, 600_000, 5000)
    }
  })
})
