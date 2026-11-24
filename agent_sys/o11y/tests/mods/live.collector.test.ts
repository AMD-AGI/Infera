// Live acceptance criterion 19, collector clause (spec §12; design §12 map, §14 q2): two fleet sessions
// started through `o11y-launch --otel` share one collector. It stays while another fleet session is alive,
// survives a SIGKILL of the last session (nothing has swept yet), and is stopped by the next sweep; nothing
// else of the package remains. OTel goes to a loopback stand-in, never the user's endpoint; the collector is
// a test-only container (exact name, removed at the end). Needs, besides the variables of live.test.ts,
// docker and the image otel/opentelemetry-collector-contrib:latest locally.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { execFileSync, spawn } from 'node:child_process'
import { mkdirSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { makeRun, skipReason, sleep, waitFor } from './live_harness.ts'

const IMAGE = 'otel/opentelemetry-collector-contrib:latest'
const docker = (...a: string[]) => execFileSync('docker', a, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] })
function dockerReason(): string | false {
  try {
    docker('image', 'inspect', IMAGE)
    return false
  } catch {
    return `docker or the image ${IMAGE} not available`
  }
}
const skip = skipReason() || dockerReason()

// A loopback OTLP/HTTP stand-in that answers 200 to everything.
const STANDIN = `import http.server,sys
class H(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get('content-length') or 0)); self.send_response(200); self.end_headers()
    def log_message(self, *a): pass
http.server.ThreadingHTTPServer(('127.0.0.1', int(sys.argv[1])), H).serve_forever()
`

test('live: criterion 19 (collector clause) — kept while a fleet session lives, stopped by the sweep after the last one is killed', { skip, timeout: 15 * 60_000 }, async (t) => {
  const run = makeRun()
  t.diagnostic(`run directory ${run.dir}`)
  const uid = process.getuid!()
  const suffix = `live${process.pid}`
  const name = `infera-agent-sys-o11y-otelcol-${uid}-${suffix}`
  const base = 41000 + (process.pid % 500) * 4
  const up = base + 100
  const standin = spawn('python3', ['-I', '-c', STANDIN, String(up)], { stdio: 'ignore', env: run.env })
  t.after(() => {
    run.kill()
    standin.kill()
    assert.ok(name.startsWith(`infera-agent-sys-o11y-otelcol-${uid}-live`))
    for (const a of ['stop', 'rm']) try { docker(a, name) } catch {}
  })
  writeFileSync(join(run.cfg, 'settings.json'), JSON.stringify({ env: {
    CLAUDE_CODE_ENABLE_TELEMETRY: '1', OTEL_METRICS_EXPORTER: 'otlp', OTEL_LOGS_EXPORTER: 'otlp', OTEL_EXPORTER_OTLP_PROTOCOL: 'http/protobuf',
    OTEL_EXPORTER_OTLP_ENDPOINT: `http://127.0.0.1:${up}`, OTEL_METRIC_EXPORT_INTERVAL: '5000', OTEL_LOGS_EXPORT_INTERVAL: '2000',
  } }))
  const otelDir = join(run.dir, 'otel')
  mkdirSync(otelDir, { mode: 0o700 })
  const env = { CLAUDE_CODE_ENABLE_TELEMETRY: null, AGENT_SYS_O11Y_OTEL_DIR: otelDir, AGENT_SYS_O11Y_OTEL_PORT_BASE: String(base),
    AGENT_SYS_O11Y_OTEL_CONTAINER: suffix, AGENT_SYS_O11Y_OTEL_IMAGE: IMAGE }
  const launch = join(run.plugins, 'agent-sys-o11y', 'bin', 'o11y-launch')
  const running = () => {
    try {
      return docker('inspect', '-f', '{{.State.Running}}', name).trim() === 'true'
    } catch {
      return false
    }
  }
  const ours = () => [...run.procs()].filter(([, c]) => !/o11y-run|o11y_job|python3 -I -c import http/.test(c))

  const a = run.start({ otel: true, env })
  await waitFor('the collector', running, 120_000, 1000)
  await waitFor('session A as main', () => run.registry()?.sessions?.some((s: any) => s.pid === a.pid && s.active), 90_000)
  const b = run.start({ otel: true, env: { ...env, AGENT_SYS_O11Y: 'child' }, args: ['--name', 'b'] }) // a child: its trigger stays dormant
  await waitFor('session B listed', () => run.sidOf(b.pid), 60_000)

  await t.test('after /exit of one fleet session the collector stays while the other lives', async () => {
    await waitFor('A idle', () => run.liveStatus(a.pid) === 'idle', 120_000)
    run.type(a.name, '/exit')
    await waitFor('A to exit', () => !run.procs().has(a.pid), 30_000)
    await sleep(35_000) // A's clean end ran a sweep; B is a live client
    assert.equal(running(), true)
  })

  await t.test('after SIGKILL of the last session the collector stays until the next sweep, which stops it; nothing else remains', async () => {
    process.kill(b.pid, 'SIGKILL')
    await sleep(20_000)
    assert.deepEqual(ours(), [], 'no process of the package')
    assert.equal(running(), true, 'nothing has swept yet (design §14 q2)')
    execFileSync(launch, ['--sweep'], { env: { ...run.env, AGENT_SYS_O11Y_OTEL_DIR: otelDir, AGENT_SYS_O11Y_OTEL_PORT_BASE: String(base), AGENT_SYS_O11Y_OTEL_CONTAINER: suffix, AGENT_SYS_O11Y_OTEL_IMAGE: IMAGE } })
    await waitFor('the collector to stop', () => !running(), 60_000)
  })
})
