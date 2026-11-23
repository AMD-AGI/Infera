// Unit tests of the mods' protocol layer: locations (drift with env_mgr's prefix rule and the Python slug),
// schema defaults, the validator, sanitisation and the detached-spawn wrapper. Run: npm test
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { execFileSync, spawnSync } from 'node:child_process'
import { existsSync, mkdtempSync, readFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { SCHEMAS, configDirOf, detachArgv, fleetDirOf, prefixOf, sanitize, slug, validate, withDefaults } from '../../protocol/ts/index.ts'

const HERE = dirname(fileURLToPath(import.meta.url))
const O11Y = join(HERE, '..', '..')
const REPO = join(O11Y, '..', '..')

test('slug matches the shared test vectors (drift with o11y_core.fleet.slug)', async () => {
  const vec = JSON.parse(readFileSync(join(O11Y, 'protocol', 'test_vectors', 'slug.json'), 'utf8'))
  assert.ok(vec.cases.length > 0)
  for (const c of vec.cases) assert.equal(await slug(c.in), c.out, c.in)
})

test('fleet root: explicit dir, else prefix/state/o11y/fleets/<slug(launch dir)>', async () => {
  assert.equal(await fleetDirOf({ AGENT_SYS_O11Y_FLEET_DIR: '/x/f' }, '/p'), '/x/f')
  assert.equal(await fleetDirOf({ HOME: '/home/u' }, '/home/u/proj'), '/home/u/.infera_agent_sys/state/o11y/fleets/-home-u-proj')
  assert.equal(await fleetDirOf({ HOME: '/home/u', AGENT_SYS_HOME: '/opt/as' }, '/p'), '/opt/as/state/o11y/fleets/-p')
  assert.equal(prefixOf({ HOME: '/h', AGENT_SYS_HOME: 'relative' }), '/h/.infera_agent_sys') // must be absolute
  assert.equal(configDirOf({ HOME: '/h' }), '/h/.claude')
  assert.equal(configDirOf({ HOME: '/h', CLAUDE_CONFIG_DIR: '/c/' }), '/c')
})

test('prefix rule matches env_mgr.prefix.Prefix.resolve (drift)', (t) => {
  const py = spawnSync('python3', ['-c', 'import os,sys; sys.path.insert(0, sys.argv[1]); from agent_sys.env_mgr.prefix import Prefix; print(Prefix.resolve(os.environ).root)', REPO], {
    env: { PATH: process.env.PATH, HOME: '/home/drift', AGENT_SYS_HOME: '/opt/drift/as' },
    encoding: 'utf8',
  })
  if (py.status !== 0) return t.skip(`env_mgr not importable: ${py.stderr.split('\n').slice(-2).join(' ')}`)
  assert.equal(prefixOf({ HOME: '/home/drift', AGENT_SYS_HOME: '/opt/drift/as' }), py.stdout.trim())
  const py2 = spawnSync('python3', ['-c', 'import os,sys; sys.path.insert(0, sys.argv[1]); from agent_sys.env_mgr.prefix import Prefix; print(Prefix.resolve(os.environ).root)', REPO], {
    env: { PATH: process.env.PATH, HOME: '/home/drift' },
    encoding: 'utf8',
  })
  assert.equal(prefixOf({ HOME: '/home/drift' }), py2.stdout.trim())
})

test('schema defaults make valid configs (TS defaults vs schema defaults)', () => {
  for (const name of ['config.trigger', 'config.analysis', 'config.display']) {
    const doc = withDefaults(SCHEMAS[name], { v: 1 })
    assert.deepEqual(validate(SCHEMAS[name], doc), [], name)
  }
})

test('validator reports range, type and unknown keys', () => {
  const s = SCHEMAS['config.trigger']
  assert.match(validate(s, { v: 1, kick: { interval_s: 5 } }).join(), /interval_s: must be >= 60/)
  assert.match(validate(s, { v: 1, kick: { interval_s: 'x' } }).join(), /expected number/)
  assert.match(validate(s, { v: 1, kikc: {} }).join(), /kikc: unknown key/)
})

test('sanitize strips control characters and line breaks and caps length', () => {
  assert.equal(sanitize('a\nb\r\tc\u0007d e', 100), 'a b c d e')
  assert.equal(sanitize('x'.repeat(10), 5), 'xxxx…')
})

test('detached spawn passes every value as an argument (no interpolation)', () => {
  const d = mkdtempSync(join(tmpdir(), 'o11y-mods-'))
  const log = join(d, 'we"ird $(touch pwned) log')
  const out = join(d, 'out.txt')
  const argv = detachArgv(log, ['sh', '-c', 'echo "$1" > "$2"; echo to-log', 'sh', '$(touch pwned2)', out])
  execFileSync(argv[0], argv.slice(1), { cwd: d })
  for (let i = 0; i < 50 && !existsSync(out); i++) execFileSync('sleep', ['0.1'])
  assert.equal(readFileSync(out, 'utf8'), '$(touch pwned2)\n')
  for (let i = 0; i < 50 && !readFileSync(log, { encoding: 'utf8', flag: 'a+' }).includes('to-log'); i++) execFileSync('sleep', ['0.1'])
  assert.ok(readFileSync(log, 'utf8').includes('to-log'))
  assert.ok(!existsSync(join(d, 'pwned')) && !existsSync(join(d, 'pwned2')))
})

test('committed mods/dist match a fresh build of mods/src', async () => {
  const { build } = await import('esbuild')
  const common = { bundle: true, format: 'esm' as const, platform: 'neutral' as const, target: 'es2022', external: ['claude-code'], legalComments: 'none' as const, write: false }
  for (const [entry, dist] of [
    ['plugins/agent-sys-o11y/mods/src/trigger/index.ts', 'plugins/agent-sys-o11y/mods/dist/trigger.js'],
    ['plugins/agent-sys-o11y-dash/mods/src/display/index.ts', 'plugins/agent-sys-o11y-dash/mods/dist/display.js'],
  ]) {
    const r = await build({ ...common, absWorkingDir: O11Y, entryPoints: [entry], outfile: dist })
    assert.equal(r.outputFiles![0].text, readFileSync(join(O11Y, dist), 'utf8'), `${dist} is stale: run npm run build`)
  }
})
