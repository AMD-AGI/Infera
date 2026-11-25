// Bundle each mod into its plugin's single hooks module. Not minified and platform-neutral: the engine's
// validator reads `$.noun.method` and `on("<event>")` literally, and a hooks module gets no Node APIs
// (pattern from claude-dashboard, MIT). The protocol layer and its JSON Schemas are bundled into both.
//
// Staged, because a running session hot-reloads plugins/*/mods/dist on save: the bundles are built in a
// copy of the package (.build/, git-ignored, reused, never cleaned by this script), checked there with
// `claude plugin validate` (CLAUDE_BIN, else `claude` on PATH) and the mods tests, and only then installed
// into plugins/*/mods/dist, each by a rename. O11Y_BUILD_SKIP_VALIDATE=1 skips the validator when no
// claude is available (the tests still run).
import { build } from 'esbuild'
import { spawnSync } from 'node:child_process'
import { copyFileSync, existsSync, mkdirSync, readdirSync, renameSync, symlinkSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'

const HERE = resolve(import.meta.dirname)
const STAGE = join(HERE, '.build')
const common = { bundle: true, format: 'esm', platform: 'neutral', target: 'es2022', external: ['claude-code'], legalComments: 'none', logLevel: 'warning' }
const MODS = [
  ['agent-sys-o11y', 'plugins/agent-sys-o11y/mods/src/trigger/index.ts', 'plugins/agent-sys-o11y/mods/dist/trigger.js'],
  ['agent-sys-o11y-dash', 'plugins/agent-sys-o11y-dash/mods/src/display/index.ts', 'plugins/agent-sys-o11y-dash/mods/dist/display.js'],
]
const fail = (msg) => {
  console.error(`build: ${msg}; plugins/*/mods/dist left unchanged`)
  process.exit(1)
}

// 1. A copy of what the bundles and the tests read.
mkdirSync(STAGE, { recursive: true })
for (const d of ['plugins', 'protocol', 'tests/mods']) {
  mkdirSync(join(STAGE, d), { recursive: true })
  const r = spawnSync('cp', ['-a', '--', `${join(HERE, d)}/.`, join(STAGE, d)]) // merges into the reused copy
  if (r.status !== 0) fail(`staging ${d}: ${r.stderr}`)
}
copyFileSync(join(HERE, 'package.json'), join(STAGE, 'package.json'))
if (!existsSync(join(STAGE, 'node_modules'))) symlinkSync(join(HERE, 'node_modules'), join(STAGE, 'node_modules'))

// 2. Build there.
for (const [, entry, dist] of MODS) await build({ ...common, absWorkingDir: STAGE, entryPoints: [entry], outfile: dist })

// 3. Check there: the engine's validator, then the mods tests (live ones skip without their variables).
if (process.env.O11Y_BUILD_SKIP_VALIDATE !== '1') {
  for (const [plugin] of MODS) {
    const r = spawnSync(process.env.CLAUDE_BIN || 'claude', ['plugin', 'validate', join(STAGE, 'plugins', plugin)], { encoding: 'utf8' })
    if (r.error) fail(`claude not runnable (${r.error.message}); set CLAUDE_BIN, or O11Y_BUILD_SKIP_VALIDATE=1`)
    if (!/Validation passed/.test(r.stdout)) fail(`${plugin} fails claude plugin validate:\n${r.stdout}${r.stderr}`)
  }
}
const env = Object.fromEntries(Object.entries(process.env).filter(([k]) => !k.startsWith('O11Y_LIVE')))
const tests = readdirSync(join(STAGE, 'tests/mods')).filter((n) => n.endsWith('.test.ts')).map((n) => `tests/mods/${n}`)
const t = spawnSync(process.execPath, ['--test', ...tests], { cwd: STAGE, env, encoding: 'utf8' })
if (t.status !== 0) fail(`mods tests fail on the staged bundles:\n${t.stdout.split('\n').filter((l) => /^(✖|ℹ (pass|fail))/.test(l)).join('\n')}`)

// 4. Install: each bundle replaces the live one by a rename in its own directory.
for (const [, , dist] of MODS) {
  const to = join(HERE, dist)
  const tmp = `${to}.tmp.${process.pid}`
  mkdirSync(dirname(to), { recursive: true })
  copyFileSync(join(STAGE, dist), tmp)
  renameSync(tmp, to)
  console.log(`build: installed ${dist}`)
}
