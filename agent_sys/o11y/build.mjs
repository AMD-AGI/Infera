// Bundle each mod into its plugin's single hooks module. Not minified and platform-neutral: the engine's
// validator reads `$.noun.method` and `on("<event>")` literally, and a hooks module gets no Node APIs
// (pattern from claude-dashboard, MIT). The protocol layer and its JSON Schemas are bundled into both.
import { build } from 'esbuild'

const common = { bundle: true, format: 'esm', platform: 'neutral', target: 'es2022', external: ['claude-code'], legalComments: 'none', logLevel: 'info' }

await build({ ...common, entryPoints: ['plugins/agent-sys-o11y/mods/src/trigger/index.ts'], outfile: 'plugins/agent-sys-o11y/mods/dist/trigger.js' })
await build({ ...common, entryPoints: ['plugins/agent-sys-o11y-dash/mods/src/display/index.ts'], outfile: 'plugins/agent-sys-o11y-dash/mods/dist/display.js' })
