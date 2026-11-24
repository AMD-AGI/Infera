// Detached one-shot processes of the mods (the analysis jobs, the collector sweep) and the rotation of their
// shared log. Standalone: no imports.

// Default cap of a rotating append-only log (config.trigger rotate_mb; the Python store.ROTATE_BYTES twin).
export const ROTATE_BYTES = 100 * 1024 * 1024
// A detached one-shot: `sh -c` only redirects stdio; every value, the log path and cap included, is an argument.
// Rotation (design §4): a log past the cap is renamed to `<log>.1` (replacing the previous one) before the job
// appends, so the log and its `.1` stay within about twice the cap.
export const DETACH_SCRIPT =
  'log=$1; cap=$2; shift 2; [ -f "$log" ] && [ "$(wc -c <"$log")" -gt "$cap" ] && mv -f "$log" "$log.1"; ' +
  'setsid -f "$@" >>"$log" 2>&1 </dev/null'
export const detachArgv = (log: string, argv: string[], capBytes: number = ROTATE_BYTES) =>
  ['sh', '-c', DETACH_SCRIPT, 'sh', log, String(Math.max(1, Math.floor(capBytes) || ROTATE_BYTES)), ...argv]
